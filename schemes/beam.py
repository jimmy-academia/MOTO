import asyncio
import uuid
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict, Optional

# MyOpto Imports
from myopto.trace.bundle import FunModule
from myopto.trace.runtime import RuntimeTracer, llm, msg
from myopto.optimizers.structure_editor import StructureEditor
from myopto.optimizers import OptoPrimeLocal

@dataclass
class Candidate:
    """
    Represents a node in the Beam Search.
    Holds the executable Bundle (FunModule) which encapsulates the code.
    """
    bundle: FunModule
    
    # Performance Metrics
    score: float = 0.0
    feedback: Any = None
    trace_ir: Dict = field(default_factory=dict)
    
    # Execution Artifacts (Captured for backward pass)
    output_node: Any = None
    live_parameters: List[Any] = field(default_factory=list) 

    # Metadata
    id: str = ""
    parent_id: Optional[str] = None
    mutation_type: str = "init"

    def __lt__(self, other):
        return self.score < other.score

    @property
    def code(self) -> str:
        """Helper to access the source code from the bundle."""
        if self.bundle.parameter is not None:
            return self.bundle.parameter._data
        return self.bundle.info.get("source", "")


class BeamInnerLoopEngine:
    def __init__(self, editor: StructureEditor, beam_width: int = 3, verbose: bool = False):
        self.editor = editor
        self.beam_width = beam_width
        self.verbose = verbose

    async def run(
        self, 
        seed_code: str, 
        feed_fn: Any, 
        context: str, 
        batch: List[Tuple[Any, Any]], 
        iterations: int = 3
    ):
        # 1. Initialization: Bootstrap the first Bundle from the seed code string
        # We assume seed_code is a valid python string defining a @bundle decorated function.
        # We load it using the editor to get the function object.
        initial_func = self.editor.load_function(seed_code, extra_globals={"llm": llm, "msg": msg})
        
        # Ensure it is a FunModule (it should be if decorated with @bundle)
        if not isinstance(initial_func, FunModule):
            # Fallback: if user didn't decorate it, we wrap it? 
            # But we need trainable=True for code injection.
            # Assuming the user provided code *has* @bundle(trainable=True).
            raise ValueError("Seed code must define a function decorated with @bundle(trainable=True)")

        population = [Candidate(bundle=initial_func, id=self._id(), mutation_type="seed")]
        trajectory = []

        for it in range(iterations):
            if self.verbose:
                print(f"--- Iteration {it} (Population: {len(population)}) ---")

            # 2. Evaluation
            await self._evaluate_population(population, feed_fn, context, batch)

            # 3. Selection
            valid = [c for c in population if c.score > -1]
            valid.sort(key=lambda c: c.score, reverse=True)
            survivors = valid[:self.beam_width]

            # Record Best
            best = survivors[0] if survivors else None
            trajectory.append({
                "it": it,
                "best_score": best.score if best else 0,
                "best_code": best.code if best else "",
            })
            
            if self.verbose and best:
                print(f"Best: {best.score:.2f} ({best.mutation_type})")

            # 4. Expansion
            if it < iterations - 1:
                population = await self._expand_survivors(survivors)

        return (population[0] if population else None), trajectory

    # --- Core Mechanics ---

    async def _evaluate_population(self, population, feed_fn, context, batch):
        tasks = [self._run_single_candidate(c, feed_fn, context, batch) for c in population]
        results = await asyncio.gather(*tasks)
        
        for cand, res in zip(population, results):
            cand.score = res["score"]
            cand.feedback = res["fb"]
            cand.trace_ir = res["trace"]
            cand.output_node = res["output_node"]
            cand.live_parameters = res["parameters"]

    async def _run_single_candidate(self, candidate, feed_fn, context, batch):
        try:
            # 1. Execute the Bundle
            # FunModule.forward handles compiling self.parameter (code) if needed
            x, y = batch[0]
            
            # We must use RuntimeTracer to capture the *live* prompt parameters
            with RuntimeTracer(domain="workflow", trainable_prompt_templates=True) as wf_tr:
                # Execute the bundle
                pred = await candidate.bundle(context, x)
            
            # 2. Capture Artifacts
            # wf_tr.parameters() returns the ParameterNodes for PROMPTS created during this run
            params = wf_tr.parameters() 
            out_node = getattr(pred, "node", None)

            # 3. Feedback
            signal = {"trace": wf_tr.to_ir(), "ground_truth": y}
            with RuntimeTracer(domain="feedback") as fb_tr:
                fb_result = await feed_fn(pred, signal)
            
            return {
                "score": self._parse_score(fb_result), 
                "fb": fb_result, 
                "trace": wf_tr.to_ir(),
                "output_node": out_node,
                "parameters": params
            }

        except Exception as e:
            if self.verbose: print(f"Crash {candidate.id}: {e}")
            return {
                "score": -1.0, "fb": str(e), "trace": {}, 
                "output_node": None, "parameters": []
            }

    async def _expand_survivors(self, survivors: List[Candidate]) -> List[Candidate]:
        next_gen = []
        for parent in survivors:
            # Elitism: Clone parent bundle
            next_gen.append(Candidate(
                bundle=parent.bundle.detach(), 
                parent_id=parent.id, 
                score=parent.score, 
                mutation_type="elite"
            ))

            # Child A: Structure Edit
            child_struct = await self._mutate_structure(parent)
            if child_struct:
                next_gen.append(child_struct)

            # Child B: Prompt Edit
            child_prompt = await self._mutate_prompts(parent)
            if child_prompt:
                next_gen.append(child_prompt)
                
        return next_gen

    # --- Mutators ---

    async def _mutate_structure(self, parent: Candidate) -> Optional[Candidate]:
        if not parent.trace_ir:
            return None

        # 1. Rewrite Logic
        res = self.editor.rewrite_function(
            func_or_code=parent.code,
            ir=parent.trace_ir,
            feedback=f"FIX LOGIC: {parent.feedback}",
        )
        
        if not res.ok:
            return None

        # 2. Create Child Bundle
        # We clone the parent bundle, then inject the new code
        child_bundle = parent.bundle.detach()
        if child_bundle.parameter is not None:
            child_bundle.parameter._data = res.code
        else:
            # If parent wasn't trainable, we might be stuck unless we reload entirely.
            # Assuming trainable=True as per initialization.
            return None

        return Candidate(
            bundle=child_bundle,
            id=self._id(),
            parent_id=parent.id, 
            mutation_type="structure"
        )

    async def _mutate_prompts(self, parent: Candidate) -> Optional[Candidate]:
        """
        Optimizes prompts using OptoPrime, patches the code, and creates a new Bundle.
        """
        if not parent.output_node or not parent.live_parameters:
            return None

        # 1. Optimize (In-Memory update of live_parameters)
        opt = OptoPrimeLocal(parent.live_parameters)
        opt.zero_feedback()
        try:
            # OptoPrime updates p.data for each p in live_parameters
            opt.backward(parent.output_node, parent.feedback, visualize=False)
            opt.step(mode="per_param")
        except Exception:
            return None

        # 2. Patch the Source Code
        # We read the UPDATED values from the live parameters and replace them in the code string.
        current_code = parent.code
        new_code = current_code
        changed = False
        
        for p in parent.live_parameters:
            # p.initial_value is what was in the code before this step
            # p.data is the new optimized string
            if p.data != p.initial_value:
                if p.initial_value in new_code:
                    new_code = new_code.replace(p.initial_value, p.data)
                    changed = True
        
        if not changed:
            return None

        # 3. Create Child Bundle with patched code
        child_bundle = parent.bundle.detach()
        if child_bundle.parameter is not None:
            child_bundle.parameter._data = new_code
        else:
            return None

        return Candidate(
            bundle=child_bundle,
            id=self._id(),
            parent_id=parent.id,
            mutation_type="prompt"
        )

    # --- Utilities ---

    def _id(self):
        return str(uuid.uuid4())[:8]

    def _parse_score(self, fb_result: Any) -> float:
        if isinstance(fb_result, (int, float)):
            return float(fb_result)
        if isinstance(fb_result, dict):
            return float(fb_result.get("score", 0.0))
        text = str(fb_result).strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else 0.0
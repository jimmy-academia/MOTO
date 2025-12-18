import asyncio
import uuid
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict, Optional

from myopto.trace.bundle import Bundle
from myopto.trace.runtime import RuntimeTracer, llm, msg
from myopto.optimizers.structure_editor import StructureEditor
from myopto.optimizers import OptoPrimeLocal

@dataclass
class Candidate:
    code: str
    score: float = 0.0
    feedback: Any = None
    trace_ir: Dict = field(default_factory=dict)
    id: str = ""
    parent_id: Optional[str] = None
    mutation_type: str = "init"

    def __lt__(self, other):
        return self.score < other.score

class BeamInnerLoopEngine:
    def __init__(self, editor: StructureEditor, beam_width: int = 3, verbose: bool = False):
        self.editor = editor
        self.beam_width = beam_width
        self.verbose = verbose

    async def run(self, seed_code: str, feed_fn: Bundle, context: str, batch: List[Tuple[Any, Any]], iterations: int = 3):
        population = [Candidate(code=seed_code, id=self._id(), mutation_type="seed")]
        trajectory = []

        for it in range(iterations):
            if self.verbose:
                print(f"--- Iteration {it} (Population: {len(population)}) ---")

            # 1. Evaluation
            await self._evaluate_population(population, feed_fn, context, batch)

            # 2. Selection
            valid_candidates = [c for c in population if c.score > -1]
            valid_candidates.sort(key=lambda c: c.score, reverse=True)
            survivors = valid_candidates[:self.beam_width]

            # 3. Trajectory
            best_of_gen = survivors[0] if survivors else None
            trajectory.append({
                "it": it,
                "best_score": best_of_gen.score if best_of_gen else 0,
                "best_code": best_of_gen.code if best_of_gen else ""
            })
            
            if self.verbose and best_of_gen:
                print(f"Best: {best_of_gen.score:.2f} ({best_of_gen.mutation_type})")

            # 4. Expansion
            if it < iterations - 1:
                population = await self._expand_survivors(survivors)

        return (population[0] if population else None), trajectory

    # --- Helpers ---

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

    async def _evaluate_population(self, population, feed_fn, context, batch):
        tasks = [self._run_single_candidate(c, feed_fn, context, batch) for c in population]
        results = await asyncio.gather(*tasks)
        
        for cand, res in zip(population, results):
            cand.score = res["score"]
            cand.feedback = res["fb"]
            cand.trace_ir = res["trace"]

    async def _run_single_candidate(self, candidate, feed_fn, context, batch):
        try:
            # Load code into a unique function instance
            func = self.editor.load_function(candidate.code, func_name=None, extra_globals={"llm": llm, "msg": msg})
            x, y = batch[0]

            with RuntimeTracer(domain="workflow") as wf_tr:
                pred = await func(context, x)
            
            signal = {"trace": wf_tr.to_ir(), "ground_truth": y}
            with RuntimeTracer(domain="feedback") as fb_tr:
                fb_result = await feed_fn(pred, signal)
            
            score = self._parse_score(fb_result)
            return {"score": score, "fb": fb_result, "trace": wf_tr.to_ir()}

        except Exception as e:
            if self.verbose: print(f"Crash {candidate.id}: {e}")
            return {"score": -1.0, "fb": str(e), "trace": {"_error": str(e)}}

    async def _expand_survivors(self, survivors):
        next_gen = []
        for parent in survivors:
            # Elitism
            next_gen.append(Candidate(
                code=parent.code, parent_id=parent.id, 
                score=parent.score, mutation_type="elite"
            ))

            # Mutation A: Structure
            code_struct = await self._mutate_structure(parent)
            if code_struct:
                next_gen.append(Candidate(
                    code=code_struct, parent_id=parent.id, mutation_type="structure"
                ))

            # Mutation B: Prompt
            code_prompt = await self._mutate_prompts(parent)
            if code_prompt:
                next_gen.append(Candidate(
                    code=code_prompt, parent_id=parent.id, mutation_type="prompt"
                ))
        return next_gen

    async def _mutate_structure(self, parent: Candidate) -> Optional[str]:
        if not parent.trace_ir:
            return None

        # Use trace to fix logic/structure
        res = self.editor.rewrite_function(
            func_or_code=parent.code,
            ir=parent.trace_ir,
            feedback=f"FIX LOGIC: {parent.feedback}",
        )
        return res.code if res.ok else None

    async def _mutate_prompts(self, parent: Candidate) -> Optional[str]:
        # 1. Load function to access in-memory ParameterNodes
        try:
            func = self.editor.load_function(parent.code, extra_globals={"llm": llm, "msg": msg})
        except:
            return None

        params = func.parameters()
        if not params:
            return None

        # 2. Optimize parameters using textual feedback
        opt = OptoPrimeLocal(params)
        opt.step(feedback=str(parent.feedback))

        # 3. Patch source code (Serialization)
        new_code = parent.code
        changed = False
        for p in params:
            if p.data != p.initial_value:
                # Replace old prompt string with new one
                if p.initial_value in new_code:
                    new_code = new_code.replace(p.initial_value, p.data)
                    changed = True
        
        return new_code if changed else None
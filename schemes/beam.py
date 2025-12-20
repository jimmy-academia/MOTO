# schemes/beam.py
"""
Beam Search Inner Loop Engine for Test-Time Adaptation.

This module implements a beam search optimization that maintains a population
of candidate workflows, evaluates them, and expands the best ones through
structure and prompt mutations.

Key Design: Uses plain functions + code strings instead of FunModule bundles.
Tracing is handled by RuntimeTracer which captures llm() calls and creates
trainable ParameterNodes for prompt templates.
"""
import asyncio
import uuid
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict, Optional, Callable

from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto.optimizers.structure_editor import StructureEditor
from myopto.optimizers import OptoPrimeLocal

from utils.logs import logger


@dataclass
class Candidate:
    """
    Represents a node in the Beam Search.
    
    Holds an executable function and its source code. The function is a plain
    Python callable (not a FunModule). Tracing happens via RuntimeTracer context.
    """
    fn: Callable                # The executable function
    code: str                   # The source code string
    
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


# Alias for backward compatibility
BeamCandidate = Candidate


class BeamInnerLoopEngine:
    """
    Beam Search optimization engine for test-time workflow adaptation.
    
    Maintains a population of candidate workflows, evaluates them using a
    feedback function, selects the best performers, and expands them through
    structure edits (code rewrites) and prompt edits (template optimization).
    """
    
    def __init__(
        self, 
        editor: StructureEditor, 
        beam_width: int = 3, 
        verbose: bool = False,
        max_iterations: int = None  # Added for VETO compatibility (ignored, use iterations in run())
    ):
        self.editor = editor
        self.beam_width = beam_width
        self.verbose = verbose
        # max_iterations stored but not used here - iterations passed to run() instead

    async def run(
        self, 
        seed_code: str, 
        feed_fn: Callable, 
        context: str, 
        batch: List[Tuple[Any, Any]], 
        iterations: int = 3
    ) -> Tuple[Optional[Candidate], List[Dict]]:
        """
        Run beam search optimization.
        
        Args:
            seed_code: Source code string for the initial workflow function
            feed_fn: Feedback function that scores predictions
            context: Context string passed to workflow
            batch: List of (input, ground_truth) tuples. ground_truth can be None.
            iterations: Number of optimization iterations
            
        Returns:
            (best_candidate, trajectory) - Best candidate found and optimization history
        """
        logger.info('[Beam] running beam search')

        # 1. Initialize: Load the seed function from code string
        initial_func = self.editor.load_function(
            seed_code, 
            extra_globals={"llm": llm, "msg": msg}
        )
        
        population = [Candidate(
            fn=initial_func, 
            code=seed_code, 
            id=self._id(), 
            mutation_type="seed"
        )]
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

            # 4. Expansion (skip on last iteration)
            if it < iterations - 1:
                population = await self._expand_survivors(survivors)

        return (population[0] if population else None), trajectory

    # --------------------------------------------------
    # Core Mechanics
    # --------------------------------------------------

    async def _evaluate_population(
        self, 
        population: List[Candidate], 
        feed_fn: Callable, 
        context: str, 
        batch: List[Tuple[Any, Any]]
    ) -> None:
        """Evaluate all candidates in parallel."""
        tasks = [
            self._run_single_candidate(c, feed_fn, context, batch) 
            for c in population
        ]
        results = await asyncio.gather(*tasks)
        
        for cand, res in zip(population, results):
            cand.score = res["score"]
            cand.feedback = res["fb"]
            cand.trace_ir = res["trace"]
            cand.output_node = res["output_node"]
            cand.live_parameters = res["parameters"]

    async def _run_single_candidate(
        self, 
        candidate: Candidate, 
        feed_fn: Callable, 
        context: str, 
        batch: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """
        Execute a single candidate and collect feedback.
        
        The workflow function is executed inside a RuntimeTracer context which:
        - Captures all llm() calls as traced nodes
        - Creates trainable ParameterNodes for prompt templates
        - Builds an IR (intermediate representation) of the execution
        """
        try:
            x, y = batch[0]
            
            # 1. Execute the workflow inside RuntimeTracer
            with RuntimeTracer(domain="workflow", trainable_prompt_templates=True) as wf_tr:
                pred = candidate.fn(context, x)
            
            # 2. Capture Artifacts
            params = wf_tr.parameters() 
            out_node = getattr(pred, "node", None) or wf_tr.output_node
            
            # Clean the prediction string
            pred_str = strip_trace_tags(str(pred)) if pred else ""

            # 3. Run Feedback function
            signal = {"trace": wf_tr.to_ir(), "ground_truth": y}
            with RuntimeTracer(domain="feedback", trainable_prompt_templates=False) as fb_tr:
                fb_result = feed_fn(pred_str, signal)
            
            return {
                "score": self._parse_score(fb_result), 
                "fb": fb_result, 
                "trace": wf_tr.to_ir(),
                "output_node": out_node,
                "parameters": params
            }

        except Exception as e:
            if self.verbose:
                print(f"Crash {candidate.id}: {e}")
            return {
                "score": -1.0, 
                "fb": str(e), 
                "trace": {}, 
                "output_node": None, 
                "parameters": []
            }

    async def _expand_survivors(self, survivors: List[Candidate]) -> List[Candidate]:
        """
        Expand surviving candidates into next generation.
        
        Creates children through:
        1. Structure edits (code rewrites via StructureEditor)
        2. Prompt edits (template optimization via OptoPrime)
        """
        next_gen = []
        
        for parent in survivors:
            # Keep parent
            next_gen.append(parent)
            
            # Try structure edit
            struct_child = await self._structure_edit(parent)
            if struct_child:
                next_gen.append(struct_child)
            
            # Try prompt edit
            prompt_child = await self._prompt_edit(parent)
            if prompt_child:
                next_gen.append(prompt_child)
        
        return next_gen

    async def _structure_edit(self, parent: Candidate) -> Optional[Candidate]:
        """
        Create a child candidate via structure edit.
        
        Uses the StructureEditor to rewrite the workflow code based on feedback.
        """
        if not parent.feedback:
            return None

        try:
            new_code = self.editor.edit(
                parent.code,
                feedback=str(parent.feedback),
            )
            
            if not new_code or new_code == parent.code:
                return None
            
            new_fn = self.editor.load_function(
                new_code, 
                extra_globals={"llm": llm, "msg": msg}
            )
            
            return Candidate(
                fn=new_fn,
                code=new_code,
                id=self._id(),
                parent_id=parent.id,
                mutation_type="structure"
            )
        except Exception as e:
            if self.verbose:
                print(f"Structure edit failed: {e}")
            return None

    async def _prompt_edit(self, parent: Candidate) -> Optional[Candidate]:
        """
        Create a child candidate via prompt optimization.
        
        Uses OptoPrimeLocal to update prompt templates, then patches the
        source code with the new prompts and reloads the function.
        """
        if not parent.output_node or not parent.live_parameters:
            return None

        # 1. Optimize prompts in-memory
        try:
            opt = OptoPrimeLocal(parent.live_parameters)
            opt.zero_feedback()
            opt.backward(parent.output_node, str(parent.feedback), visualize=False)
            opt.step(mode="per_param")
        except Exception as e:
            if self.verbose:
                print(f"Prompt optimization failed: {e}")
            return None

        # 2. Patch the source code with updated prompt values
        new_code = parent.code
        changed = False
        
        for p in parent.live_parameters:
            initial = getattr(p, 'initial_value', None) or getattr(p, '_initial_data', None)
            if initial and p.data != initial:
                if initial in new_code:
                    new_code = new_code.replace(initial, p.data)
                    changed = True
        
        if not changed:
            return None

        # 3. Load the new function from patched code
        try:
            new_fn = self.editor.load_function(
                new_code, 
                extra_globals={"llm": llm, "msg": msg}
            )
        except Exception as e:
            if self.verbose:
                print(f"Failed to load prompt-patched code: {e}")
            return None

        return Candidate(
            fn=new_fn,
            code=new_code,
            id=self._id(),
            parent_id=parent.id,
            mutation_type="prompt"
        )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _id(self) -> str:
        """Generate a short unique ID."""
        return str(uuid.uuid4())[:8]

    def _parse_score(self, fb_result: Any) -> float:
        """Extract a numeric score from feedback result."""
        if isinstance(fb_result, (int, float)):
            return float(fb_result)
        if isinstance(fb_result, dict):
            return float(fb_result.get("score", 0.0))
        # Try to parse from string
        text = str(fb_result).strip()
        match = re.search(r'"score"\s*:\s*([-+]?\d*\.?\d+)', text)
        if match:
            return float(match.group(1))
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else 0.0
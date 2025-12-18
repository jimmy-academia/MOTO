# schemes/ecwo.py
"""
Edge-Cloud Workflow Optimization (ECWO) Scheme.

This scheme implements test-time adaptation using beam search optimization.
For each query, it runs a beam search loop to adapt the seed workflow to
the specific problem/context.

Key Design: Uses plain functions instead of FunModule bundles. Tracing is
handled by RuntimeTracer which captures llm() calls during execution.
"""
import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto.optimizers.structure_editor import StructureEditor
from myopto.utils.llm_router import get_llm
from myopto.utils.usage import get_total_cost, reset_usage

from .base import BaseScheme
from .beam import BeamInnerLoopEngine

# -------------------------------------------------------------------------
# Default Code Templates
# -------------------------------------------------------------------------

SEED_WORKFLOW_CODE = """
def seed_workflow(context: str, problem: str) -> str:
    # A simple Chain-of-Thought starter
    plan = llm(f"Given context: {context}\\nProblem: {problem}\\nDraft a plan to solve this.", call_tag="plan")
    answer = llm(f"Execute this plan:\\n{plan}\\n\\nReturn the final answer.", call_tag="answer")
    return answer
""".lstrip()

FEEDBACK_WORKFLOW_CODE = """
def feedback_workflow(pred: str, signal: dict) -> str:
    # Extracts ground_truth from signal if available (training),
    # otherwise acts as a self-critic (inference).
    
    ground_truth = signal.get("ground_truth")
    trace = signal.get("trace", {})
    
    if ground_truth:
        # Supervised Mode (Training/Oracle): Strict comparison
        check = llm(
            f"Compare the prediction with the ground truth.\\n"
            f"Prediction: {pred}\\n"
            f"Ground Truth: {ground_truth}\\n\\n"
            f"Is the prediction correct? Return JSON {{'score': 1.0}} or {{'score': 0.0, 'critique': '...'}}.",
            call_tag="judge_gt"
        )
    else:
        # Unsupervised Mode (Edge/Inference): Self-Correction / Critique
        check = llm(
            f"Critique this answer for logical consistency and clarity.\\n"
            f"Answer: {pred}\\n\\n"
            f"Return JSON {{'score': <0.0-1.0>, 'critique': '...'}}.",
            call_tag="judge_self"
        )
    
    return check
""".lstrip()


class ECWOScheme(BaseScheme):
    """
    Edge-Cloud Workflow Optimization (ECWO) Scheme.
    
    Mode: Test-Time Adaptation / Inference-Only.
    Behavior: For each query, runs a Beam Search optimization loop to adapt 
              the seed workflow to the specific problem/context.
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Configuration
        self.beam_width = getattr(args, "beam_width", 3)
        self.iterations = getattr(args, "inner_loop_iters", 3)
        self.verbose = getattr(args, "verbose", False)
        
        # Code templates
        self.seed_code = SEED_WORKFLOW_CODE
        self.feedback_code = FEEDBACK_WORKFLOW_CODE
        
        # Structure editor for code manipulation
        self.editor = StructureEditor(
            llm=get_llm(role="optimizer"),
            max_tokens=getattr(args, "structure_max_tokens", 12000),
            require_call_tag=True,
            forbid_strip_trace_tags=True,
            forbid_imports=True,
            verbose=self.verbose,
        )
        
        # Load initial functions (plain callables, not FunModules)
        self.seed_fn = self._load(self.seed_code, "seed_workflow")
        self.feedback_fn = self._load(self.feedback_code, "feedback_workflow")

    def _load(self, code: str, fn_name: str) -> Callable:
        """
        Load code into a callable function.
        
        Uses StructureEditor.load_function which exec's the code and returns
        the named function. The function is a plain Python callable.
        """
        return self.editor.load_function(
            code, 
            fn_name, 
            extra_globals={"llm": llm, "msg": msg}
        )

    async def train_one_batch(
        self, 
        batch: List[dict], 
        calculate_score: Callable
    ) -> Dict[str, Any]:
        """
        Runs the Inner Loop (Adaptation) on a training batch.
        
        Since ECWO is an inference-time adaptation scheme, "training" here acts as 
        an evaluation of the seed's adaptability. We run the full beam search 
        for each training example and report the final score against ground truth.
        """
        # Unpack batch: BaseScheme.iter_batches yields tuple of columns
        if len(batch) == 3:
            contexts, questions, answers = batch
        else:
            questions, answers = batch
            contexts = [""] * len(questions)

        scores = []
        costs = []

        # Process batch sequentially
        for i in range(len(questions)):
            q, a, c = questions[i], answers[i], contexts[i]
            
            # Run Inference (Adaptation)
            # We treat 'a' (ground truth) as hidden from the inner loop,
            # only used for final evaluation here.
            pred, cost = await self.inference(q)
            
            # Evaluate using benchmark scorer
            try:
                score_result = calculate_score(a, pred)
                # Handle both (score,) and (score, extra) return formats
                score = score_result[0] if isinstance(score_result, tuple) else score_result
            except TypeError:
                # Some scorers have reversed arg order
                score_result = calculate_score(pred, a)
                score = score_result[0] if isinstance(score_result, tuple) else score_result
                
            scores.append(float(score))
            costs.append(cost)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_cost = sum(costs) / len(costs) if costs else 0.0
        
        return {
            "score": avg_score,
            "cost": avg_cost,
            "n_samples": len(questions)
        }

    async def inference(self, input_text: str) -> Tuple[str, float]:
        """
        Main entry point for Test-Time Adaptation.
        
        Args:
            input_text: The user query (can include context).
            
        Returns:
            (best_answer, cost_usd)
        """
        reset_usage()
        
        # 1. Setup Engine
        engine = BeamInnerLoopEngine(
            editor=self.editor,
            beam_width=self.beam_width,
            verbose=self.verbose
        )
        
        # 2. Run Beam Search
        # In inference mode, ground_truth is None.
        # The inner loop relies on self.feedback_fn (Self-Critic).
        best_cand, trajectory = await engine.run(
            seed_code=self.seed_code,
            feed_fn=self.feedback_fn,
            context="", 
            batch=[(input_text, None)],  # No ground truth available
            iterations=self.iterations
        )
        
        cost = float(get_total_cost())
        
        # 3. Extract Result
        if not best_cand:
            return "Failure: No valid candidate found.", cost
            
        # Execute the best candidate one final time to get clean output
        try:
            with RuntimeTracer(domain="inference_final") as tr:
                final_res = best_cand.fn("", input_text)  # Plain function call
            return strip_trace_tags(str(final_res)), cost
        except Exception as e:
            return f"Error executing best candidate: {e}", cost

    def save_model(self, epoch: Optional[int] = None) -> None:
        """Persist the base templates."""
        self.scheme_file.parent.mkdir(parents=True, exist_ok=True)
        code = (
            "# ECWO Scheme State\n"
            f"SEED_WORKFLOW_CODE = {repr(self.seed_code)}\n"
            f"FEEDBACK_WORKFLOW_CODE = {repr(self.feedback_code)}\n"
        )
        self.scheme_file.write_text(code, encoding="utf-8")
        
        if epoch is not None:
            snap = self.scheme_file.parent / f"scheme_epoch_{epoch}.py"
            snap.write_text(code, encoding="utf-8")

    def load(self, path: Path) -> bool:
        """Load state from disk."""
        if not path.exists():
            return False
        try:
            ns = {}
            exec(path.read_text("utf-8"), ns, ns)
            self.seed_code = ns.get("SEED_WORKFLOW_CODE", self.seed_code)
            self.feedback_code = ns.get("FEEDBACK_WORKFLOW_CODE", self.feedback_code)
            # Reload functions
            self.seed_fn = self._load(self.seed_code, "seed_workflow")
            self.feedback_fn = self._load(self.feedback_code, "feedback_workflow")
            return True
        except Exception:
            return False
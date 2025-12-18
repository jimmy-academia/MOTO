# schemes/ecwo.py
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from myopto.trace.bundle import FunModule, bundle
from myopto.trace.runtime import RuntimeTracer, llm, msg
from myopto.optimizers.structure_editor import StructureEditor
from myopto.utils.llm_router import get_llm
from myopto.utils.usage import get_total_cost, reset_usage

from .base import BaseScheme
from .beam import BeamInnerLoopEngine

# -------------------------------------------------------------------------
# Default Code Templates
# -------------------------------------------------------------------------

SEED_WORKFLOW_CODE = """
@bundle(trainable=True)
def seed_workflow(context: str, problem: str) -> str:
    # A simple Chain-of-Thought starter
    plan = llm(f"Given context: {context}\\nProblem: {problem}\\nDraft a plan to solve this.", call_tag="plan")
    answer = llm(f"Execute this plan:\\n{plan}\\n\\nReturn the final answer.", call_tag="answer")
    return answer
""".lstrip()

FEEDBACK_WORKFLOW_CODE = """
@bundle(trainable=True)
def feedback_workflow(pred: str, signal: dict) -> dict:
    # Extracts ground_truth from signal if available (training),
    # otherwise acts as a self-critic (inference).
    
    ground_truth = signal.get("ground_truth")
    trace = signal.get("trace", {})
    
    if ground_truth:
        # Supervised Mode (Training/Oracle): Strict comparison
        # Note: In pure Edge setting, this branch is usually not taken inside the inner loop
        # unless we are simulating an 'Oracle Critic' for upper-bound testing.
        check = llm(
            f"Compare the prediction with the ground truth.\\n"
            f"Prediction: {pred}\\n"
            f"Ground Truth: {ground_truth}\\n\\n"
            f"Is the prediction correct? Return JSON {{'score': 1.0}} or {{'score': 0.0, 'critique': '...'}}.",
            call_tag="judge_gt"
        )
    else:
        # Unsupervised Mode (Edge/Inference): Self-Correction / Critique
        # We look for internal consistency or policy violations
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
        
        # Initialization
        self.seed_code = SEED_WORKFLOW_CODE
        self.feedback_code = FEEDBACK_WORKFLOW_CODE
        
        self.editor = StructureEditor(
            llm=get_llm(role="optimizer"),
            max_tokens=getattr(args, "structure_max_tokens", 12000),
            require_call_tag=True,
            forbid_strip_trace_tags=True,
            forbid_imports=True,
            verbose=self.verbose,
        )
        
        # Load initial bundles
        self.seed_bundle = self._load(self.seed_code, "seed_workflow")
        self.feedback_bundle = self._load(self.feedback_code, "feedback_workflow")

    def _load(self, code: str, fn_name: str) -> FunModule:
        """Load code into a FunModule."""
        fn = self.editor.load_function(code, fn_name, extra_globals={"llm": llm, "msg": msg})
        if not isinstance(fn, FunModule):
            raise ValueError(f"Function {fn_name} must be decorated with @bundle.")
        return fn

    async def train_one_batch(self, batch: List[dict], calculate_score: Any) -> Dict[str, Any]:
        """
        Runs the Inner Loop (Adaptation) on a training batch.
        
        Since ECWO is an inference-time adaptation scheme, "training" here acts as 
        an evaluation of the seed's adaptability. We run the full beam search 
        for each training example and report the final score against ground truth.
        """
        # Unpack batch: BaseScheme.iter_batches yields tuple of columns
        # Assuming batch is (questions, answers) or (contexts, questions, answers)
        if len(batch) == 3:
            contexts, questions, answers = batch
        else:
            questions, answers = batch
            contexts = [""] * len(questions)

        scores = []
        costs = []

        # Process batch sequentially (or parallel if safe)
        for i in range(len(questions)):
            q, a, c = questions[i], answers[i], contexts[i]
            
            # Run Inference (Adaptation)
            # We treat 'a' (ground truth) as hidden from the inner loop, 
            # only used for final evaluation here.
            pred, cost = await self.inference(q)
            
            # Evaluate using benchmark scorer
            score = await calculate_score(a, pred)
            scores.append(score)
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
        # The inner loop relies on self.feedback_bundle (Self-Critic).
        best_cand, trajectory = await engine.run(
            seed_code=self.seed_code, # Engine expects code string to bootstrap
            feed_fn=self.feedback_bundle,
            context="", 
            batch=[(input_text, None)], # No Ground Truth available to inner loop
            iterations=self.iterations
        )
        
        cost = float(get_total_cost())
        
        # 3. Extract Result
        if not best_cand:
            return "Failure: No valid candidate found.", cost
            
        # Execute the best bundle one last time to get the clean output string
        # (or return the output stored in the candidate if available)
        try:
            # We use the best candidate's bundle directly
            with RuntimeTracer(domain="inference_final") as tr:
                # Assuming context is handled inside input_text or empty
                final_res = await best_cand.bundle("", input_text)
            return str(final_res), cost
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

    def load(self, path: Path) -> bool:
        """Load state."""
        if not path.exists():
            return False
        try:
            ns = {}
            exec(path.read_text("utf-8"), ns, ns)
            self.seed_code = ns.get("SEED_WORKFLOW_CODE", self.seed_code)
            self.feedback_code = ns.get("FEEDBACK_WORKFLOW_CODE", self.feedback_code)
            # Reload bundles
            self.seed_bundle = self._load(self.seed_code, "seed_workflow")
            self.feedback_bundle = self._load(self.feedback_code, "feedback_workflow")
            return True
        except Exception:
            return False
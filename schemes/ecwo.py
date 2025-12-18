# schemes/ecwo.py
"""
Edge-Cloud Workflow Optimization (ECWO) Scheme.

This scheme implements test-time adaptation using beam search optimization.

Key Design:
- Training (train_one_batch): Uses ground truth to guide beam search optimization
- Inference: Executes the trained workflow directly for evaluation
- Test-Time Adaptation (optional): Can run beam search without GT via adapt()

The separation ensures that:
1. Training properly leverages ground truth for supervised adaptation
2. Inference is fast and used correctly by benchmark evaluation (run_baseline)
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
from utils.logs import logger

# -------------------------------------------------------------------------
# Default Contexts for Supervised Benchmarks (no context provided)
# -------------------------------------------------------------------------

DEFAULT_CONTEXTS = {
    "math": "Solve the following mathematical problem. Show your reasoning step by step and provide the final answer.",
    "gsm8k": "Solve the following grade school math word problem. Break it down into steps and compute the final numerical answer.",
    "drop": "Answer the following reading comprehension question. Use discrete reasoning over the provided passage.",
    "hotpotqa": "Answer the following multi-hop question. You may need to combine information from multiple sources.",
    "humaneval": "Write a Python function to solve the following programming problem. Ensure the code is correct and handles edge cases.",
    "mbpp": "Write a Python function to solve the following programming problem. The function should pass all test cases.",
}

DEFAULT_CONTEXT_FALLBACK = "Solve the following problem carefully. Show your reasoning and provide a clear final answer."

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
    
    Key Methods:
    - train_one_batch: Runs beam search WITH ground truth (supervised)
    - inference: Executes the trained workflow directly (for evaluation)
    - adapt: Runs beam search WITHOUT ground truth (test-time adaptation)
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Store benchmark name for default context lookup
        self.benchmark_name = getattr(args, "benchmark", "").lower()
        
        # Configuration
        self.beam_width = getattr(args, "beam_width", 3)
        self.iterations = getattr(args, "inner_loop_iters", 3)
        self.verbose = getattr(args, "verbose", False)
        
        logger.info(f"[ECWO] Initializing scheme for benchmark: {self.benchmark_name}")
        logger.info(f"[ECWO] Config: beam_width={self.beam_width}, iterations={self.iterations}")
        
        # Code templates
        self.seed_code = SEED_WORKFLOW_CODE
        self.feedback_code = FEEDBACK_WORKFLOW_CODE
        
        # Structure editor for code manipulation
        logger.info("[ECWO] Setting up StructureEditor...")
        self.editor = StructureEditor(
            llm=get_llm(role="optimizer"),
            max_tokens=getattr(args, "structure_max_tokens", 12000),
            require_call_tag=True,
            forbid_strip_trace_tags=True,
            forbid_imports=True,
            verbose=self.verbose,
        )
        
        # Load initial functions (plain callables, not FunModules)
        logger.info("[ECWO] Loading seed workflow and feedback functions...")
        self.seed_fn = self._load(self.seed_code, "seed_workflow")
        self.feedback_fn = self._load(self.feedback_code, "feedback_workflow")
        logger.info("[ECWO] Initialization complete.")

    def _get_default_context(self) -> str:
        """
        Get default context string based on benchmark type.
        
        For supervised benchmarks without explicit context, this provides
        task-specific instructions to guide the workflow.
        """
        context = DEFAULT_CONTEXTS.get(self.benchmark_name, DEFAULT_CONTEXT_FALLBACK)
        logger.debug(f"[ECWO] Using default context for '{self.benchmark_name}': {context[:50]}...")
        return context

    def _load(self, code: str, fn_name: str) -> Callable:
        """
        Load code into a callable function.
        
        Uses StructureEditor.load_function which exec's the code and returns
        the named function. The function is a plain Python callable.
        """
        logger.debug(f"[ECWO] Loading function: {fn_name}")
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
        Runs the Inner Loop (Adaptation) on a training batch WITH ground truth.
        
        This is the TRAINING function - it uses ground truth to guide the
        beam search optimization for learning better workflows.
        
        Args:
            batch: Tuple of (questions, answers) or (contexts, questions, answers)
            calculate_score: Scoring function from benchmark
            
        Returns:
            Dictionary with training metrics (score, cost, n_samples)
        """
        # Unpack batch: BaseScheme.iter_batches yields tuple of columns
        if len(batch) == 3:
            contexts, questions, answers = batch
            logger.info(f"[ECWO] train_one_batch: received batch with contexts (n={len(questions)})")
        else:
            questions, answers = batch
            # Use default context for supervised benchmarks without context
            default_ctx = self._get_default_context()
            contexts = [default_ctx] * len(questions)
            logger.info(f"[ECWO] train_one_batch: using default context for {len(questions)} samples")

        scores = []
        costs = []

        logger.info(f"[ECWO] Starting training batch: {len(questions)} samples, beam_width={self.beam_width}, iterations={self.iterations}")

        # Process batch sequentially
        for i in range(len(questions)):
            q, a, c = questions[i], answers[i], contexts[i]
            
            logger.info(f"[ECWO] Training sample {i+1}/{len(questions)}: question_len={len(q)}, answer_len={len(str(a))}")
            
            # Run beam search optimization WITH ground truth
            # The ground truth is passed to the feedback function for supervised learning
            pred, cost = await self._run_training(c, q, a)
            
            logger.debug(f"[ECWO] Sample {i+1} prediction_len={len(pred)}, cost={cost:.4f}")
            
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
            logger.info(f"[ECWO] Sample {i+1} score: {score:.4f}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_cost = sum(costs) / len(costs) if costs else 0.0
        
        logger.info(f"[ECWO] Batch complete: avg_score={avg_score:.4f}, avg_cost={avg_cost:.4f}, n_samples={len(questions)}")
        
        return {
            "score": avg_score,
            "cost": avg_cost,
            "n_samples": len(questions)
        }

    async def _run_training(
        self, 
        context: str, 
        question: str, 
        ground_truth: str
    ) -> Tuple[str, float]:
        """
        Run beam search optimization WITH ground truth (training mode).
        
        The feedback function receives ground_truth in the signal dict,
        enabling supervised comparison and critique.
        
        Args:
            context: Context string
            question: Input question
            ground_truth: Expected answer (used for feedback/scoring)
            
        Returns:
            (prediction, cost_usd)
        """
        reset_usage()
        logger.debug("[ECWO] _run_training: Starting beam search with ground truth")
        
        # Setup Engine
        engine = BeamInnerLoopEngine(
            editor=self.editor,
            beam_width=self.beam_width,
            verbose=self.verbose
        )
        
        logger.debug(f"[ECWO] _run_training: Running {self.iterations} iterations of beam search")
        
        # Run Beam Search with ground truth
        # The ground_truth is passed to feedback_fn via the batch tuple
        best_cand, trajectory = await engine.run(
            seed_code=self.seed_code,
            feed_fn=self.feedback_fn,
            context=context, 
            batch=[(question, ground_truth)],  # Ground truth IS available
            iterations=self.iterations
        )
        
        cost = float(get_total_cost())
        logger.debug(f"[ECWO] _run_training: Beam search complete, cost={cost:.4f}")
        
        # Extract Result
        if not best_cand:
            logger.warning("[ECWO] _run_training: No valid candidate found")
            return "Failure: No valid candidate found.", cost
            
        logger.info(f"[ECWO] _run_training: Best candidate score={best_cand.score:.4f}, mutation_type={best_cand.mutation_type}")
        
        # Update seed code with the best candidate found during training
        if best_cand.code and best_cand.score > 0:
            logger.info("[ECWO] _run_training: Updating seed code with best candidate")
            self.seed_code = best_cand.code
            self.seed_fn = best_cand.fn
            
        # Execute the best candidate one final time to get clean output
        try:
            logger.debug("[ECWO] _run_training: Executing best candidate for final output")
            with RuntimeTracer(domain="training_final") as tr:
                final_res = best_cand.fn(context, question)
            return strip_trace_tags(str(final_res)), cost
        except Exception as e:
            logger.error(f"[ECWO] _run_training: Error executing best candidate: {e}")
            return f"Error executing best candidate: {e}", cost

    async def inference(self, input_text: str) -> Tuple[str, float]:
        """
        Inference function for evaluation (used by benchmark.run_baseline).
        
        This method executes the trained workflow directly WITHOUT running
        beam search optimization. It's designed for fast evaluation.
        
        For test-time adaptation (beam search without GT), use adapt() instead.
        
        Args:
            input_text: The user query (can include context).
            
        Returns:
            (prediction, cost_usd)
        """
        reset_usage()
        logger.debug(f"[ECWO] inference: Starting (input_len={len(input_text)})")
        
        # Use default context for supervised benchmarks
        context = self._get_default_context()
        
        try:
            logger.debug("[ECWO] inference: Executing trained workflow")
            # Execute the trained workflow directly
            with RuntimeTracer(domain="inference") as tr:
                result = self.seed_fn(context, input_text)
            
            cost = float(get_total_cost())
            pred = strip_trace_tags(str(result))
            logger.debug(f"[ECWO] inference: Complete (pred_len={len(pred)}, cost={cost:.4f})")
            return pred, cost
            
        except Exception as e:
            cost = float(get_total_cost())
            logger.error(f"[ECWO] inference: Error - {e}")
            return f"Error: {e}", cost

    async def inference_with_meta(
        self, 
        context: str, 
        x: str
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Inference with metadata (compatible with clovertoy benchmark).
        
        Args:
            context: Context string
            x: Input question
            
        Returns:
            (prediction, cost_usd, metadata_dict)
        """
        reset_usage()
        logger.debug(f"[ECWO] inference_with_meta: Starting (context_len={len(context)}, x_len={len(x)})")
        
        # Use provided context, or fall back to default if empty
        effective_context = context if context.strip() else self._get_default_context()
        
        try:
            logger.debug("[ECWO] inference_with_meta: Executing workflow")
            with RuntimeTracer(domain="inference") as tr:
                result = self.seed_fn(effective_context, x)
            
            cost = float(get_total_cost())
            pred = strip_trace_tags(str(result))
            
            meta = {
                "cost_usd": cost,
                "iterations": 0,  # No iterations in direct inference
                "sample_cost_usd": cost,
            }
            
            logger.debug(f"[ECWO] inference_with_meta: Complete (pred_len={len(pred)}, cost={cost:.4f})")
            return pred, cost, meta
            
        except Exception as e:
            cost = float(get_total_cost())
            logger.error(f"[ECWO] inference_with_meta: Error - {e}")
            return f"Error: {e}", cost, {"cost_usd": cost, "error": str(e)}

    async def adapt(self, input_text: str) -> Tuple[str, float]:
        """
        Test-Time Adaptation: Runs beam search WITHOUT ground truth.
        
        This is the unsupervised adaptation mode where the feedback function
        acts as a self-critic. Use this for test-time workflow optimization.
        
        Note: This is more expensive than inference() as it runs multiple
        iterations of beam search.
        
        Args:
            input_text: The user query (can include context).
            
        Returns:
            (best_answer, cost_usd)
        """
        reset_usage()
        logger.info(f"[ECWO] adapt: Starting test-time adaptation (input_len={len(input_text)})")
        
        # Use default context
        context = self._get_default_context()
        
        # Setup Engine
        engine = BeamInnerLoopEngine(
            editor=self.editor,
            beam_width=self.beam_width,
            verbose=self.verbose
        )
        
        logger.info(f"[ECWO] adapt: Running beam search (iterations={self.iterations}, beam_width={self.beam_width})")
        
        # Run Beam Search without ground truth
        # The inner loop relies on self.feedback_fn (Self-Critic mode)
        best_cand, trajectory = await engine.run(
            seed_code=self.seed_code,
            feed_fn=self.feedback_fn,
            context=context, 
            batch=[(input_text, None)],  # No ground truth available
            iterations=self.iterations
        )
        
        cost = float(get_total_cost())
        logger.info(f"[ECWO] adapt: Beam search complete (cost={cost:.4f})")
        
        # Extract Result
        if not best_cand:
            logger.warning("[ECWO] adapt: No valid candidate found")
            return "Failure: No valid candidate found.", cost
        
        logger.info(f"[ECWO] adapt: Best candidate score={best_cand.score:.4f}")
            
        # Execute the best candidate one final time to get clean output
        try:
            logger.debug("[ECWO] adapt: Executing best candidate for final output")
            with RuntimeTracer(domain="adapt_final") as tr:
                final_res = best_cand.fn(context, input_text)
            pred = strip_trace_tags(str(final_res))
            logger.debug(f"[ECWO] adapt: Complete (pred_len={len(pred)})")
            return pred, cost
        except Exception as e:
            logger.error(f"[ECWO] adapt: Error executing best candidate - {e}")
            return f"Error executing best candidate: {e}", cost

    def save_model(self, epoch: Optional[int] = None) -> None:
        """Persist the base templates."""
        logger.info(f"[ECWO] save_model: Saving to {self.scheme_file}" + (f" (epoch={epoch})" if epoch else ""))
        
        self.scheme_file.parent.mkdir(parents=True, exist_ok=True)
        code = (
            "# ECWO Scheme State\n"
            f"SEED_WORKFLOW_CODE = {repr(self.seed_code)}\n"
            f"FEEDBACK_WORKFLOW_CODE = {repr(self.feedback_code)}\n"
        )
        self.scheme_file.write_text(code, encoding="utf-8")
        logger.debug(f"[ECWO] save_model: Wrote {len(code)} bytes")
        
        if epoch is not None:
            snap = self.scheme_file.parent / f"scheme_epoch_{epoch}.py"
            snap.write_text(code, encoding="utf-8")
            logger.debug(f"[ECWO] save_model: Created snapshot at {snap}")

    def load(self, path: Path) -> bool:
        """Load state from disk."""
        logger.info(f"[ECWO] load: Attempting to load from {path}")
        
        if not path.exists():
            logger.warning(f"[ECWO] load: File not found: {path}")
            return False
        try:
            ns = {}
            exec(path.read_text("utf-8"), ns, ns)
            self.seed_code = ns.get("SEED_WORKFLOW_CODE", self.seed_code)
            self.feedback_code = ns.get("FEEDBACK_WORKFLOW_CODE", self.feedback_code)
            # Reload functions
            logger.debug("[ECWO] load: Reloading functions from loaded code")
            self.seed_fn = self._load(self.seed_code, "seed_workflow")
            self.feedback_fn = self._load(self.feedback_code, "feedback_workflow")
            logger.info("[ECWO] load: Successfully loaded scheme state")
            return True
        except Exception as e:
            logger.error(f"[ECWO] load: Failed to load - {e}")
            return False
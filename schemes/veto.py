# schemes/veto.py
"""
VETO: Variational Edge Test-time Optimization

A variant of ECWO optimized for local Small Language Models (SLMs).
Uses LocalSLM backend for both optimizer and executor roles.

Key Differences from ECWO:
- Shorter, more structured prompts for SLM compatibility
- Simplified feedback format (JSON-friendly)
- Reduced beam width for memory efficiency
- Optional MLX acceleration for Apple Silicon
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import re

from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto.optimizers.structure_editor import StructureEditor
from myopto.utils.llm_router import get_llm, set_role_models, set_role_config
from myopto.utils.usage import get_total_cost, reset_usage, resolve_model_name
from .base import BaseScheme
from .beam import BeamInnerLoopEngine
from utils.logs import logger


# --------------------------------------------------
# SLM-Optimized Prompts (Shorter, More Structured)
# --------------------------------------------------

VETO_SEED_CODE = """
def seed_workflow(context: str, problem: str) -> str:
    # Direct approach for SLM
    answer = llm(
        f"Context: {context}\\nProblem: {problem}\\nAnswer:",
        call_tag="answer"
    )
    return answer
""".lstrip()


VETO_FEEDBACK_CODE = """
def feedback_workflow(pred: str, signal: dict) -> str:
    gt = signal.get("ground_truth")
    if gt:
        # Supervised: simple comparison
        check = llm(
            f"Compare:\\nPrediction: {pred}\\nAnswer: {gt}\\n"
            f"Score 0-1, critique briefly. JSON: {{'score':X,'critique':'...'}}",
            call_tag="judge"
        )
    else:
        # Self-critique
        check = llm(
            f"Critique this answer briefly.\\n{pred}\\n"
            f"JSON: {{'score':X,'critique':'...'}}",
            call_tag="judge"
        )
    return check
""".lstrip()


VETO_EDIT_PROMPT = """Improve this workflow code based on feedback.

Current Code:
```python
{code}
```

Feedback: {feedback}

Rules:
- Keep llm() calls with call_tag parameter
- Return final answer as string
- Be concise

Improved Code:
```python
""".lstrip()


VETO_MERGE_PROMPT = """Select the best workflow from these candidates.

{candidates}

Feedback Summary: {feedback}

Reply with just the number (1, 2, etc.) of the best one.""".lstrip()


# --------------------------------------------------
# VETO Scheme Implementation
# --------------------------------------------------

class VETOScheme(BaseScheme):
    """
    VETO: Test-time adaptation using local SLMs.
    
    This scheme optimizes workflows at test time using small language models
    that can run locally on edge devices. It implements beam search over
    workflow candidates with SLM-friendly prompts and reduced resource usage.
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # SLM Configuration - use short model name from -e flag
        self.slm_model = getattr(args, "executor", "qwen2-0.5b")
        self.device = getattr(args, "device", "mps")
        
        # Configure SLM for both roles
        self._configure_slm()
        
        # Reduced beam width for SLM (memory efficiency)
        self.beam_width = min(getattr(args, "beam_width", 2), 3)
        self.iterations = getattr(args, "inner_loop_iters", 2)
        
        logger.info(f"[VETO] Using SLM: {self.slm_model} (device={self.device})")
        
        # SLM-optimized code templates
        self.seed_code = VETO_SEED_CODE
        self.feedback_code = VETO_FEEDBACK_CODE
        
        # Structure editor with smaller token limit
        self.editor = StructureEditor(
            llm=get_llm(role="optimizer"),
            max_tokens=getattr(args, "structure_max_tokens", 4000),
            require_call_tag=True,
            forbid_strip_trace_tags=True,
            forbid_imports=True,
            verbose=getattr(args, "verbose", False),
        )
        
        # Load functions
        self.seed_fn = self._load(self.seed_code, "seed_workflow")
        self.feedback_fn = self._load(self.feedback_code, "feedback_workflow")
        
        # Beam engine for inner loop optimization
        self.beam_engine: Optional[BeamInnerLoopEngine] = None
        
        # State tracking
        self.current_workflow_code = self.seed_code
        self.best_score = 0.0
        self.history: List[Dict[str, Any]] = []

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    def _configure_slm(self):
        """Configure LocalSLM for optimizer and executor roles."""
        # Use set_role_models with short model names
        set_role_models(executor=self.slm_model, optimizer=self.slm_model)
        
        # Additional config for token limits
        set_role_config("executor", max_new_tokens=150, device=self.device)
        set_role_config("optimizer", max_new_tokens=300, device=self.device)
        
        logger.info(f"[VETO] Configured LocalSLM: {self.slm_model}")

    def _load(self, code: str, fn_name: str) -> Callable:
        """Load code into a callable function."""
        return self.editor.load_function(
            code, 
            fn_name, 
            extra_globals={"llm": llm, "msg": msg, "strip_trace_tags": strip_trace_tags}
        )

    # --------------------------------------------------
    # Workflow Execution
    # --------------------------------------------------

    def execute_workflow(
        self, 
        context: str, 
        problem: str,
        workflow_fn: Optional[Callable] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute a workflow and return result with trace.
        
        Args:
            context: Background context for the problem
            problem: The specific problem to solve
            workflow_fn: Optional custom workflow function
            
        Returns:
            Tuple of (answer, trace_dict)
        """
        fn = workflow_fn or self.seed_fn
        
        tracer = RuntimeTracer()
        with tracer:
            try:
                result = fn(context, problem)
            except Exception as e:
                logger.warning(f"[VETO] Workflow execution failed: {e}")
                result = f"Error: {str(e)}"
        
        trace = tracer.get_trace()
        return result, trace

    def get_feedback(
        self, 
        prediction: str, 
        signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get feedback on a prediction using the feedback workflow.
        
        Args:
            prediction: Model's prediction
            signal: Signal dict (may contain ground_truth)
            
        Returns:
            Feedback dict with score and critique
        """
        tracer = RuntimeTracer()
        with tracer:
            raw_feedback = self.feedback_fn(prediction, signal)
        
        # Parse JSON from SLM output
        feedback = self._parse_feedback(raw_feedback)
        return feedback

    def _parse_feedback(self, raw: str) -> Dict[str, Any]:
        """Parse JSON feedback from SLM output."""
        # Try to extract JSON from response
        try:
            # Look for JSON pattern
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: extract score manually
        score = 0.5
        if "1" in raw or "correct" in raw.lower():
            score = 1.0
        elif "0" in raw or "wrong" in raw.lower():
            score = 0.0
            
        return {"score": score, "critique": raw[:200]}

    # --------------------------------------------------
    # Beam Search Optimization
    # --------------------------------------------------

    def init_beam_engine(self):
        """Initialize the beam search engine for inner loop."""
        self.beam_engine = BeamInnerLoopEngine(
            beam_width=self.beam_width,
            editor=self.editor,
            max_iterations=self.iterations,
        )

    def generate_candidates(
        self, 
        current_code: str, 
        feedback: Dict[str, Any]
    ) -> List[str]:
        """
        Generate candidate workflow improvements.
        
        Args:
            current_code: Current workflow code
            feedback: Feedback from evaluation
            
        Returns:
            List of candidate code strings
        """
        candidates = [current_code]  # Keep original
        
        optimizer = get_llm(role="optimizer")
        
        # Generate variants
        for i in range(self.beam_width - 1):
            prompt = VETO_EDIT_PROMPT.format(
                code=current_code,
                feedback=feedback.get("critique", "Improve accuracy")
            )
            
            try:
                response = optimizer(prompt)
                new_code = self._extract_code(response)
                if new_code and self._validate_code(new_code):
                    candidates.append(new_code)
            except Exception as e:
                logger.warning(f"[VETO] Candidate generation failed: {e}")
        
        return candidates

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Look for code block
        match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Look for def statement
        match = re.search(r'(def \w+.*?)(?=\n\S|\Z)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return None

    def _validate_code(self, code: str) -> bool:
        """Validate that code is safe and well-formed."""
        # Basic checks
        if "import" in code and "forbid_imports" not in code:
            return False
        if "exec(" in code or "eval(" in code:
            return False
        if "def " not in code:
            return False
        if "llm(" not in code:
            return False
        
        # Try to compile
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False

    def select_best(
        self, 
        candidates: List[str], 
        scores: List[float],
        feedback_summary: str
    ) -> Tuple[str, float]:
        """
        Select the best candidate based on scores.
        
        Args:
            candidates: List of candidate codes
            scores: Corresponding scores
            feedback_summary: Summary of feedback
            
        Returns:
            Tuple of (best_code, best_score)
        """
        if not candidates:
            return self.seed_code, 0.0
        
        # Simple argmax selection (SLM-friendly)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return candidates[best_idx], scores[best_idx]

    # --------------------------------------------------
    # Main Optimization Loop
    # --------------------------------------------------

    def optimize(
        self, 
        context: str, 
        problem: str, 
        signal: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run VETO optimization on a single problem.
        
        Args:
            context: Background context
            problem: Problem to solve
            signal: Optional supervision signal
            
        Returns:
            Tuple of (best_answer, optimization_info)
        """
        reset_usage()
        signal = signal or {}
        
        logger.info(f"[VETO] Starting optimization (beam={self.beam_width}, iters={self.iterations})")
        
        current_code = self.current_workflow_code
        best_answer = ""
        best_score = 0.0
        
        for iteration in range(self.iterations):
            logger.info(f"[VETO] Iteration {iteration + 1}/{self.iterations}")
            
            # Execute current workflow
            current_fn = self._load(current_code, "seed_workflow")
            answer, trace = self.execute_workflow(context, problem, current_fn)
            
            # Get feedback
            feedback = self.get_feedback(answer, signal)
            score = feedback.get("score", 0.5)
            
            logger.info(f"[VETO] Score: {score:.2f} - {feedback.get('critique', '')[:50]}")
            
            # Update best
            if score > best_score:
                best_score = score
                best_answer = answer
                self.current_workflow_code = current_code
            
            # Early exit on perfect score
            if score >= 0.99:
                logger.info("[VETO] Perfect score achieved, stopping early")
                break
            
            # Generate and evaluate candidates
            candidates = self.generate_candidates(current_code, feedback)
            
            if len(candidates) > 1:
                # Evaluate each candidate
                scores = []
                for cand in candidates:
                    try:
                        cand_fn = self._load(cand, "seed_workflow")
                        cand_answer, _ = self.execute_workflow(context, problem, cand_fn)
                        cand_feedback = self.get_feedback(cand_answer, signal)
                        scores.append(cand_feedback.get("score", 0.0))
                    except Exception:
                        scores.append(0.0)
                
                # Select best candidate
                current_code, iter_best_score = self.select_best(
                    candidates, scores, feedback.get("critique", "")
                )
                
                if iter_best_score > best_score:
                    best_score = iter_best_score
                    best_fn = self._load(current_code, "seed_workflow")
                    best_answer, _ = self.execute_workflow(context, problem, best_fn)
        
        # Record history
        self.history.append({
            "problem": problem[:100],
            "answer": best_answer[:200],
            "score": best_score,
            "iterations": iteration + 1,
            "cost": get_total_cost(),
        })
        
        info = {
            "score": best_score,
            "iterations": iteration + 1,
            "final_code": self.current_workflow_code,
            "cost": get_total_cost(),
        }
        
        logger.info(f"[VETO] Optimization complete. Best score: {best_score:.2f}")
        
        return best_answer, info

    # --------------------------------------------------
    # Batch Processing
    # --------------------------------------------------

    def run_batch(
        self, 
        dataset: List[Dict[str, Any]],
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run VETO on a batch of problems.
        
        Args:
            dataset: List of problem dicts with context, problem, and optional ground_truth
            max_samples: Optional limit on samples to process
            
        Returns:
            Results dict with predictions and metrics
        """
        if max_samples:
            dataset = dataset[:max_samples]
        
        results = []
        total_score = 0.0
        
        for i, item in enumerate(dataset):
            logger.info(f"[VETO] Processing {i+1}/{len(dataset)}")
            
            context = item.get("context", "")
            problem = item.get("problem", item.get("question", ""))
            signal = {"ground_truth": item.get("ground_truth", item.get("answer"))}
            
            answer, info = self.optimize(context, problem, signal)
            
            results.append({
                "problem": problem,
                "prediction": answer,
                "score": info["score"],
                "iterations": info["iterations"],
            })
            total_score += info["score"]
        
        return {
            "results": results,
            "avg_score": total_score / len(dataset) if dataset else 0.0,
            "total_cost": get_total_cost(),
            "history": self.history,
        }

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def reset(self):
        """Reset scheme state."""
        self.current_workflow_code = self.seed_code
        self.best_score = 0.0
        self.history = []
        self.seed_fn = self._load(self.seed_code, "seed_workflow")
        reset_usage()

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.history:
            return {"runs": 0}
        
        scores = [h["score"] for h in self.history]
        return {
            "runs": len(self.history),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "total_cost": sum(h.get("cost", 0) for h in self.history),
        }

    def save_workflow(self, path: str):
        """Save current workflow code to file."""
        Path(path).write_text(self.current_workflow_code)
        logger.info(f"[VETO] Saved workflow to {path}")

    def load_workflow(self, path: str):
        """Load workflow code from file."""
        code = Path(path).read_text()
        if self._validate_code(code):
            self.current_workflow_code = code
            self.seed_fn = self._load(code, "seed_workflow")
            logger.info(f"[VETO] Loaded workflow from {path}")
        else:
            raise ValueError(f"Invalid workflow code in {path}")
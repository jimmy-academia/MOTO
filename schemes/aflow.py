# schemes/aflow.py
"""
AFlow Scheme - Agentic Workflow Generation (ICLR 2025)
Integrated with llm_router for centralized LLM management.

AFlow uses Monte Carlo Tree Search to explore and optimize code-based
LLM workflows. This scheme integrates AFlow with the main codebase's
llm_router for unified LLM configuration.

Reference: https://arxiv.org/abs/2410.10762
"""
import os
import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from .base import BaseScheme
from myopto.utils.llm_router import get_llm
from myopto.utils.usage import get_total_cost, reset_usage
from utils.logs import logger


# --------------------------------------------------
# Dataset Configurations
# --------------------------------------------------

# Map main codebase benchmark names to AFlow dataset names
BENCHMARK_TO_AFLOW_DATASET = {
    "math": "MATH",
    "gsm8k": "GSM8K",
    "drop": "DROP",
    "hotpotqa": "HotpotQA",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
}

DATASET_CONFIGS = {
    "DROP": {
        "question_type": "qa",
        "operators": ["Custom", "AnswerGenerate", "ScEnsemble"],
    },
    "HotpotQA": {
        "question_type": "qa",
        "operators": ["Custom", "AnswerGenerate", "ScEnsemble"],
    },
    "MATH": {
        "question_type": "math",
        "operators": ["Custom", "ScEnsemble", "Programmer"],
    },
    "GSM8K": {
        "question_type": "math",
        "operators": ["Custom", "ScEnsemble", "Programmer"],
    },
    "MBPP": {
        "question_type": "code",
        "operators": ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    },
    "HumanEval": {
        "question_type": "code",
        "operators": ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    },
}


# --------------------------------------------------
# Usage Tracker (defined BEFORE LLMAdapter)
# --------------------------------------------------

class UsageTracker:
    """Tracks token usage and costs for LLM calls."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
    
    def track(self, response: Any):
        """Track usage from a response."""
        self.call_count += 1
        
        # Try to extract usage info
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens"):
                self.total_input_tokens += usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                self.total_output_tokens += usage.completion_tokens
    
    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": get_total_cost(),
            "call_count": self.call_count,
        }


# --------------------------------------------------
# LLM Adapter (bridges llm_router to AFlow's async interface)
# --------------------------------------------------

class LLMAdapter:
    """
    Adapter that wraps llm_router's LLM interface to match AFlow's AsyncLLM interface.
    
    This allows AFlow's code to use get_llm(role=...) without modification to the
    core AFlow logic. The adapter provides both sync and async calling conventions.
    """
    
    def __init__(self, role: str = "executor"):
        self.role = role
        self._llm = None
        self._usage_tracker = UsageTracker()
    
    @property
    def llm(self):
        """Lazy initialization of the underlying LLM."""
        if self._llm is None:
            self._llm = get_llm(role=self.role)
        return self._llm
    
    async def __call__(self, prompt: str) -> str:
        """Async call interface matching AFlow's AsyncLLM."""
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages)
        
        # Extract text from response
        text = self._extract_text(response)
        
        # Track usage
        self._usage_tracker.track(response)
        
        return text
    
    async def call_with_format(self, prompt: str, formatter: Any) -> Dict[str, Any]:
        """
        Call LLM with structured output formatting.
        
        Args:
            prompt: The prompt to send
            formatter: XmlFormatter instance for parsing response
            
        Returns:
            Parsed response as dictionary
        """
        raw_response = await self(prompt)
        return formatter.parse(raw_response)
    
    def _extract_text(self, response: Any) -> str:
        """Extract text content from various response formats."""
        if isinstance(response, str):
            return response
        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content
        if isinstance(response, dict):
            if "choices" in response:
                return response["choices"][0]["message"]["content"]
            if "content" in response:
                return response["content"]
        return str(response)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return self._usage_tracker.get_summary()


# --------------------------------------------------
# AFlow Scheme
# --------------------------------------------------

class AFlowScheme(BaseScheme):
    """
    AFlow: Automating Agentic Workflow Generation (ICLR 2025)
    
    Uses graph-based workflow optimization with Monte Carlo Tree Search
    to explore and optimize code-based LLM workflows.
    
    Key features:
    - Separate optimizer/executor LLMs (via llm_router roles)
    - Code-represented workflows with operators
    - MCTS-based workflow exploration
    - Experience-driven optimization
    
    Attributes:
        dataset: AFlow dataset name (MATH, GSM8K, etc.)
        question_type: Task type (math, code, qa)
        operators: Available operators for workflow construction
        optimized_path: Path for storing optimized workflows
        max_rounds: Maximum optimization rounds
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Map benchmark name to AFlow dataset
        benchmark = getattr(args, "benchmark", "math").lower()
        self.dataset = BENCHMARK_TO_AFLOW_DATASET.get(benchmark, benchmark.upper())
        
        # Get dataset config
        if self.dataset not in DATASET_CONFIGS:
            logger.warning(f"[AFlow] Unknown dataset '{self.dataset}', using MATH config")
            self.dataset = "MATH"
        
        config = DATASET_CONFIGS[self.dataset]
        self.question_type = config["question_type"]
        self.operators = config["operators"]
        
        # Optimization parameters
        self.output_dir = getattr(args, "output_dir", "output")
        self.sub_dir = f"aflow_{benchmark}"
        self.optimized_path = os.path.join(self.output_dir, self.sub_dir)
        self.template_path = "schemes/AFlow/workspace"

        self.max_rounds = getattr(args, "epochs", 20)
        self.sample = getattr(args, "batch_size", 4)
        self.validation_rounds = getattr(args, "val_interval", 1)
        self.check_convergence = True
        self.initial_round = 1
        
        # Create LLM adapters using llm_router
        self.optimize_llm = LLMAdapter(role="optimizer")
        self.execute_llm = LLMAdapter(role="executor")
        
        # State
        self._optimizer = None
        self._current_workflow = None
        self._best_round = None
        
        logger.info(f"[AFlow] Initialized: dataset={self.dataset}, question_type={self.question_type}")
        logger.info(f"[AFlow] Operators: {self.operators}")
        logger.info(f"[AFlow] Max rounds: {self.max_rounds}, Sample: {self.sample}")
    
    def _ensure_workspace_initialized(self):
        """Copy initial workflow templates to output workspace if needed."""
        import shutil
        
        source_dir = os.path.join(self.template_path, self.dataset)
        round1_target = os.path.join(self.optimized_path, "round_1")
        graph_file = os.path.join(round1_target, "graph.py")
        
        # Check for actual graph.py file, not just directory existence
        if not os.path.exists(graph_file):
            os.makedirs(round1_target, exist_ok=True)
            # Create __init__.py files for Python imports
            for path in [
                os.path.join(self.output_dir, "__init__.py"),
                os.path.join(self.output_dir, self.sub_dir, "__init__.py"),
                os.path.join(self.optimized_path, "__init__.py"),
            ]:
                if not os.path.exists(path):
                    Path(path).touch()
            # Copy workflow files to round_1
            for f in ["__init__.py", "graph.py", "prompt.py"]:
                src = os.path.join(source_dir, f)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(round1_target, f))
            logger.info(f"[AFlow] Initialized round_1 at {round1_target}")

    def _get_optimizer(self):
        """
        Lazy initialization of AFlow Optimizer.
        
        This delays the import and initialization of AFlow components until needed,
        allowing the scheme to be registered even if AFlow sources aren't yet moved.
        """
        if self._optimizer is not None:
            return self._optimizer

        self._ensure_workspace_initialized()

        try:
            # Import AFlow components (assumes AFlow is in schemes/AFlow/)
            from schemes.AFlow.scripts.optimizer import Optimizer
            
            self._optimizer = Optimizer(
                dataset=self.dataset,
                question_type=self.question_type,
                opt_llm=self.optimize_llm,
                exec_llm=self.execute_llm,
                operators=self.operators,
                sample=self.sample,
                check_convergence=self.check_convergence,
                optimized_path=self.optimized_path,
                initial_round=self.initial_round,
                max_rounds=self.max_rounds,
                validation_rounds=self.validation_rounds,
                train_benchmark=self._train_benchmark,    # NEW
                train_indices=self._train_indices,
            )
            return self._optimizer
            
        except ImportError as e:
            logger.error(f"[AFlow] Failed to import AFlow components: {e}")
            logger.error("[AFlow] Ensure AFlow source is in schemes/AFlow/")
            raise
    
    async def train(
        self,
        train_benchmark: Any,
        train_indices: Any,
        test_benchmark: Optional[Any] = None,
        test_indices: Optional[Any] = None,
    ) -> None:
        """
        Run AFlow optimization loop with periodic test evaluation.
        
        Unlike optimizer.optimize() which runs all rounds internally,
        this method controls the loop to enable periodic test evaluation
        using test_benchmark (like BaseScheme does).
        """
        logger.info(f"[AFlow] Starting training: {self.max_rounds} rounds, val_interval={self.validation_rounds}")
        reset_usage()
        
        # Store benchmarks for use in optimizer
        self._train_benchmark = train_benchmark
        self._train_indices = list(train_indices) if train_indices else None
        
        self.prep_train()
        
        try:
            optimizer = self._get_optimizer()
            
            # Control the loop here (instead of optimizer.optimize())
            for round_idx in range(self.max_rounds):
                # Run ONE optimization round with retries
                score = await self._run_one_round(optimizer)
                
                optimizer.round += 1
                logger.info(f"[AFlow] Round {optimizer.round}: score={score}")
                
                self.save_model(epoch=optimizer.round)
                
                # Periodic test evaluation (like BaseScheme)
                if (test_benchmark is not None 
                    and test_indices is not None 
                    and len(test_indices) > 0
                    and optimizer.round % self.validation_rounds == 0):
                    
                    logger.info(f"[AFlow] Periodic test evaluation at round {optimizer.round}")
                    self._best_round = self._find_best_round()
                    self.prep_test()
                    
                    await test_benchmark.run_baseline(
                        agent=self.inference,
                        specific_indices=list(test_indices),
                        max_concurrent_tasks=10,
                    )
                    
                    self.prep_train()
                
                # Check convergence
                converged, conv_round, final_round = optimizer.convergence_utils.check_convergence(top_k=3)
                if converged and optimizer.check_convergence:
                    logger.info(f"[AFlow] Converged at round {conv_round}, final round: {final_round}")
                    optimizer.convergence_utils.print_results()
                    break
                
                await asyncio.sleep(1)
            
            # Find best round after optimization
            self._best_round = self._find_best_round()
            logger.info(f"[AFlow] Training complete. Best round: {self._best_round}")
            
        except Exception as e:
            logger.error(f"[AFlow] Training failed: {e}")
            raise
        
        self.save_model()
    
    async def _run_one_round(self, optimizer) -> Optional[float]:
        """Run one optimization round with retries."""
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                score = await optimizer._optimize_graph()
                return score
            except Exception as e:
                logger.warning(f"[AFlow] Round error: {e}. Retry {retry + 1}/{max_retries}")
                if retry == max_retries - 1:
                    logger.error("[AFlow] Max retries reached, skipping round")
                    return None
                await asyncio.sleep(5 * (retry + 1))
        
        return None
    
    def _find_best_round(self) -> int:
        """Find the best performing round from optimization results."""
        try:
            from schemes.AFlow.scripts.optimizer_utils.data_utils import DataUtils
            
            data_utils = DataUtils(f"{self.optimized_path}/{self.dataset}")
            top_rounds = data_utils.get_top_rounds(sample=2, mode="Graph")
            
            if top_rounds and top_rounds[1]:
                return top_rounds[1]["round"]
        except Exception as e:
            logger.warning(f"[AFlow] Could not determine best round: {e}")
        
        return 1
    
    def _load_workflow_class(self, round_num: int):
        """
        Dynamically load the workflow class from a specific round.
        
        AFlow generates Python workflow files for each round. This method
        loads and returns the Workflow class from the specified round.
        """
        graph_path = Path(self.optimized_path) / f"round_{round_num}" / "graph.py"
        
        if not graph_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {graph_path}")
        
        spec = importlib.util.spec_from_file_location("workflow_module", str(graph_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["workflow_module"] = module
        spec.loader.exec_module(module)
        
        return module.Workflow
    
    async def inference(self, input_text: str) -> Tuple[str, float]:
        """
        Run inference using the best trained workflow.
        
        Args:
            input_text: The problem to solve
            
        Returns:
            (answer, cost_usd)
        """
        reset_usage()
        
        # Ensure workspace is initialized (copies template if needed)
        self._ensure_workspace_initialized()
        
        # Determine which round to use
        round_num = self._best_round or 1
        
        try:
            # Load the workflow class
            WorkflowClass = self._load_workflow_class(round_num)
            
            # Create workflow instance
            from schemes.AFlow.scripts.async_llm import LLMsConfig
            llm_config = LLMsConfig.default().get("executor")
            
            workflow = WorkflowClass(
                name=f"{self.dataset}_workflow",
                llm_config=llm_config,
                dataset=self.dataset,
            )
            
            # Run inference
            answer, cost = await workflow(input_text)
            
            return answer, cost
            
        except Exception as e:
            logger.error(f"[AFlow] Inference failed: {e}")
            # Fallback to simple LLM call
            llm = get_llm(role="executor")
            response = llm(f"Solve this problem:\n{input_text}")
            return str(response), get_total_cost()
    
    def save_model(self, epoch: Optional[int] = None) -> None:
        """
        Save AFlow state.
        
        AFlow manages its own workflow files, so we just save metadata
        about the best round and configuration.
        """
        logger.info(f"[AFlow] Saving model state to {self.scheme_file}")
        
        self.scheme_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "best_round": self._best_round,
            "dataset": self.dataset,
            "optimized_path": self.optimized_path,
            "max_rounds": self.max_rounds,
        }
        
        code = f"# AFlow Scheme State\nAFLOW_STATE = {repr(state)}\n"
        self.scheme_file.write_text(code, encoding="utf-8")
        
        if epoch is not None:
            snap = self.scheme_file.parent / f"scheme_epoch_{epoch}.py"
            snap.write_text(code, encoding="utf-8")
    
    def load(self, path: Path) -> bool:
        """
        Load AFlow state from disk.
        
        Args:
            path: Path to state file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        path = Path(path)
        logger.info(f"[AFlow] Loading state from {path}")
        
        if not path.exists():
            logger.warning(f"[AFlow] State file not found: {path}")
            return False
        
        try:
            ns = {}
            exec(path.read_text("utf-8"), ns, ns)
            
            state = ns.get("AFLOW_STATE", {})
            
            if state:
                self._best_round = state.get("best_round", 1)
                self.optimized_path = state.get("optimized_path", self.optimized_path)
                logger.info(f"[AFlow] Loaded state: best_round={self._best_round}")
                return True
            
        except Exception as e:
            logger.error(f"[AFlow] Failed to load state: {e}")
        
        return False
    
    def prep_train(self) -> None:
        """Prepare for training."""
        logger.debug("[AFlow] prep_train: resetting usage tracking")
        reset_usage()
    
    def prep_test(self) -> None:
        """Prepare for testing/inference."""
        logger.debug("[AFlow] prep_test: resetting usage tracking")
        reset_usage()
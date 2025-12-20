# schemes/AFlow/scripts/evaluator.py
# ----------------------------------------------------------------------------------------------------
# AFlow Evaluator - Uses main codebase benchmarks
# ----------------------------------------------------------------------------------------------------
# This evaluator imports from ./benchmarks/ (main codebase), NOT ./schemes/AFlow/benchmarks/
# ----------------------------------------------------------------------------------------------------

from typing import Dict, Literal, Tuple, Any

# Import from main codebase benchmarks
from benchmarks.benchmark import BaseBenchmark
from benchmarks.drop import DROPBenchmark
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.hotpotqa import HotpotQABenchmark
from benchmarks.humaneval import HumanEvalBenchmark
from benchmarks.math import MATHBenchmark
from benchmarks.mbpp import MBPPBenchmark

# Try importing LiveCodeBench (may not exist in all setups)
try:
    from benchmarks.livecodebench import LiveCodeBench
    HAS_LIVECODEBENCH = True
except ImportError:
    LiveCodeBench = None
    HAS_LIVECODEBENCH = False

from utils.logs import logger

# Dataset type definition
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "LiveCodeBench"]


class Evaluator:
    """
    Evaluates workflow performance on different datasets.
    
    Uses the main codebase's benchmark classes from ./benchmarks/
    """
    
    def __init__(self, eval_path: str):
        self.eval_path = eval_path
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "HumanEval": HumanEvalBenchmark,
            "HotpotQA": HotpotQABenchmark,
            "MBPP": MBPPBenchmark,
            "DROP": DROPBenchmark,
        }
        if HAS_LIVECODEBENCH:
            self.dataset_configs["LiveCodeBench"] = LiveCodeBench
    
    async def graph_evaluate(
        self,
        dataset: DatasetType,
        graph: Any,
        params: dict,
        path: str,
        is_test: bool = False
    ) -> Tuple[float, float, float]:
        """
        Evaluate a workflow graph.
        
        Args:
            dataset: Dataset name
            graph: Workflow class to evaluate
            params: Configuration parameters
            path: Output path for logs
            is_test: Whether using test or validation data
            
        Returns:
            Tuple of (score, avg_cost, total_cost)
        """
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]
        benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=path)
        
        # Configure the graph with params
        configured_graph = await self._configure_graph(dataset, graph, params)
        
        # For validation, can optionally limit indices (None = all)
        va_list = None
        
        return await benchmark.run_evaluation(configured_graph, va_list)
    
    async def _configure_graph(self, dataset: str, graph: Any, params: dict):
        """Configure a workflow graph with given parameters."""
        dataset_config = params.get("dataset", dataset)
        llm_config = params.get("llm_config", {})
        return graph(name=dataset, llm_config=llm_config, dataset=dataset_config)
    
    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        """Get the data file path for a dataset."""
        base_path = f"data/datasets/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_validate.jsonl"
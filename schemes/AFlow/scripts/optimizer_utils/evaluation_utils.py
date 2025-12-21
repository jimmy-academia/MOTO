# schemes/AFlow/scripts/optimizer_utils/evaluation_utils.py
# ----------------------------------------------------------------------------------------------------
# AFlow Evaluation Utilities
# ----------------------------------------------------------------------------------------------------

from typing import Any, List, Tuple

from ..evaluator import Evaluator
from utils.logs import logger


class EvaluationUtils:
    """
    Utilities for evaluating workflows.
    
    Wraps the Evaluator class to provide convenient methods for
    evaluating workflows during optimization rounds.
    """
    
    def __init__(self, root_path: str):
        self.root_path = root_path
    
    async def evaluate_initial_round(
        self,
        optimizer: Any,
        graph_path: str,
        directory: str,
        validation_n: int,
        data: List
    ) -> List:
        """Evaluate the initial round."""
        optimizer.graph = optimizer.graph_utils.load_graph(optimizer.round, graph_path)
        evaluator = Evaluator(eval_path=directory)
        
        # Check for external benchmark from main.py
        ext_benchmark = getattr(optimizer, 'train_benchmark', None)
        ext_indices = getattr(optimizer, 'train_indices', None)
        
        # Use external benchmark: run once and return
        if ext_benchmark is not None:
            n_samples = len(ext_indices) if ext_indices else "all"
            logger.info(f"[EvaluationUtils] Using external benchmark ({n_samples} samples)")
            
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
                ext_benchmark=ext_benchmark,
                ext_indices=ext_indices,
            )
            new_data = optimizer.data_utils.create_result_data(
                optimizer.round, score, avg_cost, total_cost
            )
            data.append(new_data)
            result_path = optimizer.data_utils.get_results_file_path(graph_path)
            optimizer.data_utils.save_results(result_path, data)
            return data
        
        # Original approach: loop validation_n times
        for i in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
            )
            
            new_data = optimizer.data_utils.create_result_data(
                optimizer.round, score, avg_cost, total_cost
            )
            data.append(new_data)
            
            result_path = optimizer.data_utils.get_results_file_path(graph_path)
            optimizer.data_utils.save_results(result_path, data)
        
        return data
    
    async def evaluate_graph(
        self,
        optimizer: Any,
        directory: str,
        validation_n: int,
        data: List,
        initial: bool = False
    ) -> float:
        """Evaluate a workflow graph."""
        evaluator = Evaluator(eval_path=directory)
        
        # Check for external benchmark from main.py
        ext_benchmark = getattr(optimizer, 'train_benchmark', None)
        ext_indices = getattr(optimizer, 'train_indices', None)
        
        # Use external benchmark: run once and return
        if ext_benchmark is not None:
            n_samples = len(ext_indices) if ext_indices else "all"
            logger.info(f"[EvaluationUtils] Using external benchmark ({n_samples} samples)")
            
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
                ext_benchmark=ext_benchmark,
                ext_indices=ext_indices,
            )
            cur_round = optimizer.round + 1 if not initial else optimizer.round
            new_data = optimizer.data_utils.create_result_data(
                cur_round, score, avg_cost, total_cost
            )
            data.append(new_data)
            result_path = optimizer.data_utils.get_results_file_path(
                f"{optimizer.root_path}/workflows"
            )
            optimizer.data_utils.save_results(result_path, data)
            return score
        
        # Original approach: loop validation_n times
        sum_score = 0
        for i in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
            )
            
            cur_round = optimizer.round + 1 if not initial else optimizer.round
            new_data = optimizer.data_utils.create_result_data(
                cur_round, score, avg_cost, total_cost
            )
            data.append(new_data)
            
            result_path = optimizer.data_utils.get_results_file_path(
                f"{optimizer.root_path}/workflows"
            )
            optimizer.data_utils.save_results(result_path, data)
            
            sum_score += score
        
        return sum_score / validation_n if validation_n > 0 else 0.0
    
    async def evaluate_graph_test(
        self,
        optimizer: Any,
        directory: str,
        is_test: bool = True
    ) -> Tuple[float, float, float]:
        """Evaluate graph on test data."""
        evaluator = Evaluator(eval_path=directory)
        return await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
            directory,
            is_test=is_test,
        )
# schemes/AFlow/scripts/optimizer.py
# ----------------------------------------------------------------------------------------------------
# AFlow Optimizer - Adapted for llm_router integration
# ----------------------------------------------------------------------------------------------------
# Original: @Date 8/12/2024 22:00 PM, @Author issac
# Adapted for main codebase integration with llm_router
# ----------------------------------------------------------------------------------------------------

import asyncio
import time
from typing import List, Literal, Dict, Any, Union

from pydantic import BaseModel, Field

try:
    from scripts.evaluator import DatasetType
    from scripts.optimizer_utils.convergence_utils import ConvergenceUtils
    from scripts.optimizer_utils.data_utils import DataUtils
    from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
    from scripts.optimizer_utils.experience_utils import ExperienceUtils
    from scripts.optimizer_utils.graph_utils import GraphUtils
    from scripts.async_llm import AsyncLLM, create_llm_instance
    from scripts.formatter import XmlFormatter, FormatError
    from utils.logs import logger
except ImportError:
    # Running from schemes/AFlow context
    from .evaluator import DatasetType
    from .optimizer_utils.convergence_utils import ConvergenceUtils
    from .optimizer_utils.data_utils import DataUtils
    from .optimizer_utils.evaluation_utils import EvaluationUtils
    from .optimizer_utils.experience_utils import ExperienceUtils
    from .optimizer_utils.graph_utils import GraphUtils
    from .async_llm import AsyncLLM, create_llm_instance
    from .formatter import XmlFormatter, FormatError
    
    import logging
    logger = logging.getLogger(__name__)


QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]


# ----------------------------------------------------------------------------------------------------
# Data Models
# ----------------------------------------------------------------------------------------------------

class GraphOptimize(BaseModel):
    """Model for graph optimization response."""
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


# ----------------------------------------------------------------------------------------------------
# Optimizer Class
# ----------------------------------------------------------------------------------------------------

class Optimizer:
    """
    AFlow Optimizer using MCTS-based workflow exploration.
    
    This optimizer explores the space of code-based LLM workflows using
    Monte Carlo Tree Search, evaluating and refining workflows based on
    their performance on validation data.
    
    Adapted to use llm_router via opt_llm and exec_llm parameters which
    can be LLMAdapter instances from the AFlowScheme.
    """
    
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm: Any = None,           # LLMAdapter or config
        exec_llm: Any = None,          # LLMAdapter or config  
        opt_llm_config: Any = None,    # Legacy: for backward compatibility
        exec_llm_config: Any = None,   # Legacy: for backward compatibility
        operators: List[str] = None,
        sample: int = 4,
        check_convergence: bool = False,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 5,
    ) -> None:
        """
        Initialize the AFlow Optimizer.
        
        Args:
            dataset: Dataset name (MATH, GSM8K, etc.)
            question_type: Task type (math, code, qa)
            opt_llm: Optimizer LLM (LLMAdapter or AsyncLLM)
            exec_llm: Executor LLM (LLMAdapter or AsyncLLM)
            opt_llm_config: Legacy config for optimizer LLM
            exec_llm_config: Legacy config for executor LLM
            operators: List of operator names
            sample: Number of workflow samples per round
            check_convergence: Enable early stopping
            optimized_path: Path for workflow storage
            initial_round: Starting round number
            max_rounds: Maximum optimization rounds
            validation_rounds: Validation iterations per round
        """
        # Handle both new (opt_llm) and legacy (opt_llm_config) interfaces
        if opt_llm is not None:
            self.optimize_llm = self._wrap_llm(opt_llm, "optimizer")
        elif opt_llm_config is not None:
            self.optimize_llm = create_llm_instance(opt_llm_config, role="optimizer")
        else:
            self.optimize_llm = create_llm_instance(role="optimizer")
        
        if exec_llm is not None:
            self.execute_llm = self._wrap_llm(exec_llm, "executor")
            self.execute_llm_config = exec_llm
        elif exec_llm_config is not None:
            self.execute_llm = create_llm_instance(exec_llm_config, role="executor")
            self.execute_llm_config = exec_llm_config
        else:
            self.execute_llm = create_llm_instance(role="executor")
            self.execute_llm_config = None

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators or []

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds

        # Initialize utility classes
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)
    
    def _wrap_llm(self, llm: Any, role: str) -> AsyncLLM:
        """Wrap an LLM adapter or config into AsyncLLM."""
        if isinstance(llm, AsyncLLM):
            return llm
        return create_llm_instance(llm, role=role)
    
    def optimize(self, mode: OptimizerType = "Graph"):
        """
        Main optimization entry point.
        
        Args:
            mode: "Graph" for workflow optimization, "Test" for evaluation
        """
        if mode == "Test":
            test_n = 1
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test())
            return None

        for opt_round in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    score = loop.run_until_complete(self._optimize_graph())
                    break
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Error occurred: {e}. Retrying... ({retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.error("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)

            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected at round {convergence_round}, final round: {final_round}"
                )
                self.convergence_utils.print_results()
                break

            time.sleep(2)

    async def test(self):
        """Run evaluation on test data."""
        rounds = [1]  # Can be extended
        data = []

        graph_path = f"{self.root_path}/workflows_test"
        json_file_path = self.data_utils.get_results_file_path(graph_path)
        data = self.data_utils.load_results(graph_path)

        for round_num in rounds:
            directory = self.graph_utils.create_round_directory(graph_path, round_num)
            self.graph = self.graph_utils.load_graph(round_num, graph_path)

            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test(
                self, directory, is_test=True
            )

            new_data = self.data_utils.create_result_data(round_num, score, avg_cost, total_cost)
            data.append(new_data)
            self.data_utils.save_results(json_file_path, data)
        
        return data

    async def _optimize_graph(self):
        """
        Single round of graph optimization.
        
        This is the core MCTS-based workflow optimization loop that:
        1. Evaluates current best workflows
        2. Generates new workflow candidates
        3. Evaluates and selects best candidates
        """
        validation_n = self.validation_rounds
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)

        # Initial round setup
        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            avg_score = await self.evaluation_utils.evaluate_graph(
                self, directory, validation_n, data, initial=True
            )

        # Generate and evaluate new workflows
        while True:
            directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)

            top_rounds = self.data_utils.get_top_rounds(self.sample)
            sample = self.data_utils.select_round(top_rounds)

            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
            graph = self.graph_utils.extract_solve_graph(graph_load)

            processed_experience = self.experience_utils.load_experience()
            experience = self.experience_utils.format_experience(processed_experience, sample["round"])

            operator_description = self.graph_utils.load_operators_description(self.operators)
            log_data = self.data_utils.load_log(sample["round"])

            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience, sample["score"], graph[0], prompt, operator_description, self.type, log_data
            )

            # Call optimizer LLM
            try:
                graph_formatter = XmlFormatter.from_model(GraphOptimize)
                response = await self.optimize_llm.call_with_format(
                    graph_optimize_prompt,
                    graph_formatter
                )
                logger.info("Graph optimization response received successfully")
            except Exception as e:
                logger.error(f"Format error in graph optimization: {e}")
                raw_response = await self.optimize_llm(graph_optimize_prompt)
                response = self._extract_fields_from_response(raw_response)
                if not response:
                    continue

            # Validate and save new workflow
            check_result, error = self.graph_utils.check_syntax(response)
            if not check_result:
                logger.warning(f"Syntax error in generated graph: {error}")
                continue

            self.graph_utils.write_graph_files(directory, response, self.round + 1, self.dataset)

            # Load and evaluate new graph
            self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)
            avg_score = await self.evaluation_utils.evaluate_graph(
                self, directory, validation_n, data
            )

            # Save experience
            self.experience_utils.save_experience(
                round_num=self.round + 1,
                score=avg_score,
                modification=response.get("modification", ""),
            )

            logger.info(f"Round {self.round + 1} average score: {avg_score}")
            return avg_score
    
    def _extract_fields_from_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Extract graph optimization fields from raw response.
        
        Fallback parser when structured formatting fails.
        """
        result = {
            "modification": "",
            "graph": "",
            "prompt": "",
        }
        
        # Simple extraction heuristics
        if "<modification>" in raw_response:
            start = raw_response.find("<modification>") + len("<modification>")
            end = raw_response.find("</modification>")
            if end > start:
                result["modification"] = raw_response[start:end].strip()
        
        if "<graph>" in raw_response:
            start = raw_response.find("<graph>") + len("<graph>")
            end = raw_response.find("</graph>")
            if end > start:
                result["graph"] = raw_response[start:end].strip()
        
        if "<prompt>" in raw_response:
            start = raw_response.find("<prompt>") + len("<prompt>")
            end = raw_response.find("</prompt>")
            if end > start:
                result["prompt"] = raw_response[start:end].strip()
        
        return result if result["graph"] else None
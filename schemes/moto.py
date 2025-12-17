# schemes/moto.py
import os
import random
import asyncio
from pathlib import Path

from llm import global_llm, configure_llm, request_cost

from myopto.trace import bundle
from myopto.optimizers import OptoPrime
from myopto import trace
from myopto.trace.utils import render_opt_step

from tqdm import tqdm

from .base import BaseScheme
from .prompt.moto import META_PROMPTS

from utils.logs import logger
from utils import writef

from debug import check

@bundle(trainable=False,traceable_code=True)
def llm(prompt: str):
    return global_llm(str(prompt))

@bundle(trainable=True, traceable_code=True)
def solution_workflow(problem: str) -> str:
    """
    Solves a math problem and extracts the answer.
    """
    # Plan
    plan = llm(f"Create a step-by-step plan to solve: {problem}")
    
    # Execute
    solution = llm(f"Solve the problem following this plan: {plan}. Problem: {problem}")
    
    # Extract
    final_answer = llm(f"Extract exactly the final answer from this text: {solution}. Return ONLY the answer.")
    
    return final_answer


def get_feedback_str(problem: str, gold: str, pred: str, is_correct: bool) -> str:
    if is_correct:
        return f"Test Case Passed! (Problem: {problem})"
    
    return (
        "Test Case Failed.\n"
        f"Problem: {problem}\n"
        f"Expected: {gold}\n"
        f"Got: {pred}\n\n"
        "DIAGNOSIS:\n"
        "1. If the numbers match but format differs, adjust the extraction logic.\n"
        "2. If the numbers are different, the reasoning is flawed.\n"
        f"{META_PROMPTS}"
    )

@bundle(trainable=False)
def concat(*items):
    out = ""
    for i, item in enumerate(items):
        out += f"ID {i}: {item}\n"
    return out

# --- 2. The Scheme Class ---
class MotoScheme(BaseScheme):
    def __init__(self, args):
        super().__init__(args)
        configure_llm(model=self.args.exe_model, usage=False)
        self.optimizer = OptoPrime(solution_workflow.parameters())

    async def train(self, train_benchmark, train_indices, test_benchmark=None, test_indices=None, test_freq=1):
        logger.info(f"\n=== Starting MOTO Training ({self.args.epochs} epochs) ===")
        
        # Load Raw Data
        train_data = await train_benchmark.load_data(specific_indices=train_indices)
        logger.info(f"Loaded {len(train_data)} training examples.")

        for epoch in range(1, self.args.epochs+1):
            logger.info(f"\n--- Epoch {epoch}/{self.args.epochs} ---")

            # Optional: Test during training
            logger.warning('skip mid-training validation')
            # if test_benchmark and test_indices and (epoch+3) % self.args.val_interval == 0:
            #     logger.info(f"Running Mid-Training Validation on epoch {epoch}..")
            #     self.prep_test()
            #     await test_benchmark.run_baseline(self.inference, specific_indices=test_indices)
            #     configure_llm(usage=False)
            # random.shuffle(train_data)
            # Batch Processing

            batch_size = self.args.batch_size
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i : i + batch_size]
                outputs = []
                feedbacks = []
                desc = f"Processing Batch {i//batch_size + 1}/{(len(train_data) - 1) // batch_size + 1}"
                for example in tqdm(batch, ncols=88, desc=desc):
                    problem = example[train_benchmark.q_key]
                    gold = example[train_benchmark.a_key]
                    try:
                        prediction_node = solution_workflow(problem)
                        prediction_str = prediction_node.data
                        score, cleaned = train_benchmark.calculate_score(gold, prediction_str)
                        is_correct = (score == 1)
                        if not is_correct:
                            exp, pred = cleaned
                            logger.info(f"\n>>> Expected: {exp} \n <<< Got: {pred}")
                        fb_text = get_feedback_str(problem, gold, prediction_str, is_correct)

                    except trace.ExecutionError as e:
                        prediction_node = e.exception_node
                        fb_text = str(e.exception_node.data)

                    logger.debug(fb_text)
                    outputs.append(prediction_node)
                    feedbacks.append(fb_text)
                    
                batched_outputs = concat(*outputs)      # MessageNode[str-ish]
                batched_feedback = concat(*feedbacks)   # MessageNode[str]

                self.optimizer.zero_feedback()
                self.optimizer.backward(batched_outputs, batched_feedback.data, visualize=True, print_limit=1e5)
                input('pause after optimizer.backward()')

                tg = self.optimizer.trace_graph
                logger.info(f"TraceGraph size = {len(tg.graph)}")

                for lvl, node in tg.graph:   # no limit
                    logger.info(
                        f"[L{lvl}] {node.py_name} | "
                        f"parents={[p.py_name for p in node.parents]}"
                    )
                input('pause after optimizer.trace_graph')

                from myopto.optimizers.optoprime import node_to_function_feedback

                ff = node_to_function_feedback(tg)

                logger.info("=== FUNCTION GRAPH (FULL) ===")
                for lvl, line in ff.graph:   # no limit
                    logger.info(f"[L{lvl}] {line}")

                logger.info("Roots:")
                for k in ff.roots:
                    logger.info(f"  {k}")

                logger.info("Outputs:")
                for k in ff.output:
                    logger.info(f"  {k}")

                input("pause before step")

                check()
                self.optimizer.step()

            # Save Checkpoint
            self.save_model(epoch)

    def prep_test(self):
        configure_llm(usage=True)
        self._free_solution = solution_workflow.fun

    async def inference(self, input_text: str) -> tuple[str, float]:
        token = request_cost.set(0.0)

        def _run_free():
            result = self._free_solution(input_text)
            total = request_cost.get()
            return str(result), total

        try:
            # Get both prediction and cost from the worker thread
            prediction_str, total_cost = await asyncio.to_thread(_run_free)
            return prediction_str, total_cost
        except Exception as e:
            return f"Error: {str(e)}", 0.0
        finally:
            # Restore previous ContextVar state for this async task
            request_cost.reset(token)


    def save_model(self, epoch=None):
        code = solution_workflow.parameters()[0].data
        path = self.scheme_file
        if epoch is not None:
            writef(path.with_name(f"{path.name}.{epoch}"), code)
        writef(path, code)

    def load(self, path: Path):
        path = Path(path)
        if not path.exists():
            return False
        
        logger.info(f"Loading scheme from {path}")
        with path.open('r') as f:
            code = f.read()
        solution_workflow.parameters()[0]._set(code)
        return True
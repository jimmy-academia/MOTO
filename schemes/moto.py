# schemes/moto.py
import os
import random
from pathlib import Path

from llm import LLMClient, request_cost

from myopto.trace import bundle
from myopto.optimizers import OptoPrime
from myopto import trace

from tqdm import tqdm

# patch for parallel batch
# https://docs.google.com/document/d/1Jk-GwRUlLyUYK4qpcQrMBhU8sgmg2WlOl0obf2Akhz0/edit?usp=sharing

from .base import BaseScheme
from ._moto_prompt import META_PROMPTS

from utils.logs import logger
from utils import writef

# --- 1. The Agent (Synchronous Bundle) ---
llm_client = LLMClient(model="gpt-4o-mini") 

def llm(prompt: str) -> str:
    return llm_client.answer(prompt)

@bundle(trainable=True)
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
        self.optimizer = OptoPrime(solution_workflow.parameters())
        self.scheme_file = self.scheme_file.with_name("code.py")

    async def inference(self, input_text: str) -> tuple[str, float]:
        # Reset cost for this specific async task context
        # .set() returns a token we can use to reset if needed, but here we just want to start at 0
        token = request_cost.set(0.0)
        try:
            # 1. Run the synchronous bundle
            # Disabling tracing to eliminate unnecessary graph overhead in evaluation.
            with trace.stop_tracing():
                # Since solution_workflow returns a MessageNode, we access .data
                prediction_node = solution_workflow(input_text)
                prediction_str = str(prediction_node.data)
            
            # 2. Return format (Answer, Cost)
            total_cost = request_cost.get()
            return prediction_str, total_cost
        
        except Exception as e:
            return f"Error: {str(e)}", 0.0
        finally:
            # Good practice to reset context, though not strictly required if tasks are isolated
            request_cost.reset(token)

    async def train(self, train_benchmark, train_indices, test_benchmark=None, test_indices=None, test_freq=1):
        logger.info(f"\n=== Starting MOTO Training ({self.args.epochs} epochs) ===")
        
        # Load Raw Data
        train_data = await train_benchmark.load_data(specific_indices=train_indices)
        logger.info(f"Loaded {len(train_data)} training examples.")

        for epoch in range(self.args.epochs):
            logger.info(f"\n--- Epoch {epoch+1}/{self.args.epochs} ---")

            # Optional: Test during training
            if test_benchmark and test_indices and epoch % self.args.val_interval == 0:
                logger.info("Running Mid-Training Validation...")
                await test_benchmark.run_baseline(self.inference, specific_indices=test_indices)

            random.shuffle(train_data)
            
            # Batch Processing
            batch_size = self.args.batch_size
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i : i + batch_size]
                outputs = []
                feedbacks = []
                
                desc = f"Processing Batch {i//batch_size + 1}/{len(train_data)//batch_size + 1}"
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
                self.optimizer.backward(batched_outputs, batched_feedback.data)
                self.optimizer.step()

            # Save Checkpoint
            self.save_model(epoch)
            
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
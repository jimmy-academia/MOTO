import os
import random
from pathlib import Path

from llm import LLMClient

from myopto.trace import bundle
from myopto.optimizers import OptoPrime

from .base import BaseScheme

from utils.logs import logger

# --- 1. The Agent (Synchronous Bundle) ---
llm_client = LLMClient(model="gpt-4o-mini") 

def llm_call(prompt: str) -> str:
    return llm_client.answer(prompt)

@bundle(trainable=True)
def solution_workflow(problem: str) -> str:
    """The synchronous logic graph."""
    plan = llm_call(f"Create a step-by-step plan to solve: {problem}")
    solution = llm_call(f"Solve the problem following this plan: {plan}. Problem: {problem}")
    return solution

# --- 2. The Scheme Class ---
class MotoScheme(BaseScheme):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = OptoPrime(solution_workflow.parameters())

    # [THE FIX] The benchmark calls 'await graph(input)'. 
    # We must provide an async function that returns (answer_str, cost_float).
    async def inference(self, input_text: str) -> tuple[str, float]:
        try:
            # 1. Run the synchronous bundle
            # Since solution_workflow returns a MessageNode, we access .data
            prediction_node = solution_workflow(input_text)
            prediction_str = str(prediction_node.data)
            
            # 2. Return format (Answer, Cost)
            # You can calculate actual cost here if your LLMClient tracks it
            return prediction_str, 0.0 
        except Exception as e:
            return f"Error: {str(e)}", 0.0

    async def train(self, train_benchmark, train_indices, val_benchmark=None, val_indices=None):
        logger.info(f"\n=== Starting MOTO Training ({self.args.epochs} epochs) ===")
        
        # Load Raw Data
        train_data = await train_benchmark.load_data(specific_indices=train_indices)
        logger.info(f"Loaded {len(train_data)} training examples.")

        for epoch in range(self.args.epochs):
            logger.info(f"\n--- Epoch {epoch+1}/{self.args.epochs} ---")
            random.shuffle(train_data)
            
            # Batch Processing
            batch_size = self.args.batch_size
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i : i + batch_size]
                outputs = []
                feedbacks = []
                
                for example in batch:
                    try:
                        # 1. Forward Pass (Trace Graph)
                        prediction_node = solution_workflow(example['problem'])
                        prediction_str = prediction_node.data
                        
                        # 2. Score (Using Benchmark Logic)
                        score, _ = train_benchmark.calculate_score(example['answer'], prediction_str)
                        
                        # 3. Generate Feedback
                        if score == 1:
                            fb = "Correct."
                        else:
                            fb = (f"Failed.\nProblem: {example['problem']}\n"
                                  f"Expected: {example['answer']}\nGot: {prediction_str}")
                            # Only optimize on failure? Or both? (Strategy dependent)
                            outputs.append(prediction_node)
                            feedbacks.append(fb)
                            
                    except Exception as e:
                        logger.error(f"Error processing item: {e}")

                # 4. Backward Pass (Optimize)
                if outputs:
                    self.optimizer.zero_feedback()
                    for node, fb in zip(outputs, feedbacks):
                        self.optimizer.backward(node, fb)
                    self.optimizer.step()
                    
                logger.info(f"Batch {i//batch_size + 1}/{len(train_data)//batch_size + 1} Processed")

            # Optional: Test during training
            if val_benchmark and val_indices:
                logger.info("Running Mid-Training Validation...")
                await val_benchmark.run_baseline(self.inference, specific_indices=val_indices)

            # Save Checkpoint
            self.save_model()
            
    def save_model(self):
        code = solution_workflow.parameters()[0].data
        path = Path(self.args.output_dir) / f"{self.model_name}_optimized.py"
        self.save(path, code)

    def load(self, path: Path):
        path = Path(path)
        if not path.exists():
            return False
        
        logger.info(f"Loading scheme from {path}")
        with path.open('r') as f:
            code = f.read()
        solution_workflow.parameters()[0].data = code
        return True
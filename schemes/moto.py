import os
import random
from pathlib import Path

from llm import LLMClient

from myopto.trace import bundle
from myopto.optimizers import OptoPrime
from myopto import trace

# patch for parallel batch
# https://docs.google.com/document/d/1Jk-GwRUlLyUYK4qpcQrMBhU8sgmg2WlOl0obf2Akhz0/edit?usp=sharing

from .base import BaseScheme

from utils.logs import logger

# --- 1. The Agent (Synchronous Bundle) ---
llm_client = LLMClient(model="gpt-4o-mini") 

def llm_call(prompt: str) -> str:
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

META_PROMPTS = """
You are an expert AI Systems Engineer optimizing a Python function `solution_workflow` to solve complex math problems.

Your goal is to modify the code to maximize accuracy on unseen test cases.

### DIAGNOSIS INSTRUCTIONS
Analyze the "Expected" vs "Got" values in the error log above:
1. **Logic Error** (Wrong number/result): The LLM reasoning failed.
   - *Fix:* Introduce "Chain of Thought" (Ask for step-by-step reasoning).
   - *Fix:* Add a "Planning" step before the solution step.
   - *Fix:* Add a "Verification" step where an LLM reviews the previous answer.
2. **Extraction Error** (Correct number buried in text): The LLM solved it, but the function returned extra words.
   - *Fix:* Improve the Python string parsing (use `re` module, split lines).
   - *Fix:* Enforce stricter output formats in the prompt (e.g., "Output ONLY the number").
   - *Fix:* Add a specific "Extraction" LLM call to isolate the final answer.

### OPTIMIZATION STRATEGIES (Use these!)
- **Architecture:** Don't just rely on one prompt. Build a pipeline: `Plan -> Solve -> Verify -> Sanitize`.
- **Python Power:** Use Python logic to make the code robust.
   - Use `try/except` blocks to handle parsing failures.
   - Use `if` statements to check if an answer looks empty or invalid, and retry if needed.
   - Use `re` (regex) to find numbers or patterns like `\boxed{...}`.
- **Prompt Engineering:**
   - Assign roles ("You are a math expert...").
   - Use delimiters (e.g., "Put your final answer inside <ANSWER> tags") to make extraction easier.

### CONSTRAINTS
1. **Generalization:** Do NOT hard-code answers for the specific problem in the log. The code must solve *any* similar math problem.
2. **Validity:** The output must be valid, runnable Python code.
3. **Efficiency:** Keep the code readable. Do not add infinite loops.

### OUTPUT
Return **only** the full, updated Python source code for `solution_workflow`.
"""

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
                    problem = example[train_benchmark.q_key]
                    gold = example[train_benchmark.a_key]
                    try:
                        prediction_node = solution_workflow(problem)
                        prediction_str = prediction_node.data
                        score, _ = train_benchmark.calculate_score(gold, pred_str)
                        is_correct = (score == 1)
                        if not is_correct:
                            logger.info(f"Expected: {gold} | Got: {pred}")
                        fb_text = get_feedback_str(problem, gold, pred_str, is_correct)

                    except trace.ExecutionError as e:
                        pred_node = e.exception_node
                        fb = str(e.exception_node.data)

                    logger.debug(fb)
                    correctness = pred_node.eq(gold)
                    outputs.append(correctness)
                    feedbacks.append(fb)
                    
                batched_outputs = concat(*outputs)      # MessageNode[str-ish]
                batched_feedback = concat(*feedbacks)   # MessageNode[str]

                self.optimizer.zero_feedback()
                self.optimizer.backward(batched_outputs, batched_feedback.data)
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
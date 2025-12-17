import os
import random
import asyncio
from pathlib import Path

# --- Imports from your project ---
from llm import LLMClient
from myopto.trace import bundle, ExecutionError
from myopto.optimizers import OptoPrime
from .base import BaseScheme
from utils.logs import logger
from utils import writef # Helper to save file

# ==============================================================================
# 1. AGENT & PROMPTS (Global Scope)
# ==============================================================================

llm_client = LLMClient(model="gpt-4o-mini")

def llm_call(prompt: str) -> str:
    """Synchronous LLM call wrapper."""
    return llm_client.answer(prompt)

@bundle(trainable=True)
def solution_workflow(problem: str) -> str:
    """
    Solves a math problem and extracts the answer.
    The optimizer will rewrite the code inside this function.
    """
    # Plan
    plan = llm_call(f"Create a step-by-step plan to solve: {problem}")
    
    # Execute
    solution = llm_call(f"Solve the problem following this plan: {plan}. Problem: {problem}")
    
    # Extract
    final_answer = llm_call(f"Extract exactly the final answer from this text: {solution}. Return ONLY the answer.")
    
    return final_answer

@bundle(trainable=False)
def concat(*items):
    """Utility to combine multiple outputs into one Trace Node for batch processing."""
    out = ""
    for i, item in enumerate(items):
        out += f"ID {i}: {item}\n"
    return out

# --- Feedback Logic ---

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
   - Use `re` (regex) to find numbers or patterns like `\\boxed{...}`.
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

# ==============================================================================
# 2. MOTO SCHEME CLASS
# ==============================================================================

class MotoScheme(BaseScheme):
    def __init__(self, args):
        super().__init__(args)
        # Initialize Optimizer tracking the global bundle
        self.optimizer = OptoPrime(solution_workflow.parameters())

    async def inference(self, input_text: str) -> tuple[str, float]:
        """
        Wrapper to allow the synchronous agent to be called asynchronously by the benchmark.
        """
        try:
            # Run in thread to prevent blocking the event loop
            prediction_node = await asyncio.to_thread(solution_workflow, input_text)
            prediction_str = str(prediction_node.data)
            return prediction_str, 0.0
        except Exception as e:
            return f"Error: {str(e)}", 0.0

    async def train(self, train_benchmark, train_indices, val_benchmark=None, val_indices=None):
        logger.info(f"\n=== Starting MOTO Training ({self.args.epochs} epochs) ===")
        
        # Load Data
        train_data = train_benchmark.load_data(limit=self.args.train_limit, shuffle=True)
        logger.info(f"Loaded {len(train_data)} training examples.")

        for epoch in range(self.args.epochs):
            logger.info(f"\n--- Epoch {epoch+1}/{self.args.epochs} ---")
            random.shuffle(train_data)
            
            batch_size = self.args.batch_size
            
            # --- BATCH LOOP ---
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i : i + batch_size]
                
                # A. Parallel Execution Helper
                async def process_example(example):
                    problem = example[train_benchmark.q_key]
                    gold = example[train_benchmark.a_key]
                    try:
                        # 1. Forward Pass (in thread)
                        # We MUST wrap this because solution_workflow is synchronous
                        prediction_node = await asyncio.to_thread(solution_workflow, problem)
                        pred_str = str(prediction_node.data)
                        
                        # 2. Evaluation
                        # benchmark.calculate_score returns (1, parsed_ans) or (0, parsed_ans)
                        score, _ = train_benchmark.calculate_score(gold, pred_str)
                        is_correct = (score == 1)
                        
                        # 3. Generate Feedback
                        fb_text = get_feedback_str(problem, gold, pred_str, is_correct)
                        
                        return prediction_node, fb_text, is_correct
                        
                    except ExecutionError as e:
                        # Handle Trace internal errors (e.g., LLM failure inside bundle)
                        logger.error(f"Trace Execution Error: {e}")
                        # Return the node that caused exception if available
                        node = e.exception_node if hasattr(e, 'exception_node') else None
                        return node, f"Execution Failed: {e}", False
                    except Exception as e:
                        logger.error(f"General Error: {e}")
                        return None, None, False

                # B. Execute Batch concurrently
                results = await asyncio.gather(*[process_example(ex) for ex in batch])
                
                # C. Aggregate Results
                outputs = []
                feedbacks = []
                correct_count = 0
                
                for res in results:
                    node, fb, correct = res
                    if node is not None:
                        outputs.append(node)
                        feedbacks.append(fb)
                        if correct: correct_count += 1

                # D. Backward Pass (Optimization)
                if outputs:
                    # 1. Combine inputs/outputs into single batch nodes
                    batched_outputs = concat(*outputs)
                    batched_feedback = concat(*feedbacks) # Returns a Node
                    
                    # 2. Step
                    self.optimizer.zero_feedback()
                    # Pass the Node for outputs, but the string data for feedback
                    self.optimizer.backward(batched_outputs, batched_feedback.data)
                    self.optimizer.step()

                logger.info(f"Batch {i//batch_size + 1}/{len(train_data)//batch_size + 1} | Acc: {correct_count}/{len(batch)}")

            # --- END OF EPOCH ---
            
            # Optional: Validation
            if val_benchmark and val_indices:
                logger.info("Running Mid-Training Validation...")
                # We use specific_indices to limit validation size
                await val_benchmark.run_baseline(
                    agent=self.inference, 
                    specific_indices=val_indices,
                    max_concurrent_tasks=10
                )

            # Save Checkpoint
            self.save_model()

    def save_model(self):
        # Extract the optimized python code
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
        
        # Inject the loaded code into the live bundle
        solution_workflow.parameters()[0].data = code
        return True
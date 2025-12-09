import os
from llm import get_key, LLMClient
from utils import writef, Logger
import argparse

# global logger instance
log = Logger()
log.init()

# opt_model = "gpt-5-mini"
opt_model = "gpt-5.1"
log(f"optimizing with {opt_model}")

os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LiteLLM"
os.environ["OPENAI_API_KEY"] = get_key()
os.environ["TRACE_LITELLM_MODEL"] = opt_model
os.environ["LITELLM_LOG"] = "INFO"

from opto.trace import bundle
from opto.optimizers import OptoPrime
from opto import trace

from tqdm import tqdm

from data import MATH_EXAMPLES

parser = argparse.ArgumentParser()
parser.add_argument("-m", type=int, default=4, help="Which model to use for llm() execution.",)
args = parser.parse_args()

if args.m == 5:
    model = "gpt-5-nano"
elif args.m == 4:
    model = 'gpt-4o-mini'

log(f'executing with {model}')
llm_client = LLMClient(model=model)

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

def get_feedback(problem: str, gold: str, pred: str) -> str:
    gold_str = gold.strip()
    pred_str = pred.strip()
    if pred_str == gold_str:
        return f"test case passed! (Problem: {problem})"
    else:
        return (
            "test case failed.\n"
            f"Problem: {problem}\n"
            f"Expected final answer: {gold_str}\n"
            f"Got: {pred_str}\n"
            "Please improve the body of `solution_workflow` so that it produces precisely the expected final answer"
            f"{META_PROMPTS}"
        )

@bundle(trainable=False)
def concat(*items):
    out = ""
    for i, item in enumerate(items):
        out += f"ID {i}: {item}\n"
    return out


def train_math(MATH_EXAMPLES, epochs: int = 2, batch_size: int = 7):
    optimizer = OptoPrime(solution_workflow.parameters())
    MATH_EXAMPLES = MATH_EXAMPLES[:10]
    n = len(MATH_EXAMPLES)
    train_samples = MATH_EXAMPLES[:]
    batch_size = 2
    for epoch in range(epochs):
        log(f"\n=== Epoch {epoch} ===")

        total = num_correct = start = end = 0
        sub_correct = 0
        fail_samples = []
        # one tqdm bar per epoch
        with tqdm(total=n, ncols=88, desc=f"Epoch {epoch}", leave=True) as pbar:

            if batch_size < 5 and sub_correct == batch_size:
                batch_size += 1

            sub_correct = 0
            while end != n:
                end = min(start + batch_size, n)
                batch = MATH_EXAMPLES[start:end]
                start = end

            # for start in range(0, n, batch_size):
                # end = min(start + batch_size, n)
                # batch = MATH_EXAMPLES[start:end]

                outputs = []    # list[MessageNode[bool]] for this batch
                feedbacks = []  # list[str] for this batch

                for ex in batch:
                    problem = ex["problem"]
                    gold = ex["answer"]

                    try:
                        # Directly call the trainable function
                        pred_node = solution_workflow(problem)   # MessageNode[str]
                        pred_str = pred_node.data          # underlying string
                        fb = get_feedback(problem, gold, pred_str)
                    except trace.ExecutionError as e:
                        # If the function crashes, treat that as the prediction node
                        pred_node = e.exception_node
                        fb = str(e.exception_node.data)

                    log(fb, say=False)
                    correctness = pred_node.eq(gold)   # MessageNode[bool]
                    outputs.append(correctness)
                    feedbacks.append(fb)

                    total += 1
                    if bool(correctness.data):
                        num_correct += 1
                        sub_correct += 1
                        log('correct!')
                    else:
                        log('incorrect!')
                    running_acc = num_correct / total if total > 0 else 0.0
                    sub_running_acc = sub_correct / batch_size

                    pbar.set_postfix(
                        acc=f"{running_acc:.3f}",
                        sacc=f"{sub_running_acc:.3f}",
                        correct=f"{num_correct}/{total}",
                        bs = batch_size
                    )
                # one optimization step per batch
                batched_outputs = concat(*outputs)      # MessageNode[str-ish]
                batched_feedback = concat(*feedbacks)   # MessageNode[str]

                optimizer.zero_feedback()
                optimizer.backward(batched_outputs, batched_feedback.data)
                optimizer.step()

                # update tqdm
                pbar.update(len(batch))
                running_acc = num_correct / total if total > 0 else 0.0
                pbar.set_postfix(
                    acc=f"{running_acc:.3f}",
                    correct=f"{num_correct}/{total}",
                )
                src = solution_workflow.parameters()[0].data

                log("\nUpdated solution_workflow source:\n")
                log(src)
                log("-" * 60)

        epoch_acc = num_correct / total if total > 0 else 0.0
        log(f"Epoch {epoch} accuracy: {num_correct}/{total} = {epoch_acc:.3f}")
        # 🔥 NEW: log the updated function
        
        train_samples = fail_samples[:]

        if total > 0 and num_correct == total and len(train_samples) == len(MATH_EXAMPLES):
            log("All training examples solved.")
            break
        else:
            train_samples = MATH_EXAMPLES[:]


        final_src = solution_workflow.parameters()[0].data
        writef(f"output/{model}_math_solver.py", final_src)
        log("Saved optimized solver to math_solver.py")
        log.saveto(f"output/{model}_log")
    return solution_workflow


if __name__ == "__main__":

    trained_solver = train_math(MATH_EXAMPLES)


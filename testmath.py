import os
from llm import get_key, LLMClient
from utils import writef
import argparse

os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LiteLLM"
os.environ["OPENAI_API_KEY"] = get_key()
os.environ["TRACE_LITELLM_MODEL"] = "gpt-5-mini"  # optimizer LLM
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

print(f'executing with {model}')
llm_client = LLMClient(model=model)


def llm(prompt: str) -> str:
    return llm_client.answer(prompt)


# -----------------------------------------------------------
# 1. Single trainable Python solver (directly optimized)
# -----------------------------------------------------------
@bundle(trainable=True)
def math_script(problem: str):
    """
    Trainable Python solver for MATH.

    GOAL
    - Learn a solver that generalizes to unseen MATH problems.
    - Do NOT overfit to this specific batch of training problems.

    STRICTLY FORBIDDEN (for the editing LLM)
    - Do NOT add code that hard-codes answers for particular phrasings, e.g.:
          if "convex pentagon" in problem.lower():
              return "135"
    - Do NOT detect long, specific substrings (full sentences, weird decimals,
      exact names) and map them directly to fixed answers.
    - Do NOT build long chains of `if "...something very specific..." in problem`
      each returning a constant.

    ALLOWED / ENCOURAGED
    - Use `llm(...)` as the main reasoning engine; multiple calls are fine.
    - Use intermediate variables s1, s2, s3, ... to store stepwise reasoning.
    - Build prompts with f-strings including `problem` and earlier steps.
    - Ask the LLM to put the final answer on a clearly marked last line, then
      parse that answer in Python.
    - Any branching you add must correspond to reusable patterns (problem types),
      not single memorized questions.

    OUTPUT
    - Return a STRING containing ONLY the final answer (no explanation).
    """
    s1 = llm(
        f"""You are an expert competition mathematician.

Problem: {problem}

1. Think step by step and solve the problem.
2. On the LAST line, write: ANSWER: <final answer only, numeric or simplified expression>"""
    )

    # Extract final answer from the last line containing 'ANSWER:'
    lines = s1.strip().splitlines()
    for line in reversed(lines):
        if "ANSWER:" in line:
            ans = line.split("ANSWER:", 1)[1].strip()
            if ans:
                return ans

    # Fallback: if the model ignored the format, just return trimmed output
    return s1.strip()

    
# -----------------------------------------------------------
# 2. Utility for batching feedback
# -----------------------------------------------------------
@bundle(trainable=False)
def concat(*items):
    out = ""
    for i, item in enumerate(items):
        out += f"ID {i}: {item}\n"
    return out


# -----------------------------------------------------------
# 3. Feedback + training loop
# -----------------------------------------------------------
def get_feedback(problem: str, gold: str, pred: str) -> str:
    gold_str = gold.strip()
    pred_str = pred.strip()
    if pred_str == gold_str:
        return "test case passed!"
    else:
        return (
            "test case failed.\n"
            f"Problem: {problem}\n"
            f"Expected final answer: {gold_str}\n"
            f"Got: {pred_str}\n"
            "Please improve the body of `math_script` so that it produces the correct "
            "numeric answer while following the inline comments. "
        )


def train_math(MATH_EXAMPLES, epochs: int = 5, batch_size: int = 5):
    optimizer = OptoPrime(math_script.parameters())
    MATH_EXAMPLES = MATH_EXAMPLES[:5]
    n = len(MATH_EXAMPLES)
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch} ===")

        total = 0
        num_correct = 0

        # one tqdm bar per epoch
        with tqdm(total=n, ncols=88, desc=f"Epoch {epoch}", leave=True) as pbar:

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = MATH_EXAMPLES[start:end]

                outputs = []    # list[MessageNode[bool]] for this batch
                feedbacks = []  # list[str] for this batch

                for ex in batch:
                    problem = ex["problem"]
                    gold = ex["answer"]

                    try:
                        # Directly call the trainable function
                        pred_node = math_script(problem)   # MessageNode[str]
                        pred_str = pred_node.data          # underlying string
                        fb = get_feedback(problem, gold, pred_str)
                    except trace.ExecutionError as e:
                        # If the function crashes, treat that as the prediction node
                        pred_node = e.exception_node
                        fb = str(e.exception_node.data)

                    correctness = pred_node.eq(gold)   # MessageNode[bool]
                    outputs.append(correctness)
                    feedbacks.append(fb)

                    total += 1
                    if bool(correctness.data):
                        num_correct += 1

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

        epoch_acc = num_correct / total if total > 0 else 0.0
        print(f"Epoch {epoch} accuracy: {num_correct}/{total} = {epoch_acc:.3f}")
        # 🔥 NEW: print the updated function
        src = math_script.parameters()[0].data

        print("\nUpdated math_script source:\n")
        print(src)
        print("-" * 60)
        if total > 0 and num_correct == total:
            print("All training examples solved.")
            break

    final_src = math_script.parameters()[0].data
    writef(f"output/{model}_math_solver.py", final_src)
    print("Saved optimized solver to math_solver.py")
    # After training, math_script is the optimized solver
    return math_script


if __name__ == "__main__":

    trained_solver = train_math(MATH_EXAMPLES)


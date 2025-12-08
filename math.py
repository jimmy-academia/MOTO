import os
from llm import get_key, LLMClient
from utils import writef

os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LiteLLM"
os.environ["OPENAI_API_KEY"] = get_key()
os.environ["TRACE_LITELLM_MODEL"] = "gpt-5-mini"  # optimizer LLM
os.environ["LITELLM_LOG"] = "INFO"

from opto.trace import bundle
from opto.optimizers import OptoPrime
from opto import trace

from tqdm import tqdm

from data import MATH_EXAMPLES

llm_client = LLMClient(model="gpt-5-nano")


def llm(prompt: str) -> str:
    return llm_client.answer(prompt)


# -----------------------------------------------------------
# 1. Single trainable Python solver (directly optimized)
# -----------------------------------------------------------
@bundle(trainable=True)
def math_script(problem: str):
    """
    Trainable Python solver for MATH.

    Guidelines (for the editing LLM):
    - Use `llm(...)` as the main reasoning engine; you may call it multiple times.
    - Prefer intermediate variables named s1, s2, s3, ... to store stepwise reasoning.
    - Build prompts using f-strings that include `problem` and earlier steps, e.g.:
          s2 = llm(f"Given the problem: {problem} and the reasoning so far: {s1} ...")
    - You may use loops, lists, and standard Python expressions to organize the workflow.
    - At the end, return a STRING containing ONLY the final numeric answer.
    """
    # Baseline: single-shot answer extraction.
    s1 = llm(
        f"You are an expert competition mathematician.\n"
        f"Problem: {problem}\n\n"
        "Solve this problem carefully and respond with ONLY the final numeric answer."
    )
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
            "Do NOT change the function name or its parameters."
        )


def train_math(MATH_EXAMPLES, epochs: int = 5, batch_size: int = 5):
    optimizer = OptoPrime(math_script.parameters())
    MATH_EXAMPLES = MATH_EXAMPLES[:33]
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
        print("\nUpdated math_script source:\n")
        print(math_script.data)
        print("-" * 60)
        if total > 0 and num_correct == total:
            print("All training examples solved.")
            break

    ffinal_src = math_script.data
    writef("math_solver.py", final_src)
    print("Saved optimized solver to math_solver.py")
    # After training, math_script is the optimized solver
    return math_script


if __name__ == "__main__":
    trained_solver = train_math(MATH_EXAMPLES)


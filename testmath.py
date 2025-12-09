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
    var_1 = llm(f"solve the given input {problem}")
    var_2 = llm(f"do something with step 1 intermediate result {var_1}")
    var_3 = llm(f"do something with step 2 intermediate result {var_2}")
    var_4 = llm(f"output final answer from previous results {var_1}, {var_2}, {var_3}")
    return var_4

META_PROMPTS = """
You are editing Python code that defines an LLM-based solution workflow.

Goal:
- Transform the function into a robust, general-purpose solver for many unseen problems, not just the given training examples.

LLM steps:
- You may create multiple intermediate LLM calls as needed by using:
      s1 = llm(f"... prompt with input {{var}} ...")
- Store each call’s result in variables (e.g., s1, s2, s3, plan, check) and pass them into later steps.
- Encourage explicit reasoning: plan → reason → verify → finalize.

Allowed edits:
- STRUCTURAL: split a large `llm(...)` call into smaller steps; merge redundant steps; add planning, checking, or repair steps.
- Hint: you can try multiple steps in parallel, or plan then solve, or check then revise answer, or breakdown into smaller steps.
- Note: different prompt will provide different effects. e.g. Chain-of-thought: "let's think step by step" can provide grounded reasoning.
- PYTHON LOGIC: add helper functions, loops, conditionals, or data structures if they clarify or stabilize the workflow.
- PROMPTS: rewrite prompts to be concrete and structured, with clear input/output expectations.

Error handling:
- Distinguish syntax errors (code does not run) from reasoning errors (bad logic or incorrect answers).
- Fix both: ensure valid Python and improve the reasoning process.

Constraints:
- NEVER hard-code answers or patterns tied to specific training questions.
- Do NOT overfit; design a general workflow that can handle new inputs.
- Keep the function readable and maintainable.

Output:
- Return the full, updated function source code.
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
    MATH_EXAMPLES = MATH_EXAMPLES[:35]
    n = len(MATH_EXAMPLES)
    train_samples = MATH_EXAMPLES[:]
    batch_size = 1
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


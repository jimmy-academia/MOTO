import os
from llm import get_key, LLMClient
from utils import writef, Logger
import argparse


# global logger instance
log = Logger()
log.init()

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


# -----------------------------------------------------------
# 1. Single trainable Python solver (directly optimized)
# -----------------------------------------------------------
@bundle(trainable=True)
def math_script(problem: str) -> str:
    """
    Trainable workflow for competition-style MATH problems.

    OVERALL GOAL
    - Learn a general-purpose solver that works on unseen problems.
    - Use the `llm(...)` function as the main reasoning engine.
    - Keep the function signature fixed: math_script(problem: str) -> str (answer only).

    ALLOWED STRUCTURAL EDITS (OPERATOR-LIKE MOVES)
    - SPLIT_STEP:
        - You may replace a single large llm(...) call with multiple smaller, named steps
          (e.g., classification, planning, solving, verification).
    - MERGE_STEPS:
        - You may merge redundant or unhelpful steps into a simpler workflow.
    - ADD_VERIFIER:
        - You may add steps that check earlier outputs, detect likely errors, and revise
          the solution before returning an answer.
    - BRANCH_BY_TYPE:
        - You may classify the problem into a small set of coarse types
          (e.g., algebra / geometry / number theory / combinatorics)
          and route through specialized sub-workflows.
        - This routing must be based on high-level features, not memorized phrases.
    - REFINE_PROMPTS:
        - You may rewrite prompts to be more explicit, add input/output schemas,
          request multiple candidate solutions and then select among them, etc.
    - RETRY_OR_LOOP:
        - When a verifier step finds issues, you may re-call llm(...) with a clarified
          or corrected prompt.

    FORBIDDEN BEHAVIORS (TO AVOID OVERFITTING)
    - Do NOT hard-code answers or partial answers for specific problems.
    - Do NOT key off exact question IDs, long rare substrings, or oddly specific text.
    - Do NOT create large chains of `if "some very specific sentence" in problem: ...`
      that map directly to constant answers.
    - Any branching must correspond to reusable patterns (problem families), not
      individual training questions.

    IMPLEMENTATION GUIDELINES
    - Use intermediate variables s1, s2, s3, ... to store stepwise llm(...) outputs.
    - Build prompts with f-strings that include `problem` and previous steps, e.g.:
          s2 = llm(f"... Problem: {problem} ... Prior reasoning: {s1} ...")
    - Prefer prompts that:
        * Ask for full chain-of-thought internally, AND
        * Put the final answer on a clearly marked last line, e.g. "ANSWER: <...>".
    - When modifying the workflow, favor explicit reasoning over shortcuts:
        break the problem into subgoals, write intermediate conclusions, and let
        later steps inspect and critique those reasoning traces.  # <-- NEW HINT
    - You may introduce a small internal `state` dict to store structured information
      (classification, plan, solution, checks), but the **return value must stay**:
      a single STRING with ONLY the final answer.

    RETURN CONTRACT
    - Return ONLY the final answer as a string (no explanation, no extra words).
    - Strip obvious whitespace / formatting noise.
    """

    # BASELINE WORKFLOW (YOU MAY REFACTOR THIS INTO MULTI-STEP FORM)
    s1 = llm(
        f"""You are an expert competition mathematician.

Problem:
{problem}

1. Think step by step and derive the correct solution.
2. Show your full reasoning.
3. On the LAST line, write exactly:
   ANSWER: <final answer only>"""
    )

    # SIMPLE PARSING LOGIC (YOU MAY IMPROVE OR REPLACE THIS)
    lines = s1.strip().splitlines()
    for line in reversed(lines):
        if "ANSWER:" in line:
            ans = line.split("ANSWER:", 1)[1].strip()
            if ans:
                return ans

    # Fallback: if the format was not followed, return a best-effort trimmed answer.
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
        return "test case passed! (Problem: {problem})"
    else:
        return (
            "test case failed.\n"
            f"Problem: {problem}\n"
            f"Expected final answer: {gold_str}\n"
            f"Got: {pred_str}\n"
            "Please improve the body of `math_script` so that it produces precisely the expected final answer"
        )


def train_math(MATH_EXAMPLES, epochs: int = 5, batch_size: int = 5):
    optimizer = OptoPrime(math_script.parameters())
    MATH_EXAMPLES = MATH_EXAMPLES[:5]
    n = len(MATH_EXAMPLES)
    

    train_samples = MATH_EXAMPLES
    batch_size = 0
    for epoch in range(epochs):
        log(f"\n=== Epoch {epoch} ===")

        total = 0
        num_correct = 0
        if batch_size < 5:
            batch_size += 1

        fail_samples = []
        # one tqdm bar per epoch
        with tqdm(total=n, ncols=88, desc=f"Epoch {epoch} bs {batch_size}", leave=True) as pbar:
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

                    log(fb)
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
                src = math_script.parameters()[0].data

                log("\nUpdated math_script source:\n")
                log(src)
                log("-" * 60)

        epoch_acc = num_correct / total if total > 0 else 0.0
        log(f"Epoch {epoch} accuracy: {num_correct}/{total} = {epoch_acc:.3f}")
        # 🔥 NEW: log the updated function
        
        if total > 0 and num_correct == total:
            log("All training examples solved.")
            break

        train_samples = fail_samples[:]

    final_src = math_script.parameters()[0].data
    writef(f"output/{model}_math_solver.py", final_src)
    log("Saved optimized solver to math_solver.py")
    log.saveto(f"output/{model}_log")
    return math_script


if __name__ == "__main__":

    trained_solver = train_math(MATH_EXAMPLES)


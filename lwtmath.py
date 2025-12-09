# anot.py

import os
import json
from lwt import LWTExecutor
from llm import get_key, LLMClient
from utils import writef, Logger

# global logger instance
log = Logger()
log.init()

# traindata

os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LiteLLM"
os.environ["OPENAI_API_KEY"] = get_key()
os.environ["TRACE_LITELLM_MODEL"] = "gpt-5-mini"  # optimizer LLM
os.environ["LITELLM_LOG"] = "INFO"

from opto.trace import bundle
from opto.optimizers import OptoPrime
from opto import trace

from tqdm import tqdm

from data import MATH_EXAMPLES
MATH_EXAMPLES = MATH_EXAMPLES[:35]

# -------------------------------------------------------------------
# 0. Shared executor (uses gpt-5-nano for actual workflow execution)
# -------------------------------------------------------------------
llm_client = LLMClient(model='gpt-4o-mini')
EXECUTOR = LWTExecutor(llm_client)

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

# -------------------------------------------------------------------
# 2. Trainable LWT script (string)
# -------------------------------------------------------------------

@bundle(trainable=True)
def solution_workflow():
    '''
    Edit only the string assigned to the variable `script`.
    The content of `script` must follow the LWT script format.
    '''
    script = r'''
(0)=LLM("solve the given problem: {(input)}")
'''.strip()
    return script

@bundle(trainable=False)
def concat(*items):
    out = ""
    for i, item in enumerate(items):
        out += f"ID {i}: {item}\n"
    return out

@bundle(trainable=False)
def run_triage(script: str, case_text: str):
    # IMPORTANT: this returns a MessageNode (Trace wraps the return value)
    return EXECUTOR.solve_query(script, case_text)

@bundle(trainable=False)
def first(pair):
    # pair is (output, cache)
    return pair[0]

def get_feedback(pred: str, target: str, cache: dict) -> str:
    if pred.strip() == target.strip():
        return "test case passed!"
    else:
        return (
            f"test case failed! expected: {target}, got: {pred}. "
            f"Intermediate results: {json.dumps(cache, ensure_ascii=False)}"
        )

def train_lwt(epochs: int = 3):
    optimizer = OptoPrime(solution_workflow.parameters())

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch} ===")

        
        total = num_correct = 0

        batch_size = 5
        for start in range(0, 35, batch_size):
            script_node = solution_workflow()
            script_text = script_node.data
            print("Current script:\n", script_text)

            end = min(start + batch_size, 35)
            batch = MATH_EXAMPLES[start:end]
            outputs = []    # correctness MessageNodes
            feedbacks = []  # strings
        
            for ex in tqdm(batch, ncols=88):
                x = ex["problem"]
                y = ex["answer"]

                try:
                    pred_and_cache = run_triage(script_node, x)   # MessageNode[(str, dict)]
                    pred_node = first(pred_and_cache)             # MessageNode[str]
                    pred_str, cache = pred_and_cache.data         # (str, dict)
                    fb = get_feedback(pred_str, y, cache)
                except trace.ExecutionError as e:
                    pred_node = e.exception_node
                    fb = str(e.exception_node.data)

                correctness = pred_node.eq(y)   # MessageNode[bool]
                outputs.append(correctness)
                feedbacks.append(fb)
                
                total += 1
                if bool(correctness.data):
                    num_correct += 1

            feedbacks[0] = META_PROMPTS + "\n" + feedbacks[0]
            batched_feedback = concat(*feedbacks)   # MessageNode[str]
            batched_outputs  = concat(*outputs)     # MessageNode[bool/str]

            optimizer.zero_feedback()
            optimizer.backward(batched_outputs, batched_feedback.data)
            # optimizer.step(verbose=True)
            optimizer.step()

            updated_script_node = solution_workflow()
            print("Updated script:\n", updated_script_node.data)

        all_correct = (num_correct == total)
        print(f"Accuracy: {num_correct}/{total} = {num_correct/total:.2f}")

        # prepend meta + triage spec to the first feedback string
        
        if all_correct:
            print("all correct")
            break

        
    final_node = solution_workflow()
    print("\nFinal script:\n", final_node.data)
    return final_node.data


if __name__ == "__main__":
    train_lwt()


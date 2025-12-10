# anot.py

import os
import json
from lwt import get_key, LWTExecutor, LLMClient
# HEALTHCARE_EXAMPLES

from data import HEALTHCARE_EXAMPLES, TRIAGE_SPEC

os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LiteLLM"
os.environ["OPENAI_API_KEY"] = get_key()
os.environ["TRACE_LITELLM_MODEL"] = "gpt-5-mini"  # optimizer LLM
os.environ["LITELLM_LOG"] = "INFO"

from opto.trace import bundle
from opto.optimizers import OptoPrime
from opto import trace

from tqdm import tqdm

# -------------------------------------------------------------------
# 0. Shared executor (uses gpt-5-nano for actual workflow execution)
# -------------------------------------------------------------------
llm_client = LLMClient(model="gpt-5-nano")
EXECUTOR = LWTExecutor(llm_client)

# -------------------------------------------------------------------
# 1. Triage spec in natural language
# -------------------------------------------------------------------
# TRIAGE_SPEC = """
# Home care: normal oxygen (SpO₂ ≥ 95%), mild symptoms only, fever < 38°C, age < 70, no major comorbidities acting up, no red-flag symptoms.

# Urgent clinical evaluation: borderline oxygen (SpO₂ 93–94%), moderate fever 38–39.9°C, age ≥ 70 with new weakness or shortness of breath, or relevant comorbidities with worsening symptoms, as long as no ER criteria are present.

# ER referral: low oxygen (SpO₂ ≤ 92%), very high fever ≥ 40°C, or any severe symptoms such as marked breathlessness, confusion, chest pain, or signs of shock.
# """.strip()

LWT_META_SPEC = """
You are editing an LWT script.

- The script is a sequence of lines: (k)=LLM("instruction").
- You may add, delete, or reorder lines (0), (1), (2), ...
- You may rewrite the natural-language instructions inside each LLM("...").

Do NOT:
- Change the syntactic format (must remain (k)=LLM("...")).
- Introduce other code or control structures.

Your goal is to modify the script so that its outputs on the training examples
match the target labels as accurately as possible. You may refactor the script
into multiple steps if helpful.
""".strip()

# -------------------------------------------------------------------
# 2. Trainable LWT script (string)
# -------------------------------------------------------------------

@bundle(trainable=True)
def triage_script():
    '''
    Edit only the string assigned to the variable `script`.
    The content of `script` must follow the LWT script format.
    '''
    script = r'''
(0)=LLM("Consider the description: {(input)} -- decide what care the patient should receive.")
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
    optimizer = OptoPrime(triage_script.parameters())

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch} ===")

        script_node = triage_script()
        script_text = script_node.data
        print("Current script:\n", script_text)

        outputs = []    # correctness MessageNodes
        feedbacks = []  # strings
        total = num_correct = 0

        for ex in tqdm(HEALTHCARE_EXAMPLES, ncols=88):
            x = ex["query"]
            y = ex["label"]

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

        all_correct = (num_correct == total)
        print(f"Accuracy: {num_correct}/{total} = {num_correct/total:.2f}")

        # prepend meta + triage spec to the first feedback string
        header = LWT_META_SPEC+"\n\nTriage rule:\n"+TRIAGE_SPEC+"\n"
        if all_correct:
            print("all correct")
            break

        feedbacks[0] = header + "\n" + feedbacks[0]
        batched_feedback = concat(*feedbacks)   # MessageNode[str]
        batched_outputs  = concat(*outputs)     # MessageNode[bool/str]

        optimizer.zero_feedback()
        optimizer.backward(batched_outputs, batched_feedback.data)
        # optimizer.step(verbose=True)
        optimizer.step()

        updated_script_node = triage_script()
        print("Updated script:\n", updated_script_node.data)

    final_node = triage_script()
    print("\nFinal script:\n", final_node.data)
    return final_node.data


if __name__ == "__main__":
    train_lwt()


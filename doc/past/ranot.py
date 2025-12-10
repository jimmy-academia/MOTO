# ranot.py ==> private anot
import re
import os
import json

from lwt import get_key, LWTExecutor, LLMClient
from data import HEALTHCARE_EXAMPLES, TRIAGE_SPEC

os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LiteLLM"
os.environ["OPENAI_API_KEY"] = get_key()
os.environ["TRACE_LITELLM_MODEL"] = "gpt-5-mini"  # optimizer LLM
os.environ["LITELLM_LOG"] = "INFO"

from opto.trace import bundle
from opto.optimizers import OptoPrime
from opto import trace

from tqdm import tqdm

# Edge-side LLM (conceptually private)
llm_client = LLMClient(model="gpt-5-nano")
EXECUTOR = LWTExecutor(llm_client)

# -------------------------------------------------------------------
# 1. Meta spec for script editing
# -------------------------------------------------------------------

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


def prep_prompt(privacy_flag: str) -> str:
    error_info = {
        "black": "only whether each case passed or failed",
        "dark": "expected ground truth versus output of each case (no patient details)",
        "gray": (
            "expected ground truth versus output of each case and a coarse failing "
            "step index (one or more comma-separated integers, or UNKNOWN)"
        ),
    }[privacy_flag]

    LWT_PRIVACY_SPEC = f"""
You are optimizing an LWT script using ONLY abstract, privacy-preserving feedback.

- You NEVER see raw patient text.
- You NEVER see intermediate cache contents or structured patient data.
- You only receive anonymized error summaries consisting of {error_info}.
- You must improve the script using only:
  (a) the triage rule description,
  (b) the current script,
  (c) anonymized feedback about prediction vs. label.

Do NOT ask for raw data or detailed examples. They are not available.
""".strip()

    return LWT_PRIVACY_SPEC


# -------------------------------------------------------------------
# 2. Trainable LWT script (string)
# -------------------------------------------------------------------

@bundle(trainable=True)
def triage_script():
    """
    Edit only the string assigned to the variable `script`.
    The content of `script` must follow the LWT script format.
    """
    script = r'''
(0)=LLM("Consider the description: {(input)} -- decide what care the patient should receive. Output exactly one of: 'ER referral', 'Urgent clinical evaluation', or 'Home care'.")
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


# -------------------------------------------------------------------
# 3. Coarse feedback (edge-only)
# -------------------------------------------------------------------

def validate_step_output(raw: str) -> str:
    """
    Validate that `raw` is either:
      - 'UNKNOWN'
      - a single integer        e.g. '0'
      - a comma-separated list  e.g. '0,1,3'

    Returns:
        The cleaned string if valid.
        'UNKNOWN' if invalid or unsafe.
    """
    if not isinstance(raw, str):
        return "UNKNOWN"

    s = raw.strip().upper()

    # Case 1: exactly UNKNOWN
    if s == "UNKNOWN":
        return "UNKNOWN"

    # Case 2: match comma-separated integers: "0", "1", "0,1", "2,3,4"
    if re.fullmatch(r"[0-9]+(,[0-9]+)*", s):
        return s  # already safe

    # Everything else is unsafe → default to UNKNOWN
    return "UNKNOWN"


def get_coarse_block(script_text: str, pred: str, target: str, cache_str: str) -> str:
    """
    Edge-only helper: use llm_client to guess which step indices are likely at fault.
    This sees ONLY script text, labels, and an execution trace summary.
    """
    prompt = f"""
You are an error analyzer for an LWT (LLM-Workflow Template) script.

The script is a list of numbered steps in the format:
    (k)=LLM("...")

You will see:
- The current LWT script.
- The final model prediction.
- The ground truth label.
- The execution trace showing the output of each step.

Your task:
Identify which STEP INDICES are most likely responsible for the incorrect outcome.
A step is "responsible" if its output appears inconsistent with the ground truth label,
or if it seems to propagate an error into later steps.

Rules:
1. You may output ONE step index (e.g., 1) if the failure is localized.
2. You may output MULTIPLE indices (e.g., 1,2 or 0,3) if multiple steps seem suspicious.
3. If you cannot determine the source, output ONLY: UNKNOWN
4. Output MUST contain only a comma-separated list of integers (like 0,2,3)
   OR the exact token UNKNOWN.

Do NOT explain your reasoning. Do NOT include extra text.

Current LWT script:
{script_text}

Final outputs:
- Ground truth label: {target}
- Model prediction: {pred}

Execution trace (per-step outputs):
{cache_str}

Return ONLY the step indices (comma-separated) OR UNKNOWN.
""".strip()

    raw = llm_client.answer(prompt)
    output = validate_step_output(raw)
    return output


def get_feedback(privacy_flag: str, pred: str, target: str, coarse: str) -> str:
    pred = pred.strip()
    target = target.strip()

    if pred == target:
        return "test case passed!"

    expect_v_get = f"expected: {target}, got: {pred}."
    coarse_str = f"Possible failure step indices: {coarse}."

    if privacy_flag == "black":
        msg = "test case failed!"
    elif privacy_flag == "dark":
        msg = f"test case failed! {expect_v_get}"
    elif privacy_flag == "gray":
        msg = f"test case failed! {expect_v_get} {coarse_str}"
    else:
        # Fallback – should never happen if flags are controlled
        msg = f"test case failed! {expect_v_get}"

    return msg


# -------------------------------------------------------------------
# 4. Training loop
# -------------------------------------------------------------------

def train_lwt(privacy_flag: str, epochs: int = 3):
    optimizer = OptoPrime(triage_script.parameters())

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch} ({privacy_flag}) ===")

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

                if privacy_flag == "gray" and pred_str.strip() != y.strip():
                    # Use full JSON trace here; it's still edge-only
                    cache_str = json.dumps(cache, ensure_ascii=False)
                    coarse = get_coarse_block(script_text, pred_str, y, cache_str)
                else:
                    coarse = "UNKNOWN"

                fb = get_feedback(privacy_flag, pred_str, y, coarse)

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

        # prepend meta + privacy + triage spec to the first feedback string
        privacy_spec = prep_prompt(privacy_flag)
        header = (
            LWT_META_SPEC
            + "\n\n"
            + privacy_spec
            + "\n\nTriage rule:\n"
            + TRIAGE_SPEC
            + "\n"
        )

        if all_correct:
            print("all correct")
            break

        feedbacks[0] = header + "\n" + feedbacks[0]
        batched_feedback = concat(*feedbacks)   # MessageNode[str]
        batched_outputs  = concat(*outputs)     # MessageNode[bool]

        optimizer.zero_feedback()
        optimizer.backward(batched_outputs, batched_feedback.data)
        optimizer.step()

        updated_script_node = triage_script()
        # print("Updated script:\n", updated_script_node.data)

    final_node = triage_script()
    print("\nFinal script:\n", final_node.data)
    return final_node.data


if __name__ == "__main__":
    for privacy_flag in ["black", "dark", "gray"]:
        print(f"\n==================== Privacy mode: {privacy_flag} ====================")
        train_lwt(privacy_flag)

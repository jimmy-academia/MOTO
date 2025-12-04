import os
from lwt import LLMClient, get_key

os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LiteLLM"
os.environ["OPENAI_API_KEY"] = get_key()
os.environ["TRACE_LITELLM_MODEL"] = "gpt-5-mini"  # optimizer LLM
os.environ["LITELLM_LOG"] = "INFO"

from opto.trace import bundle, node, GRAPH
from opto.optimizers import OptoPrime

from tqdm import tqdm 

# HEALTHCARE_EXAMPLES
# from anot import TRIAGE_SPEC

TRIAGE_SPEC = "If (O₂ saturation < 92 OR systolic BP < 90 OR HR > 130):\n    → severity = critical\nElse If (O₂ < 95 OR Temp > 101°F OR RR > 24):\n    → severity = moderate\nElse:\n    → severity = mild\n\nIf (≥2 chronic conditions OR age ≥ 70):\n    → risk = high\nElse:\n    → risk = standard\n\nIf (severity = critical):\n    → recommend ER referral\nElse If (severity = moderate AND risk = high):\n    → recommend urgent clinical evaluation\nElse If (severity = moderate AND risk = standard):\n    → recommend outpatient evaluation\nElse:\n    → recommend home care"

HEALTHCARE_EXAMPLES = [
    {
        "query": "A 51-year-old patient presents with loss of smell for 1 days, chest pain for 7 days, headache for 7 days. Oxygen saturation is 90%, and body temperature is 102.0°F. Medical history includes HIV. Recent travel to outbreak area.",
        "label": "ER referral"
    },
    {
        "query":"A 89-year-old patient presents with headache for 8 days, rash for 6 days, chest pain for 4 days, dry cough for 8 days, shortness of breath for 3 days. Oxygen saturation is 94%, and body temperature is 98.6°F. Medical history includes chronic kidney disease. Recent travel abroad.",
        "label":"Urgent clinical evaluation"
    },
    {
        "query":"A 16-year-old patient presents with fever for 8 days, shortness of breath for 4 days. Oxygen saturation is 90%, and body temperature is 103.0°F. Medical history includes CHF, hypertension. Recent travel abroad.",
        "label":"ER referral"
    },
    {
        "query":"A 37-year-old patient presents with loss of smell for 7 days, fever for 9 days. Oxygen saturation is 97%, and body temperature is 99.5°F. Medical history includes none. Recent travel abroad.",
        "label":"Home care"
    },
    {
        "query":"A 59-year-old patient presents with chest pain for 10 days, nausea for 7 days, fatigue for 4 days. Oxygen saturation is 97%, and body temperature is 99.5°F. Medical history includes diabetes, hypertension. Recent travel abroad.",
        "label":"Home care"
    },
]


INPROMPT = True
# INPROMPT = False
    

EXECUTOR = LLMClient(model="gpt-5-nano")  # your execution model

INITIAL_PROMPT = (
    "You are a medical triage assistant. "
    "Read the patient case and output the correct triage label "
)

    # "{Home care, Urgent clinical evaluation, ER referral} with no extra text."

trainable_prompt = node(
    INITIAL_PROMPT,
    trainable=True,
    name="prompt_text",
    constraint=None if INPROMPT else TRIAGE_SPEC,
)

@bundle()
def run_triage(prompt_text: str, query: str) -> str:
    """
    One LLM call: combine the trainable prompt with the user query.
    """
    triage_spec_block = (
        "=== TRIAGE SPEC ===\n" + TRIAGE_SPEC + "\n\n"
        if INPROMPT
        else ""
    )
    full_prompt = (
        f"{prompt_text}\n\n"
        f"{triage_spec_block}"
        "=== PATIENT CASE ===\n"
        f"{query}\n\n"
        "Triage label:"
    )
    return EXECUTOR.answer(full_prompt)

# ---------------------------------------------------------------------
# 3. Feedback construction
# ---------------------------------------------------------------------

def build_feedback(pred: str, gold: str) -> str:
    pred_norm = pred.strip().upper()
    gold_norm = gold.strip().upper()

    verdict = "The triage decision is CORRECT.\n" if pred_norm == gold_norm else "The triage decision is INCORRECT.\n"
    return verdict

# ---------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------

def train(epochs: int = 3):
    # Optimizer over ONLY this node.
    # OptoPrime will use your default optimizer LLM config
    # (e.g., from env / config file).
    optimizer = OptoPrime([trainable_prompt])

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch} ===")

        total = correct = 0
        ## measure accuracy
        pbar = tqdm(HEALTHCARE_EXAMPLES, ncols=88)
        for i, ex in enumerate(pbar):
            x = ex["query"]
            y = ex["label"]

            # IMPORTANT: clear graph each iteration
            GRAPH.clear()

            # Forward: one triage call, using trainable_prompt
            output_node = run_triage(trainable_prompt, x)
            pred = output_node.data  # underlying string from the node

            if pred.strip().upper() == y.strip().upper():
                correct += 1
            total += 1

            feedback = build_feedback(pred, y)

            # Backward: send feedback to optimizer
            optimizer.zero_feedback()
            optimizer.backward(output_node, feedback)
            optimizer.step()

            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix(acc=acc)
        # print(f"\nAccuracy after epoch {epoch}: {correct}/{total} = {acc:.3f}")

        # Show current prompt at end of epoch
        print("\nCurrent prompt after epoch", epoch)
        print("-" * 80)
        print(trainable_prompt.data)
        print("-" * 80)
        if acc == 1:
            break

if __name__ == "__main__":
    train(epochs=3)
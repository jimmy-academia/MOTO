from opto.trace import bundle, node, GRAPH
from opto.optimizers import OptoPrime

from lwt import get_key, LLMClient, HEALTHCARE_EXAMPLES

from anot import TRIAGE_SPEC

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
    triage_spec = "=== TRIAGE SPEC ===\n" + f"{TRIAGE_SPEC}\n" if INPROMPT else ""
    full_prompt = (
        f"{prompt_text}\n\n"
        triage_spec
        "=== PATIENT CASE ===\n"
        f"{query}\n\n"
        "Triage label:"
    )
    return EXECUTOR.llm_answer(full_prompt)

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
        for i, ex in enumerate(HEALTHCARE_EXAMPLES):
            x = ex["query"]
            y = ex["label"]

            # IMPORTANT: clear graph each iteration
            GRAPH.clear()?????

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
        print(f"\nAccuracy after epoch {epoch}: {correct}/{total} = {acc:.3f}")

        # Show current prompt at end of epoch
        print("\nCurrent prompt after epoch", epoch)
        print("-" * 80)
        print(trainable_prompt.data)
        print("-" * 80)


if __name__ == "__main__":
    train(epochs=3, variant)
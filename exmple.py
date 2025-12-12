#example.py
import os
import json
from llm import get_key, global_llm
from pprint import pprint

os.environ.setdefault("TRACE_DEFAULT_LLM_BACKEND", "LiteLLM")
os.environ.setdefault("TRACE_LITELLM_MODEL", "gpt-5-nano")
os.environ.setdefault("LITELLM_LOG", "DEBUG") #"INFO"
os.environ["OPENAI_API_KEY"] = get_key()


from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto import trace
from myopto.optimizers import OptoPrimeLocal

def workflow(problem_text: str) -> str:
    # Wrap input so it also becomes a traced root (optional but useful)
    problem = msg(problem_text, name="problem")

    attempts = []
    for i in range(2):
        attempt = llm(
            f"Attempt {i+1}: Solve the problem.\n"
            f"Problem:\n{problem}\n"
            f"Return only the final answer."
            , call_tag="attempt"
        )
        attempts.append(attempt)

    best = llm(
        f"Pick the best final answer from the attempts below.\n"
        f"Problem:\n{problem}\n\n"
        f"Attempts:\n" + "\n".join(attempts) + "\n\n"
        f"Return only the final answer."
        , call_tag="select"
    )

    return best

def make_feedback(problem_text: str, expected: str, got: str) -> str:
    got_s = (got or "").strip()
    exp_s = (expected or "").strip()

    if got_s == exp_s:
        # If you want the optimizer to *still* do something, don’t say “passed”.
        # Tell it an improvement objective (format robustness, fewer retries, etc.)
        return (
            "ID 0: Test Case Passed.\n"
            f"Problem: {problem_text}\n"
            f"Expected: {expected}\n"
            f"Got: {got}\n\n"
            "Improve robustness: always return ONLY the final numeric answer with no extra tokens."
        )

    return (
        "ID 0: Test Case Failed.\n"
        f"Problem: {problem_text}\n"
        f"Expected: {expected}\n"
        f"Got: {got}\n\n"
        "Fix the prompts to reliably return ONLY the final numeric answer."
    )

def main():
    
    rt = RuntimeTracer(
        backend=lambda system, user: global_llm(user, system_prompt=system),
        clear_graph_on_enter=True,
    )

    problem_text = "A particular convex pentagon has two congruent, acute angles. The measure of each of the other interior angles is equal to the sum of the measures of the two acute angles. What is the common measure of the large angles, in degrees?"
    expected = "135"

    # ---- Forward pass (collect graph + templates)
    with rt:
        best_msg = workflow(problem_text)
        output_node = best_msg.node  # <-- THIS is what we backprop from
        got = strip_trace_tags(str(best_msg))

    print("\n--- BEFORE OPT ---")
    print("answer:", got)
    print(trace.GRAPH.summary())
    ir = rt.to_ir()
    print("\nIR:")
    pprint(ir, width=120, sort_dicts=False)

    # ---- Build optimizer on trainable prompt templates
    # Runtime tracer creates trainable prompt templates as ParameterNodes. :contentReference[oaicite:6]{index=6}
    params = list(rt.prompt_templates.values())
    opt = OptoPrimeLocal(params)

    feedback = make_feedback(problem_text, expected, got)

    # ---- Backward + Step (one iteration)
    opt.zero_feedback()
    opt.backward(output_node, feedback, visualize=False)
    opt.step(mode="per_param", verbose="output")
    # verbose="output" prints LLM response in OptoPrime.call_llm :contentReference[oaicite:7]{index=7}

    print("\n--- UPDATED PROMPT TEMPLATES ---")
    for p in params:
        print(f"\n[{p.name}]")
        print(p.data)

    # ---- Rerun after update (same tracer => reuses same ParameterNodes by callsite)
    with rt:
        best_msg2 = workflow(problem_text)
        got2 = strip_trace_tags(str(best_msg2))

    print("\n--- AFTER OPT ---")
    print("answer:", got2)
    print(trace.GRAPH.summary())


if __name__ == "__main__":
    main()



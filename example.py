#example.py
import os
import json
from llm import get_key, global_llm
from pprint import pprint
import inspect, textwrap, difflib

os.environ.setdefault("TRACE_DEFAULT_LLM_BACKEND", "LiteLLM")
os.environ.setdefault("TRACE_LITELLM_MODEL", "gpt-5-nano")
os.environ.setdefault("LITELLM_LOG", "DEBUG") #"INFO"
os.environ["OPENAI_API_KEY"] = get_key()


from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto import trace
from myopto.optimizers import OptoPrimeLocal, StructureEditor

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

def get_source(func) -> str:
    return textwrap.dedent(inspect.getsource(func)).strip()

def print_code_diff(old: str, new: str, from_name="before", to_name="after"):
    diff = difflib.unified_diff(
        old.splitlines(True),
        new.splitlines(True),
        fromfile=from_name,
        tofile=to_name,
    )
    print("".join(diff))
    
def main():
    rt = RuntimeTracer(
        backend=lambda system, user: global_llm(user, system_prompt=system),
        clear_graph_on_enter=True,
    )

    problem_text = "A particular convex pentagon has two congruent, acute angles. ..."
    expected = "135"

    # ---------- Forward pass #1 (original structure) ----------
    old_code = get_source(workflow)

    with rt:
        best_msg = workflow(problem_text)
        output_node = best_msg.node
        got = strip_trace_tags(str(best_msg))

    ir = rt.to_ir()
    feedback = make_feedback(problem_text, expected, got)

    print("\n--- BEFORE (ORIGINAL) ---")
    print("answer:", got)
    print(trace.GRAPH.summary())
    print("\nIR:")
    pprint(ir, width=120, sort_dicts=False)

    # ---------- Structure edit ----------
    if got.strip() != expected.strip():
        editor = StructureEditor(verbose=True)
        res = editor.rewrite_function(
            workflow,
            ir=ir,
            feedback=feedback,
            func_name="workflow",
            required_call_tags=["attempt", "select"],
            max_retries=2,
        )

        print("\n--- STRUCTURE EDIT RESULT ---")
        print("ok:", res.ok)
        if not res.ok:
            print("errors:", res.errors)
            return

        print("reasoning:", res.reasoning)
        print("\n[NEW WORKFLOW CODE]\n")
        print(res.code)

        print("\n--- DIFF (ORIGINAL -> NEW) ---")
        print_code_diff(old_code, res.code, from_name="workflow_before.py", to_name="workflow_after.py")

        # Load rewritten function
        workflow2 = editor.load_function(
            res.code,
            func_name="workflow",
            extra_globals={"llm": llm, "msg": msg},
        )

        # IMPORTANT: switch to the new workflow
        global workflow
        workflow = workflow2

        # ---------- Forward pass #2 (new structure) ----------
        # Clear graph + retrace so params/output_node correspond to new structure
        with rt:
            best_msg = workflow(problem_text)
            output_node = best_msg.node
            got = strip_trace_tags(str(best_msg))

        ir = rt.to_ir()
        feedback = make_feedback(problem_text, expected, got)

        print("\n--- AFTER STRUCTURE EDIT (NEW FORWARD) ---")
        print("answer:", got)
        print(trace.GRAPH.summary())
        print("\nIR:")
        pprint(ir, width=120, sort_dicts=False)

    # ---------- Prompt optimization (on CURRENT structure/run) ----------
    params = list(rt.prompt_templates.values())
    opt = OptoPrimeLocal(params)

    opt.zero_feedback()
    opt.backward(output_node, feedback, visualize=False)
    opt.step(mode="per_param", verbose="output")

    print("\n--- UPDATED PROMPT TEMPLATES ---")
    for p in params:
        print(f"\n[{p.name}]")
        print(p.data)

    # ---------- Rerun after prompt update ----------
    with rt:
        best_msg2 = workflow(problem_text)
        got2 = strip_trace_tags(str(best_msg2))

    print("\n--- AFTER PROMPT OPT ---")
    print("answer:", got2)
    print(trace.GRAPH.summary())

if __name__ == "__main__":
    main()



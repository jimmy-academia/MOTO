#example.py
import json
from pprint import pprint
from llm import global_llm 
from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto import trace  # for trace.GRAPH.summary()

def workflow(problem_text: str) -> str:
    # Wrap input so it also becomes a traced root (optional but useful)
    problem = msg(problem_text, name="problem")

    attempts = []
    for i in range(3):
        attempt = llm(
            f"Attempt {i+1}: Solve the problem.\n"
            f"Problem:\n{problem}\n"
            f"Return only the final answer."
        )
        attempts.append(attempt)

        check = llm(
            f"Quick check: is this answer plausible?\n"
            f"Problem:\n{problem}\n"
            f"Answer:\n{attempt}\n"
            f"Return YES or NO."
        )
        if "YES" in check:
            break

    best = llm(
        f"Pick the best final answer from the attempts below.\n"
        f"Problem:\n{problem}\n\n"
        f"Attempts:\n" + "\n".join(attempts) + "\n\n"
        f"Return only the final answer."
    )

    return strip_trace_tags(str(best))


# Run it under tracing
with RuntimeTracer(backend=lambda system, user: global_llm(user, system_prompt=system)) as rt:
    ans = workflow("What is 12 * 13?")
    print(ans)

print('graph summary')
print(trace.GRAPH.summary(limit=None))
ir = rt.to_ir()
# print(ir)
print('\n\nir')
pprint(ir, width=120, sort_dicts=False)

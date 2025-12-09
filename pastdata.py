import re
from pathlib import Path
from utils import loadjl

root_dir = Path('../aflow_data/')

def extract_boxed_answer(solution: str) -> str:
    """
    Extract the final numeric answer from a MATH-style solution.

    Priority:
    1. Look for \\boxed{...}
    2. Then \\boxed(...)
    3. Fallback: last integer / simple fraction in the text
    """
    # \boxed{...}
    m = re.search(r"\\boxed\{([^}]*)\}", solution)
    if not m:
        # \boxed(...)
        m = re.search(r"\\boxed\(([^)]*)\)", solution)

    if m:
        ans = m.group(1).strip()
        # clean minor LaTeX artifacts
        ans = ans.replace("\\,", "").strip()
        return ans

    # fallback: last number or fraction
    candidates = re.findall(r"[-+]?\d+(?:/\d+)?", solution)
    if candidates:
        return candidates[-1].strip()

    # worst case, give back the whole solution (should almost never happen)
    return solution.strip()


# load the AFLOW-style validation file
val_math_raw = loadjl(root_dir / "math_validate.jsonl")
test_math_raw = loadjl(root_dir / "math_test.jsonl")

# build MATH_EXAMPLES in the format your trainer expects
MATH_EXAMPLES = [
    {
        "problem": ex["problem"],
        "answer": extract_boxed_answer(ex["solution"]),
    }
    for ex in val_math_raw
]

MATH_TEST = [
    {
        "problem": ex["problem"],
        "answer": extract_boxed_answer(ex["solution"]),
    }
    for ex in val_math_raw
]


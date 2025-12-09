@bundle(trainable=True)
def solution_workflow(problem: str) -> str:
    var_1 = llm(f"solve the given input {problem}")
    var_2 = llm(f"do something with step 1 intermediate result {var_1}")
    var_3 = llm(f"do something with step 2 intermediate result {var_2}")
    var_4 = llm(f"output final answer from previous results {var_1}, {var_2}, {var_3}")
    return var_4

META_PROMPTS = """
Your goal is to edit the function code representing an LLM solution workflow.
You are allowed to create multiple LLM intermediate step as needed.
To create an LLM intermediate step, simply call `llm` with the correctly formatted prompt sting:
    llm(f"this is a prompt with input field {{var}}")
You can consider structural edits --- split, merge, add steps
You can consider python script that will be helpful
Consider whether the error is syntact -- requring cleanup, or content -- requiring better reasoning.
You must NEVER hard code or overfit to the given training question. Your goal is to create a general workflow.
"""
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
@bundle(trainable=True)
def math_script(problem: str):
    """
    Trainable Python solver for MATH.

    GOAL
    - Learn a solver that generalizes to unseen MATH problems.
    - Do NOT overfit to this specific batch of training problems.

    STRICTLY FORBIDDEN (for the editing LLM)
    - Do NOT add code that hard-codes answers for particular phrasings, e.g.:
          if "convex pentagon" in problem.lower():
              return "135"
    - Do NOT detect long, specific substrings (full sentences, weird decimals,
      exact names) and map them directly to fixed answers.
    - Do NOT build long chains of `if "...something very specific..." in problem`
      each returning a constant.

    ALLOWED / ENCOURAGED
    - Use `llm(...)` as the main reasoning engine; multiple calls are fine.
    - Use intermediate variables s1, s2, s3, ... to store stepwise reasoning.
    - Build prompts with f-strings including `problem` and earlier steps.
    - Ask the LLM to put the final answer on a clearly marked last line, then
      parse that answer in Python.
    - Any branching you add must correspond to reusable patterns (problem types),
      not single memorized questions.

    OUTPUT
    - Return a STRING containing ONLY the final answer (no explanation).
    """
    s1 = llm(f"solve the problem: {problem}")
    return s1

@bundle(trainable=True)
def math_script(problem: str):
    """
    Trainable Python solver for competition-style MATH problems.

    You (the editing LLM) may modify ONLY the body of this function.
    Keep the signature `math_script(problem: str) -> str` and keep using
    the helper `llm(...)` defined above.

    OBJECTIVE
    - Learn a general-purpose solver that works on unseen problems, not just
      the small training batch.
    - Improve the function structure and prompt composition systematically
      over iterations.

    INPUT / OUTPUT CONTRACT
    - Input: `problem` is a single math problem in natural language or LaTeX.
    - Output: a STRING containing ONLY the final numeric answer (e.g. "5",
      "7/2", "3.5"), stripped of whitespace and extra words.

    DESIGN GUIDELINES
    - Use `llm(...)` as the reasoning engine; multiple calls are allowed
      (plan → solve → verify).
    - Store intermediate reasoning in variables s1, s2, s3, ... for clarity.
    - Build prompts using f-strings that incorporate `problem` and prior steps.
    - Ask the model to output the final answer on a clearly marked line, e.g.
      `FINAL_ANSWER: <number>`, then parse it in Python.
    - You may branch on broad mathematical patterns (algebra vs geometry, etc.),
      but not on memorized problem text.

    STRICTLY FORBIDDEN
    - Do NOT hard-code or memorize specific training problems in any form,
      including fixed answers, keyword lookups, or substring-based rules.
    - Do NOT use external files, data, or internet access.
    - Do NOT create unbounded loops or recursive calls to `llm(...)`.

    OPTIMIZATION GOAL
    - Each revision should make the solver more robust and general, improving
      accuracy across diverse problems rather than fitting particular examples.
    """
    s1 = llm(f"solve the problem: {problem}")
    return s1


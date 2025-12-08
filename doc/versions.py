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
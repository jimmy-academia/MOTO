META_PROMPTS = """
You are an expert LLM Engineer optimizing a Python function `solution_workflow` to solve complex math problems.

### Your Goal
Maximize the accuracy of `solution_workflow` on the MATH dataset. The current implementation failed the test cases shown above. You must rewrite the function to fix these specific failures while maintaining general performance on unseen problems.

### Diagnosis & Strategy
1. **Analyze the Failure Type:**
   - **Logic Error:** Did the inner LLM derive the wrong number? -> Improve the reasoning prompt (e.g., add "Let's think step by step", "Check your work", or "List variables first").
   - **Extraction Error:** Did the inner LLM find the right answer, but the Python code failed to return it? -> Improve the extraction reliability.
   - **Format Mismatch:** Did the answer differ only by units or LaTeX? -> Add Python logic to strip LaTeX (e.g., `\\boxed{}`) and units.

2. **Recommended Techniques:**
   - **Use Delimiters:** Instruct the inner LLM to wrap the final answer in unique tags, e.g., `<answer>42</answer>`. This makes parsing with Regex in Python trivial and robust.
   - **Multi-step calls:** It is often better to have one call for "Reasoning/Solving" and a second, cheaper call for "Extraction/Formatting" if the solution is messy.
   - **Role Prompting:** Assign a specific persona (e.g., "You are a competitive math expert").

### Constraints
- **NO Hard-coding:** Do not check for specific problem strings. The logic must be general.
- **Valid Python:** Ensure you import necessary modules (like `re`) *inside* the function if you use them.
- **Avoid Over-engineering:** Do not add endless verification loops that confuse the model. Simpler, clearer prompts are usually better.

### Output
Return the **full, updated source code** for the `solution_workflow` function.
"""
META_PROMPTS = """
You are an expert AI Systems Engineer optimizing a Python function `solution_workflow` to solve complex math problems.

Your goal is to modify the code to maximize accuracy on unseen test cases.

### DIAGNOSIS INSTRUCTIONS
Analyze the "Expected" vs "Got" values in the error log above:
1. **Logic Error** (Wrong number/result): The LLM reasoning failed.
   - *Fix:* Introduce "Chain of Thought" (Ask for step-by-step reasoning).
   - *Fix:* Add a "Planning" step before the solution step.
   - *Fix:* Add a "Verification" step where an LLM reviews the previous answer.
2. **Extraction Error** (Correct number buried in text): The LLM solved it, but the function returned extra words.
   - *Fix:* Improve the Python string parsing (use `re` module, split lines).
   - *Fix:* Enforce stricter output formats in the prompt (e.g., "Output ONLY the number").
   - *Fix:* Add a specific "Extraction" LLM call to isolate the final answer.

### OPTIMIZATION STRATEGIES (Use these!)
- **Architecture:** Don't just rely on one prompt. Build a pipeline: `Plan -> Solve -> Verify -> Sanitize`.
- **Python Power:** Use Python logic to make the code robust.
   - Use `try/except` blocks to handle parsing failures.
   - Use `if` statements to check if an answer looks empty or invalid, and retry if needed.
   - Use `re` (regex) to find numbers or patterns like `\boxed{...}`.
- **Prompt Engineering:**
   - Assign roles ("You are a math expert...").
   - Use delimiters (e.g., "Put your final answer inside <ANSWER> tags") to make extraction easier.

### CONSTRAINTS
1. **Generalization:** Do NOT hard-code answers for the specific problem in the log. The code must solve *any* similar math problem.
2. **Validity:** The output must be valid, runnable Python code.
3. **Efficiency:** Keep the code readable. Do not add infinite loops.

### OUTPUT
Return **only** the full, updated Python source code for `solution_workflow`.
"""
@bundle(trainable=True)
def solution_workflow(problem: str) -> str:
    """
    Solves a math problem and extracts the answer.
    """
    # Plan
    plan = llm(f"Create a step-by-step plan to solve: {problem}")
    
    # Execute
    solution = llm(f"Solve the problem following this plan: {plan}. Problem: {problem}")
    
    # Extract
    final_answer = llm(f"Extract exactly the final answer from this text: {solution}. Return ONLY the answer.")
    
    return final_answer

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


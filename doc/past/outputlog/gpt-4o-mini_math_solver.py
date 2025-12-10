def solution_workflow(problem: str) -> str:
    """
    Solves a math problem and extracts the answer.
    """
    import re

    # Helper to detect obviously invalid / non-answers from the LLM
    def looks_like_non_answer(text: str) -> bool:
        if not isinstance(text, str):
            return False
        lowered = text.strip().lower()
        if not lowered:
            return True
        generic_failure_patterns = [
            "no solution",
            "no common word found",
            "cannot be solved",
            "can't be solved",
            "insufficient information",
            "not enough information",
            "does not have a solution",
            "there is no answer",
            "there is no such",
        ]
        return any(p in lowered for p in generic_failure_patterns)

    # 1) Planning step: outline a solution strategy without solving yet
    plan_prompt = (
        "You are a careful, expert math and logic problem solver. "
        "Your first task is to design a clear, ordered plan to solve the problem below. "
        "Do NOT carry out calculations yet; only describe the steps you will take.\n\n"
        f"Problem: {problem}\n\n"
        "Requirements for the plan:\n"
        "- Break the approach into small, numbered steps.\n"
        "- Ensure the plan covers all constraints mentioned in the problem.\n"
        "- If the problem involves searching (e.g., over numbers or words), include a systematic search strategy.\n"
        "- Do not guess the final answer or say that no solution exists at this stage.\n\n"
        "Output only the plan as an ordered list of steps. Do not include any final answer or partial computations."
    )
    plan = llm(plan_prompt)

    # 2) Solve step: follow the plan, show reasoning, and produce a single final answer
    solve_prompt = (
        "You are a careful, expert math and logic problem solver. "
        "Now follow the given plan step by step to fully solve the problem. "
        "Show your detailed reasoning and calculations.\n\n"
        f"Problem: {problem}\n\n"
        f"Plan:\n{plan}\n\n"
        "Important instructions:\n"
        "- Follow the plan, but you may refine it if needed while solving.\n"
        "- Show clear, step-by-step reasoning and intermediate computations.\n"
        "- Avoid vague statements such as 'no common word found' or 'no solution' unless you fully justify why no solution can exist using precise logical arguments.\n"
        "- In most contest-style math or puzzle problems, a valid solution DOES exist; make a serious attempt to find it.\n"
        "- At the end, clearly state ONE final answer.\n"
        "- The final answer must be exactly what the question is asking for "
        "(for example, a number, an algebraic expression, a word, or a short phrase), "
        "not an intermediate value.\n"
        "- Wrap the final answer (and only the final answer expression, with no words) "
        "inside tags exactly like this: <FINAL_ANSWER>...</FINAL_ANSWER>.\n"
        "- Do not put more than one different value inside <FINAL_ANSWER> tags.\n"
    )
    solution = llm(solve_prompt)

    # 3) Verification step: double-check reasoning and answer
    verify_prompt = (
        "You are a meticulous math and logic checker. Review the problem, the solution attempt, "
        "and the stated final answer. Check for arithmetic mistakes, logical errors, and "
        "whether the answer actually satisfies the problem conditions.\n\n"
        f"Problem: {problem}\n\n"
        f"Solution attempt (including reasoning and final answer tags):\n{solution}\n\n"
        "Your job:\n"
        "1. Carefully re-check each step of the reasoning.\n"
        "2. If you find any error, or if the reasoning concludes with something like 'no solution', "
        "   'no common word found', or another generic failure without rigorous proof, then you must "
        "   correct the reasoning and compute a new correct final answer if one exists.\n"
        "3. If the final answer is already correct and well-justified, keep it.\n\n"
        "Important: The final answer you output must be exactly the kind of thing the "
        "question is asking for (e.g., a number, algebraic expression, word, or phrase).\n\n"
        "At the end, output the final answer again in the form:\n"
        "<FINAL_ANSWER>...</FINAL_ANSWER>\n"
        "The text inside <FINAL_ANSWER> must contain ONLY the final answer, with no explanation or extra words.\n"
        "Do not include more than one candidate answer inside <FINAL_ANSWER> tags."
    )
    verified = llm(verify_prompt)

    # Prefer the verified answer if it looks usable
    text_to_parse = (
        verified if isinstance(verified, str) and verified.strip() else solution
    )

    def extract_final_answer(text: str) -> str:
        """Extract the final answer from LLM output using tags and robust fallbacks.

        Works for numeric, algebraic, and textual answers.
        """
        if not isinstance(text, str):
            return str(text)

        # 1) Prefer explicit <FINAL_ANSWER> tags (case-insensitive)
        tag_matches = re.findall(
            r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", text, re.DOTALL | re.IGNORECASE
        )
        if tag_matches:
            # If there are multiple tags, prefer the last non-empty one
            for candidate in reversed(tag_matches):
                candidate = candidate.strip()
                if candidate:
                    return candidate

        # 2) Fallback: try common LaTeX-style wrappers like \boxed{}
        boxed_match = re.search(r"\\boxed\{([^}]*)\}", text)
        if boxed_match:
            answer = boxed_match.group(1).strip()
            if answer:
                return answer

        # 3) Try to grab the last non-empty line and strip extra wording
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            last = lines[-1]
            # Remove common prefixes
            last = re.sub(
                r"^(Answer|Final answer|Thus|So, the answer is)[:\s-]*",
                "",
                last,
                flags=re.IGNORECASE,
            )
            last = last.strip()
            if last:
                return last

        # 4) As a last resort, return the whole text stripped
        return text.strip()

    initial_answer = extract_final_answer(text_to_parse)

    # If the candidate answer clearly looks like a non-answer, try one more focused solve
    if looks_like_non_answer(initial_answer):
        direct_solve_prompt = (
            "You are a precise math and logic problem solver. Solve the following problem directly, "
            "showing your reasoning, and then state a single final answer.\n\n"
            f"Problem: {problem}\n\n"
            "Important:\n"
            "- Do NOT answer with phrases like 'no solution', 'no common word found', or 'cannot be solved' "
            "  unless you provide a rigorous, step-by-step logical proof that no solution can exist.\n"
            "- In typical contest and textbook problems, a valid solution exists. Try hard to find it.\n"
            "- At the end, output the final answer inside <FINAL_ANSWER>...</FINAL_ANSWER> tags, "
            "  with no extra words inside the tags."
        )
        fallback_solution = llm(direct_solve_prompt)
        # Use this new attempt for extraction
        initial_answer = extract_final_answer(fallback_solution)

    # 4) Sanitize step: ask LLM to cleanly isolate the final answer if needed
    sanitize_prompt = (
        "You are an answer extraction assistant. Given the original problem and a candidate "
        "answer, output ONLY the final answer expression exactly as it should be reported.\n\n"
        f"Problem: {problem}\n\n"
        f"Candidate answer: {initial_answer}\n\n"
        "Important:\n"
        "- Do not change the meaning or type of the answer.\n"
        "- If the correct answer is a word or phrase, output just that word or phrase.\n"
        "- If the correct answer is numeric or algebraic, output just that number or expression.\n"
        "- Do not add any explanation or extra text.\n\n"
        "Respond in exactly this format:\n"
        "<FINAL_ANSWER>...</FINAL_ANSWER>\n"
        "where ... is only the final answer (for example: 162, 3/4, 2^5, x+1, MAKE)."
    )
    sanitized = llm(sanitize_prompt)

    final_answer = extract_final_answer(
        sanitized
        if isinstance(sanitized, str) and sanitized.strip()
        else initial_answer
    )

    return final_answer

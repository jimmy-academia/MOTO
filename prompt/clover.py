# prompt/clover.py
META_PROMPT = (
    "You are a careful operations agent. Follow the policy strictly.\n"
    "Return ONLY valid JSON (no markdown, no extra text)."
)

SEED_WORKFLOW_CODE = """
def seed_workflow(policy_text: str, ticket_text: str, schema_text: str, meta_prompt: str) -> str:
    template = (
        "{meta_prompt}\\n\\n"
        "=== POLICY ===\\n"
        "{policy_text}\\n\\n"
        "=== INPUT TICKET ===\\n"
        "{ticket_text}\\n\\n"
        "=== OUTPUT SCHEMA (JSON) ===\\n"
        "{schema_text}\\n\\n"
        "Return ONLY the JSON object."
    )
    return llm(
        template,
        meta_prompt=meta_prompt,
        policy_text=policy_text,
        ticket_text=ticket_text,
        schema_text=schema_text,
        call_tag="solve",
    )
""".lstrip()

FEEDBACK_WORKFLOW_CODE = """
def feedback_workflow(context_id: str, policy_text: str, ticket_text: str, pred: str, report: dict) -> str:
    failed = report.get("failed_checks", [])
    if not isinstance(failed, list):
        failed = [failed]

    lines = [
        "The previous output failed verification.",
        f"context_id: {context_id}",
        "",
        "Failed checks:",
    ]
    for item in failed:
        if isinstance(item, dict):
            name = item.get("name", "check")
            reason = item.get("reason", "")
            expected = item.get("expected", None)
            got = item.get("got", None)
            extra = ""
            if expected is not None or got is not None:
                extra = f" (expected={expected} got={got})"
            lines.append(f"- {name}: {reason}{extra}")
        else:
            lines.append(f"- {item}")

    lines += [
        "",
        "Policy:",
        (policy_text or '').strip(),
        "",
        "Input ticket:",
        (ticket_text or '').strip(),
        "",
        "Model output:",
        (pred or '').strip(),
        "",
        "Return a corrected output that will pass verification.",
    ]
    return "\\n".join(lines)
""".lstrip()
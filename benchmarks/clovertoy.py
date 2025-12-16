# benchmarks/clovertoy.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .benchmark import BaseBenchmark
from utils.logs import logger


# -----------------------------------------------------------------------------
# Policies (Context A/B/C for validate; D for test)
# -----------------------------------------------------------------------------
POLICY_A = """AcmeOps Standard Policy
- Output must be STRICT JSON only (no extra text).
- team must be one of {security, billing, infra, app}.
- severity is based on Impact Level:
  - Critical -> critical
  - Major -> high
  - Minor -> normal
- redacted_owner: keep the full name, but redact email domain:
  "Alice Zhang <alice@corp.com>" -> "Alice Zhang <alice@REDACTED>"
- evidence_quote: must be exactly the Indicator line from the input.
- internal_action: must be one sentence, chosen to match team, and must NOT include the word "ESCALATE".
- customer_message: must be one sentence and must include "We will update you."
"""

POLICY_B = """BetaBank Security-First Policy
- STRICT JSON only.
- redacted_owner must be exactly "REDACTED".
- severity rule:
  - if team == security -> critical
  - else if Impact Level is Critical -> high
  - else -> normal
- evidence_quote: must be exactly the Indicator line.
- internal_action format:
  - if team == security: must start with "ESCALATE:"
  - otherwise: must start with "ROUTE:"
- customer_message rules:
  - must include "We are looking into this urgently."
  - must NOT contain the words: "security", "phishing", "unauthorized".
"""

POLICY_C = """GammaCare Customer-Friendly Policy
- STRICT JSON only.
- severity is based on SLA Hours:
  - SLA <= 1h -> critical
  - SLA <= 12h -> high
  - otherwise -> normal
- redacted_owner: only keep the first name (e.g., "Alice").
- evidence_quote: must be exactly the Customer line from the input.
- internal_action: must be a bullet list with exactly 2 bullets, each bullet starts with "- ".
- customer_message: must start with "Hi" and must include "Sorry for the inconvenience."
"""

POLICY_D = """DeltaCompliance Hybrid Policy (unseen at test time)
- STRICT JSON only.
- redacted_owner: must be formatted as "LastName, F."
- severity is the worse of:
  - Impact-based severity (Critical->critical, Major->high, Minor->normal)
  - SLA-based severity (SLA <= 1h->critical, SLA <= 6h->high, else normal)
- evidence_quote: must be exactly the Indicator line.
- internal_action rules:
  - must be a bullet list with exactly 2 bullets
  - if severity == critical, the first bullet must include the token "ESCALATE"
  - at least one bullet must include the indicator token (the word after "Indicator:")
- customer_message rules:
  - must include: "For privacy, we have redacted personal information."
"""


ALLOWED_TEAMS = {"security", "billing", "infra", "app"}
SEV_ORDER = {"normal": 0, "high": 1, "critical": 2}


TEAM_BY_INDICATOR = {
    # security-ish
    "unauthorized-login": "security",
    "phishing": "security",
    "api-key": "security",
    # billing-ish
    "double-charge": "billing",
    "refund": "billing",
    # infra-ish
    "dns-outage": "infra",
    "latency": "infra",
    "disk": "infra",
    # app-ish
    "ios-crash": "app",
    "ui-bug": "app",
}


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    owner_raw: str
    owner_first: str
    owner_last: str
    email_local: str
    impact_level: str
    sla_hours: float
    customer_line: str       # exact "Customer: ..."(whole line)
    indicator_line: str      # exact "Indicator: ..."(whole line)
    indicator_token: str
    raw_text: str


def _line_map(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in raw.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


_OWNER_RE = re.compile(r"^(?P<name>.+?)\s*<(?P<email>[^>]+)>$")


def parse_ticket(raw_text: str) -> Ticket:
    raw_text = raw_text.strip()
    lines = raw_text.splitlines()
    kv = _line_map(raw_text)

    ticket_id = kv.get("Ticket ID", "").strip()
    owner = kv.get("Owner", "").strip()
    impact_level = kv.get("Impact Level", "").strip()
    sla_raw = kv.get("SLA Hours", "").strip()

    # exact evidence lines (must be exact substring from input)
    customer_line = next((l.strip() for l in lines if l.strip().startswith("Customer:")), "")
    indicator_line = next((l.strip() for l in lines if l.strip().startswith("Indicator:")), "")

    m = _OWNER_RE.match(owner)
    if not m:
        # fallback: best effort
        name = owner
        email_local = ""
    else:
        name = m.group("name").strip()
        email = m.group("email").strip()
        email_local = email.split("@", 1)[0] if "@" in email else ""

    name_parts = [p for p in name.split() if p]
    owner_first = name_parts[0] if name_parts else ""
    owner_last = name_parts[-1] if len(name_parts) >= 2 else (name_parts[0] if name_parts else "")

    try:
        sla_hours = float(sla_raw)
    except Exception:
        sla_hours = 0.0

    indicator_token = indicator_line.split(":", 1)[1].strip() if ":" in indicator_line else ""

    return Ticket(
        ticket_id=ticket_id,
        owner_raw=owner,
        owner_first=owner_first,
        owner_last=owner_last,
        email_local=email_local,
        impact_level=impact_level,
        sla_hours=sla_hours,
        customer_line=customer_line,
        indicator_line=indicator_line,
        indicator_token=indicator_token,
        raw_text=raw_text,
    )


def infer_team(ticket: Ticket) -> str:
    return TEAM_BY_INDICATOR.get(ticket.indicator_token, "app")


def severity_from_impact(impact: str) -> str:
    impact = (impact or "").strip().lower()
    if impact == "critical":
        return "critical"
    if impact == "major":
        return "high"
    return "normal"


def severity_from_sla(sla_hours: float, high_cutoff: float) -> str:
    # critical <=1h always
    if sla_hours <= 1.0:
        return "critical"
    if sla_hours <= high_cutoff:
        return "high"
    return "normal"


def worse_severity(a: str, b: str) -> str:
    return a if SEV_ORDER.get(a, 0) >= SEV_ORDER.get(b, 0) else b


def make_problem_prompt(context_id: str, policy_text: str, raw_ticket: str) -> str:
    schema = """{
  "ticket_id": "INC-xxxx",
  "team": "security|billing|infra|app",
  "severity": "critical|high|normal",
  "redacted_owner": "string",
  "evidence_quote": "string (must be exact substring from input)",
  "internal_action": "string",
  "customer_message": "string"
}"""
    return (
        "You are an operations agent. Follow the policy exactly.\n\n"
        f"=== POLICY (context_id={context_id}) ===\n{policy_text}\n\n"
        "=== INPUT TICKET ===\n"
        f"{raw_ticket.strip()}\n\n"
        "=== OUTPUT REQUIREMENTS ===\n"
        "- Output MUST be STRICT JSON only (no markdown, no extra text).\n"
        "- Output MUST match this exact schema (keys must exist, no extra keys):\n"
        f"{schema}\n"
    )

# --- CloverToy prompt parser (for scheme + benchmark fallback parsing) ---
_PROMPT_RE = re.compile(
    r"=== POLICY \(context_id=(?P<context_id>[A-Za-z])\) ===\n(?P<context>.*?)\n\n=== INPUT TICKET ===\n(?P<ticket>.*?)\n\n=== OUTPUT REQUIREMENTS ===",
    re.S,
)

def split_problem_prompt(problem_text: str) -> Tuple[str, str, str]:
    """
    Parse the canonical CloverToy prompt into (context_id, policy_text, raw_ticket).
    Returns ("", "", problem_text) if it doesn't match the template.
    """
    m = _PROMPT_RE.search(problem_text or "")
    if not m:
        return "", "", (problem_text or "").strip()
    return (
        (m.group("context_id") or "").strip(),
        (m.group("context") or "").strip(),
        (m.group("ticket") or "").strip(),
    )



def generate_gt(context_id: str, policy_text: str, ticket: Ticket) -> Dict[str, Any]:
    team = infer_team(ticket)

    if context_id == "A":
        severity = severity_from_impact(ticket.impact_level)
        redacted_owner = f"{ticket.owner_first} {ticket.owner_last} <{ticket.email_local}@REDACTED>".strip()
        evidence_quote = ticket.indicator_line
        internal_action_by_team = {
            "security": "Investigate the incident and rotate any potentially compromised credentials.",
            "billing": "Audit billing records and correct any duplicate charges.",
            "infra": "Investigate infrastructure signals and mitigate the service impact.",
            "app": "Reproduce the issue and prepare a fix for the application.",
        }
        internal_action = internal_action_by_team.get(team, "Investigate and mitigate the issue.")
        customer_message = "We will update you."
    elif context_id == "B":
        if team == "security":
            severity = "critical"
        elif severity_from_impact(ticket.impact_level) == "critical":
            severity = "high"
        else:
            severity = "normal"
        redacted_owner = "REDACTED"
        evidence_quote = ticket.indicator_line
        if team == "security":
            internal_action = "ESCALATE: Investigate and contain the issue immediately."
        else:
            internal_action = "ROUTE: Triage the issue and route to the appropriate team."
        customer_message = "We are looking into this urgently."
    elif context_id == "C":
        severity = severity_from_sla(ticket.sla_hours, high_cutoff=12.0)
        redacted_owner = ticket.owner_first
        evidence_quote = ticket.customer_line
        internal_action = "- Triage the report and confirm the scope.\n- Coordinate next steps with the appropriate team."
        customer_message = "Hi, Sorry for the inconvenience."
    elif context_id == "D":
        sev_impact = severity_from_impact(ticket.impact_level)
        sev_sla = severity_from_sla(ticket.sla_hours, high_cutoff=6.0)
        severity = worse_severity(sev_impact, sev_sla)
        redacted_owner = f"{ticket.owner_last}, {ticket.owner_first[:1]}."
        evidence_quote = ticket.indicator_line
        tok = ticket.indicator_token
        if severity == "critical":
            internal_action = (
                f"- ESCALATE and investigate the {tok} indicator.\n"
                f"- Contain impact and document findings with {tok} evidence."
            )
        else:
            internal_action = (
                f"- Review and validate the {tok} indicator.\n"
                f"- Resolve the {tok} issue and document the correction."
            )
        customer_message = "For privacy, we have redacted personal information."
    else:
        raise ValueError(f"Unknown context_id={context_id}")

    return {
        "ticket_id": ticket.ticket_id,
        "team": team,
        "severity": severity,
        "redacted_owner": redacted_owner,
        "evidence_quote": evidence_quote,
        "internal_action": internal_action,
        "customer_message": customer_message,
    }


def _is_strict_json_only(s: str) -> Tuple[bool, str]:
    # allow surrounding whitespace, but disallow markdown fences
    if "```" in s:
        return False, "contains markdown code fence"
    stripped = s.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False, "does not look like a JSON object"
    try:
        obj = json.loads(stripped)
    except Exception as e:
        return False, f"json.loads failed: {e}"
    if not isinstance(obj, dict):
        return False, "parsed JSON is not an object"
    return True, ""


def _two_bullets(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) != 2:
        return False
    return all(line.startswith("- ") for line in lines)


def verify_output(context_id: str, policy_text: str, ticket: Ticket, pred_text: str, cost_usd: float = 0.0) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    tags: List[str] = []

    def fail(name: str, reason: str, expected: Any = None, got: Any = None, tag: Optional[str] = None):
        entry = {"name": name, "reason": reason}
        if expected is not None:
            entry["expected"] = expected
        if got is not None:
            entry["got"] = got
        checks.append(entry)
        if tag:
            tags.append(tag)

    # 1) strict json only
    ok, msg = _is_strict_json_only(pred_text)
    if not ok:
        fail("strict_json", msg, tag="json")
        passed_checks = 0
        total_checks = 7
        alpha = float(os.environ.get("CLOVERTOY_COST_ALPHA", "0.0"))
        score = 0.0
        return {
            "passed": False,
            "score": score,
            "score_with_cost": score - alpha * cost_usd,
            "cost_usd": cost_usd,
            "failed_checks": checks,
            "tags": tags,
            "normalized_pred": None,
        }

    pred = json.loads(pred_text.strip())

    # 2) schema keys (exactly)
    expected_keys = {
        "ticket_id",
        "team",
        "severity",
        "redacted_owner",
        "evidence_quote",
        "internal_action",
        "customer_message",
    }
    pred_keys = set(pred.keys())
    if pred_keys != expected_keys:
        missing = sorted(list(expected_keys - pred_keys))
        extra = sorted(list(pred_keys - expected_keys))
        fail("schema_keys", f"keys mismatch (missing={missing}, extra={extra})", expected=sorted(list(expected_keys)), got=sorted(list(pred_keys)), tag="schema")

    # 3) evidence exact match (+ must be substring of input)
    # expected evidence depends on context
    expected_evidence = ticket.indicator_line if context_id in {"A", "B", "D"} else ticket.customer_line
    ev = pred.get("evidence_quote", "")
    if not isinstance(ev, str):
        fail("evidence_quote_type", "evidence_quote is not a string", expected="string", got=type(ev).__name__, tag="evidence")
    else:
        if ev not in ticket.raw_text:
            fail("evidence_substring", "evidence_quote is not an exact substring of input", expected="exact substring from input", got=ev, tag="evidence")
        if ev != expected_evidence:
            fail("evidence_exact_line", "evidence_quote must match the required line exactly", expected=expected_evidence, got=ev, tag="evidence")

    # 4) redaction rule
    ro = pred.get("redacted_owner", "")
    if context_id == "A":
        expected_ro = f"{ticket.owner_first} {ticket.owner_last} <{ticket.email_local}@REDACTED>".strip()
        if ro != expected_ro:
            fail("redaction", "Policy A redaction mismatch", expected=expected_ro, got=ro, tag="redaction")
    elif context_id == "B":
        if ro != "REDACTED":
            fail("redaction", "Policy B requires exact 'REDACTED'", expected="REDACTED", got=ro, tag="redaction")
    elif context_id == "C":
        if ro != ticket.owner_first:
            fail("redaction", "Policy C requires first name only", expected=ticket.owner_first, got=ro, tag="redaction")
    elif context_id == "D":
        expected_ro = f"{ticket.owner_last}, {ticket.owner_first[:1]}."
        if ro != expected_ro:
            fail("redaction", "Policy D requires 'LastName, F.' format", expected=expected_ro, got=ro, tag="redaction")

    # 5) severity rule
    team = pred.get("team", "")
    impact_sev = severity_from_impact(ticket.impact_level)
    if context_id == "A":
        expected_sev = impact_sev
    elif context_id == "B":
        expected_sev = "critical" if team == "security" else ("high" if impact_sev == "critical" else "normal")
    elif context_id == "C":
        expected_sev = severity_from_sla(ticket.sla_hours, high_cutoff=12.0)
    else:  # D
        expected_sev = worse_severity(impact_sev, severity_from_sla(ticket.sla_hours, high_cutoff=6.0))

    sev = pred.get("severity", "")
    if sev != expected_sev:
        fail("severity", "severity does not match policy rule", expected=expected_sev, got=sev, tag="severity")

    # also enforce valid team + ticket_id correctness (helps stability)
    if pred.get("ticket_id") != ticket.ticket_id:
        fail("ticket_id", "ticket_id must match input Ticket ID", expected=ticket.ticket_id, got=pred.get("ticket_id"), tag="id")
    if team not in ALLOWED_TEAMS:
        fail("team", "team must be one of allowed set", expected=sorted(list(ALLOWED_TEAMS)), got=team, tag="team")

    # 6) internal_action format rule
    ia = pred.get("internal_action", "")
    if not isinstance(ia, str):
        fail("internal_action_type", "internal_action is not a string", expected="string", got=type(ia).__name__, tag="action")
    else:
        if context_id == "A":
            if "\n" in ia:
                fail("internal_action_format", "Policy A requires one sentence (no newlines)", tag="action")
            if "ESCALATE" in ia.upper():
                fail("internal_action_forbidden", 'Policy A forbids the word "ESCALATE"', tag="action")
        elif context_id == "B":
            prefix = "ESCALATE:" if team == "security" else "ROUTE:"
            if not ia.startswith(prefix):
                fail("internal_action_prefix", "Policy B prefix mismatch", expected=prefix, got=ia[: max(len(prefix), 16)], tag="action")
        elif context_id == "C":
            if not _two_bullets(ia):
                fail("internal_action_bullets", "Policy C requires exactly 2 bullets '- '", tag="action")
        elif context_id == "D":
            if not _two_bullets(ia):
                fail("internal_action_bullets", "Policy D requires exactly 2 bullets '- '", tag="action")
            else:
                lines = ia.splitlines()
                if expected_sev == "critical" and "ESCALATE" not in lines[0]:
                    fail("internal_action_escalate", 'Policy D requires first bullet include "ESCALATE" when severity=critical', tag="action")
                tok = ticket.indicator_token
                if tok and (tok not in ia):
                    fail("internal_action_token", "Policy D requires indicator token appears in at least one bullet", expected=tok, got=ia, tag="action")

    # 7) customer_message rule
    cm = pred.get("customer_message", "")
    if not isinstance(cm, str):
        fail("customer_message_type", "customer_message is not a string", expected="string", got=type(cm).__name__, tag="message")
    else:
        if context_id == "A":
            if "We will update you." not in cm:
                fail("customer_message_required", 'Policy A requires including "We will update you."', tag="message")
        elif context_id == "B":
            if "We are looking into this urgently." not in cm:
                fail("customer_message_required", 'Policy B requires including "We are looking into this urgently."', tag="message")
            banned = ["security", "phishing", "unauthorized"]
            cm_lower = cm.lower()
            for w in banned:
                if w in cm_lower:
                    fail("customer_message_banned", f"Policy B forbids the word '{w}'", expected=f"not contain {w}", got=cm, tag="message")
                    break
        elif context_id == "C":
            if not cm.startswith("Hi"):
                fail("customer_message_prefix", 'Policy C requires customer_message starts with "Hi"', tag="message")
            if "Sorry for the inconvenience." not in cm:
                fail("customer_message_required", 'Policy C requires including "Sorry for the inconvenience."', tag="message")
        elif context_id == "D":
            req = "For privacy, we have redacted personal information."
            if req not in cm:
                fail("customer_message_required", f'Policy D requires including "{req}"', tag="message")

    total_checks = 7
    passed_checks = total_checks - len({c["name"] for c in checks})  # unique check names
    score = max(0.0, passed_checks / total_checks)

    alpha = float(os.environ.get("CLOVERTOY_COST_ALPHA", "0.0"))
    return {
        "passed": len(checks) == 0,
        "score": score,
        "score_with_cost": score - alpha * cost_usd,
        "cost_usd": cost_usd,
        "failed_checks": checks,
        "tags": tags,
        "normalized_pred": pred,
    }


class CloverToyBenchmark(BaseBenchmark):
    """
    BaseBenchmark-compatible CloverToy benchmark.

    Notes:
    - Uses deterministic verifier (verify_output) for scoring.
    - Default concurrency is forced to 1 because RuntimeTracer is backed by a global graph.
    """

    def __init__(self, name: str, file_path: Path, log_path: Path):
        super().__init__(name, file_path, log_path)
        # Clover scheme uses a global trace graph; keep evaluation single-threaded by default.
        self.max_concurrent_tasks = int(os.environ.get("CLOVERTOY_MAX_CONCURRENCY", "1"))

    def get_result_columns(self) -> List[str]:
        return [
            "ticket_id",
            "context_id",
            "passed",
            "score",            # solve rate (0/1)
            "score_raw",        # verifier fractional score
            "score_with_cost",
            "iterations",
            "sample_cost_usd",  # per-sample cost
            "cost",             # cumulative cost (for legacy BaseBenchmark aggregation)
            "tags",
            "failed_checks",
            "prediction",
        ]

    def calculate_score(self, expected_output: str, prediction: str) -> float:
        # Not used (we override evaluate_problem), but required by BaseBenchmark.
        try:
            obj = json.loads(prediction)
            return 1.0 if isinstance(obj, dict) else 0.0
        except Exception:
            return 0.0

    async def evaluate_problem(self, problem: Dict[str, Any], agent) -> Tuple[Any, ...]:
        prompt = problem.get(self.q_key, "")

        # Prefer structured fields if present in the JSONL record.
        context_id = problem.get("context_id", "")
        policy_text = problem.get("context", "")
        raw_ticket = problem.get("input", "")

        # Fallback: parse from prompt if record is missing fields.
        if not (context_id and policy_text and raw_ticket):
            p_ctx_id, p_policy, p_ticket = split_problem_prompt(prompt)
            context_id = context_id or p_ctx_id
            policy_text = policy_text or p_policy
            raw_ticket = raw_ticket or p_ticket

        ticket = parse_ticket(raw_ticket)

        # Agent call: (prediction, cost) or (prediction, cost, meta)
        scheme = getattr(agent, "__self__", None)
        if scheme is not None and hasattr(scheme, "inference_with_meta"):
            prediction, cost, meta = await scheme.inference_with_meta(prompt)
        else:
            prediction, cost = await agent(prompt)
            meta = {}

        prediction = (prediction or "").strip()
        iterations = int(meta.get("iterations", 1))
        sample_cost_usd = float(meta.get("sample_cost_usd", 0.0))

        report = verify_output(context_id, policy_text, ticket, prediction, cost_usd=sample_cost_usd)

        passed = bool(report.get("passed", False))
        score_raw = float(report.get("score", 0.0))
        score_with_cost = float(report.get("score_with_cost", score_raw))
        score = 1.0 if passed else 0.0

        tags = report.get("tags", [])
        failed_checks = report.get("failed_checks", [])

        return (
            ticket.ticket_id,
            context_id,
            passed,
            score,
            score_raw,
            score_with_cost,
            iterations,
            sample_cost_usd,
            float(cost),
            ",".join(tags) if isinstance(tags, list) else str(tags),
            json.dumps(failed_checks, ensure_ascii=False),
            prediction,
        )

    def save_results_to_csv(self, results: List[Tuple[Any, ...]]) -> Tuple[float, float, float]:
        import pandas as pd

        columns = self.get_result_columns()
        df = pd.DataFrame(results, columns=columns)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.log_path, index=False)

        solve_rate = float(df["passed"].mean()) if len(df) else 0.0
        avg_iters = float(df["iterations"].mean()) if len(df) else 0.0

        # Cost: prefer per-sample sum if available
        if "sample_cost_usd" in df.columns and df["sample_cost_usd"].notna().any():
            total_cost = float(df["sample_cost_usd"].fillna(0.0).sum())
        else:
            total_cost = float(df["cost"].max()) if len(df) else 0.0
        avg_cost = total_cost / len(df) if len(df) else 0.0

        logger.info(
            f"[clovertoy] solve_rate={solve_rate:.3f} avg_iters={avg_iters:.2f} "
            f"avg_cost=${avg_cost:.6f} total_cost=${total_cost:.6f}"
        )
        return solve_rate, avg_cost, total_cost

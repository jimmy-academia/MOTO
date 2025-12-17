#!/usr/bin/env python3
"""
Generate CloverToy dataset files:

- data/clovertoy_validate.jsonl  (contexts A/B/C × 10 fixed tickets)
- data/clovertoy_test.jsonl      (context D × 10 fixed tickets)

Run:
  python scripts/make_clovertoy_data.py

This will overwrite existing files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


SCHEMA = {
    "ticket_id": "INC-xxxx",
    "team": "security|billing|infra|app",
    "severity": "critical|high|normal",
    "redacted_owner": "string",
    "evidence_quote": "string (must be exact substring from input)",
    "internal_action": "string",
    "customer_message": "string",
}

POLICIES: Dict[str, str] = {
    "A": """AcmeOps Standard Policy

- Output must be STRICT JSON only (no extra text).
- team must be one of {security, billing, infra, app}.
- severity is based on Impact Level in the input:
  - Critical -> critical
  - Major -> high
  - Minor -> normal
- redacted_owner: keep the full name, but redact email domain:
  "Alice Zhang <alice@corp.com>" -> "Alice Zhang <alice@REDACTED>"
- evidence_quote: must be exactly the Indicator line from the input.
- internal_action: must be one sentence and must NOT include the word "ESCALATE".
- customer_message: must be one sentence and must include "We will update you.".
""",
    "B": """BetaBank Security-First Policy

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
""",
    "C": """GammaCare Customer-Friendly Policy

- STRICT JSON only.
- severity is based on SLA Hours in the input:
  - SLA <= 1h -> critical
  - SLA <= 12h -> high
  - otherwise -> normal
- redacted_owner: only keep the first name (e.g., "Alice").
- evidence_quote: must be exactly the Customer line from the input.
- internal_action: must be a bullet list with exactly 2 bullets, each bullet starts with "- ".
- customer_message: must start with "Hi" and must include "Sorry for the inconvenience.".
""",
    "D": """DeltaCompliance Hybrid Policy (Unseen at test time)

- STRICT JSON only.
- redacted_owner must be formatted as "LastName, F."
  Example: "Alice Zhang ..." -> "Zhang, A."
- severity is the worse of:
  - Impact-based severity (Critical->critical, Major->high, Minor->normal)
  - SLA-based severity (SLA <= 1h -> critical, SLA <= 6h -> high, else normal)
- evidence_quote: must be exactly the Indicator line.
- internal_action rules:
  - must be a bullet list with exactly 2 bullets
  - if severity == critical, the first bullet must include token "ESCALATE"
  - at least one bullet must include the indicator token (the word after "Indicator:")
- customer_message must include:
  "For privacy, we have redacted personal information."
""",
}


TICKETS: List[str] = [
    """Ticket ID: INC-1001
Owner: Alice Zhang <alice.zhang@corp.com>
Impact Level: Critical
SLA Hours: 1
Customer: "Users cannot log in after the latest update."
Indicator: unauthorized-login
Details: Multiple reports mention repeated login prompts and suspicious IP addresses.
""",
    """Ticket ID: INC-1002
Owner: Brian Chen <brian.chen@corp.com>
Impact Level: Major
SLA Hours: 12
Customer: "I was charged twice for the same invoice."
Indicator: double-charge
Details: The customer reports duplicate payment entries and requests an immediate fix.
""",
    """Ticket ID: INC-1003
Owner: Clara Wu <clara.wu@corp.com>
Impact Level: Critical
SLA Hours: 2
Customer: "Our site is down and shows DNS errors."
Indicator: dns-outage
Details: Monitoring shows NXDOMAIN spikes and timeouts across multiple regions.
""",
    """Ticket ID: INC-1004
Owner: David Park <david.park@corp.com>
Impact Level: Major
SLA Hours: 24
Customer: "The iOS app crashes on launch after updating to iOS 17."
Indicator: ios-crash
Details: Crash seems to occur during startup initialization on iPhone 13/14.
""",
    """Ticket ID: INC-1005
Owner: Evan Lin <evan.lin@corp.com>
Impact Level: Critical
SLA Hours: 0.5
Customer: "I clicked a suspicious link and now my account behaves strangely."
Indicator: phishing
Details: Customer mentions a fake login page and unexpected password reset emails.
""",
    """Ticket ID: INC-1006
Owner: Fiona Li <fiona.li@corp.com>
Impact Level: Major
SLA Hours: 6
Customer: "Everything is extremely slow today."
Indicator: latency
Details: API p95 latency increased 4x, mostly on read endpoints.
""",
    """Ticket ID: INC-1007
Owner: George Wang <george.wang@corp.com>
Impact Level: Minor
SLA Hours: 24
Customer: "I want a refund because the subscription was canceled unexpectedly."
Indicator: refund
Details: Customer requests refund confirmation and timeline.
""",
    """Ticket ID: INC-1008
Owner: Helen Zhao <helen.zhao@corp.com>
Impact Level: Minor
SLA Hours: 48
Customer: "The settings button does nothing on the new UI."
Indicator: ui-bug
Details: The click event is ignored intermittently; no error shown.
""",
    """Ticket ID: INC-1009
Owner: Ian Zhou <ian.zhou@corp.com>
Impact Level: Critical
SLA Hours: 1
Customer: "We found unknown API calls from our account."
Indicator: api-key
Details: Logs suggest possible key leakage and unauthorized usage patterns.
""",
    """Ticket ID: INC-1010
Owner: Julia Kim <julia.kim@corp.com>
Impact Level: Major
SLA Hours: 8
Customer: "The service is failing intermittently with storage errors."
Indicator: disk
Details: Disk usage is near capacity and error rates spike during peak traffic.
""",
]

TEAM_BY_INDICATOR = {
    "unauthorized-login": "security",
    "phishing": "security",
    "api-key": "security",
    "double-charge": "billing",
    "refund": "billing",
    "dns-outage": "infra",
    "latency": "infra",
    "disk": "infra",
    "ios-crash": "app",
    "ui-bug": "app",
}

def parse_field(ticket: str, prefix: str) -> str:
    for line in ticket.splitlines():
        if line.startswith(prefix):
            return line.split(prefix, 1)[1].strip()
    raise ValueError(f"Missing line: {prefix}")

def parse_line(ticket: str, prefix: str) -> str:
    for line in ticket.splitlines():
        if line.startswith(prefix):
            return line.rstrip("\n")
    raise ValueError(f"Missing line: {prefix}")

def impact_to_sev(impact: str) -> str:
    return {"Critical":"critical","Major":"high","Minor":"normal"}[impact]

def sla_to_sev(sla: float, critical_le: float, high_le: float) -> str:
    if sla <= critical_le:
        return "critical"
    if sla <= high_le:
        return "high"
    return "normal"

def worse(a: str, b: str) -> str:
    rank = {"normal":0,"high":1,"critical":2}
    return a if rank[a] >= rank[b] else b

def redact_owner(context_id: str, owner_line: str) -> str:
    # owner_line example: 'Alice Zhang <alice.zhang@corp.com>'
    name = owner_line.split("<",1)[0].strip()
    email = owner_line.split("<",1)[1].rstrip(">").strip()
    local = email.split("@",1)[0]
    first = name.split()[0]
    last = name.split()[-1]

    cid = context_id.upper()
    if cid == "A":
        return f"{name} <{local}@REDACTED>"
    if cid == "B":
        return "REDACTED"
    if cid == "C":
        return first
    if cid == "D":
        return f"{last}, {first[0]}."
    raise ValueError(context_id)

def severity(context_id: str, *, team: str, impact: str, sla: float) -> str:
    cid = context_id.upper()
    if cid == "A":
        return impact_to_sev(impact)
    if cid == "B":
        if team == "security":
            return "critical"
        if impact == "Critical":
            return "high"
        return "normal"
    if cid == "C":
        return sla_to_sev(sla, critical_le=1.0, high_le=12.0)
    if cid == "D":
        impact_sev = impact_to_sev(impact)
        sla_sev = sla_to_sev(sla, critical_le=1.0, high_le=6.0)
        return worse(impact_sev, sla_sev)
    raise ValueError(context_id)

def evidence_quote(context_id: str, ticket: str) -> str:
    cid = context_id.upper()
    if cid in {"A","B","D"}:
        return parse_line(ticket, "Indicator:")
    if cid == "C":
        return parse_line(ticket, "Customer:")
    raise ValueError(context_id)

def internal_action(context_id: str, *, team: str, sev: str, indicator_token: str) -> str:
    cid = context_id.upper()
    if cid == "A":
        return f"Investigate and resolve the {indicator_token} issue."
    if cid == "B":
        prefix = "ESCALATE:" if team == "security" else "ROUTE:"
        return f"{prefix} Investigate and resolve the {indicator_token} issue."
    if cid == "C":
        return "- Review the report and confirm the scope.\n- Proceed with the next step to address the issue."
    if cid == "D":
        if sev == "critical":
            b1 = f"- ESCALATE and investigate indicator {indicator_token}."
        else:
            b1 = f"- Investigate indicator {indicator_token}."
        b2 = "- Document findings and apply remediation steps."
        return b1 + "\n" + b2
    raise ValueError(context_id)

def customer_message(context_id: str) -> str:
    cid = context_id.upper()
    if cid == "A":
        return "We will update you."
    if cid == "B":
        return "We are looking into this urgently."
    if cid == "C":
        return "Hi, Sorry for the inconvenience."
    if cid == "D":
        return "For privacy, we have redacted personal information."
    raise ValueError(context_id)

def build_prompt(context_id: str, policy: str, ticket: str) -> str:
    return (
        f"Context Policy (ID={context_id}):\n{policy}\n\n"
        f"Ticket:\n{ticket.strip()}\n\n"
        "Return STRICT JSON ONLY. Do not include any extra text.\n"
        "The output JSON MUST have exactly these keys:\n"
        f"{list(SCHEMA.keys())}\n\n"
        "Schema (values are free-form unless constrained by policy):\n"
        f"{json.dumps(SCHEMA, indent=2)}\n"
    )

def gt_output(context_id: str, ticket: str) -> str:
    tid = parse_field(ticket, "Ticket ID:")
    owner = parse_field(ticket, "Owner:")
    impact = parse_field(ticket, "Impact Level:")
    sla = float(parse_field(ticket, "SLA Hours:"))
    indicator_token = parse_field(ticket, "Indicator:")
    team = TEAM_BY_INDICATOR[indicator_token]
    sev = severity(context_id, team=team, impact=impact, sla=sla)

    obj = {
        "ticket_id": tid,
        "team": team,
        "severity": sev,
        "redacted_owner": redact_owner(context_id, owner),
        "evidence_quote": evidence_quote(context_id, ticket),
        "internal_action": internal_action(context_id, team=team, sev=sev, indicator_token=indicator_token),
        "customer_message": customer_message(context_id),
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def make_records(context_id: str) -> List[dict]:
    policy = POLICIES[context_id]
    records = []
    for ticket in TICKETS:
        tid = parse_field(ticket, "Ticket ID:")
        rec = {
            "context_id": context_id,
            "context": policy,
            "ticket_id": tid,
            "input": ticket.strip("\n"),
            "problem": build_prompt(context_id, policy, ticket),
            "output": gt_output(context_id, ticket),
        }
        records.append(rec)
    return records

def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    data_dir = Path("data")
    validate_path = data_dir / "clovertoy_validate.jsonl"
    test_path = data_dir / "clovertoy_test.jsonl"

    validate_records = []
    for cid in ["A", "B", "C"]:
        validate_records.extend(make_records(cid))

    test_records = make_records("D")

    write_jsonl(validate_path, validate_records)
    write_jsonl(test_path, test_records)

    print(f"Wrote {len(validate_records)} records -> {validate_path}")
    print(f"Wrote {len(test_records)} records -> {test_path}")

if __name__ == "__main__":
    main()

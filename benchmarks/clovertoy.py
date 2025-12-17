import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from benchmarks.benchmark import BaseBenchmark


SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ticket_id": {"type": "string"},
        "approved": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["ticket_id", "approved", "reason"],
    "additionalProperties": False,
}


@dataclass
class Ticket:
    ticket_id: str
    text: str


def parse_ticket(prompt: str) -> Ticket:
    # Legacy helper (kept for compatibility; benchmark no longer relies on parsing prompts).
    m = re.search(r"Ticket ID:\s*(\S+)", prompt)
    tid = m.group(1) if m else "unknown"
    return Ticket(ticket_id=tid, text=prompt)


def verify_output(prediction: str) -> Tuple[bool, List[str]]:
    # Legacy verifier (kept for compatibility; benchmark no longer calls this).
    failed = []
    try:
        obj = json.loads(prediction)
    except Exception:
        return False, ["invalid_json"]

    for k in ("ticket_id", "approved", "reason"):
        if k not in obj:
            failed.append(f"missing_{k}")

    if "approved" in obj and not isinstance(obj["approved"], bool):
        failed.append("approved_not_bool")

    if "ticket_id" in obj and not isinstance(obj["ticket_id"], str):
        failed.append("ticket_id_not_str")

    if "reason" in obj and not isinstance(obj["reason"], str):
        failed.append("reason_not_str")

    return len(failed) == 0, failed


def _strip_code_fences(s: str) -> str:
    s2 = (s or "").strip()
    if not s2.startswith("```"):
        return s2
    lines = s2.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_substring(text: str) -> str:
    t = _strip_code_fences(text).strip()

    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        return t

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = t.find(open_ch)
        end = t.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            return t[start : end + 1].strip()

    return t


class CloverToyBenchmark(BaseBenchmark):
    def __init__(self, data_file, keys=None):
        super().__init__(data_file, keys=keys)
        # Safe default: trace backend isn't concurrency-safe yet
        self.max_concurrent_tasks = 1

    def get_result_columns(self) -> List[str]:
        return [
            "ticket_id",
            "context_id",
            "passed",
            "score",
            "score_raw",
            "score_with_cost",
            "iterations",
            "sample_cost_usd",
            "cost_usd",
            "tags",
            "failed_checks",
            "pred",
        ]

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, Any]:
        """Score is 1.0 iff prediction JSON parses and equals the expected JSON object."""
        # Parse expected
        expected_obj: Any = None
        if isinstance(expected_output, str):
            try:
                expected_obj = json.loads(expected_output)
            except Exception:
                expected_obj = expected_output.strip()
        else:
            expected_obj = expected_output

        # Parse prediction (tolerate fences / leading text)
        cleaned = _extract_json_substring(prediction or "")
        try:
            pred_obj = json.loads(cleaned)
        except Exception:
            return 0.0, cleaned

        score = 1.0 if pred_obj == expected_obj else 0.0
        return float(score), cleaned

    async def evaluate_problem(self, agent, problem, *args, **kwargs):
        """Treat dataset fields as authoritative: context + input + output.

        No prompt parsing. No verifier.
        """
        context_id = problem.get("context_id", "")
        ticket_id = problem.get("ticket_id", "")

        # Authoritative fields from dataset
        context = problem.get("context", "")
        x = problem.get("input", "")
        answer = problem.get("answer", problem.get("output", ""))

        # Optional pre-built prompt for baseline agents (does not require parsing)
        full_prompt = problem.get("problem", "")

        prediction = ""
        cost_usd = 0.0
        iterations = None
        sample_cost_usd = None

        scheme = getattr(agent, "__self__", None)

        # Option A: structured scheme API (recommended)
        if scheme is not None and hasattr(scheme, "inference_with_meta"):
            try:
                prediction, cost_usd, meta = await scheme.inference_with_meta(context, x)
                if isinstance(meta, dict):
                    iterations = meta.get("iterations", None)
                    sample_cost_usd = meta.get("sample_cost_usd", None)
            except Exception:
                prediction, cost_usd = await agent(full_prompt or x)
        else:
            # Fallback: plain agent(prompt)
            prediction, cost_usd = await agent(full_prompt or x)

        score, _cleaned = self.calculate_score(answer, prediction)
        passed = bool(score == 1 or score == 1.0)

        # No verifier => keep these empty but preserve schema
        tags: List[str] = []
        failed_checks: List[str] = []

        score_raw = float(score)
        score_with_cost = float(score)

        return (
            ticket_id,
            context_id,
            passed,
            float(score),
            score_raw,
            score_with_cost,
            iterations,
            sample_cost_usd,
            float(cost_usd),
            tags,
            failed_checks,
            prediction,
        )

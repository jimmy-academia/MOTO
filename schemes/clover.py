# schemes/clover.py
from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from schemes.base import BaseScheme
from utils.logs import logger

# myopto runtime tracer + optimizers (from trace/ package)
from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto.optimizers import OptoPrimeLocal, StructureEditor
from myopto.utils.llm_router import set_role_models
from myopto.utils.usage import configure_usage, reset_usage, get_total_cost


# Keep schema text stable (and we also pass it as a msg(...) placeholder to prevent edits)
CLOVERTOY_SCHEMA = """{
  "ticket_id": "INC-xxxx",
  "team": "security|billing|infra|app",
  "severity": "critical|high|normal",
  "redacted_owner": "string",
  "evidence_quote": "string (must be exact substring from input)",
  "internal_action": "string",
  "customer_message": "string"
}""".strip()

# Parse the benchmark prompt produced by benchmarks/clovertoy.py
# Expected structure:
#   === POLICY (context_id=A) ===
#   ...
#   === INPUT TICKET ===
#   ...
#   === OUTPUT REQUIREMENTS ===
_POLICY_BLOCK_RE = re.compile(
    r"=== POLICY\s*\(context_id=(?P<context_id>[^)]+)\)\s*===\n"
    r"(?P<policy>.*?)\n"
    r"\n=== INPUT TICKET ===\n"
    r"(?P<ticket>.*?)\n"
    r"\n=== OUTPUT REQUIREMENTS ===",
    re.DOTALL,
)

_TICKET_ID_RE = re.compile(r"Ticket ID:\s*(?P<ticket_id>INC-\d+)")


def _parse_problem_text(full_prompt: str) -> Tuple[str, str, str]:
    """
    Return (context_id, policy_text, raw_ticket_text).
    Falls back to ("", "", full_prompt) if parsing fails.
    """
    m = _POLICY_BLOCK_RE.search(full_prompt)
    if not m:
        return "", "", full_prompt
    return (
        (m.group("context_id") or "").strip(),
        (m.group("policy") or "").strip(),
        (m.group("ticket") or "").strip(),
    )


def _extract_ticket_id(ticket_text: str) -> str:
    m = _TICKET_ID_RE.search(ticket_text)
    return (m.group("ticket_id") if m else "").strip()


def _build_feedback(
    context_id: str,
    policy_text: str,
    ticket_text: str,
    pred_text: str,
    verifier_report: Dict[str, Any],
) -> str:
    """
    Feedback string for both prompt optimizer and structure editor.
    """
    failed = verifier_report.get("failed_checks", []) or []
    tags = verifier_report.get("tags", []) or []

    lines: List[str] = []
    lines.append("You MUST fix the output to pass a deterministic verifier.")
    lines.append(f"context_id: {context_id}")
    if tags:
        lines.append(f"tags: {tags}")

    lines.append("\nPOLICY (must follow exactly):")
    lines.append(policy_text)

    lines.append("\nINPUT TICKET:")
    lines.append(ticket_text)

    lines.append("\nMODEL OUTPUT (current, failing):")
    lines.append(pred_text)

    lines.append("\nFAILED CHECKS (must fix all):")
    for fc in failed:
        name = fc.get("name", "unknown")
        reason = fc.get("reason", "")
        expected = fc.get("expected", None)
        got = fc.get("got", None)
        lines.append(f"- {name}: {reason}")
        if expected is not None:
            lines.append(f"  expected: {expected}")
        if got is not None:
            lines.append(f"  got: {got}")

    lines.append(
        "\nREQUIREMENTS REMINDER:\n"
        "- Output MUST be STRICT JSON only (no markdown, no extra text).\n"
        "- Output MUST contain exactly these 7 keys:\n"
        '  ticket_id, team, severity, redacted_owner, evidence_quote, internal_action, customer_message\n'
        "- evidence_quote MUST be an exact substring from the INPUT TICKET, per POLICY.\n"
        "- redacted_owner / severity / internal_action / customer_message MUST follow POLICY.\n"
    )

    return "\n".join(lines)


# -----------------------------
# Seed workflow (structure editor will rewrite this function)
# -----------------------------
def workflow(policy_text: str, ticket_text: str, schema_text: str) -> Any:
    """
    Minimal seed workflow:
    - 1 LLM call tagged 'solve'
    - Input placeholders: policy, ticket, schema are msg(...) roots (protected)
    """
    policy = msg(policy_text, name="policy")
    ticket = msg(ticket_text, name="ticket")
    schema = msg(schema_text, name="schema")

    prompt = (
        "You are an operations agent.\n"
        "Follow the POLICY exactly.\n"
        "Return ONLY a single JSON object (no markdown, no extra text).\n"
        "\n"
        "=== POLICY ===\n"
        f"{policy}\n"
        "\n"
        "=== INPUT TICKET ===\n"
        f"{ticket}\n"
        "\n"
        "=== OUTPUT REQUIREMENTS ===\n"
        "- Output MUST be STRICT JSON only.\n"
        "- Output MUST match this exact schema:\n"
        f"{schema}\n"
        "\n"
        "Return only the JSON object.\n"
    )

    # call_tag is critical for stable prompt template tracking
    return llm(prompt, call_tag="solve")


class CloverScheme(BaseScheme):
    """
    Commit 5: Minimal closed-loop:
      - parse {context, x} from benchmark prompt
      - run seed workflow with RuntimeTracer -> (y, ir)
      - verifier feedback (no-GT) from benchmarks/clovertoy.verify_output
      - if fail: 2 candidates:
          1) prompt-only patch via OptoPrimeLocal on runtime prompt templates
          2) structure patch via StructureEditor.rewrite_function + load_function
      - pick best and iterate until pass or budget
    """

    def __init__(self, args):
        super().__init__(args)

        # Enable myopto usage tracking so we can return cost
        configure_usage(enabled=True)

        # Make executor use args.exe_model and optimizer use args.opt_model
        # (main.py sets TRACE_LITELLM_MODEL globally; this overrides per-role)
        exe_model = getattr(args, "exe_model", None)
        opt_model = getattr(args, "opt_model", None)
        set_role_models(executor=exe_model, optimizer=opt_model, metaoptimizer=opt_model)

        self.max_iters = int(os.environ.get("CLOVER_MAX_ITERS", "3"))
        self.max_cost_usd = float(os.environ.get("CLOVER_MAX_COST_USD", "5.0"))
        self.enable_structure = os.environ.get("CLOVER_ENABLE_STRUCTURE", "1") == "1"

        # Required call tags for StructureEditor (keep at least 'solve')
        self.required_call_tags = ["solve"]

    async def train(
        self,
        train_benchmark,
        train_indices,
        val_benchmark,
        val_indices,
    ):
        """
        Minimal: create scheme_file so main.py won't try to retrain next run.
        Clover is primarily runtime closed-loop, so no offline training artifact needed.
        """
        self.scheme_dir.mkdir(parents=True, exist_ok=True)
        self.scheme_file.write_text(
            "CloverScheme: closed-loop at inference time. No offline training artifact.\n",
            encoding="utf-8",
        )
        logger.info(f"[CLOVER] Wrote scheme file: {self.scheme_file}")

    def load(self):
        # Nothing to load for minimal commit 5
        return

    def prep_test(self):
        # Nothing special needed
        return

    async def inference(self, input_text: str) -> Tuple[str, float]:
        """
        Returns (answer_str, cost_usd).
        """
        reset_usage()

        context_id, policy_text, ticket_text = _parse_problem_text(input_text)
        ticket_id = _extract_ticket_id(ticket_text)

        # Import verifier lazily to avoid circular imports at module import time
        from benchmarks.clovertoy import parse_ticket, verify_output

        def verify(pred: str, cost_usd: float) -> Dict[str, Any]:
            try:
                ticket = parse_ticket(ticket_text)
            except Exception as e:
                return {
                    "passed": False,
                    "score": 0.0,
                    "score_with_cost": 0.0,
                    "tags": ["parse_ticket_error"],
                    "failed_checks": [{"name": "parse_ticket", "reason": str(e)}],
                    "normalized_pred": None,
                }
            return verify_output(context_id, policy_text, ticket, pred, cost_usd=cost_usd)

        def run_once(
            tracer: RuntimeTracer,
            wf: Callable[[str, str, str], Any],
        ) -> Tuple[str, Any, Dict[str, Any], float]:
            """
            Run workflow once with a tracer, return:
              (pred_str, out_msg, ir, incremental_cost_usd)
            """
            cost_before = get_total_cost()
            with tracer:
                out = wf(policy_text, ticket_text, CLOVERTOY_SCHEMA)
            pred_str = strip_trace_tags(str(out)).strip()
            ir = tracer.to_ir()
            cost_after = get_total_cost()
            return pred_str, out, ir, float(cost_after - cost_before)

        # Current "best" state
        best_pred = ""
        best_report: Dict[str, Any] = {"score_with_cost": -1.0, "score": -1.0, "passed": False}
        best_iters = 0

        # Current candidate state (workflow + tracer)
        current_wf = workflow
        current_tracer = RuntimeTracer(tags=[f"clover/{context_id or 'unknown'}"])

        for it in range(1, self.max_iters + 1):
            # Stop on budget
            if get_total_cost() >= self.max_cost_usd:
                logger.info(
                    f"[CLOVER] budget stop (cost=${get_total_cost():.4f} >= ${self.max_cost_usd:.4f}) "
                    f"ticket={ticket_id} ctx={context_id} it={it}"
                )
                break

            # Baseline run
            base_pred, base_out, base_ir, base_cost = run_once(current_tracer, current_wf)
            base_report = verify(base_pred, cost_usd=base_cost)

            candidates: List[Tuple[str, Callable, RuntimeTracer, str, Dict[str, Any]]] = [
                ("base", current_wf, current_tracer, base_pred, base_report)
            ]

            # Track best so far
            if base_report.get("score_with_cost", 0.0) > best_report.get("score_with_cost", -1.0):
                best_pred, best_report, best_iters = base_pred, base_report, it

            if bool(base_report.get("passed", False)):
                best_pred, best_report, best_iters = base_pred, base_report, it
                break

            feedback = _build_feedback(context_id, policy_text, ticket_text, base_pred, base_report)

            # Candidate 1: Prompt-only patch (OptoPrimeLocal) using current_tracer prompt templates
            try:
                params = list(current_tracer.prompt_templates.values())
                if params:
                    opt = OptoPrimeLocal(params)
                    opt.backward(base_out.node, feedback, visualize=False)
                    opt.step(mode="per_param", verbose="output")

                    p_pred, p_out, p_ir, p_cost = run_once(current_tracer, current_wf)
                    p_report = verify(p_pred, cost_usd=p_cost)
                    candidates.append(("prompt", current_wf, current_tracer, p_pred, p_report))

                    if p_report.get("score_with_cost", 0.0) > best_report.get("score_with_cost", -1.0):
                        best_pred, best_report, best_iters = p_pred, p_report, it
            except Exception as e:
                logger.warning(f"[CLOVER] prompt candidate failed: {e}")

            # Candidate 2: Structure patch (StructureEditor)
            if self.enable_structure:
                try:
                    editor = StructureEditor(verbose=False, forbid_imports=True)
                    edit_res = editor.rewrite_function(
                        current_wf,
                        base_ir,
                        feedback,
                        required_call_tags=self.required_call_tags,
                        func_name="workflow",
                        max_retries=1,
                    )
                    if getattr(edit_res, "ok", False) and getattr(edit_res, "code", ""):
                        new_wf = editor.load_function(
                            edit_res.code,
                            func_name="workflow",
                            extra_globals={"llm": llm, "msg": msg},
                        )
                        s_tracer = RuntimeTracer(tags=[f"clover_struct/{context_id or 'unknown'}"])
                        s_pred, s_out, s_ir, s_cost = run_once(s_tracer, new_wf)
                        s_report = verify(s_pred, cost_usd=s_cost)
                        candidates.append(("struct", new_wf, s_tracer, s_pred, s_report))

                        if s_report.get("score_with_cost", 0.0) > best_report.get("score_with_cost", -1.0):
                            best_pred, best_report, best_iters = s_pred, s_report, it
                except Exception as e:
                    logger.warning(f"[CLOVER] structure candidate failed: {e}")

            # Pick best candidate for next iteration
            def cand_key(c):
                _name, _wf, _tr, _pred, _rep = c
                return float(_rep.get("score_with_cost", _rep.get("score", 0.0)))

            chosen = max(candidates, key=cand_key)
            chosen_name, chosen_wf, chosen_tracer, chosen_pred, chosen_report = chosen
            current_wf, current_tracer = chosen_wf, chosen_tracer

            logger.info(
                f"[CLOVER] ticket={ticket_id} ctx={context_id} it={it} "
                f"chosen={chosen_name} passed={bool(chosen_report.get('passed', False))} "
                f"score={chosen_report.get('score', 0.0):.3f} "
                f"cost_total=${get_total_cost():.4f}"
            )

            if bool(chosen_report.get("passed", False)):
                best_pred, best_report, best_iters = chosen_pred, chosen_report, it
                break

        # Final per-sample cost
        total_cost = float(get_total_cost())

        # Log summary per sample
        logger.info(
            f"[CLOVER] DONE ticket={ticket_id} ctx={context_id} "
            f"passed={bool(best_report.get('passed', False))} "
            f"best_score={best_report.get('score', 0.0):.3f} "
            f"iters={best_iters} cost=${total_cost:.4f}"
        )

        return best_pred, total_cost

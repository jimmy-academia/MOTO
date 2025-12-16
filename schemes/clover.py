from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from schemes.base import BaseScheme
from utils.logs import logger

from myopto.optimizers.optoprime_local import OptoPrimeLocal
from myopto.optimizers.structure_editor import StructureEditor
from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto.utils.llm_router import set_role_models
from myopto.utils.usage import configure_usage, get_total_cost, reset_usage

from prompt.clover import DEFAULT_META_PROMPT, DEFAULT_SEED_WORKFLOW_CODE, DEFAULT_FEEDBACK_WORKFLOW_CODE

# IMPORTANT: use a *stable* prompt template string + kwargs.
# If we build the final string with f-strings, the template becomes example-specific and
# prompt optimization becomes meaningless.

# Deterministic, but persisted so outer-loop optimization can rewrite it.



@dataclass
class CloverRun:
    pred: str
    out_msg: Any
    trace_ir: Dict[str, Any]
    total_cost_usd: float


@dataclass
class Candidate:
    kind: str  # "base" | "prompt" | "struct"
    wf: Callable[..., Any]
    wf_code: str
    tracer: RuntimeTracer
    run: CloverRun
    report: Dict[str, Any]
    # If kind=="prompt", these are the optimized prompt template values to apply.
    prompt_values: Optional[List[Any]] = None


class _CloverAdapter:
    """
    Small adapter surface so CloverScheme isn't hard-wired to CloverToy.

    For commit-5 we only implement CloverToy; adding a new benchmark should only
    require adding a new adapter, not touching scheme logic.
    """

    name: str
    schema_text: str

    def split_problem_prompt(self, prompt: str) -> Tuple[str, str, str]:
        """Return (context_id, policy_text, raw_input_text)."""
        raise NotImplementedError

    def parse_input(self, raw_input_text: str) -> Any:
        """Parse benchmark-specific input into a structured object (optional)."""
        return raw_input_text

    def verify(
        self,
        context_id: str,
        policy_text: str,
        parsed_input: Any,
        pred: str,
        cost_usd: float,
    ) -> Dict[str, Any]:
        """Return report dict with at least: passed, score, score_with_cost, failed_checks."""
        raise NotImplementedError


class _CloverToyAdapter(_CloverAdapter):
    name = "clovertoy"

    def __init__(self) -> None:
        from benchmarks.clovertoy import CLOVERTOY_SCHEMA, parse_ticket, split_problem_prompt, verify_output

        self.schema_text = CLOVERTOY_SCHEMA
        self._parse_ticket = parse_ticket
        self._split_problem_prompt = split_problem_prompt
        self._verify_output = verify_output

    def split_problem_prompt(self, prompt: str) -> Tuple[str, str, str]:
        return self._split_problem_prompt(prompt)

    def parse_input(self, raw_input_text: str) -> Any:
        return self._parse_ticket(raw_input_text)

    def verify(
        self,
        context_id: str,
        policy_text: str,
        parsed_input: Any,
        pred: str,
        cost_usd: float,
    ) -> Dict[str, Any]:
        return self._verify_output(context_id, policy_text, parsed_input, pred, cost_usd=cost_usd)


def _score_from_report(report: Dict[str, Any]) -> float:
    if "score_with_cost" in report:
        try:
            return float(report["score_with_cost"])
        except Exception:
            pass
    try:
        return float(report.get("score", 0.0))
    except Exception:
        return 0.0


class CloverScheme(BaseScheme):
    """
    Minimal Clover closed-loop scheme (runtime tracer + prompt candidate + structure candidate).

    - Benchmark-specific parsing/verifying is isolated behind an adapter.
    - Persisted state lives in self.scheme_file (code.py):
        - META_PROMPT
        - SEED_WORKFLOW_CODE
        - FEEDBACK_WORKFLOW_CODE
    """

    def __init__(self, args: Any):
        super().__init__(args)

        self.benchmark_name = getattr(args, "benchmark", "clovertoy")

        # Role routing for myopto LLM calls.
        set_role_models(
            executor=getattr(args, "exe_model", None),
            optimizer=getattr(args, "opt_model", None),
            metaoptimizer=getattr(args, "opt_model", None),
        )
        configure_usage(enabled=True)

        # State persisted via code.py
        self.meta_prompt: str = DEFAULT_META_PROMPT

        self.workflow_func_name: str = "seed_workflow"
        self.seed_workflow_code: str = DEFAULT_SEED_WORKFLOW_CODE

        self.feedback_func_name: str = "feedback_workflow"
        self.feedback_workflow_code: str = DEFAULT_FEEDBACK_WORKFLOW_CODE

        self.required_call_tags: List[str] = ["solve"]

        # Runtime knobs
        self.max_inner_iters: int = int(os.environ.get("CLOVER_MAX_ITERS", "3"))
        self.max_cost_usd: float = float(os.environ.get("CLOVER_MAX_COST_USD", "1000000000"))
        self.enable_structure: bool = os.environ.get("CLOVER_ENABLE_STRUCTURE", "1") == "1"

        # Adapter (benchmark-specific)
        self.adapter: _CloverAdapter = self._make_adapter(self.benchmark_name)

        # Compiled workflow fns
        self.workflow_fn: Callable[..., Any] = self._compile_function(self.seed_workflow_code, self.workflow_func_name)
        self.feedback_fn: Callable[..., Any] = self._compile_function(self.feedback_workflow_code, self.feedback_func_name)

        # Book-keeping for "inner-loop -> outer-loop message passing"
        self.observations_path = self.output_subdir / "observations.jsonl"
        self.last_meta: Dict[str, Any] = {}

    # ----------------------------
    # BaseScheme hooks
    # ----------------------------

    def prep_test(self) -> None:
        configure_usage(enabled=True)
        set_role_models(
            executor=getattr(self.args, "exe_model", None),
            optimizer=getattr(self.args, "opt_model", None),
            metaoptimizer=getattr(self.args, "opt_model", None),
        )

    async def train_on_batch(self, batch: List[dict], train_benchmark: Any) -> Dict[str, Any]:
        # Commit-5 minimal: no outer-loop learning yet. We still persist a seed artifact.
        return {"noop": True, "batch": len(batch)}

    def save_model(self, epoch: Optional[int] = None) -> None:
        self._save_state(self.scheme_file)

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False

        scope: Dict[str, Any] = {}
        exec(path.read_text(encoding="utf-8"), scope)

        self.meta_prompt = str(scope.get("META_PROMPT", self.meta_prompt))

        self.workflow_func_name = str(scope.get("WORKFLOW_FUNC_NAME", self.workflow_func_name))
        self.seed_workflow_code = str(scope.get("SEED_WORKFLOW_CODE", self.seed_workflow_code))

        self.feedback_func_name = str(scope.get("FEEDBACK_FUNC_NAME", self.feedback_func_name))
        self.feedback_workflow_code = str(scope.get("FEEDBACK_WORKFLOW_CODE", self.feedback_workflow_code))

        req = scope.get("REQUIRED_CALL_TAGS", self.required_call_tags)
        if isinstance(req, (list, tuple)) and all(isinstance(x, str) for x in req):
            self.required_call_tags = list(req)

        self.workflow_fn = self._compile_function(self.seed_workflow_code, self.workflow_func_name)
        self.feedback_fn = self._compile_function(self.feedback_workflow_code, self.feedback_func_name)
        return True

    async def inference(self, input_text: str) -> Tuple[str, float]:
        pred, cost, _meta = await self.inference_with_meta(input_text)
        return pred, cost

    # ----------------------------
    # Clover API (used by CloverToyBenchmark)
    # ----------------------------

    async def inference_with_meta(self, input_text: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Returns (prediction, total_cost_usd, meta)

        meta contains:
          - iterations
          - passed
          - score
          - score_with_cost
          - best_candidate
          - trace_ir (best)
          - sample_cost_usd
        """
        reset_usage()
        self.last_meta = {}

        context_id, policy_text, raw_input_text = self.adapter.split_problem_prompt(input_text)
        parsed_input = self.adapter.parse_input(raw_input_text)

        # Current inner-loop state
        current_wf = self.workflow_fn
        current_wf_code = self.seed_workflow_code
        current_tracer = RuntimeTracer(trainable_prompt_templates=True, clear_graph_on_enter=True)

        best: Optional[Candidate] = None
        best_iter = 0

        last_feedback: Optional[str] = None

        for it in range(1, self.max_inner_iters + 1):
            if get_total_cost() >= self.max_cost_usd:
                logger.info(f"[clover] budget hit: cost=${get_total_cost():.4f} >= ${self.max_cost_usd:.4f}")
                break

            # 1) Base run
            base_run = self._run_once(
                tracer=current_tracer,
                wf=current_wf,
                policy_text=policy_text,
                raw_input_text=raw_input_text,
                schema_text=self.adapter.schema_text,
                meta_prompt=self.meta_prompt,
            )
            base_report = self.adapter.verify(
                context_id=context_id,
                policy_text=policy_text,
                parsed_input=parsed_input,
                pred=base_run.pred,
                cost_usd=base_run.total_cost_usd,
            )

            base_cand = Candidate(
                kind="base",
                wf=current_wf,
                wf_code=current_wf_code,
                tracer=current_tracer,
                run=base_run,
                report=base_report,
            )

            best, best_iter = self._maybe_update_best(best, best_iter, base_cand, it)
            if bool(base_report.get("passed")):
                logger.info(f"[clover] PASS in {it} iter(s) | score={base_report.get('score')}")
                best = base_cand
                best_iter = it
                break

            # Feedback (inner-loop -> outer-loop observation)
            feedback = self._make_feedback(
                context_id=context_id,
                policy_text=policy_text,
                raw_input_text=raw_input_text,
                pred=base_run.pred,
                report=base_report,
            )
            last_feedback = feedback

            candidates: List[Candidate] = [base_cand]

            # 2) Prompt candidate (prompt-only patch via OptoPrimeLocal)
            prompt_cand = self._prompt_candidate(
                tracer=current_tracer,
                wf=current_wf,
                wf_code=current_wf_code,
                output_node=getattr(base_run.out_msg, "node", None),
                feedback=feedback,
                context_id=context_id,
                policy_text=policy_text,
                raw_input_text=raw_input_text,
                parsed_input=parsed_input,
            )
            if prompt_cand is not None:
                candidates.append(prompt_cand)
                best, best_iter = self._maybe_update_best(best, best_iter, prompt_cand, it)

            # 3) Structure candidate (function rewrite)
            if self.enable_structure:
                struct_cand = self._structure_candidate(
                    current_wf=current_wf,
                    current_wf_code=current_wf_code,
                    base_ir=base_run.trace_ir,
                    feedback=feedback,
                    context_id=context_id,
                    policy_text=policy_text,
                    raw_input_text=raw_input_text,
                    parsed_input=parsed_input,
                )
                if struct_cand is not None:
                    candidates.append(struct_cand)
                    best, best_iter = self._maybe_update_best(best, best_iter, struct_cand, it)

            # 4) Choose best candidate for next inner iteration
            chosen = max(candidates, key=lambda c: _score_from_report(c.report))
            logger.info(
                f"[clover] iter={it} choose={chosen.kind} score={chosen.report.get('score')} "
                f"score_with_cost={chosen.report.get('score_with_cost')} total_cost=${chosen.run.total_cost_usd:.4f}"
            )

            if chosen.kind == "prompt" and chosen.prompt_values is not None:
                # Apply the optimized prompt template values to the CURRENT tracer
                params = list(current_tracer.prompt_templates.values())
                if len(params) == len(chosen.prompt_values):
                    for p, v in zip(params, chosen.prompt_values):
                        p.data = v  # type: ignore[attr-defined]
                # Keep wf the same
            elif chosen.kind == "struct":
                current_wf = chosen.wf
                current_wf_code = chosen.wf_code
                current_tracer = chosen.tracer
            else:
                # base: no state change
                pass

            if bool(chosen.report.get("passed")):
                best = chosen
                best_iter = it
                break

        # Finalize
        if best is None:
            best_pred = ""
            best_report = {"passed": False, "score": 0.0, "score_with_cost": 0.0, "failed_checks": ["no_runs"]}
            best_ir: Dict[str, Any] = {}
            best_kind = "none"
        else:
            best_pred = best.run.pred
            best_report = best.report
            best_ir = best.run.trace_ir
            best_kind = best.kind

        total_cost = float(get_total_cost())
        meta = {
            "iterations": int(best_iter or 1),
            "passed": bool(best_report.get("passed")),
            "score": float(best_report.get("score", 0.0) or 0.0),
            "score_with_cost": float(best_report.get("score_with_cost", best_report.get("score", 0.0)) or 0.0),
            "best_candidate": best_kind,
            "trace_ir": best_ir,
            "sample_cost_usd": total_cost,
            "last_feedback": last_feedback,
        }
        self.last_meta = meta
        self._append_observation(context_id=context_id, meta=meta)
        return best_pred, total_cost, meta

    # ----------------------------
    # Internals
    # ----------------------------

    def _make_adapter(self, benchmark_name: str) -> _CloverAdapter:
        if benchmark_name == "clovertoy":
            return _CloverToyAdapter()
        logger.warning(f"[clover] No adapter for benchmark={benchmark_name}; using CloverToyAdapter as fallback.")
        return _CloverToyAdapter()

    def _compile_function(self, code: str, func_name: str) -> Callable[..., Any]:
        editor = StructureEditor(verbose=False, forbid_imports=True)
        return editor.load_function(
            code,
            func_name=func_name,
            extra_globals={"llm": llm, "msg": msg},
        )

    def _save_state(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _py_triple(s: str) -> str:
            return '"""' + s.replace('"""', r'\\"""') + '"""'

        content = "\n".join(
            [
                "# Auto-generated CloverScheme artifact (code.py)",
                f"META_PROMPT = {_py_triple(self.meta_prompt)}",
                f"WORKFLOW_FUNC_NAME = {_py_triple(self.workflow_func_name)}",
                f"FEEDBACK_FUNC_NAME = {_py_triple(self.feedback_func_name)}",
                f"REQUIRED_CALL_TAGS = {json.dumps(self.required_call_tags)}",
                'SEED_WORKFLOW_CODE = r"""',
                self.seed_workflow_code.rstrip(),
                '"""',
                'FEEDBACK_WORKFLOW_CODE = r"""',
                self.feedback_workflow_code.rstrip(),
                '"""',
                "",
            ]
        )
        path.write_text(content, encoding="utf-8")
        logger.info(f"[clover] saved scheme state to {path}")

    def _append_observation(self, context_id: str, meta: Dict[str, Any]) -> None:
        try:
            rec = {"context_id": context_id, **meta}
            with self.observations_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[clover] failed to append observation: {e}")

    def _run_once(
        self,
        tracer: RuntimeTracer,
        wf: Callable[..., Any],
        policy_text: str,
        raw_input_text: str,
        schema_text: str,
        meta_prompt: str,
    ) -> CloverRun:
        with tracer:
            out = wf(
                policy_text=policy_text,
                ticket_text=raw_input_text,
                schema_text=schema_text,
                meta_prompt=meta_prompt,
            )
        pred = strip_trace_tags(str(out)).strip()
        ir = tracer.to_ir()
        total_cost = float(get_total_cost())
        return CloverRun(pred=pred, out_msg=out, trace_ir=ir, total_cost_usd=total_cost)

    def _maybe_update_best(
        self,
        best: Optional[Candidate],
        best_iter: int,
        cand: Candidate,
        it: int,
    ) -> Tuple[Optional[Candidate], int]:
        if best is None:
            return cand, it
        if _score_from_report(cand.report) > _score_from_report(best.report):
            return cand, it
        return best, best_iter

    def _make_feedback(
        self,
        context_id: str,
        policy_text: str,
        raw_input_text: str,
        pred: str,
        report: Dict[str, Any],
    ) -> str:
        try:
            return str(
                self.feedback_fn(
                    context_id=context_id,
                    policy_text=policy_text,
                    ticket_text=raw_input_text,
                    pred=pred,
                    report=report,
                )
            )
        except Exception as e:
            logger.warning(f"[clover] feedback_workflow failed; falling back to builtin: {e}")
            return self._fallback_feedback(context_id, policy_text, raw_input_text, pred, report)

    def _fallback_feedback(
        self,
        context_id: str,
        policy_text: str,
        raw_input_text: str,
        pred: str,
        report: Dict[str, Any],
    ) -> str:
        failed = report.get("failed_checks", [])
        if not isinstance(failed, list):
            failed = [str(failed)]
        return "\n".join(
            [
                "The previous output failed verification.",
                f"context_id: {context_id}",
                "",
                "Failed checks:",
                *[f"- {f}" for f in failed],
                "",
                "Policy:",
                (policy_text or "").strip(),
                "",
                "Input ticket:",
                (raw_input_text or "").strip(),
                "",
                "Model output:",
                (pred or "").strip(),
                "",
                "Return a corrected output that will pass verification.",
            ]
        )

    def _prompt_candidate(
        self,
        tracer: RuntimeTracer,
        wf: Callable[..., Any],
        wf_code: str,
        output_node: Any,
        feedback: str,
        context_id: str,
        policy_text: str,
        raw_input_text: str,
        parsed_input: Any,
    ) -> Optional[Candidate]:
        if not tracer.prompt_templates:
            return None
        if output_node is None:
            return None

        params = list(tracer.prompt_templates.values())
        try:
            orig_values = [p.data for p in params]  # type: ignore[attr-defined]
        except Exception:
            return None

        cand: Optional[Candidate] = None
        try:
            opt = OptoPrimeLocal(params)
            opt.backward(output_node, feedback, visualize=False)
            opt.step(mode="per_param", verbose="output")

            opt_values = [p.data for p in params]  # type: ignore[attr-defined]

            p_run = self._run_once(
                tracer=tracer,
                wf=wf,
                policy_text=policy_text,
                raw_input_text=raw_input_text,
                schema_text=self.adapter.schema_text,
                meta_prompt=self.meta_prompt,
            )
            p_report = self.adapter.verify(
                context_id=context_id,
                policy_text=policy_text,
                parsed_input=parsed_input,
                pred=p_run.pred,
                cost_usd=p_run.total_cost_usd,
            )
            cand = Candidate(
                kind="prompt",
                wf=wf,
                wf_code=wf_code,
                tracer=tracer,
                run=p_run,
                report=p_report,
                prompt_values=opt_values,
            )
        finally:
            # Always restore the original prompt templates for fair candidate selection.
            for p, v in zip(params, orig_values):
                try:
                    p.data = v  # type: ignore[attr-defined]
                except Exception:
                    pass

        return cand

    def _structure_candidate(
        self,
        current_wf: Callable[..., Any],
        current_wf_code: str,
        base_ir: Dict[str, Any],
        feedback: str,
        context_id: str,
        policy_text: str,
        raw_input_text: str,
        parsed_input: Any,
    ) -> Optional[Candidate]:
        editor = StructureEditor(verbose=False, forbid_imports=True)
        edit = editor.rewrite_function(
            current_wf,
            base_ir,
            feedback,
            required_call_tags=self.required_call_tags,
            func_name=self.workflow_func_name,
            max_retries=1,
        )
        if not edit.ok or not edit.code:
            return None

        try:
            new_wf = editor.load_function(
                edit.code,
                func_name=self.workflow_func_name,
                extra_globals={"llm": llm, "msg": msg},
            )
        except Exception as e:
            logger.warning(f"[clover] load_function failed: {e}")
            return None

        s_tracer = RuntimeTracer(trainable_prompt_templates=True, clear_graph_on_enter=True)
        s_run = self._run_once(
            tracer=s_tracer,
            wf=new_wf,
            policy_text=policy_text,
            raw_input_text=raw_input_text,
            schema_text=self.adapter.schema_text,
            meta_prompt=self.meta_prompt,
        )
        s_report = self.adapter.verify(
            context_id=context_id,
            policy_text=policy_text,
            parsed_input=parsed_input,
            pred=s_run.pred,
            cost_usd=s_run.total_cost_usd,
        )
        return Candidate(
            kind="struct",
            wf=new_wf,
            wf_code=edit.code,
            tracer=s_tracer,
            run=s_run,
            report=s_report,
        )

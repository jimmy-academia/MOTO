# schemes/clover.py
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from myopto.optimizers import OptoPrimeLocal
from myopto.optimizers.structure_editor import StructureEditor
from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto.utils.llm_call import llm_json
from myopto.utils.llm_router import get_llm
from myopto.utils.usage import get_total_cost, reset_usage

from schemes.base import BaseScheme
from prompt.clover import META_PROMPT, SEED_WORKFLOW_CODE, FEEDBACK_WORKFLOW_CODE


@dataclass
class CloverTrajectoryStep:
    iteration: int
    pred: str
    feedback: str
    wf_code_snippet: str
    wf_code_hash: str
    cost_usd: float


class CloverScheme(BaseScheme):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.meta_prompt = META_PROMPT
        self.seed_workflow_code = SEED_WORKFLOW_CODE
        self.feedback_workflow_code = FEEDBACK_WORKFLOW_CODE

        self.editor = StructureEditor(
            llm=get_llm(role="optimizer"),
            max_tokens=getattr(args, "structure_max_tokens", 12000),
            require_call_tag=True,
            forbid_strip_trace_tags=True,
            forbid_imports=True,
            verbose=getattr(args, "verbose", False),
        )

        # OptoPrimeLocal (and friends) have had signature drift across myopto versions.
        # Build kwargs defensively so we don't call it with the wrong constructor args.
        opt_kwargs: Dict[str, Any] = {"max_tokens": getattr(args, "opt_max_tokens", 12000)}
        try:
            sig = inspect.signature(OptoPrimeLocal)
            params = sig.parameters
            if "llm" in params:
                opt_kwargs["llm"] = get_llm(role="optimizer")
            if "log" in params:
                opt_kwargs["log"] = False
            elif "logging" in params:
                opt_kwargs["logging"] = False
        except Exception:
            # Historical constructor path.
            opt_kwargs["log"] = False

        self.prompt_optimizer = OptoPrimeLocal(**opt_kwargs)

        self.seed_workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")
        self.feedback_workflow_fn = self._load(self.feedback_workflow_code, "feedback_workflow")

    def _vprint(self, *args, **kwargs):
        if getattr(self.args, "verbose", False):
            print(*args, **kwargs)

    def _load(self, code: str, fn_name: str):
        # Allow workflows to call llm/msg at runtime.
        return self.editor.load_function(code, fn_name, extra_globals={"llm": llm, "msg": msg})

    def _sha12(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

    def _make_tracer(self, *, trainable_prompt_templates: bool, name: Optional[str] = None) -> RuntimeTracer:
        """
        Create a RuntimeTracer while staying compatible with multiple myopto tracer signatures.

        Key behavior:
        - Keep the trace graph available after leaving the context so the optimizer can read it.
        - Optionally route disk traces to args.trace_dir/trace_root if supported.
        """
        kwargs: Dict[str, Any] = {
            "trainable_prompt_templates": trainable_prompt_templates,
            "clear_graph_on_enter": True,
        }

        try:
            sig = inspect.signature(RuntimeTracer)
            params = sig.parameters

            # Some implementations clear graph on exit; disable if knob exists so optimizers can read the trace.
            if "clear_graph_on_exit" in params:
                kwargs["clear_graph_on_exit"] = False

            # Support directing traces to a specific directory (if tracer supports it).
            trace_root = (
                getattr(self.args, "trace_dir", None)
                or getattr(self.args, "trace_root", None)
                or getattr(self.args, "trace_path", None)
            )
            if trace_root is not None:
                if "trace_dir" in params:
                    kwargs["trace_dir"] = trace_root
                elif "trace_root" in params:
                    kwargs["trace_root"] = trace_root
                elif "save_dir" in params:
                    kwargs["save_dir"] = trace_root

            if name:
                if "name" in params:
                    kwargs["name"] = name
                elif "run_name" in params:
                    kwargs["run_name"] = name
                elif "call_tag" in params:
                    kwargs["call_tag"] = name
                elif "session_name" in params:
                    kwargs["session_name"] = name
        except Exception:
            pass

        return RuntimeTracer(**kwargs)

    def _finalize_tracer(self, tracer: RuntimeTracer) -> None:
        """Best-effort tracer finalization for implementations that flush to disk on finalize/close."""
        for method_name in ("finalize", "flush", "close"):
            fn = getattr(tracer, method_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def _call_workflow(self, wf_fn, context: str, x: str, tracer: RuntimeTracer) -> Tuple[str, Any, dict, float]:
        # Cost delta for this workflow call only (usage bucket is shared across batch).
        start_cost = float(get_total_cost())
        with tracer:
            out_msg = wf_fn(context, x)

        # Ensure any on-exit flushes happen before we read IR / trace files.
        self._finalize_tracer(tracer)

        pred = strip_trace_tags(str(out_msg))
        try:
            ir = tracer.to_ir()
        except Exception:
            ir = {}

        end_cost = float(get_total_cost())
        return pred, out_msg, ir, end_cost - start_cost

    def _call_feedback(self, feedback_fn, trace_ir: dict, pred: str, tracer: RuntimeTracer) -> str:
        with tracer:
            out_msg = feedback_fn(trace_ir, pred)

        self._finalize_tracer(tracer)
        return strip_trace_tags(str(out_msg))

    def _invoke_optimizer_method(
        self,
        fn,
        *,
        tracer: RuntimeTracer,
        out_msg: Any,
        prompt_templates: Any,
        trace_ir: Optional[dict],
        trace_dir: Any,
        objective: Any,
        call_tag: str,
        verbose: bool,
    ):
        """
        Call an optimizer method while adapting to signature drift across myopto versions.

        We try to satisfy required parameters by name (tracer/templates/trace_ir/trace_dir/output)
        and pass objective+call_tag+verbose when accepted.
        """
        try:
            sig = inspect.signature(fn)
        except Exception:
            # Signature not inspectable: try common call shapes.
            common_attempts = [
                ((), {"tracer": tracer, "objective": objective, "call_tag": call_tag, "verbose": verbose}),
                ((tracer,), {"objective": objective, "call_tag": call_tag, "verbose": verbose}),
                ((prompt_templates,), {"objective": objective, "call_tag": call_tag, "verbose": verbose}),
                ((trace_ir,), {"objective": objective, "call_tag": call_tag, "verbose": verbose}),
                ((out_msg, objective), {"call_tag": call_tag, "verbose": verbose}),
                ((tracer, objective), {"call_tag": call_tag, "verbose": verbose}),
                ((objective,), {"call_tag": call_tag, "verbose": verbose}),
                ((), {}),
            ]
            last_err: Optional[Exception] = None
            for args, kwargs in common_attempts:
                try:
                    return fn(*args, **kwargs)
                except TypeError as e:
                    last_err = e
                    continue
            if last_err is not None:
                raise last_err
            return fn()

        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        kwargs: Dict[str, Any] = {}

        # objective naming
        if accepts_var_kw or "objective" in params:
            kwargs["objective"] = objective
        elif accepts_var_kw or "feedback" in params:
            kwargs["feedback"] = objective
        elif accepts_var_kw or "loss" in params:
            kwargs["loss"] = objective

        if accepts_var_kw or "call_tag" in params:
            kwargs["call_tag"] = call_tag
        if accepts_var_kw or "verbose" in params:
            kwargs["verbose"] = verbose

        # Required args (excluding self/*args/**kwargs and excluding kwargs already supplied).
        required_names: List[str] = []
        for name, p in params.items():
            if name == "self":
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if p.default is inspect._empty and name not in kwargs:
                required_names.append(name)

        def resolve_required(name: str) -> Any:
            # Tracer / trace tape
            if name in ("tracer", "trace", "runtime_tracer", "rt", "tape", "graph"):
                return tracer

            # Prompt/template collections
            if name in ("prompt_templates", "prompt_template", "templates", "params", "parameters", "prompts"):
                return prompt_templates if prompt_templates is not None else tracer

            # Trace IR / graph dict
            if name in ("trace_ir", "ir", "trace_dict", "trace_graph"):
                return trace_ir if trace_ir is not None else {}

            # Trace directory / path on disk
            if name in ("trace_dir", "trace_path", "trace_root", "save_dir"):
                return trace_dir

            # Output/result node
            if name in ("node", "out", "output", "output_node", "result", "root", "y", "pred", "prediction"):
                return out_msg if out_msg is not None else tracer

            # Feedback/objective
            if name in ("feedback", "objective", "loss"):
                return objective

            raise TypeError(f"Unsupported required argument '{name}' for {fn!r}")

        args: List[Any] = [resolve_required(n) for n in required_names]

        if not accepts_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in params}

        return fn(*args, **kwargs)

    def _run_prompt_optimizer(
        self,
        *,
        tracer: RuntimeTracer,
        out_msg: Any,
        trace_ir: Optional[dict],
        feedback: str,
        call_tag: str,
    ) -> bool:
        """
        Apply OptoPrimeLocal (or compatible myopto optimizers) correctly.

        Fixes vs buggy version:
        - Do NOT assume optimizer.step(tracer, objective=msg(...)) is the only valid API.
        - Prefer plain-string objectives; fall back to msg(feedback) only if needed.
        - If optimizer exposes zero_feedback/backward/step, use the full 3-stage API.
        - Provide tracer/prompt_templates/trace_ir/trace_dir depending on what the optimizer expects.
        """
        opt = getattr(self, "prompt_optimizer", None)
        if opt is None:
            return False

        verbose = bool(getattr(self.args, "verbose", False))

        prompt_templates = (
            getattr(tracer, "prompt_templates", None)
            or getattr(tracer, "templates", None)
            or getattr(tracer, "prompt_template", None)
        )
        trace_dir = (
            getattr(tracer, "trace_dir", None)
            or getattr(tracer, "trace_path", None)
            or getattr(tracer, "trace_root", None)
            or getattr(tracer, "save_dir", None)
        )

        # Objective candidates: string first (most optimizers expect this), msg(...) second as fallback.
        objectives: List[Any] = []
        if isinstance(feedback, str):
            objectives.append(feedback)
        try:
            objectives.append(msg(feedback))
        except Exception:
            pass

        zero_feedback_fn = getattr(opt, "zero_feedback", None)
        backward_fn = getattr(opt, "backward", None)
        step_fn = getattr(opt, "step", None)

        last_err: Optional[Exception] = None

        # Trace-style API: zero_feedback -> backward -> step
        if callable(backward_fn) and callable(step_fn):
            for obj in objectives:
                try:
                    if callable(zero_feedback_fn):
                        self._invoke_optimizer_method(
                            zero_feedback_fn,
                            tracer=tracer,
                            out_msg=out_msg,
                            prompt_templates=prompt_templates,
                            trace_ir=trace_ir,
                            trace_dir=trace_dir,
                            objective=obj,
                            call_tag=call_tag,
                            verbose=verbose,
                        )

                    self._invoke_optimizer_method(
                        backward_fn,
                        tracer=tracer,
                        out_msg=out_msg,
                        prompt_templates=prompt_templates,
                        trace_ir=trace_ir,
                        trace_dir=trace_dir,
                        objective=obj,
                        call_tag=call_tag,
                        verbose=verbose,
                    )
                    self._invoke_optimizer_method(
                        step_fn,
                        tracer=tracer,
                        out_msg=out_msg,
                        prompt_templates=prompt_templates,
                        trace_ir=trace_ir,
                        trace_dir=trace_dir,
                        objective=obj,
                        call_tag=call_tag,
                        verbose=verbose,
                    )
                    return True
                except Exception as e:
                    last_err = e
                    continue

        # One-shot step API: step(...)
        if callable(step_fn):
            for obj in objectives:
                try:
                    self._invoke_optimizer_method(
                        step_fn,
                        tracer=tracer,
                        out_msg=out_msg,
                        prompt_templates=prompt_templates,
                        trace_ir=trace_ir,
                        trace_dir=trace_dir,
                        objective=obj,
                        call_tag=call_tag,
                        verbose=verbose,
                    )
                    return True
                except Exception as e:
                    last_err = e
                    continue

        if verbose and last_err is not None:
            self._vprint(f"[CloverScheme] prompt optimizer failed ({type(last_err).__name__}): {last_err}")
        return False

    async def train_one_batch(self, batch, calculate_score):
        # One usage bucket per batch.
        reset_usage()

        contexts, xs, answers = batch
        obs: List[Dict[str, Any]] = []
        passed_count = 0

        for ctx, x, ans in zip(contexts, xs, answers):
            record = self.inner_loop(ctx, x)
            pred = record["pred"]

            score = 0.0
            try:
                score, _extra = calculate_score(ans, pred)
            except TypeError:
                # Back-compat: some benchmarks use calculate_score(pred, ans)
                score, _extra = calculate_score(pred, ans)

            passed = bool(score == 1 or score == 1.0)
            if passed:
                passed_count += 1

            # Keep the batch observation self-contained for outer loop.
            record["score"] = float(score)
            record["passed"] = passed
            record["context"] = ctx
            record["input"] = x
            obs.append(record)

        batch_size = len(answers)
        all_passed = passed_count == batch_size

        # Outer loop runs once per batch, but NEVER runs on an all-pass batch.
        updated = False
        if not all_passed:
            failed_obs = [o for o in obs if not o.get("passed")]
            updated = self.outer_loop(failed_obs or obs)

        if updated:
            # Reload updated seed workflow (outer loop can only update seed_workflow).
            self.seed_workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")

        # Expose all_passed so the epoch driver can early-stop.
        return {
            "cost_usd": float(get_total_cost()),
            "passed": passed_count,
            "batch_size": batch_size,
            "all_passed": all_passed,
            "outer_updated": updated,
            "observations": obs,
        }

    def inner_loop(self, context: str, x: str) -> Dict[str, Any]:
        wf_code = self.seed_workflow_code
        wf_fn = self.seed_workflow_fn

        trajectory: List[CloverTrajectoryStep] = []
        last_pred = ""
        last_cost = 0.0

        # Stable short identifier so trace/call_tag artifacts don't collide across samples.
        sample_tag = self._sha12(f"{context}\n{x}")

        for it in range(getattr(self.args, "inner_loop_iters", 3)):
            tracer = self._make_tracer(trainable_prompt_templates=True, name=f"clover_fw_{sample_tag}_{it}")
            pred, out_msg, trace_ir, cost = self._call_workflow(wf_fn, context, x, tracer)
            last_pred = pred
            last_cost = float(cost)

            # Feedback has access to trace_ir and pred but not ground truth.
            tracer_fb = self._make_tracer(trainable_prompt_templates=False, name=f"clover_fb_{sample_tag}_{it}")
            feedback = self._call_feedback(self.feedback_workflow_fn, trace_ir, pred, tracer_fb)

            # Structure update (workflow rewrite).
            wf_code_new = self.editor.rewrite_code(
                code=wf_code,
                feedback=feedback,
                call_tag=f"clover_struct_{sample_tag}_{it}",
            )
            if isinstance(wf_code_new, str) and wf_code_new.strip():
                wf_code = wf_code_new
                wf_fn = self._load(wf_code, "seed_workflow")

            # Prompt update (OptoPrimeLocal or compatible optimizer).
            self._run_prompt_optimizer(
                tracer=tracer,
                out_msg=out_msg,
                trace_ir=trace_ir,
                feedback=feedback,
                call_tag=f"clover_prompt_{sample_tag}_{it}",
            )

            trajectory.append(
                CloverTrajectoryStep(
                    iteration=it,
                    pred=pred,
                    feedback=feedback,
                    wf_code_snippet=wf_code[:4000],
                    wf_code_hash=self._sha12(wf_code),
                    cost_usd=float(cost),
                )
            )

        return {
            "pred": last_pred,
            "iterations": len(trajectory),
            "sample_cost_usd": float(last_cost),
            "trajectory": [step.__dict__ for step in trajectory],
        }

    def outer_loop(self, observations: List[Dict[str, Any]]) -> bool:
        prompt = f"""You are a meta-optimizer improving a Python workflow function.

Current seed_workflow_code (seed_workflow):
{self.seed_workflow_code}

Recent batch observations (up to 10):
{json.dumps(observations[:10], ensure_ascii=False)}

Your job: decide whether to update the SEED workflow code. If yes, output a new seed_workflow_code string.

Constraints:
- Do not add imports.
- Keep function name seed_workflow fixed.
- Only return STRICT JSON with the keys:
  - update_seed_workflow: boolean
  - new_seed_workflow_code: string
"""

        schema = {
            "name": "clover_outer_update",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "update_seed_workflow": {"type": "boolean"},
                    "new_seed_workflow_code": {"type": "string"},
                },
                "required": ["update_seed_workflow", "new_seed_workflow_code"],
            },
            "strict": True,
        }

        # Be defensive: llm_json signature has drifted across versions.
        try:
            obj = llm_json(prompt, role="metaoptimizer", json_schema=schema, call_tag="clover_outer")
        except TypeError:
            try:
                obj = llm_json(prompt, role="metaoptimizer", schema=schema, call_tag="clover_outer")
            except Exception:
                return False
        except Exception:
            return False

        update = bool(obj.get("update_seed_workflow", False))
        if not update:
            return False

        new_code = obj.get("new_seed_workflow_code")
        if not isinstance(new_code, str) or not new_code.strip():
            return False

        if new_code.strip() == self.seed_workflow_code.strip():
            return False

        self.seed_workflow_code = new_code
        return True

    def save_model(self, epoch: Optional[int] = None):
        """Persist Clover state (text) so runs can resume without retraining."""
        self.scheme_file.parent.mkdir(parents=True, exist_ok=True)

        code = (
            "# Auto-generated by CloverScheme.save_model\n"
            "# Safe to exec/import to restore state.\n\n"
            f"META_PROMPT = {repr(self.meta_prompt)}\n\n"
            f"SEED_WORKFLOW_CODE = {repr(self.seed_workflow_code)}\n\n"
            f"FEEDBACK_WORKFLOW_CODE = {repr(self.feedback_workflow_code)}\n"
        )
        self.scheme_file.write_text(code, encoding="utf-8")

        if epoch is not None:
            snap = self.scheme_file.parent / f"scheme_epoch_{epoch}.py"
            snap.write_text(code, encoding="utf-8")

    def load(self, path: Path):
        if not path.exists():
            return False

        ns: Dict[str, Any] = {}
        try:
            txt = path.read_text(encoding="utf-8")
            exec(compile(txt, str(path), "exec"), ns, ns)
        except Exception:
            return False

        self.meta_prompt = ns.get("META_PROMPT", self.meta_prompt)
        self.seed_workflow_code = ns.get("SEED_WORKFLOW_CODE", self.seed_workflow_code)
        self.feedback_workflow_code = ns.get("FEEDBACK_WORKFLOW_CODE", self.feedback_workflow_code)

        self.seed_workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")
        self.feedback_workflow_fn = self._load(self.feedback_workflow_code, "feedback_workflow")
        return True

    async def inference_with_meta(self, context: str, x: str) -> Tuple[str, float, Dict[str, Any]]:
        """Preferred structured inference API (context + input)."""
        reset_usage()
        record = self.inner_loop(context, x)
        pred = record.get("pred", "")
        cost_usd = float(get_total_cost())
        meta = dict(record)
        meta["cost_usd"] = cost_usd
        return str(pred), cost_usd, meta

    async def inference(self, input_text: str) -> Tuple[str, float]:
        """Default inference API used by BaseBenchmark. Treat input_text as `x` with empty context."""
        pred, cost_usd, _meta = await self.inference_with_meta("", input_text)
        return pred, cost_usd

# schemes/clover.py
import hashlib
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

from utils.log import logger


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

        # IMPORTANT:
        # OptoPrimeLocal requires `parameters` (ParameterNode list) at construction time.
        # We do NOT have prompt template parameters until after we run a traced forward pass.
        # So we instantiate OptoPrimeLocal lazily per step (or you could cache it).
        self._opt_max_tokens = getattr(args, "opt_max_tokens", 12000)
        self._opt_verbose = getattr(args, "verbose", False)

        self.seed_workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")
        self.feedback_workflow_fn = self._load(self.feedback_workflow_code, "feedback_workflow")

    def train_one_batch(self, batch, calculate_score):
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
                score, _extra = calculate_score(pred, ans)

            passed = bool(score == 1 or score == 1.0)
            if passed:
                passed_count += 1

            record["score"] = float(score)
            record["passed"] = passed
            record["context"] = ctx
            record["input"] = x
            obs.append(record)

        batch_size = len(answers)
        all_passed = passed_count == batch_size

        updated = False
        if not all_passed:
            failed_obs = [o for o in obs if not o.get("passed")]
            updated = self.outer_loop(failed_obs or obs)

        if updated:
            self.seed_workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")

        return {
            "cost_usd": float(get_total_cost()),
            "passed": passed_count,
            "batch_size": batch_size,
            "all_passed": all_passed,
            "outer_updated": updated,
            "observations": obs,
        }

    def _load(self, code: str, fn_name: str):
        # Allow workflows to call llm/msg at runtime.
        return self.editor.load_function(code, fn_name, extra_globals={"llm": llm, "msg": msg})

    def _call_workflow(self, wf_fn, context: str, x: str, tracer: RuntimeTracer) -> Tuple[str, Any, Any, dict, float, Optional[str]]:
        start_cost = float(get_total_cost())
        with tracer:
            out_msg = wf_fn(context, x)

        pred = strip_trace_tags(str(out_msg))
        try:
            ir = tracer.to_ir()
        except Exception as e:
            ir = {
                "_error": "to_ir_failed",
                "_exception": repr(e),
            }
        cost = float(get_total_cost()) - start_cost

        # Primary: Msg.node (normal case)
        output_node = getattr(out_msg, "node", None)
        # Fallback: last LLM node executed under this tracer
        if output_node is None:
            output_node = getattr(tracer, "output_node", None)

        # If still None: no traced LLM output
        err = None
        if output_node is None:
            err = "No output_node available: indicating that workflow did not call any llm (cannot run OptoPrime backward/step)."
            logger.warning(err)
            pred = f"[CLOVER_ERROR]\n{err}\n\n[WORKFLOW_OUTPUT]\n{pred}"

        return pred, output_node, ir, cost


    def inner_loop(self, context: str, x: str) -> Dict[str, Any]:
        wf_code = self.seed_workflow_code
        wf_fn = self.seed_workflow_fn

        trajectory: List[CloverTrajectoryStep] = []
        last_pred = ""
        last_cost = 0.0

        sample_tag = self._sha12(f"{context}\n{x}")

        for it in range(getattr(self.args, "inner_loop_iters", 3)):
            # Forward trace (this is the graph OptoPrimeLocal needs)
            tracer = RuntimeTracer(trainable_prompt_templates=True, clear_graph_on_enter=True)
            pred, output_node, trace_ir, cost = self._call_workflow(wf_fn, context, x, tracer)
            last_pred = pred
            last_cost = cost


            # Feedback generation MUST NOT clear forward trace graph.
            feedback = self._call_feedback_preserving_graph(trace_ir, pred, cost)

            # Structure update (workflow rewrite)
            wf_code_new = self.editor.rewrite_code(
                code=wf_code,
                feedback=feedback,
                call_tag=f"clover_struct_{sample_tag}_{it}",
            )
            if isinstance(wf_code_new, str) and wf_code_new.strip():
                wf_code = wf_code_new
                wf_fn = self._load(wf_code, "seed_workflow")

            # Prompt update using OptoPrimeLocal correctly:
            # - parameters = tracer.prompt_templates.values()
            # - backward(output_node, feedback)
            # - step(mode="per_param")
            try:
                params = self._get_prompt_parameters(tracer)
                if params and output_node is not None:
                    opt = self._make_prompt_optimizer(params)
                    opt.zero_feedback()

                    # backward signature varies slightly; keep it simple.
                    try:
                        opt.backward(output_node, feedback, visualize=False)
                    except TypeError:
                        opt.backward(output_node, feedback)

                    # Prefer per_param (your OptoPrimeLocal supports it)
                    try:
                        opt.step(mode="per_param", verbose=("output" if getattr(self.args, "verbose", False) else False))
                    except TypeError:
                        # older variants may not have mode/verbose
                        opt.step()
            except Exception:
                # Clover should be resilient; but don't silently swallow in verbose mode.
                if getattr(self.args, "verbose", False):
                    raise

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


    def _sha12(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

    def _get_prompt_parameters(self, tracer: RuntimeTracer) -> List[Any]:
        """
        Extract trainable prompt template ParameterNodes from the tracer.
        In myopto runtime tracer, these typically live in tracer.prompt_templates (dict).
        """
        pt = getattr(tracer, "prompt_templates", None)
        if isinstance(pt, dict):
            return list(pt.values())
        if isinstance(pt, (list, tuple)):
            return list(pt)
        return []

    def _make_prompt_optimizer(self, parameters: List[Any]) -> OptoPrimeLocal:
        """
        Construct OptoPrimeLocal correctly: first arg MUST be `parameters`.
        Then pass optional knobs if supported by this myopto version.
        """
        # Try the richest constructor first; fall back if this myopto version
        # doesn't accept some kwargs.
        tried: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = [
            ((parameters,), {"max_tokens": self._opt_max_tokens, "log": False, "llm": get_llm(role="optimizer")}),
            ((parameters,), {"max_tokens": self._opt_max_tokens, "log": False}),
            ((parameters,), {"max_tokens": self._opt_max_tokens}),
            ((parameters,), {}),
        ]
        last_err: Optional[Exception] = None
        for args, kwargs in tried:
            try:
                return OptoPrimeLocal(*args, **kwargs)
            except TypeError as e:
                last_err = e
                continue
        # If we got here, surface the last error; this is not recoverable.
        raise last_err if last_err is not None else TypeError("Failed to construct OptoPrimeLocal(parameters, ...)")


    def _call_feedback_preserving_graph(self, trace_ir: dict, pred: str) -> str:
        """
        Generate feedback WITHOUT clearing the forward trace graph.

        Key fix:
        - Do NOT enter a RuntimeTracer with clear_graph_on_enter=True here.
          That would wipe the workflow trace needed for OptoPrimeLocal.backward().

        We try untraced first; if that fails (because your feedback workflow insists on a tracer),
        we trace it but with clear_graph_on_enter=False so we don't destroy the forward graph.
        """
        try:
            out_msg = self.feedback_workflow_fn(trace_ir, pred)
            return strip_trace_tags(str(out_msg))
        except Exception:
            tracer_fb = RuntimeTracer(trainable_prompt_templates=False, clear_graph_on_enter=False)
            with tracer_fb:
                out_msg = self.feedback_workflow_fn(trace_ir, pred)
            return strip_trace_tags(str(out_msg))


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
        reset_usage()
        record = self.inner_loop(context, x)
        pred = record.get("pred", "")
        cost_usd = float(get_total_cost())
        meta = dict(record)
        meta["cost_usd"] = cost_usd
        return str(pred), cost_usd, meta

    async def inference(self, input_text: str) -> Tuple[str, float]:
        pred, cost_usd, _meta = await self.inference_with_meta("", input_text)
        return pred, cost_usd

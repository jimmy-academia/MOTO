# schemes/clover.py
import hashlib
import json

from myopto.optimizers import OptoPrimeLocal
from myopto.optimizers.structure_editor import StructureEditor
from myopto.trace.runtime import llm, msg
from myopto.utils.llm_call import llm_json
from myopto.utils.llm_router import get_llm
from myopto.utils.usage import get_total_cost, reset_usage

from .base import BaseScheme
from prompt.clover import META_PROMPT, SEED_WORKFLOW_CODE, FEEDBACK_WORKFLOW_CODE
# from .ecwo import InnerLoopEngine


class CloverScheme(BaseScheme):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.meta_prompt = META_PROMPT
        self.seed_workflow_code = SEED_WORKFLOW_CODE
        self.feedback_workflow_code = FEEDBACK_WORKFLOW_CODE

        self.editor = StructureEditor(
            llm=get_llm(role="optimizer"),
            max_tokens=self._arg("structure_max_tokens", 12000),
            require_call_tag=True,
            forbid_strip_trace_tags=True,
            forbid_imports=True,
            verbose=self._arg("verbose", False),
        )

        self._opt_max_tokens = self._arg("opt_max_tokens", 12000)
        self._opt_verbose = self._arg("verbose", False)

        self.seed_workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")
        self.feedback_workflow_fn = self._load(self.feedback_workflow_code, "feedback_workflow")

    def _arg(self, name, default=None):
        if hasattr(self.args, name):
            return self.args.__dict__.get(name, default)
        return default

    def train_one_batch(self, batch, calculate_score):
        reset_usage()

        contexts, xs, answers = batch
        obs = []
        passed_count = 0

        for ctx, x, ans in zip(contexts, xs, answers):
            record = self.inner_loop(ctx, x)
            pred = record.get("pred")

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

    def _load(self, code, fn_name):
        return self.editor.load_function(code, fn_name, extra_globals={"llm": llm, "msg": msg})

    def inner_loop(self, context, x):
        wf_code = self.seed_workflow_code
        wf_fn = self.seed_workflow_fn

        engine = InnerLoopEngine(
            editor=self.editor,
            feedback_fn=self.feedback_workflow_fn,
            make_opt=self._make_prompt_optimizer,
            verbose=self._arg("verbose", False),
        )

        sample_tag = self._sha12(f"{context}\n{x}")
        result = engine.run(
            wf_code,
            wf_fn,
            context,
            x,
            iterations=self._arg("inner_loop_iters", 3),
            structure_budget=self._arg("structure_budget", 1),
            structure_late_trigger_only=self._arg("structure_late_trigger_only", True),
            sample_tag=sample_tag,
        )

        self.seed_workflow_code = result.get("wf_code", wf_code)
        self.seed_workflow_fn = result.get("wf_fn", wf_fn)

        return {
            "pred": result.get("pred"),
            "iterations": result.get("iterations"),
            "sample_cost_usd": float(result.get("sample_cost_usd", 0.0)),
            "trajectory": result.get("trajectory", []),
        }

    def _sha12(self, s):
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

    def _make_prompt_optimizer(self, parameters):
        tried = [
            ((parameters,), {"max_tokens": self._opt_max_tokens, "log": False, "llm": get_llm(role="optimizer")}),
            ((parameters,), {"max_tokens": self._opt_max_tokens, "log": False}),
            ((parameters,), {"max_tokens": self._opt_max_tokens}),
            ((parameters,), {}),
        ]
        last_err = None
        for args, kwargs in tried:
            try:
                return OptoPrimeLocal(*args, **kwargs)
            except TypeError as e:
                last_err = e
                continue
        raise last_err if last_err is not None else TypeError("Failed to construct OptoPrimeLocal(parameters, ...)")

    def outer_loop(self, observations):
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
        if not new_code or not str(new_code).strip():
            return False

        if str(new_code).strip() == str(self.seed_workflow_code).strip():
            return False

        self.seed_workflow_code = str(new_code)
        return True

    def save_model(self, epoch=None):
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

    def load(self, path):
        if not path.exists():
            return False

        ns = {}
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

    async def inference_with_meta(self, context, x):
        reset_usage()
        record = self.inner_loop(context, x)
        pred = record.get("pred", "")
        cost_usd = float(get_total_cost())
        meta = dict(record)
        meta["cost_usd"] = cost_usd
        return str(pred), cost_usd, meta

    async def inference(self, input_text):
        pred, cost_usd, _meta = await self.inference_with_meta("", input_text)
        return pred, cost_usd

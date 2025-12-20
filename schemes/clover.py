# schemes/clover.py
import hashlib
import json

from myopto.optimizers import OptoPrimeLocal
from myopto.optimizers.structure_editor import StructureEditor
from myopto.trace.runtime import llm, msg
from myopto.utils.llm_router import get_llm, extract_text, try_parse_json
from myopto.utils.usage import get_total_cost, reset_usage

from .base import BaseScheme
from prompt.clover import META_PROMPT, SEED_WORKFLOW_CODE, FEEDBACK_WORKFLOW_CODE


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
            # Run seed workflow
            try:
                result = self.seed_workflow_fn(ctx, x)
            except Exception as e:
                result = f"[ERROR] {e}"

            score, feedback = calculate_score(result, ans)
            passed_count += 1 if score >= 1.0 else 0

            obs.append({
                "context": ctx[:200] if ctx else "",
                "question": x[:200] if x else "",
                "answer": ans[:200] if isinstance(ans, str) else str(ans)[:200],
                "result": str(result)[:200],
                "score": score,
                "feedback": feedback[:200] if feedback else "",
            })

        # Run outer update
        updated = self._outer_update(obs)

        cost = get_total_cost()
        return {
            "passed": passed_count,
            "total": len(batch[0]),
            "cost": cost,
            "updated": updated,
        }

    def _load(self, code: str, fn_name: str):
        """Load function from code string."""
        ns = {}
        try:
            exec(code, ns)
        except Exception as e:
            raise RuntimeError(f"Failed to load {fn_name}: {e}")
        if fn_name not in ns:
            raise RuntimeError(f"Function {fn_name} not found in code")
        return ns[fn_name]

    def _outer_update(self, observations: list) -> bool:
        """Meta-optimizer step: decide whether to update seed workflow."""
        prompt = f"""
You are a meta-optimizer for workflow code improvement.

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

        # --------------------------------------------------
        # Use get_llm + direct call with response_format
        # --------------------------------------------------
        try:
            llm = get_llm(role="metaoptimizer")
            response = llm(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": schema},
            )
            text = extract_text(response)
            obj = try_parse_json(text)
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
            exec(path.read_text(encoding="utf-8"), ns)
        except Exception:
            return False

        if "META_PROMPT" in ns:
            self.meta_prompt = ns["META_PROMPT"]
        if "SEED_WORKFLOW_CODE" in ns:
            self.seed_workflow_code = ns["SEED_WORKFLOW_CODE"]
            self.seed_workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")
        if "FEEDBACK_WORKFLOW_CODE" in ns:
            self.feedback_workflow_code = ns["FEEDBACK_WORKFLOW_CODE"]
            self.feedback_workflow_fn = self._load(self.feedback_workflow_code, "feedback_workflow")

        return True
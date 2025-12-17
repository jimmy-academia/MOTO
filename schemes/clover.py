from typing import Any, Dict, List, Tuple

from schemes.base import BaseScheme

from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto.optimizers.optoprime_local import OptoPrimeLocal
from myopto.optimizers.structure_editor import StructureEditor
from myopto.utils.usage import reset_usage, get_total_cost

from benchmarks.clovertoy import CLOVERTOY_SCHEMA, parse_ticket, verify_output
from prompt.clover import META_PROMPT, SEED_WORKFLOW_CODE, FEEDBACK_WORKFLOW_CODE


class CloverScheme(BaseScheme):
    def __init__(self, args: Any):
        super().__init__(args)
        self.args = args

        self.meta_prompt = META_PROMPT
        self.seed_workflow_code = SEED_WORKFLOW_CODE
        self.feedback_workflow_code = FEEDBACK_WORKFLOW_CODE

        self.editor = StructureEditor(verbose=False, forbid_imports=True)
        self.workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")
        self.feedback_fn = self._load(self.feedback_workflow_code, "feedback_workflow")

    def _load(self, code: str, func_name: str):
        return self.editor.load_function(code, func_name=func_name, extra_globals={"llm": llm, "msg": msg})

    async def train_one_batch(self, batch, calculate_score):
        contexts, problems, answers = batch
        reset_usage()

        obs: List[Dict[str, Any]] = []
        for ctx, x, y_true in zip(contexts, problems, answers):
            meta = self.inner_loop(ctx, x, y_true)
            obs.append(meta)

        # outer loop: meta-update the 3 artifacts (small + direct)
        updated = self.outer_loop(obs)
        if updated:
            self.workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")
            self.feedback_fn = self._load(self.feedback_workflow_code, "feedback_workflow")

        return {"cost_usd": float(get_total_cost()), "passed": sum(int(o["passed"]) for o in obs)}

    def inner_loop(self, ctx: str, x: str, y_true: str):
        iters = int(getattr(self.args, "inner_loop_iters", 3))
        enable_structure = bool(getattr(self.args, "enable_structure", True))

        wf = self.workflow_fn
        best = {"passed": False, "score": -1e9, "pred": "", "kind": "base"}

        for t in range(iters):
            # ----- run under tracer so OptoPrimeLocal can update prompt templates
            tracer = RuntimeTracer(trainable_prompt_templates=True, clear_graph_on_enter=True)
            reset_usage()
            with tracer:
                out_msg = wf(ctx, x)
            pred = strip_trace_tags(str(out_msg)).strip()

            # ----- verifier + feedback
            ticket = parse_ticket(x)
            report = verify_output("", ctx, ticket, pred, float(get_total_cost()))
            feedback = self.feedback_fn("", ctx, x, pred, {"verifier": report, "y_true": y_true})

            score = float(report.get("score_with_cost", report.get("score", 0.0)))
            passed = bool(report.get("passed", False))

            if score > best["score"]:
                best = {"passed": passed, "score": score, "pred": pred, "kind": "base", "feedback": str(feedback)}

            if passed:
                break

            # ----- (A) prompt update: use traced templates (moto-like supervised loop shape)
            params = list(tracer.prompt_templates.values())
            node = getattr(out_msg, "node", None)
            if params and node is not None:
                opt = OptoPrimeLocal(params)
                opt.zero_feedback()
                opt.backward(node, str(feedback), visualize=False)
                opt.step(mode="per_param", verbose=False)

            # ----- (B) optional structure update: rewrite workflow code using meta_prompt+feedback
            if enable_structure:
                trace_ir = tracer.to_ir()
                rewrite_prompt = (
                    f"{self.meta_prompt}\n\n"
                    "You are editing the workflow code. Apply minimal changes.\n"
                    f"Context:\n{ctx}\n\nInput:\n{x}\n\nPred:\n{pred}\n\n"
                    f"Verifier:\n{report}\n\nFeedback:\n{feedback}\n"
                )
                edit = self.editor.rewrite_code(
                    code=self.seed_workflow_code,
                    func_name="seed_workflow",
                    instruction=rewrite_prompt,
                )
                if getattr(edit, "ok", False) and getattr(edit, "code", None):
                    self.seed_workflow_code = edit.code
                    wf = self._load(self.seed_workflow_code, "seed_workflow")

        return best

    def outer_loop(self, observations: List[Dict[str, Any]]) -> bool:
        # one meta-LLM call; keep strict IO so it doesn't drift
        prompt = (
            "Update META_PROMPT, SEED_WORKFLOW_CODE, FEEDBACK_WORKFLOW_CODE for CLOVER.\n"
            "Return ONLY JSON with any subset of keys:\n"
            '  {"meta_prompt": "...", "seed_workflow_code": "...", "feedback_workflow_code": "..."}\n\n'
            "Constraints:\n"
            "- do not add imports\n"
            "- keep function names: seed_workflow, feedback_workflow\n\n"
            f"Current META_PROMPT:\n{self.meta_prompt}\n\n"
            f"Current SEED_WORKFLOW_CODE:\n{self.seed_workflow_code}\n\n"
            f"Current FEEDBACK_WORKFLOW_CODE:\n{self.feedback_workflow_code}\n\n"
            "Observations:\n"
            + "\n".join(str(o) for o in observations[:10])
        )

        raw = llm(prompt, call_tag="clover_outer")
        try:
            patch = __import__("json").loads(str(raw))
        except Exception:
            return False

        changed = False
        mp = patch.get("meta_prompt")
        sw = patch.get("seed_workflow_code")
        fw = patch.get("feedback_workflow_code")

        if isinstance(mp, str) and mp.strip() and mp != self.meta_prompt:
            self.meta_prompt, changed = mp, True
        if isinstance(sw, str) and sw.strip() and sw != self.seed_workflow_code:
            self.seed_workflow_code, changed = sw, True
        if isinstance(fw, str) and fw.strip() and fw != self.feedback_workflow_code:
            self.feedback_workflow_code, changed = fw, True

        return changed

    def save_model(self, epoch):
        pass
    def load(self, path):
        pass
    def inference(self, x):
        pass
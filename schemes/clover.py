import hashlib
import inspect
from typing import Any, Dict, List, Optional

from schemes.base import BaseScheme

from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto.optimizers.optoprime_local import OptoPrimeLocal
from myopto.optimizers.structure_editor import StructureEditor
from myopto.utils.usage import reset_usage, get_total_cost

from prompt.clover import META_PROMPT, SEED_WORKFLOW_CODE, FEEDBACK_WORKFLOW_CODE


class CloverScheme(BaseScheme):
    def __init__(self, args: Any):
        super().__init__(args)
        self.args = args

        # Meta-optimizer artifacts (outer loop edits these)
        self.meta_prompt = META_PROMPT
        self.seed_workflow_code = SEED_WORKFLOW_CODE
        self.feedback_workflow_code = FEEDBACK_WORKFLOW_CODE

        self.editor = StructureEditor(verbose=False, forbid_imports=True)
        self.workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")
        self.feedback_fn = self._load(self.feedback_workflow_code, "feedback_workflow")

    def _load(self, code: str, func_name: str):
        return self.editor.load_function(
            code,
            func_name=func_name,
            extra_globals={"llm": llm, "msg": msg},
        )

    def _sha12(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

    def _call_workflow(self, wf_fn, ctx: str, x: str):
        """
        Calls the workflow function in a signature-tolerant way.

        CloverScheme is benchmark-agnostic. If the workflow still carries optional
        params (e.g., schema/meta_prompt), we pass safe placeholders.
        """
        try:
            sig = inspect.signature(wf_fn)
        except (TypeError, ValueError):
            return wf_fn(ctx, x)

        kwargs: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name in ("ctx", "context"):
                kwargs[name] = ctx
            elif name in ("x", "input", "problem"):
                kwargs[name] = x
            elif name in ("meta_prompt", "metaprompt"):
                kwargs[name] = self.meta_prompt
            elif name in ("schema", "output_schema"):
                kwargs[name] = None
            elif name in ("system_prompt", "sys_prompt"):
                kwargs[name] = ""

        missing_required = []
        for name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is inspect._empty and name not in kwargs:
                missing_required.append(name)

        last_exc: Optional[Exception] = None

        if not missing_required:
            try:
                return wf_fn(**kwargs)
            except TypeError as e:
                last_exc = e

        # Fallback positional patterns (common historical signatures)
        candidates = [
            (ctx, x),
            (ctx, x, None),
            (ctx, x, None, self.meta_prompt),
            (ctx, x, self.meta_prompt),
            ("", ctx, x),
            ("", ctx, x, None, self.meta_prompt),
            ("", ctx, x, self.meta_prompt),
        ]
        for args in candidates:
            try:
                return wf_fn(*args)
            except TypeError as e:
                last_exc = e
                continue

        if last_exc is not None:
            raise last_exc
        return wf_fn(ctx, x)

    def _call_feedback(self, ctx: str, x: str, pred: str, trace_ir: Any):
        """
        Calls the feedback workflow in a signature-tolerant way.

        IMPORTANT: feedback sees NO ground truth; it only sees trace IR and pred.
        """
        artifacts = {"trace_ir": trace_ir}

        # Preferred / historical signature
        try:
            return self.feedback_fn("", ctx, x, pred, artifacts)
        except TypeError:
            pass

        # Common alternate
        try:
            return self.feedback_fn(ctx, x, pred, artifacts)
        except TypeError:
            pass

        # Last resort
        return self.feedback_fn(ctx, x, pred)

    async def train_one_batch(self, batch, calculate_score):
        """
        Commit 3: scoring is training/eval-only via calculate_score(answer, pred).
        calculate_score is the ONLY evaluator.
        """
        contexts, problems, answers = batch

        # Reset once per batch so get_total_cost() is meaningful across the batch.
        reset_usage()

        obs: List[Dict[str, Any]] = []
        passed_count = 0

        for ctx, x, answer in zip(contexts, problems, answers):
            inner = self.inner_loop(ctx, x)
            pred = inner.get("pred", "")

            # Training/eval-only scoring: ground truth only touched here.
            score = 0.0
            extra = None
            try:
                res = calculate_score(answer, pred)
                if isinstance(res, tuple) and len(res) >= 1:
                    score = float(res[0])
                    extra = res[1] if len(res) > 1 else None
                else:
                    score = float(res)
            except TypeError:
                # Some benchmarks may use the opposite arg order
                try:
                    res = calculate_score(pred, answer)
                    if isinstance(res, tuple) and len(res) >= 1:
                        score = float(res[0])
                        extra = res[1] if len(res) > 1 else None
                    else:
                        score = float(res)
                except Exception:
                    score = 0.0
                    extra = None
            except Exception:
                score = 0.0
                extra = None

            passed = bool(score == 1 or score == 1.0 or score is True)
            if passed:
                passed_count += 1

            record = dict(inner)
            record.update({"score": score, "passed": passed})
            if extra is not None:
                record["score_extra"] = extra
            obs.append(record)

        # Outer loop left unchanged (per your instructions).
        updated = self.outer_loop(obs)
        if updated:
            self.workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")
            self.feedback_fn = self._load(self.feedback_workflow_code, "feedback_workflow")

        return {"cost_usd": float(get_total_cost()), "passed": passed_count}

    def inner_loop(self, ctx: str, x: str) -> Dict[str, Any]:
        """
        Commit 2: benchmark-agnostic inner loop (no CloverToy imports, no verifier).
        Commit 4: single-path loop: run -> feedback(trace_ir) -> structure -> prompt.
                  self.seed_workflow_code is NOT mutated here.

        Returns:
          {pred, iterations, sample_cost_usd, trajectory}
        """
        iters = int(getattr(self.args, "inner_loop_iters", 3))
        enable_structure = bool(getattr(self.args, "enable_structure", True))

        # Local ephemeral state (no global mutation)
        wf_code = self.seed_workflow_code
        wf_fn = self.workflow_fn

        trajectory: List[Dict[str, Any]] = []

        sample_cost_start = float(get_total_cost())
        last_pred = ""

        for t in range(iters):
            iter_cost_start = float(get_total_cost())

            # (1) Executor run under tracer
            tracer = RuntimeTracer(trainable_prompt_templates=True, clear_graph_on_enter=True)
            with tracer:
                out_msg = self._call_workflow(wf_fn, ctx, x)

            pred = strip_trace_tags(str(out_msg)).strip()
            last_pred = pred

            trace_ir = tracer.to_ir()

            run_cost_mid = float(get_total_cost())
            run_cost_usd = run_cost_mid - iter_cost_start

            # (2) Feedback driven by full trace IR (no verifier, no GT)
            feedback = self._call_feedback(ctx, x, pred, trace_ir)
            feedback_str = strip_trace_tags(str(feedback)).strip()

            # (3) Structure update (local wf_code only)
            structure_updated = False
            if enable_structure:
                rewrite_prompt = (
                    f"{self.meta_prompt}\n\n"
                    "You are editing the workflow code to improve correctness.\n"
                    "Constraints:\n"
                    "- Make minimal changes.\n"
                    "- Do not add imports.\n"
                    "- Keep the function name: seed_workflow.\n\n"
                    f"Context:\n{ctx}\n\n"
                    f"Input:\n{x}\n\n"
                    f"Prediction:\n{pred}\n\n"
                    f"Feedback:\n{feedback_str}\n\n"
                    f"Trace IR:\n{trace_ir}\n"
                )
                edit = self.editor.rewrite_code(
                    code=wf_code,
                    func_name="seed_workflow",
                    instruction=rewrite_prompt,
                )
                if getattr(edit, "ok", False) and getattr(edit, "code", None):
                    if edit.code != wf_code:
                        wf_code = edit.code
                        wf_fn = self._load(wf_code, "seed_workflow")
                        structure_updated = True

            # (4) Prompt update (OptoPrimeLocal)
            params = list(getattr(tracer, "prompt_templates", {}).values())
            node = getattr(out_msg, "node", None)
            if params and node is not None:
                opt = OptoPrimeLocal(params)
                opt.zero_feedback()
                opt.backward(node, feedback_str, visualize=False)
                opt.step(mode="per_param", verbose=False)

            iter_cost_end = float(get_total_cost())
            iter_cost_usd = iter_cost_end - iter_cost_start

            trajectory.append(
                {
                    "iter": t,
                    "pred": pred,
                    "feedback": feedback_str,
                    "cost_usd": iter_cost_usd,
                    "run_cost_usd": run_cost_usd,
                    "wf_code_sha12": self._sha12(wf_code),
                    "wf_code_preview": wf_code[:200],
                    "structure_updated": structure_updated,
                }
            )

        sample_cost_end = float(get_total_cost())
        sample_cost_usd = sample_cost_end - sample_cost_start

        return {
            "pred": last_pred,
            "iterations": len(trajectory),
            "sample_cost_usd": sample_cost_usd,
            "trajectory": trajectory,
        }

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

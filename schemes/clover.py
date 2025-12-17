from schemes.base import BaseScheme


class CloverScheme(BaseScheme):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.seed_workflow_code: str = args.seed_workflow
        self.meta_prompt: str = args.meta_prompt
        self.feedback_workflow_code: str = args.feedback_workflow

        self.editor = StructureEditor(
            llm=get_llm(role="optimizer"),
            max_tokens=getattr(args, "structure_max_tokens", 12000),
            require_call_tag=True,
            forbid_strip_trace_tags=True,
            forbid_imports=True,
            verbose=getattr(args, "verbose", False),
        )
        self.prompt_optimizer = OptoPrimeLocal(max_tokens=getattr(args, "opt_max_tokens", 12000), log=False)

        self.seed_workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")
        self.feedback_workflow_fn = self._load(self.feedback_workflow_code, "feedback_workflow")

    def _call_workflow(self, wf_fn, context: str, x: str, tracer: RuntimeTracer) -> Tuple[str, dict, float]:
        with tracer:
            start_cost = float(get_total_cost())
            out_msg = wf_fn(context, x)
            pred = strip_trace_tags(str(out_msg))
            ir = tracer.to_ir()
            end_cost = float(get_total_cost())
        return pred, ir, end_cost - start_cost

    def _call_feedback(self, feedback_fn, trace_ir: dict, pred: str, tracer: RuntimeTracer) -> str:
        with tracer:
            out_msg = feedback_fn(trace_ir, pred)
            fb = strip_trace_tags(str(out_msg))
        return fb

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

        # Commit 5: outer loop runs once per batch, but NEVER runs on an all-pass batch.
        updated = False
        if not all_passed:
            failed_obs = [o for o in obs if not o.get("passed")]
            updated = self.outer_loop(failed_obs or obs)

        if updated:
            # Reload updated seed workflow (outer loop can only update seed_workflow).
            self.seed_workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")

        # Commit 5: expose all_passed so the epoch driver can early-stop.
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

        for it in range(getattr(self.args, "inner_loop_iters", 3)):
            tracer = RuntimeTracer(trainable_prompt_templates=True, clear_graph_on_enter=True)
            pred, trace_ir, cost = self._call_workflow(wf_fn, context, x, tracer)
            last_pred = pred
            last_cost = float(cost)

            # feedback has access to trace_ir and pred but not ground truth
            tracer_fb = RuntimeTracer(trainable_prompt_templates=False, clear_graph_on_enter=True)
            feedback = self._call_feedback(self.feedback_workflow_fn, trace_ir, pred, tracer_fb)

            # structure update (workflow rewrite)
            wf_code_new = self.editor.rewrite_code(code=wf_code, feedback=feedback, call_tag=f"clover_struct_{it}")
            if isinstance(wf_code_new, str) and wf_code_new.strip():
                wf_code = wf_code_new
                wf_fn = self._load(wf_code, "seed_workflow")

            # prompt update via OptoPrimeLocal over tracer.prompt_templates
            try:
                objective = msg(feedback)
                self.prompt_optimizer.step(tracer, objective=objective, verbose=False)
            except Exception:
                pass

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

        # Commit 7: schema-validated JSON decoding (OpenAI json_schema when supported).
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
# schemes/ecwo.py
import hashlib

from utils.logs import logger
from myopto.trace.runtime import RuntimeTracer, strip_trace_tags, llm, msg
from myopto.optimizers import OptoPrimeLocal
from myopto.optimizers.structure_editor import StructureEditor
from myopto.utils.llm_router import get_llm
from myopto.utils.usage import get_total_cost, reset_usage

from .base import BaseScheme

class ExecutionResult:
    def __init__(self, ok, pred, outcome, output_node, trace_ir, cost_usd, error):
        self.ok = bool(ok)
        self.pred = pred
        self.outcome = outcome
        self.output_node = output_node
        self.trace_ir = trace_ir
        self.cost_usd = float(cost_usd)
        self.error = error


def build_outcome(pred, err, extra=None):
    parts = []
    if err:
        parts.append(f"[WORKFLOW_ERROR]\n{err}")
    if extra:
        parts.append(f"[STRUCTURE_EDIT_ERROR]\n{extra}")
    if pred:
        parts.append(f"[WORKFLOW_OUTPUT]\n{pred}")
    return "\n\n".join(parts) if parts else ""


def execute_workflow(wf_fn, context, x, trainable=True, clear=True):
    tracer = RuntimeTracer(trainable_prompt_templates=trainable, clear_graph_on_enter=clear)
    start_cost = float(get_total_cost())
    pred = ""
    output_node = None
    trace_ir = {}
    err = None
    try:
        with tracer:
            out_msg = wf_fn(context, x)
        pred = strip_trace_tags(str(out_msg))
        if hasattr(out_msg, "node"):
            output_node = out_msg.node
        elif hasattr(tracer, "output_node"):
            output_node = tracer.output_node
        trace_ir = tracer.to_ir()
    except Exception as e:
        err = repr(e)
        trace_ir = {"_error": "workflow_failed", "_exception": err}
    cost_usd = float(get_total_cost()) - start_cost
    if output_node is None and "_error" not in trace_ir:
        err = err or "No output_node available"
        trace_ir["_error"] = "no_output_node"
    outcome = build_outcome(pred, err)
    ok = not err
    return ExecutionResult(ok, pred, outcome, output_node, trace_ir, cost_usd, err), tracer


def run_feedback(feedback_fn, trace_ir, outcome, trace_domain=None):
    if trace_domain:
        tracer = RuntimeTracer(trainable_prompt_templates=False, clear_graph_on_enter=True, domain=trace_domain)
        with tracer:
            out_msg = feedback_fn(trace_ir, outcome)
        return strip_trace_tags(str(out_msg)), tracer.to_ir()
    out_msg = feedback_fn(trace_ir, outcome)
    return strip_trace_tags(str(out_msg)), {}


def propose_structure_edit(editor, wf_code, feedback_text, call_tag):
    new_code = editor.rewrite_code(code=wf_code, feedback=feedback_text, call_tag=call_tag)
    if new_code:
        text = str(new_code)
        if text.strip():
            return text
    return None


def _structure_error_text(label, exc):
    return f"{label}: {repr(exc)}"


def validate_structure_edit(editor, new_code, context, x):
    trace_ir = {}
    try:
        new_fn = editor.load_function(new_code, "seed_workflow", extra_globals={"llm": llm, "msg": msg})
    except Exception as e:
        err = _structure_error_text("load_failed", e)
        trace_ir = {"_error": "structure_load_failed", "_exception": err}
        return False, None, err, trace_ir
    tracer = RuntimeTracer(trainable_prompt_templates=False, clear_graph_on_enter=True)
    try:
        with tracer:
            out_msg = new_fn(context, x)
        trace_ir = tracer.to_ir()
        strip_trace_tags(str(out_msg))
    except Exception as e:
        err = _structure_error_text("smoke_failed", e)
        trace_ir = tracer.to_ir() if trace_ir else {"_error": "structure_smoke_failed", "_exception": err}
        return False, None, err, trace_ir
    return True, new_fn, None, trace_ir


def run_prompt_optimization(make_opt, params, output_node, feedback_text, verbose=False):
    if not params or output_node is None:
        return False
    opt = make_opt(params)
    opt.zero_feedback()
    try:
        opt.backward(output_node, feedback_text, visualize=False)
    except TypeError:
        opt.backward(output_node, feedback_text)
    try:
        opt.step(mode="per_param", verbose=("output" if verbose else False))
    except TypeError:
        opt.step()
    return True


class InnerLoopEngine:
    def __init__(self, editor, feedback_fn, make_opt, verbose=False):
        self.editor = editor
        self.feedback_fn = feedback_fn
        self.make_opt = make_opt
        self.verbose = verbose

    def run(self, wf_code, wf_fn, context, x, iterations=3, structure_budget=1, structure_late_trigger_only=True, sample_tag="sample"):
        wf_current_code = wf_code
        wf_current_fn = wf_fn
        trajectory = []
        pending_structure_error = None
        last_pred = ""
        last_cost = 0.0
        for it in range(iterations):
            res, tracer = execute_workflow(wf_current_fn, context, x, trainable=True, clear=True)
            last_pred = res.pred
            last_cost = res.cost_usd
            failure = not res.ok or res.output_node is None
            if pending_structure_error:
                if not res.trace_ir.get("_error"):
                    res.trace_ir["_error"] = "structure_edit_error"
                res.trace_ir["_structure_error"] = pending_structure_error
                failure = True
            outcome = build_outcome(res.pred, res.error, pending_structure_error)
            feedback, feedback_ir = run_feedback(self.feedback_fn, res.trace_ir, outcome, trace_domain="feedback_workflow")
            pending_structure_error = None
            do_structure = False
            if it < structure_budget:
                do_structure = True
            elif not structure_late_trigger_only:
                do_structure = True
            elif failure:
                do_structure = True
            structure_note = None
            if do_structure:
                new_code = propose_structure_edit(self.editor, wf_current_code, feedback, f"clover_struct_{sample_tag}_{it}")
                if new_code:
                    ok, new_fn, err_text, val_ir = validate_structure_edit(self.editor, new_code, context, x)
                    if ok:
                        wf_current_code = new_code
                        wf_current_fn = new_fn
                    else:
                        pending_structure_error = err_text
                        res.trace_ir["_validation_ir"] = val_ir
                        structure_note = err_text
                        logger.warning(err_text)
                        if self.verbose:
                            raise Exception(err_text)
            params = None
            if hasattr(tracer, "prompt_templates"):
                pt = tracer.prompt_templates
                if pt:
                    if hasattr(pt, "values"):
                        params = list(pt.values())
                    else:
                        params = list(pt)
            run_prompt_optimization(self.make_opt, params, res.output_node, feedback, verbose=self.verbose)
            trajectory.append({
                "iteration": it,
                "pred": res.pred,
                "feedback": feedback,
                "wf_code_snippet": wf_current_code[:4000],
                "wf_code_hash": _sha12(wf_current_code),
                "cost_usd": float(res.cost_usd),
                "trace_error": structure_note,
                "feedback_ir": feedback_ir,
                "trace_ir": res.trace_ir,
            })
        return {
            "pred": last_pred,
            "iterations": len(trajectory),
            "sample_cost_usd": float(last_cost),
            "trajectory": trajectory,
            "wf_code": wf_current_code,
            "wf_fn": wf_current_fn,
        }


def _sha12(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# How to run: python debug/clover_inner_demo.py

DEFAULT_WORKFLOW_CODE = """
def seed_workflow(context, question):
    text = context.strip()
    if text:
        return llm(f"Use the given context to answer the question. Context: {text}\nQuestion: {question}")
    return llm(f"Answer the question directly. Question: {question}")
""".lstrip()


def _feedback_with_answer(answer):
    ans = str(answer).strip()
    def fn(trace_ir, outcome):
        notes = ["Ground truth answer:", ans]
        err = trace_ir.get("_error")
        if err:
            notes.append(f"Workflow error: {err}")
        if outcome:
            notes.append("Previous attempt:")
            notes.append(outcome)
        notes.append("Rewrite the workflow output so it matches the ground truth exactly.")
        return "\n".join(notes)
    return fn


def _feedback_no_answer():
    def fn(trace_ir, outcome):
        hints = []
        err = trace_ir.get("_error")
        if err:
            hints.append(f"Workflow error: {err}")
        if outcome:
            hints.append(outcome)
        hints.append("Return a clearer, checkable solution even without a provided label.")
        return "\n".join(hints)
    return fn


class ECWOScheme(BaseScheme):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.seed_workflow_code = DEFAULT_WORKFLOW_CODE

        self.editor = StructureEditor(
            llm=get_llm(role="optimizer"),
            max_tokens=self._arg("structure_max_tokens", 8000),
            require_call_tag=True,
            forbid_strip_trace_tags=True,
            forbid_imports=True,
            verbose=self._arg("verbose", False),
        )

        self._opt_max_tokens = self._arg("opt_max_tokens", 8000)
        self.seed_workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")

    def _arg(self, name, default=None):
        if hasattr(self.args, name):
            return self.args.__dict__.get(name, default)
        return default

    def _load(self, code, fn_name):
        return self.editor.load_function(code, fn_name, extra_globals={"llm": llm, "msg": msg})

    def _make_prompt_optimizer(self, parameters):
        options = [
            ({"max_tokens": self._opt_max_tokens, "log": False, "llm": get_llm(role="optimizer")}),
            ({"max_tokens": self._opt_max_tokens, "log": False}),
            ({"max_tokens": self._opt_max_tokens}),
            ({})
        ]
        for kw in options:
            try:
                return OptoPrimeLocal(parameters, **kw)
            except TypeError:
                continue
        return OptoPrimeLocal(parameters)

    def _select_feedback(self, answer):
        if self._arg("blind_feedback", False) or answer is None:
            return _feedback_no_answer()
        return _feedback_with_answer(answer)

    def _run_inner(self, context, question, answer):
        engine = InnerLoopEngine(
            editor=self.editor,
            feedback_fn=self._select_feedback(answer),
            make_opt=self._make_prompt_optimizer,
            verbose=self._arg("verbose", False),
        )

        tag = _sha12(f"{context}\n{question}")
        result = engine.run(
            self.seed_workflow_code,
            self.seed_workflow_fn,
            context,
            question,
            iterations=self._arg("inner_loop_iters", 3),
            structure_budget=self._arg("structure_budget", 1),
            structure_late_trigger_only=self._arg("structure_late_trigger_only", True),
            sample_tag=tag,
        )

        self.seed_workflow_code = result.get("wf_code", self.seed_workflow_code)
        self.seed_workflow_fn = result.get("wf_fn", self.seed_workflow_fn)

        return {
            "pred": result.get("pred"),
            "iterations": result.get("iterations"),
            "sample_cost_usd": float(result.get("sample_cost_usd", 0.0)),
            "trajectory": result.get("trajectory", []),
        }

    def train_one_batch(self, batch, calculate_score):
        reset_usage()
        obs = []
        passed_count = 0

        if self.args.batch_mode == "meta":
            contexts, questions, answers = batch
        else:
            questions, answers = batch
            contexts = [""] * len(questions)

        for ctx, q, ans in zip(contexts, questions, answers):
            record = self._run_inner(ctx, q, ans)
            pred = record.get("pred")

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
            record["input"] = q
            obs.append(record)

        batch_size = len(answers)
        return {
            "cost_usd": float(get_total_cost()),
            "passed": passed_count,
            "batch_size": batch_size,
            "all_passed": passed_count == batch_size,
            "observations": obs,
        }

    async def train(self, train_benchmark, train_indices, test_benchmark=None, test_indices=None, test_freq=1):
        self.prep_train()

        data = await train_benchmark.load_data(train_indices)
        keys = [train_benchmark.q_key, train_benchmark.a_key]
        if self.args.batch_mode == "meta":
            keys = [train_benchmark.c_key] + keys

        epochs = max(1, int(self._arg("epochs", 1)))
        batch_size = max(1, int(self._arg("batch_size", 1)))
        total_steps = (len(data) + batch_size - 1) // batch_size

        bench_name = ""
        if hasattr(self.args, "benchmark"):
            bench_name = self.args.benchmark
        logger.info(f"[train] scheme=ecwo benchmark={bench_name} epochs={epochs} batch_size={batch_size} n_train={len(data)}")

        for epoch in range(1, epochs + 1):
            for step, batch in enumerate(self.iter_batches(data, batch_size, keys), start=1):
                metrics = self.train_one_batch(batch, train_benchmark.calculate_score)
                if metrics:
                    logger.info(f"[train] epoch={epoch} step={step/total_steps} metrics={metrics}")

            self.save_model(epoch=epoch)

            if test_benchmark and test_indices:
                if epoch % max(1, int(test_freq)) == 0:
                    logger.info(f"[test] epoch={epoch} n_test={len(test_indices)}")
                    self.prep_test()
                    max_tasks = 50
                    if hasattr(test_benchmark, "max_concurrent_tasks"):
                        max_tasks = test_benchmark.max_concurrent_tasks
                    await test_benchmark.run_baseline(
                        agent=self.inference,
                        specific_indices=list(test_indices),
                        max_concurrent_tasks=max_tasks,
                    )
                    self.prep_train()

        self.save_model()

    def save_model(self, epoch=None):
        self.scheme_file.parent.mkdir(parents=True, exist_ok=True)

        code = (
            "# Auto-generated by ECWOScheme.save_model\n"
            "# Safe to exec/import to restore state.\n\n"
            f"SEED_WORKFLOW_CODE = {repr(self.seed_workflow_code)}\n"
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

        self.seed_workflow_code = ns.get("SEED_WORKFLOW_CODE", self.seed_workflow_code)
        self.seed_workflow_fn = self._load(self.seed_workflow_code, "seed_workflow")
        return True

    async def inference_with_meta(self, context, x):
        reset_usage()
        record = self._run_inner(context, x, None)
        pred = record.get("pred", "")
        cost_usd = float(get_total_cost())
        meta = dict(record)
        meta["cost_usd"] = cost_usd
        return str(pred), cost_usd, meta

    async def inference(self, input_text):
        pred, cost_usd, _meta = await self.inference_with_meta("", input_text)
        return pred, cost_usd

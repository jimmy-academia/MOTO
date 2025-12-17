# schemes/ecwo.py
from utils.log import logger
from myopto.trace.runtime import RuntimeTracer, strip_trace_tags, llm, msg
from myopto.utils.usage import get_total_cost

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
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# How to run: python debug/clover_inner_demo.py

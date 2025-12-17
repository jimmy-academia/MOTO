from schemes.base import BaseScheme

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from myopto.optimizers.optoprime_local import OptoPrimeLocal
from myopto.trace.runtime import RuntimeTracer, llm, msg, strip_trace_tags
from myopto.utils.usage import get_total_cost, reset_usage
from myopto.optimizers.structure_editor import StructureEditor

from benchmarks.clovertoy import CLOVERTOY_SCHEMA, parse_ticket, split_problem_prompt, verify_output

from prompt.clover import META_PROMPT, SEED_WORKFLOW_CODE, FEEDBACK_WORKFLOW_CODE

class CloverScheme(BaseScheme):
    def __init__(self, args: Any):
        super().__init__(args)
        self.meta_prompt = META_PROMPT
        self.seed_workflow_code = SEED_WORKFLOW_CODE
        self.feedback_workflow_code = FEEDBACK_WORKFLOW_CODE
        
    async def train_one_batch(self, batch, calculate_score):
    
        contexts, problems, answers = batch

        tracer = RuntimeTracer(trainable_prompt_templates=True, clear_graph_on_enter=True)

        records = []
        reset_usage()

        with tracer:
            editor = StructureEditor(verbose=False, forbid_imports=True)
            self.workflow_fn = editor.load_function(
                self.seed_workflow_code,
                func_name="seed_workflow",
                extra_globals={"llm": llm, "msg": msg},
            )
            self.feedback_fn = editor.load_function(
                self.feedback_workflow_code,
                func_name="feedback_workflow",
                extra_globals={"llm": llm, "msg": msg},
            )
            for ctx, x, y_true in zip(contexts, problems, answers):
                out_msg, feedback = self.inner_loop(ctx, x)
                records.append(
                    {
                        "output_node": getattr(out_msg, "node", None),
                        "feedback": feedback,
                        "y_true": y_true,
                    }
                )

        params = list(tracer.prompt_templates.values())
        opt = OptoPrimeLocal(params)
        opt.zero_feedback()
        for r in records:
            if r["output_node"] is None:
                continue
            opt.backward(r["output_node"], r["feedback"], visualize=False)

        opt.step(mode="per_param", verbose=False)


    def inner_loop(self, ctx, x):
        for __ in range(args.inner_loop_iters):
            reset_usage()
            out_msg = self.workflow_fn(x)
            pred = strip_trace_tags(str(out_msg)).strip()
            trace_ir = tracer.to_ir()
            cost_usd = float(get_total_cost())

            feedback = self.feedback_fn(ctx, pred, trace_ir)
            if feedback['passed']:
                break

            opt = optimizers...
            opt.zero_feedback()
            opt.backward(...)
            opt.step()
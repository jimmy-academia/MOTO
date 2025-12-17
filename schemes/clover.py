from schemes.base import BaseScheme

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from myopto.optimizers.optoprime_local import OptoPrimeLocal
from myopto.trace.runtime import RuntimeTracer
from myopto.utils.usage import get_total_cost, reset_usage

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

        return {
            "batch": len(records),
            "cost_usd": float(get_total_cost()),
        }


    def inner_loop(self, ctx, x):
        pass
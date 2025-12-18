# schemes/ecwo.py
import asyncio
import inspect
import textwrap
from typing import Any, List, Tuple, Dict

from myopto.trace.bundle import Bundle
from myopto.trace.runtime import RuntimeTracer, llm, msg
from myopto.optimizers.structure_editor import StructureEditor
from myopto.optimizers import OptoPrimeLocal


SEED_WORKFLOW_CODE = """
def seed_workflow(context: str, problem: str) -> str:
    answer = llm(f"Solve the given problem: {problem} given the context {context}")
    return answer
""".lstrip()


class InnerLoopEngine:
    def __init__(self, editor: StructureEditor, verbose: bool = False):
        self.editor = editor  # Use the editor passed from the scheme
        self.verbose = verbose

    async def run(self, wf_code: str, feed_fn: Bundle, context: str, batch: List[Tuple[Any, Any]], iterations: int = 3):
        
        current_code = wf_code
        current_fn = self.editor.load_function(current_code, extra_globals={"llm": llm, "msg": msg})
        trajectory = []
        
        for it in range(iterations):
            async def run_single(x, y):
                with RuntimeTracer(domain="workflow") as wf_tr:
                    pred = await current_fn(context, x)
                
                signal = {"trace": wf_tr.to_ir(), "ground_truth": y}
                with RuntimeTracer(domain="feedback") as fb_tr:
                    fb = await feed_fn(pred, signal)
                
                return {"pred": pred, "fb": fb, "wf_tr": wf_tr, "fb_tr": fb_tr, "y": y}

            results = await asyncio.gather(*[run_single(x, y) for x, y in batch])
            
            # --- 1. Aggregate Batch Feedback & Parameters ---
            combined_fb = "\n".join([f"Sample {i}: {r['fb']}" for i, r in enumerate(results)])
            # Harvest all unique parameters (prompts) from all execution traces
            opt = OptoPrimeLocal(current_fn.parameters())
            
            # --- 2. Stage 1: Prompt Optimization (Local Parameter Update) ---
            if params:
                opt = OptoPrimeLocal(list(params))
                # Pass the last sample's output node; feedback IDs will link to others
                opt.backward(results[-1]["wf_tr"].output_node, combined_fb)
                opt.step(mode="per_param", verbose=self.verbose)

            # --- 3. Stage 2: Structure Optimization (Code Rewriting) ---
            # Identify a failed trace to guide the editor
            fail_res = next((r for r in results if "fail" in str(r["fb"]).lower()), results[0])
            
            res = self.editor.rewrite_function(
                func_or_code=current_code, # Use the persistent source string
                ir=fail_res["wf_tr"].to_ir(),
                feedback=combined_fb
            )
            
            if res.ok:
                current_code = res.code
                current_fn = self.editor.load_function(current_code, extra_globals={"llm": llm, "msg": msg})

            trajectory.append({"it": it, "results": results, "code": current_code})
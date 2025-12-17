from myopto.trace.runtime import Tracer

class InnerLoopEngine:
    def run(self, wf_fn: Bundle, feed_fn: Callable, context: str, batch: List[Tuple[Any, Any]], iterations: int = 3) -> Bundle:
        current_fn = wf_fn
        
        for it in range(iterations):
            # --- 1. THE BATCH GATHERING ---
            batch_feedback = []
            batch_success = True
            
            for x, y in batch:
                with Tracer() as tracer:
                    output_node = wf_fn(context, x)

                fb_text = feedback_fn(output_node, y)
                res, tracer = execute_workflow(wf_fn, context, x)
                fb_text = feedback_fn(res.output_node, y) 
                
                batch_data.append({"res": res, "feedback": fb_text})
                            batch_feedback.append(res)



            # --- THE OPTIMIZATION & COMPILATION ---
            # We extract the code from the function object only when needed
            # and produce a new one.
            new_fn = self.optimize_and_compile(current_fn, batch_feedback)
            
            # Update the pointer
            current_fn = new_fn

        return current_fn
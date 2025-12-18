# schemes/optoprime_local.py
from __future__ import annotations

from typing import Any, Dict, Union

from myopto.trace.nodes import ParameterNode
from myopto.trace.propagators import GraphPropagator
from myopto.optimizers.optoprime import OptoPrime, node_to_function_feedback


class OptoPrimeLocal(OptoPrime):
    """
    OptoPrime variant supporting per-parameter local optimization.
    """

    def _local_summary_for_param(self, param: ParameterNode):
        assert isinstance(self.propagator, GraphPropagator)

        fb_graph = self.propagator.aggregate(param.feedback)
        # Check if graph is valid
        if not getattr(fb_graph, "graph", None) and not isinstance(fb_graph, dict):
             return None 
             
        # Defensive check for empty feedback
        if not fb_graph: 
            return None

        summary = node_to_function_feedback(fb_graph)

        if param.py_name not in summary.roots:
            return None

        # Isolate this parameter as the ONLY variable
        summary.variables = {param.py_name: summary.roots[param.py_name]}
        summary.inputs = {k: v for k, v in summary.roots.items() if k != param.py_name}
        return summary

    def _step_one_param(
        self,
        param: ParameterNode,
        *,
        verbose: Union[bool, str] = False,
        mask=None,
    ) -> Dict[ParameterNode, Any]:
        summary = self._local_summary_for_param(param)
        if summary is None:
            return {}

        system_prompt, user_prompt = self.construct_prompt(summary, mask=mask)
        
        # Apply symbol remapping
        system_prompt = self.replace_symbols(system_prompt, self.prompt_symbols)
        user_prompt = self.replace_symbols(user_prompt, self.prompt_symbols)

        response = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            verbose=verbose,
            max_tokens=self.max_tokens,
        )

        if "TERMINATE" in response:
            return {}

        suggestion = self.extract_llm_suggestion(response)
        if not isinstance(suggestion, dict):
            return {}

        if param.py_name not in suggestion:
            return {}
            
        suggestion_one = {param.py_name: suggestion[param.py_name]}
        update_dict = self.construct_update_dict(suggestion_one)
        
        # Log if needed
        if self.log is not None:
             self.log.append({
                 "mode": "per_param", 
                 "target": param.py_name,
                 "response": response
             })

        return update_dict

    def propose_per_param(
        self,
        *,
        verbose: Union[bool, str] = False,
        mask=None,
    ) -> Dict[ParameterNode, Any]:
        assert isinstance(self.propagator, GraphPropagator)

        updates: Dict[ParameterNode, Any] = {}
        # Iterate over unique parameters
        for p in self.parameters:
            if not getattr(p, "trainable", False):
                continue
            
            # Simple check if feedback exists
            if not p.feedback:
                continue

            # Run optimization for this specific parameter
            local_updates = self._step_one_param(p, verbose=verbose, mask=mask)
            updates.update(local_updates)

        return updates

    def step(
        self,
        *,
        mode: str = "global",
        verbose: Union[bool, str] = False,
        mask=None,
        **kwargs,
    ) -> Dict[ParameterNode, Any]:
        if mode in (None, "global"):
            return super().step(verbose=verbose, mask=mask, **kwargs)

        if mode != "per_param":
            raise ValueError(f"Unknown mode={mode!r}")

        update_dict = self.propose_per_param(verbose=verbose, mask=mask)
        self.update(update_dict)
        return update_dict
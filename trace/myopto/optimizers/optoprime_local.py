# trace/myopto/optimizer/optoprime_local.py
from __future__ import annotations

from typing import Any, Dict, Union

from myopto.trace.nodes import ParameterNode
from myopto.trace.propagators import GraphPropagator
from myopto.optimizers.optoprime import OptoPrime, node_to_function_feedback


class OptoPrimeLocal(OptoPrime):
    """
    OptoPrime variant that supports:
      - mode="global"   : original behavior (one LLM call updates all variables)
      - mode="per_param": one LLM call per trainable ParameterNode, using that param's local feedback graph
    """

    def _local_summary_for_param(self, param: ParameterNode):
        """
        Build a FunctionFeedback summary scoped to a *single* trainable parameter.
        Only `param` goes into #Variables; all other roots become #Inputs.
        """
        assert isinstance(self.propagator, GraphPropagator)

        fb_graph = self.propagator.aggregate(param.feedback)
        if not getattr(fb_graph, "graph", None):
            return None  # no feedback reached this param

        summary = node_to_function_feedback(fb_graph)

        # Make ONLY this param editable.
        if param.py_name not in summary.roots:
            # Defensive: if feedback exists, the param should be a root; but don't crash if not.
            return None

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
        """
        One local optimization step for one ParameterNode.
        Returns an update_dict with at most one entry.
        """
        summary = self._local_summary_for_param(param)
        if summary is None:
            return {}

        system_prompt, user_prompt = self.construct_prompt(summary, mask=mask)

        # Apply any symbol remapping (same as OptoPrime._step).
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

        # HARD filter: only allow updating this one param (even if the model suggests others).
        if param.py_name not in suggestion:
            return {}
        suggestion_one = {param.py_name: suggestion[param.py_name]}

        update_dict = self.construct_update_dict(suggestion_one)

        # Optional logging similar to OptoPrime._step
        if self.log is not None:
            self.log.append(
                {
                    "mode": "per_param",
                    "target": param.py_name,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": response,
                }
            )
            if self.summary_log is not None:
                self.summary_log.append(
                    {
                        "mode": "per_param",
                        "target": param.py_name,
                        "problem_instance": self.problem_instance(summary),
                        "summary": summary,
                    }
                )

        return update_dict

    def propose_per_param(
        self,
        *,
        verbose: Union[bool, str] = False,
        mask=None,
    ) -> Dict[ParameterNode, Any]:
        """
        Collect updates for each trainable parameter with *separate* LLM calls.
        Does NOT mutate parameters; caller should call self.update(update_dict).
        """
        assert isinstance(self.propagator, GraphPropagator)

        updates: Dict[ParameterNode, Any] = {}
        for p in self.parameters:
            if not getattr(p, "trainable", False):
                continue
            # Skip params with no feedback reaching them.
            if not getattr(self.propagator.aggregate(p.feedback), "graph", None):
                continue

            updates.update(self._step_one_param(p, verbose=verbose, mask=mask))

        return updates

    def step(
        self,
        *,
        mode: str = "global",
        verbose: Union[bool, str] = False,
        mask=None,
        **kwargs,
    ) -> Dict[ParameterNode, Any]:
        """
        mode="global"   -> original OptoPrime (one call updates all variables)
        mode="per_param"-> one call per parameter (local graph), then apply all updates
        """
        if mode in (None, "global"):
            # Fall back to original Optimizer.step -> OptoPrime._step behavior. :contentReference[oaicite:2]{index=2}
            return super().step(verbose=verbose, mask=mask, **kwargs)

        if mode != "per_param":
            raise ValueError(f"Unknown mode={mode!r}. Expected 'global' or 'per_param'.")

        update_dict = self.propose_per_param(verbose=verbose, mask=mask)
        self.update(update_dict)
        return update_dict

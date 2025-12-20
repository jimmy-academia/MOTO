from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_15.prompt as prompt_custom
from schemes.AFlow.scripts.async_llm import create_llm_instance


from schemes.AFlow.scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble()

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial solution using custom method
        initial_solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        
        # Use ScEnsemble to refine the solution by selecting the most frequent one
        refined_solution = self.sc_ensemble(solutions=[initial_solution['response']], problem=problem)
        
        return refined_solution['response'], self.llm.get_usage_summary()["total_cost"]

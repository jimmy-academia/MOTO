from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_4.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

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
        # Generate multiple solutions using the custom operator
        solutions = []
        for _ in range(5):  # Generate 5 solutions for ensemble
            response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
            solutions.append(response['response'])

        # Use ScEnsemble to select the most frequent solution
        final_solution = self.sc_ensemble(solutions=solutions, problem=problem)['response']
        
        return final_solution, self.llm.get_usage_summary()["total_cost"]

import random
from pathlib import Path

from llm import get_key, LLMClient

os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LiteLLM"
os.environ["OPENAI_API_KEY"] = get_key()
os.environ["TRACE_LITELLM_MODEL"] = opt_model
os.environ["LITELLM_LOG"] = "INFO"

from opto.trace import bundle
from opto.optimizers import OptoPrime
from .base import BaseScheme

from utils.logs import logger

# --- 1. The Agent (Synchronous Bundle) ---
llm_client = LLMClient(model="gpt-4o-mini") 

def llm_call(prompt: str) -> str:
    return llm_client.answer(prompt)

@bundle(trainable=True)
def solution_workflow(problem: str) -> str:
    """The synchronous logic graph."""
    plan = llm_call(f"Create a step-by-step plan to solve: {problem}")
    solution = llm_call(f"Solve the problem following this plan: {plan}. Problem: {problem}")
    return solution

# --- 2. The Scheme Class ---
class MotoScheme(BaseScheme):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = OptoPrime(solution_workflow.parameters())

    # [THE FIX] The benchmark calls 'await graph(input)'. 
    # We must provide an async function that returns (answer_str, cost_float).
    async def inference(self, input_text: str) -> tuple[str, float]:
        try:
            # 1. Run the synchronous bundle
            # Since solution_workflow returns a MessageNode, we access .data
            prediction_node = solution_workflow(input_text)
            prediction_str = str(prediction_node.data)
            
            # 2. Return format (Answer, Cost)
            # You can calculate actual cost here if your LLMClient tracks it
            return prediction_str, 0.0 
        except Exception as e:
            return f"Error: {str(e)}", 0.0

    async def train(self, train_benchmark, val_benchmark=None):
        logger.info(f"\n=== MOTO Training ({self.args.epochs} epochs) ===")
        
        # Use our safe loading method
        train_data = train_benchmark.load_data(limit=self.args.train_limit, shuffle=True)
        
        for epoch in range(self.args.epochs):
            # Training loop (same as before)...
            # ... [Code omitted for brevity, see previous full implementation] ...
            self.save_model()

    def save_model(self):
        code = solution_workflow.parameters()[0].data
        path = Path(self.args.output_dir) / f"{self.model_name}_optimized.py"
        self.save(path, code)

    def load(self, path: Path):
        path = Path(path)
        if not path.exists():
            return False
        
        logger.info(f"Loading scheme from {path}")
        with path.open('r') as f:
            code = f.read()
        solution_workflow.parameters()[0].data = code
        return True
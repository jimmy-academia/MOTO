from abc import ABC, abstractmethod

class BaseScheme(ABC):
    def __init__(self, args):
        self.args = args
        self.model_name = args.scheme  # e.g. 'moto'
        self.benchmark = args.benchmark
        self.scheme_file = f"{args.scheme}_{args.benchmark}_optimized.py"
        self.result_file = f"{args.scheme}_{args.benchmark}_eval_result.csv"

    @abstractmethod
    async def train(self, train_benchmark, train_indices, val_benchmark=None, val_indices=None):
        """
        Run the optimization/training loop.
        
        Args:
            train_benchmark: Benchmark object for training data
            train_indices: List of indices to use for training
            val_benchmark: (Optional) Benchmark object for validation during training
            val_indices: (Optional) List of indices for validation
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """Load a trained state/code from a file."""
        pass

    @abstractmethod
    async def inference(self, input_text: str) -> tuple[str, float]:
        """
        Run the agent on a single input.
        Returns: (answer_string, cost_float)
        """
        pass
    
    def save(self, path: str, content: str):
        """Helper to save the optimized artifact."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        print(f"Saved optimized scheme to {path}")
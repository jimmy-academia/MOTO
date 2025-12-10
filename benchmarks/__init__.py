import os
from pathlib import Path

# Import your benchmark classes
from .math import MATHBenchmark
from .gsm8k import GSM8KBenchmark
from .drop import DROPBenchmark
from .hotpotqa import HotpotQABenchmark
from .humaneval import HumanEvalBenchmark
from .mbpp import MBPPBenchmark
# Add other imports as needed (e.g., ._bbh, ._gpqa) if you want to support them

DATASET_CONFIG = {
    "math": {
        "filename": "math", # Prefix for file names
        "keys": {"q": "problem", "a": "solution"},
        "extractor": "boxed"
    },
    "gsm8k": {
        "filename": "gsm8k",
        "keys": {"q": "question", "a": "answer"},
        "extractor": "direct" 
    },
    "drop": {
        "filename": "drop",
        "keys": {"q": "context", "a": "completion"},
        "extractor": "direct"
    },
    "hotpotqa": {
        "filename": "hotpotqa",
        "keys": {"q": "question", "a": "answer"},
        "extractor": "direct"
    },
    "humaneval": {
        "filename": "humaneval",
        "keys": {"q": "prompt", "a": "canonical_solution"},
        "extractor": "direct"
    },
    "mbpp": {
        "filename": "mbpp",
        "keys": {"q": "prompt", "a": "code"},
        "extractor": "direct"
    }
}

# Map dataset names to their corresponding classes
BENCHMARK_REGISTRY = {
    "math": MATHBenchmark,
    "gsm8k": GSM8KBenchmark,
    "drop": DROPBenchmark,
    "hotpotqa": HotpotQABenchmark,
    "humaneval": HumanEvalBenchmark,
    "mbpp": MBPPBenchmark,
}

def get_benchmark(dataset_name: str, split: str = "test", data_dir: str = "data", log_dir: str = "logs"):
    """
    Factory function to initialize a benchmark instance.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'math', 'gsm8k').
        split (str): 'test' or 'validate' (default: 'test').
        data_dir (str): Root directory where .jsonl files are stored.
        log_dir (str): Directory where results/logs will be saved.

    Returns:
        BaseBenchmark: An initialized benchmark instance.
    """
    dataset_name = dataset_name.lower().strip()
    
    if dataset_name not in BENCHMARK_REGISTRY:
        available = ", ".join(BENCHMARK_REGISTRY.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")

    # 1. Determine the File Path
    # Convention: {dataset_name}_{split}.jsonl (e.g., "math_test.jsonl")
    filename = f"{dataset_name}_{split}.jsonl"
    file_path = os.path.join(data_dir, filename)

    # 2. Check if file exists
    if not os.path.exists(file_path):
        # Fallback or friendly warning
        print(f"Warning: Data file not found at {file_path}")
        print(f"Please ensure you have {filename} in your {data_dir} folder.")
        url = "https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e"
        print(f"Please manually download the dataset from:")
        print(f"  {url}")

    # 3. Initialize the class
    # The classes signature is: __init__(self, name: str, file_path: str, log_path: str)
    benchmark_cls = BENCHMARK_REGISTRY[dataset_name]
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    benchmark = benchmark_cls(
        name=dataset_name,
        file_path=file_path,
        log_path=log_dir
    )

    benchmark.q_key = DATASET_CONFIG[dataset_name]["keys"]["q"]
    benchmark.a_key = DATASET_CONFIG[dataset_name]["keys"]["a"]
    return benchmark



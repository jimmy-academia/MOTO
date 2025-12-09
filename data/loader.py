# data/loader.py
import re
import sys
import json
from pathlib import Path

# --- Configuration ---

DATA_DIR = Path(__file__).parent / "raw"

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

def extract_boxed_answer(solution: str) -> str:
    if not solution: return ""
    m = re.search(r"\\boxed\{([^}]*)\}", solution)
    if not m: m = re.search(r"\\boxed\(([^)]*)\)", solution)
    if m: return m.group(1).strip().replace("\\,", "")
    candidates = re.findall(r"[-+]?\d+(?:/\d+)?", solution)
    return candidates[-1].strip() if candidates else solution.strip()

def extract_answer(solution: str, method: str) -> str:
    if method == "boxed": return extract_boxed_answer(solution)
    return str(solution).strip()

# --- Helper Functions ---

def loadjl(filename):
    """Load a .jsonl file into a list of dictionaries."""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# --- Main Public API ---

def get_data(dset_name: str, is_train: bool) -> list:
    if dset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset '{dset_name}' not found.")
    
    config = DATASET_CONFIG[dset_name]
    
    # Construct filename (e.g., math_validate.jsonl)
    suffix = "_validate.jsonl" if is_train else "_test.jsonl"
    filename = config["filename"] + suffix
    file_path = DATA_DIR / filename

    # Check existence
    if not file_path.exists():
        url = "https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e"
        print(f"\n[ERROR] Data file missing: {file_path}")
        print(f"Please manually download the dataset from:")
        print(f"  {url}")
        print(f"\nInstructions:")
        print(f"  1. Download the zip file.")
        print(f"  2. Extract it.")
        print(f"  3. Ensure the folder is named 'aflow_data' and contains .jsonl files.")
        print(f"  4. Place it at: {DATA_DIR}")
        sys.exit(1)        

    # Load and Process
    raw_data = loadjl(file_path)
    processed_data = []
    
    q_key = config["keys"]["q"]
    a_key = config["keys"]["a"]
    extract_method = config.get("extractor", "direct")

    for ex in raw_data:
        if q_key not in ex or a_key not in ex: continue
        processed_data.append({
            "problem": ex[q_key],
            "answer": extract_answer(ex[a_key], extract_method) 
        })
        
    return processed_data

if __name__ == '__main__':
    print(get_data('math', False)[3])
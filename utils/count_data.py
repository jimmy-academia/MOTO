import os
import random
# ==============================================================================
# 1. HARDCODED DATASET SIZES
# ==============================================================================
# Run tools/count_data.py and replace this block with the output!
DATASET_SIZES = {
    'drop_test.jsonl': 800,
    'drop_validate.jsonl': 200,
    'gsm8k_test.jsonl': 1055,
    'gsm8k_validate.jsonl': 264,
    'hotpotqa_test.jsonl': 800,
    'hotpotqa_validate.jsonl': 200,
    'humaneval_public_test.jsonl': 159,
    'humaneval_test.jsonl': 131,
    'humaneval_validate.jsonl': 33,
    'math_test.jsonl': 486,
    'math_validate.jsonl': 119,
    'mbpp_public_test.jsonl': 427,
    'mbpp_test.jsonl': 341,
    'mbpp_validate.jsonl': 86,
}

# ==============================================================================
# 2. SAFE INDEX GENERATION
# ==============================================================================
def get_safe_random_indices(dataset_name, split, limit, seed=42):
    """
    Generates random indices strictly within the file's actual size.
    """
    if limit is None:
        return None
        
    filename = f"{dataset_name}_{split}.jsonl"
    
    # 1. Get exact size from hardcoded config
    max_size = DATASET_SIZES.get(filename)
    if max_size is None:
        print(f"⚠️ Unknown size for {filename}. Using default safety cap 100.")
        max_size = 100 
    
    if limit > max_size:
        print(f"Requested limit {limit} > actual size {max_size}. Cap at {max_size}.")
        limit = max_size

    # 2. Generate indices
    random.seed(seed)
    return random.sample(range(max_size), limit)

def main():
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found.")
        return

    print("# Copy this dictionary into main.py")
    print("DATASET_SIZES = {")
    
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jsonl')])
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Efficient line counting
                count = sum(1 for _ in f)
                print(f"    '{filename}': {count},")
        except Exception as e:
            print(f"    # Error reading {filename}: {e}")
            
    print("}")

if __name__ == "__main__":
    main()
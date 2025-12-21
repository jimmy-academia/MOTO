# scripts/migrate_aflow_workspace.py
"""
One-time migration to flatten AFlow workspace structure.
OLD: schemes/AFlow/workspace/DATASET/workflows/round_1/... + template/...
NEW: schemes/AFlow/workspace/DATASET/{all files flat}
"""
import os
import shutil

WORKSPACE_BASE = "schemes/AFlow/workspace"
DATASETS = ["GSM8K", "MATH", "DROP", "HotpotQA", "HumanEval", "MBPP"]

def migrate():
    for dataset in DATASETS:
        dataset_dir = os.path.join(WORKSPACE_BASE, dataset)
        
        print(f"[MIGRATE] {dataset}")
        
        # Move round_1 contents up (delete csv files)
        round1_dir = os.path.join(dataset_dir, "round_1")
        if os.path.exists(round1_dir):
            for f in os.listdir(round1_dir):
                src = os.path.join(round1_dir, f)
                if f.endswith(".csv"):
                    print(f"  [DELETE] round_1/{f}")
                    os.remove(src)
                elif f == "__pycache__":
                    print(f"  [DELETE] round_1/__pycache__/")
                    shutil.rmtree(src)
                else:
                    dst = os.path.join(dataset_dir, f)
                    print(f"  [MOVE] round_1/{f} -> {dataset}/{f}")
                    shutil.move(src, dst)
            os.rmdir(round1_dir)
        
        # Move template contents up
        template_dir = os.path.join(dataset_dir, "template")
        if os.path.exists(template_dir):
            for f in os.listdir(template_dir):
                src = os.path.join(template_dir, f)
                dst = os.path.join(dataset_dir, f)
                if f == "__pycache__":
                    print(f"  [DELETE] template/__pycache__/")
                    shutil.rmtree(src)
                else:
                    print(f"  [MOVE] template/{f} -> {dataset}/{f}")
                    shutil.move(src, dst)
            os.rmdir(template_dir)
        
    print("\nDone! New structure:")
    for dataset in DATASETS:
        dataset_dir = os.path.join(WORKSPACE_BASE, dataset)
        if os.path.exists(dataset_dir):
            files = [f for f in os.listdir(dataset_dir) if not f.startswith('.')]
            print(f"  {dataset}/: {', '.join(sorted(files))}")

if __name__ == "__main__":
    migrate()
import argparse
import asyncio
from benchmark import get_benchmark

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scheme', type=str, default='moto', help='Optimization scheme')
    past = ['math', 'gsm8k', 'drop', 'hotpotqa', 'humaneval', 'mbpp']
    mine = []
    choices = past+mine
    parser.add_argument('-b', '--benchmark', type=str, required=True, help=f'benchmark name {choices}', choices=choices)
    parser.add_argument('-o', '--opt_model', type=str, default='gpt-4o', help='Optimizer LLM')
    parser.add_argument('-e', '--exe_model', type=str, default='gpt-4o-mini', help='Executor LLM')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    return parser

async def run_main():
    parser = get_parser()
    args = parser.parse_args()
    
    print(f"Running {args.scheme} on {args.dataset}...")

    # 1. Get the Benchmark Object (for Test set)
    # This automatically finds 'data/math_test.jsonl' and sets up 'logs/'
    benchmark = get_benchmark(
        dataset_name=args.dataset, 
        split="test", 
        data_dir=args.data_dir
    )

    # 2. Load Data using the Benchmark's internal async loader
    # This ensures you get exactly what the benchmark expects
    test_data = await benchmark.load_data()
    print(f"Loaded {len(test_data)} examples from {benchmark.file_path}")

    # 3. Example: Using the Benchmark's Evaluation Logic
    # You need a dummy 'agent' (executor) to pass to evaluate_problem
    # In your MOTO scheme, this would be your optimized graph/workflow.
    
    async def my_agent_workflow(input_text):
        # Replace this with your actual executor model call
        # e.g. return await executor_llm.generate(input_text)
        return "42", 0.0 # Returns (answer, cost)

    # Run a quick baseline test
    print("\n--- Running Baseline Evaluation ---")
    avg_score, avg_cost, total_cost = await benchmark.run_baseline(
        agent=my_agent_workflow, 
        max_concurrent_tasks=10 # Adjust based on rate limits
    )
    
    print(f"\nFinal Results -> Score: {avg_score}, Total Cost: {total_cost}")

def main():
    asyncio.run(run_main())

if __name__ == '__main__':
    main()
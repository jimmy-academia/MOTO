import os
import asyncio
import argparse

from llm import get_key
from benchmarks import get_benchmark
from schemes import get_scheme
from utils.logs import logger
from utils.count_data import get_safe_random_indices
from utils import good_json_dump

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scheme', type=str, default='moto')
    # moto, optopy, optotext, (aflow)
    
    past = ['math', 'gsm8k', 'drop', 'hotpotqa', 'humaneval', 'mbpp']
    parser.add_argument('-b', '--benchmark', type=str, choices=past, help='Dataset name', default='math')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    # model
    parser.add_argument('-o', '--opt_model', type=str, default='gpt-5-mini', help='Optimizer LLM')
    parser.add_argument('-e', '--exe_model', type=str, default='gpt-4o-mini', help='Executor LLM')
    # Limits
    parser.add_argument('--train_limit', type=int, default=10)
    parser.add_argument('--test_limit', type=int, default=10)
    
    # Control Flags
    parser.add_argument('--train', action='store_true', help='Force training')
    parser.add_argument('--force_eval', action='store_true', help='Force evaluation even if results exist')
    
    # Training Params
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val_interval', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='output')
    
    return parser

async def run_main(args):

    scheme = get_scheme(args.scheme, args)
    
    # Paths
    os.makedirs(args.output_dir, exist_ok=True)
    scheme_file = os.path.join(args.output_dir, scheme.scheme_file)
    eval_result_file = os.path.join(args.output_dir, scheme.result_file)

    train_bench = get_benchmark(args.benchmark, split='validate', data_dir=args.data_dir)
    test_bench = get_benchmark(args.benchmark, split='test', data_dir=args.data_dir)
    # Safe indices generation
    train_indices = get_safe_random_indices(args.benchmark, 'validate', args.train_limit)
    test_indices = get_safe_random_indices(args.benchmark, 'test', args.test_limit, seed=123)
        

    # --------------------------------------------------------------------------
    # PHASE 1: TRAINING
    # --------------------------------------------------------------------------

    if args.train or not os.path.exists(scheme_file):
        logger.info(f"--- 🚀 Starting Training: {args.scheme} on {args.benchmark} ---")
        await scheme.train(train_benchmark=train_bench, train_indices=train_indices, test_benchmark=test_bench, test_indices=test_indices)
    else:
        logger.info(f"--- ✅ Scheme Found: {scheme_file} (Skipping Train) ---")
        scheme.load(scheme_file)

    # --------------------------------------------------------------------------
    # PHASE 2: EVALUATION
    # --------------------------------------------------------------------------
    if args.force_eval or not os.path.exists(eval_result_file):

        logger.info(f"--- 📊 Starting Evaluation: {args.benchmark} (Test Split) ---")
        
        
        
        # Run Benchmark (uses the untouched benchmark.py logic)
        # We use the ASYNC inference wrapper from the scheme
        score, cost, total_cost = await test_bench.run_baseline(
            agent=scheme.inference, 
            max_concurrent_tasks=10, 
            specific_indices=test_indices
        )
        
        with open(eval_result_file, 'w') as f:
            f.write(f"benchmark,score,cost\n{args.benchmark},{score},{cost}")
        
        logger.info(f"Saved evaluation checkpoint to {eval_result_file}")

    else:
        logger.info(f"--- ✅ Evaluation Results Found: {eval_result_file} (Skipping Eval) ---")
    
def main():
    parser = get_parser()
    args = parser.parse_args()
    logger.info(f" --- 🤖 Optimizer: {args.opt_model} | Executor: {args.exe_model} --- ")

    args.train_limit = 20
    args.test_limit = 100
    args.batch_size = 5
    args.epochs = 10
    args.val_interval = 1
    logger.debug("Arguments:\n" + good_json_dump(vars(args)))
    # for trace operation
    os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LiteLLM"
    os.environ["OPENAI_API_KEY"] = get_key()
    os.environ["TRACE_LITELLM_MODEL"] = args.opt_model
    os.environ["LITELLM_LOG"] = "INFO"
    asyncio.run(run_main(args))

if __name__ == '__main__':
    main()
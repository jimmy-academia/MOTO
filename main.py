import os
import asyncio
import argparse

from benchmarks import get_benchmark
from schemes import get_scheme

from utils import get_key
from utils.logs import logger, LogLevel
from utils.count_data import get_safe_random_indices
from utils import good_json_dump
from myopto.utils.llm_router import set_role_models

def set_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scheme', type=str, default='aflow')
    # moto, optopy, optotext, aflow, veto
    
    supervised_sets = ['math', 'gsm8k', 'drop', 'hotpotqa', 'humaneval', 'mbpp']
    meta_sets = ['clovertoy']

    parser.add_argument('-b', '--benchmark', type=str, choices=supervised_sets+meta_sets, help='Dataset name', default='gsm8k')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    # model
    parser.add_argument('-m', '--mopt_model', type=str, default='gpt-5-nano', help='Meta-Optimizer LLM')
    parser.add_argument('-o', '--opt_model', type=str, default='qwen2-0.5b', help='Optimizer LLM')
    parser.add_argument('-e', '--exe_model', type=str, default='qwen2-0.5b', help='Executor LLM')
    # trace/myopto/utils/usage.py MODEL_REGISTRY
    parser.add_argument('--batch_mode', type=str, default='sample', help='mode for making data batches.')
    # parser.add_argument("--use_mlx", action="store_true", help="Use MLX on Apple Silicon")
    # parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    # Limits
    parser.add_argument('--train_limit', type=int, default=10)
    parser.add_argument('--test_limit', type=int, default=5)
    
    # Control Flags
    parser.add_argument('--train', action='store_true', help='Force training')
    parser.add_argument('--force_eval', action='store_true', help='Force evaluation even if results exist')
    
    # Training Params
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--inner_loop_iters', type=int, default=3)
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--quick_test', action='store_true', help='Quick test mode with minimal iterations')
    args = parser.parse_args()
    if args.benchmark in meta_sets:
        args.batch_mode = "meta"
    if args.debug:
        logger.log_level = LogLevel.DEBUG.value[0]
        logger.info("Debug mode enabled")
    if args.quick_test:
        args.train_limit = 1
        args.test_limit = 100
        args.batch_size = 1
        args.epochs = 1
        args.val_interval = 1
    return args

async def run_main(args):

    scheme = get_scheme(args.scheme, args)
    
    train_bench = get_benchmark(args.benchmark, split='validate', data_dir=args.data_dir)
    test_bench = get_benchmark(args.benchmark, split='test', data_dir=args.data_dir)
    # Safe indices generation
    train_indices = get_safe_random_indices(args.benchmark, 'validate', args.train_limit)
    test_indices = get_safe_random_indices(args.benchmark, 'test', args.test_limit, seed=123)
        
    # ------------------------------------------------------------------
    # PHASE 1: TRAINING
    # ------------------------------------------------------------------

    if args.train or not scheme.scheme_file.exists():
        logger.info(f"--- 🚀 Starting Training: {args.scheme} on {args.benchmark} ---")
        await scheme.train(train_benchmark=train_bench, train_indices=train_indices, test_benchmark=test_bench, test_indices=test_indices[:20])
    else:
        logger.info(f"--- ✅ Scheme Found: {scheme.scheme_file} (Skipping Train) ---")
        scheme.load(scheme.scheme_file)

    # ------------------------------------------------------------------
    # PHASE 2: EVALUATION
    # ------------------------------------------------------------------
    if args.force_eval or not os.path.exists(scheme.result_file):

        logger.info(f"--- 📊 Starting Evaluation: {args.benchmark} (Test Split) ---")
        
        # Run Benchmark (uses the untouched benchmark.py logic)
        # We use the ASYNC inference wrapper from the scheme
        scheme.prep_test()
        score, cost, total_cost = await test_bench.run_baseline(
            agent=scheme.inference, 
            max_concurrent_tasks=10, 
            specific_indices=test_indices
        )
        
        with open(scheme.result_file, 'w') as f:
            f.write(f"benchmark,score,cost\n{args.benchmark},{score},{cost}")
        
        logger.info(f"Saved evaluation checkpoint to {scheme.result_file}")

    else:
        logger.info(f"--- ✅ Evaluation Results Found: {scheme.result_file} (Skipping Eval) ---")
    
def main():
    args = set_arguments()


    logger.info(f" --- 🤖 MetaOpt: {args.mopt_model} Optimizer: {args.opt_model} | Executor: {args.exe_model} --- ")
    logger.debug("Arguments:\n" + good_json_dump(vars(args)))
    # for trace operation
    os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "LiteLLM"
    os.environ["OPENAI_API_KEY"] = get_key()
    os.environ["LITELLM_LOG"] = "INFO"
    set_role_models(
        executor=args.exe_model,
        optimizer=args.opt_model,
        metaoptimizer=args.mopt_model,
    )
    asyncio.run(run_main(args))

if __name__ == '__main__':
    main()
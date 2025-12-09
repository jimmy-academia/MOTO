import argparse
from data import get_data

def get_parser():
    """Defines the parser and defaults in one place."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scheme', type=str, default='moto', help='Optimization scheme')
    past = ['math', 'gsm8k', 'drop', 'hotpotqa', 'humaneval', 'mbpp']
    mine = []
    parser.add_argument('-d', '--dataset', type=str, default='', help='Dataset name', choice=past+mine)
    parser.add_argument('-o', '--opt_model', type=str, default='gpt-5.1', help='Optimizer LLM')
    parser.add_argument('-e', '--exe_model', type=str, default='gpt-5-nano', help='Executor LLM')
    return parser
    # defaults = parser.parse_args([])

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Running {args.scheme} on {args.dataset} with models -- \noptimizer: {args.opt_model}, executor: {args.exe_model}")

    train_data = get_data(args.dataset, is_train=True)
    test_data = get_data(args.dataset, is_train=False)

    

if __name__ == '__main__':
    main()
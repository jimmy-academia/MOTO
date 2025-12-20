import sys
import os
import argparse

def main(args_list):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "AFlow")
    sys.path.insert(0, project_root)
    
    original_cwd = os.getcwd()
    
    os.chdir(project_root)
    
    try:
        from run import parse_args, EXPERIMENT_CONFIGS
        from data.download_data import download
        from scripts.optimizer import Optimizer
        from scripts.async_llm import LLMsConfig
        
        sys.argv = ["run.py"] + args_list
        
        args = parse_args()
        config = EXPERIMENT_CONFIGS[args.dataset]

        models_config = LLMsConfig.default()
        opt_llm_config = models_config.get(args.opt_model_name)
        if opt_llm_config is None:
            raise ValueError(
                f"The optimization model '{args.opt_model_name}' was not found in the 'models' section of the configuration file. "
                "Please add it to the configuration file or specify a valid model using the --opt_model_name flag. "
            )

        exec_llm_config = models_config.get(args.exec_model_name)
        if exec_llm_config is None:
            raise ValueError(
                f"The execution model '{args.exec_model_name}' was not found in the 'models' section of the configuration file. "
                "Please add it to the configuration file or specify a valid model using the --exec_model_name flag. "
            )

        download(["datasets"], force_download=args.if_force_download)

        optimizer = Optimizer(
            dataset=config.dataset,
            question_type=config.question_type,
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
            check_convergence=args.check_convergence,
            operators=config.operators,
            optimized_path=args.optimized_path,
            sample=args.sample,
            initial_round=args.initial_round,
            max_rounds=args.max_rounds,
            validation_rounds=args.validation_rounds,
        )

        optimizer.optimize("Graph")
        
    finally:
        os.chdir(original_cwd)

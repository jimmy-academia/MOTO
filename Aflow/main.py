import argparse
import importlib
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Central Entry Point")
    parser.add_argument("--scheme", type=str, required=True, help="Scheme to run (e.g., aflow)")
    args, remaining_args = parser.parse_known_args()

    try:
        module_name = f"schemes.{args.scheme}"
        scheme_module = importlib.import_module(module_name)
        
        if hasattr(scheme_module, "main"):
            scheme_module.main(remaining_args)
        else:
            print(f"Error: Scheme '{args.scheme}' does not have a 'main' function.")
    except ImportError as e:
        import traceback
        traceback.print_exc()
        print(f"Error: Scheme '{args.scheme}' not found in schemes directory. Details: {e}")
    except Exception as e:
        print(f"Error executing scheme '{args.scheme}': {e}")
        raise

if __name__ == "__main__":
    main()
# test.py
import os
import inspect
from llm import get_key

# --- Env / keys ---
os.environ.setdefault("TRACE_DEFAULT_LLM_BACKEND", "LiteLLM")
os.environ.setdefault("TRACE_LITELLM_MODEL", "gpt-5-nano")
os.environ.setdefault("LITELLM_LOG", "DEBUG")  # or "INFO"
os.environ["OPENAI_API_KEY"] = get_key()

# --- Imports ---
from myopto.trace.runtime import RuntimeTracer
import myopto.trace.runtime as runtime_mod

from myopto.utils.usage import configure_usage, reset_usage, get_total_cost

def _print_runtime_debug():
    print("runtime.py loaded from:", runtime_mod.__file__)
    # Check whether your patched to_ir/get_cost_summary is actually present
    try:
        src = inspect.getsource(runtime_mod.RuntimeTracer.to_ir)
        print("to_ir contains 'usage' field?:", ("usage" in src))
    except Exception as e:
        print("Could not inspect RuntimeTracer.to_ir:", e)

    print("RuntimeTracer has get_cost_summary?:", hasattr(runtime_mod.RuntimeTracer, "get_cost_summary"))

def _find_llm_node_dicts(ir: dict):
    nodes = ir.get("nodes", [])
    # Prefer nodes with call_tag == "math" (your explicit tag)
    tagged = [n for n in nodes if n.get("call_tag") == "math"]
    if tagged:
        return tagged
    # Otherwise, any node that already has usage
    with_usage = [n for n in nodes if isinstance(n, dict) and "usage" in n]
    if with_usage:
        return with_usage
    # Fallback: return all nodes (for inspection)
    return nodes

def main():
    _print_runtime_debug()

    configure_usage(True)   # enables ContextVar cost accumulation if provider returns usage
    reset_usage()

    with RuntimeTracer() as rt:
        x = rt.msg("2+2", name="x")
        y = rt.llm(f"Answer only the number: {x.clean}", call_tag="math")
        ir = rt.to_ir()

        print("\nanswer:", y.clean)

        # --- Try to pull per-call usage from IR ---
        llm_like = _find_llm_node_dicts(ir)
        print("\nIR node count:", len(ir.get("nodes", [])))

        # Print the first few node keys so you can see the schema
        for i, n in enumerate(ir.get("nodes", [])[:5]):
            if isinstance(n, dict):
                print(f"node[{i}] keys:", sorted(n.keys()))

        per_call_usage = None

        # 1) New schema: node["usage"]
        for n in llm_like:
            if isinstance(n, dict) and "usage" in n:
                per_call_usage = n["usage"]
                break

        # 2) If IR doesn't carry usage, try runtime nodes directly (if your llm() patch landed)
        if per_call_usage is None:
            try:
                if getattr(rt, "llm_nodes", None):
                    info = getattr(rt.llm_nodes[0], "info", {}) or {}
                    if isinstance(info, dict) and "usage" in info:
                        per_call_usage = info["usage"]
            except Exception:
                pass

        if per_call_usage is not None:
            print("\nper-call usage:", per_call_usage)
        else:
            print("\nper-call usage: <missing>")
            print("Reason: your current runtime.py (loaded above) is not emitting usage into to_ir(),")
            print("or RuntimeTracer.llm() isn't storing usage in node.info yet.")

        # --- Summary ---
        summary = ir.get("summary")
        if summary is not None:
            print("\nsummary (from to_ir):", summary)
        else:
            # If you added get_cost_summary but didn't wire summary into to_ir()
            if hasattr(rt, "get_cost_summary"):
                print("\nsummary (from get_cost_summary):", rt.get_cost_summary())
            else:
                print("\nsummary: <missing>")

        # --- ContextVar total (from Commit 1 tracking) ---
        print("\ncontextvar total cost:", get_total_cost())

if __name__ == "__main__":
    main()

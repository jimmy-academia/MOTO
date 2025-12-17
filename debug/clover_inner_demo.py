# debug/clover_inner_demo.py
# How to run: python debug/clover_inner_demo.py
import sys
from pathlib import Path
import importlib.util
import types

root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "trace"))

spec = importlib.util.spec_from_file_location(
    "clover_inner",
    root_dir / "schemes" / "clover_inner.py",
)
clover_inner = importlib.util.module_from_spec(spec)
log_module = types.ModuleType("utils.log")


class _StubLogger:
    def warning(self, msg):
        print("[warn]", msg)

    def info(self, msg):
        print("[info]", msg)


log_module.logger = _StubLogger()
sys.modules["utils"] = types.ModuleType("utils")
sys.modules["utils.log"] = log_module
spec.loader.exec_module(clover_inner)

InnerLoopEngine = clover_inner.InnerLoopEngine
msg = clover_inner.msg

seed_template = """
def seed_workflow(context, x):
    return msg(f"demo_pred:{context}:{x}")
"""

feedback_template = """
def feedback_workflow(trace_ir, outcome):
    return msg(f"demo_feedback:{outcome}")
"""


class FakeEditor:
    def __init__(self):
        self._used = False

    def rewrite_code(self, code, feedback, call_tag=None):
        if self._used:
            return None
        self._used = True
        return code.replace("demo_pred", "demo_pred_v2")

    def load_function(self, code, fn_name, extra_globals=None):
        ns = {}
        if extra_globals:
            ns.update(extra_globals)
        exec(code, ns, ns)
        return ns[fn_name]


class DummyOpt:
    def __init__(self, params):
        self.params = params

    def zero_feedback(self):
        return None

    def backward(self, output_node, feedback, visualize=False):
        return None

    def step(self, mode=None, verbose=False):
        print("dummy opt step", mode, verbose)
        return None


def make_opt(params):
    return DummyOpt(params)


def main():
    editor = FakeEditor()
    wf_fn = editor.load_function(seed_template, "seed_workflow", extra_globals={"msg": msg})
    feedback_fn = editor.load_function(feedback_template, "feedback_workflow", extra_globals={"msg": msg})
    engine = InnerLoopEngine(editor=editor, feedback_fn=feedback_fn, make_opt=make_opt, verbose=False)
    result = engine.run(seed_template, wf_fn, "ctx", "sample", iterations=2, structure_budget=1, structure_late_trigger_only=True, sample_tag="demo")
    print("trajectory length", len(result.get("trajectory", [])))
    print("final pred", result.get("pred"))


if __name__ == "__main__":
    main()

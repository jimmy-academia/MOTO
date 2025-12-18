# schemes/ecwo.py
import asyncio
import inspect
import textwrap
from typing import Any, List, Tuple, Dict

from myopto.trace.bundle import Bundle
from myopto.trace.runtime import RuntimeTracer, llm, msg
from myopto.optimizers.structure_editor import StructureEditor
from myopto.optimizers import OptoPrimeLocal


SEED_WORKFLOW_CODE = """
def seed_workflow(context: str, problem: str) -> str:
    answer = llm(f"Solve the given problem: {problem} given the context {context}")
    return answer
""".lstrip()


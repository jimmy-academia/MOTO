from myopto.trace.bundle import bundle, ExecutionError
from myopto.trace.modules import Module, model
from myopto.trace.containers import NodeContainer
from myopto.trace.broadcast import apply_op
import myopto.trace.propagators as propagators
import myopto.trace.operators as operators

from myopto.trace.nodes import Node, GRAPH
from myopto.trace.nodes import node


class stop_tracing:
    """A contextmanager to disable tracing."""

    def __enter__(self):
        GRAPH.TRACE = False

    def __exit__(self, type, value, traceback):
        GRAPH.TRACE = True


__all__ = [
    "node",
    "stop_tracing",
    "GRAPH",
    "Node",
    "bundle",
    "ExecutionError",
    "Module",
    "NodeContainer",
    "model",
    "apply_op",
    "propagators",
]

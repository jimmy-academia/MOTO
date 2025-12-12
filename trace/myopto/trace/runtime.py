# trace/myopto/trace/runtime.py
from __future__ import annotations

import contextvars
import hashlib
import inspect
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from myopto.trace.nodes import GRAPH, MessageNode, Node, ParameterNode, node


# -----------------------------
# Provenance tag format (Design-B)
# -----------------------------
# We wrap any Msg text as:
#   ⟦trc:<id>⟧ ...text... ⟦/trc:<id>⟧
#
# This survives most string ops and lets us recover dependencies by regex.

_TRACE_SEGMENT_RE = re.compile(
    r"⟦trc:(?P<id>[0-9a-fA-F-]+)⟧(?P<content>.*?)⟦/trc:(?P=id)⟧",
    re.DOTALL | re.IGNORECASE,
)
_TRACE_TOKEN_RE = re.compile(r"⟦/?trc:[0-9a-fA-F-]+⟧", re.IGNORECASE)
_TRACE_START_ID_RE = re.compile(r"⟦trc:([0-9a-fA-F-]+)⟧", re.IGNORECASE)


def strip_trace_tags(text: str) -> str:
    """Remove all trace tokens, leaving clean human text."""
    return _TRACE_TOKEN_RE.sub("", text)


def extract_trace_ids(text: str) -> List[str]:
    """Extract unique trace_ids in first-appearance order."""
    ids = _TRACE_START_ID_RE.findall(text)
    seen = set()
    out: List[str] = []
    for tid in ids:
        tid = tid.lower()
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


@dataclass(frozen=True)
class _TemplateSlot:
    placeholder: str
    trace_id: str


def _prompt_to_template(prompt_tagged: str) -> Tuple[str, List[_TemplateSlot]]:
    """
    Replace every tagged segment ⟦trc:id⟧...⟦/trc:id⟧ with a stable placeholder token.
    Returns:
      template_str: with placeholders, *no trace tags*
      slots: list mapping placeholder -> trace_id in appearance order
    """
    slots: List[_TemplateSlot] = []

    def repl(m: re.Match) -> str:
        tid = m.group("id").lower()
        ph = f"<<__TRACE_{len(slots)}__>>"
        slots.append(_TemplateSlot(placeholder=ph, trace_id=tid))
        return ph

    template = _TRACE_SEGMENT_RE.sub(repl, prompt_tagged)
    template = strip_trace_tags(template)  # remove any stray tokens
    return template, slots


def _get_callsite() -> Dict[str, Any]:
    """
    Pick the first frame outside myopto/trace/runtime.py.
    This is your default "identity" when no explicit tag is provided.
    """
    for frame in inspect.stack()[2:]:
        fn = frame.filename.replace("\\", "/")
        if "/myopto/trace/runtime.py" in fn or fn.endswith("runtime.py"):
            continue
        code = (frame.code_context[0].strip() if frame.code_context else None)
        return {
            "filename": frame.filename,
            "lineno": frame.lineno,
            "function": frame.function,
            "code": code,
        }
    return {"filename": "<unknown>", "lineno": -1, "function": "<unknown>", "code": None}


def _hash_callsite(callsite: Dict[str, Any]) -> str:
    raw = f"{callsite.get('filename')}:{callsite.get('lineno')}:{callsite.get('function')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


class Msg(str):
    """
    A string that carries provenance.
    Its *string value* includes tags, so plain Python string ops propagate them.
    Use .clean or strip_trace_tags() to get the clean text.
    """

    __slots__ = ("trace_id", "node", "clean")

    def __new__(cls, clean: str, *, trace_id: Optional[str] = None, node: Optional[Node] = None):
        tid = (trace_id or str(uuid.uuid4())).lower()
        tagged = f"⟦trc:{tid}⟧{clean}⟦/trc:{tid}⟧"
        obj = super().__new__(cls, tagged)
        obj.trace_id = tid
        obj.node = node
        obj.clean = clean
        return obj


# Active tracer (so user code can just call runtime.llm(...))
_ACTIVE_TRACER: contextvars.ContextVar[Optional["RuntimeTracer"]] = contextvars.ContextVar(
    "myopto_trace_runtime_tracer", default=None
)


def _default_backend(system_prompt: Optional[str], user_prompt: str) -> str:
    """
    Default backend mirrors the existing operator style that builds messages and reads choices[0].message.content.
    (See trace/myopto/trace/operators.py call_llm). 
    """
    from myopto.utils.llm import LLM

    llm = LLM()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    out = llm(messages=messages)
    if isinstance(out, str):
        return out
    try:
        return out.choices[0].message.content
    except Exception:
        return str(out)


class RuntimeTracer:
    """
    Execution-time tracer for LLM workflows.
    - Each LLM call => MessageNode
    - Dependencies derived from Msg tags found in the prompt
    - Per-callsite prompt templates are ParameterNodes (optional trainable)
    """

    def __init__(
        self,
        *,
        backend: Optional[Callable[[Optional[str], str], str]] = None,
        clear_graph_on_enter: bool = True,
        trainable_prompt_templates: bool = True,
    ):
        self._backend = backend or _default_backend
        self.clear_graph_on_enter = clear_graph_on_enter
        self.trainable_prompt_templates = trainable_prompt_templates

        # Registry for dependency resolution
        self.msg_registry: Dict[str, Msg] = {}

        # Callsite identity -> prompt template ParameterNode
        self.prompt_templates: Dict[str, ParameterNode] = {}

        # LLM nodes in execution order
        self.llm_nodes: List[MessageNode] = []

        self._token = None

    def __enter__(self) -> "RuntimeTracer":
        self._token = _ACTIVE_TRACER.set(self)
        if self.clear_graph_on_enter:
            GRAPH.clear()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _ACTIVE_TRACER.reset(self._token)
            self._token = None

    def msg(self, value: str, *, name: str = "input", info: Optional[Dict[str, Any]] = None) -> Msg:
        """
        Wrap arbitrary text as a traced root Msg (useful for function inputs).
        """
        tid = str(uuid.uuid4()).lower()
        n = node(value, name=name, trainable=False, description="[input] runtime input", info=info or {})
        m = Msg(value, trace_id=tid, node=n)
        self.msg_registry[m.trace_id] = m
        return m

    def llm(
        self,
        prompt: Union[str, Msg],
        *,
        system_prompt: Optional[str] = None,
        call_tag: Optional[str] = None,
        trainable_prompt_template: Optional[bool] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Msg]:
        """
        Traced LLM call.
        Returns Msg when tracing is enabled; returns plain str if GRAPH.TRACE is off.
        """
        # Respect global tracing switch (stop_tracing sets GRAPH.TRACE False).
        if not GRAPH.TRACE:
            return self._backend(system_prompt, strip_trace_tags(str(prompt)))

        callsite = _get_callsite()
        callsite_key = _hash_callsite(callsite)

        # Parse dependencies from tags + build template representation
        prompt_tagged = str(prompt)
        template_str, slots = _prompt_to_template(prompt_tagged)

        # Trainable template node (identity = callsite)
        if trainable_prompt_template is None:
            trainable_prompt_template = self.trainable_prompt_templates

        tmpl_node = self.prompt_templates.get(callsite_key)
        if tmpl_node is None:
            placeholders = [s.placeholder for s in slots]
            constraint = (
                "You are editing a prompt template.\n"
                "Do NOT remove or alter these placeholder tokens (keep EXACT text):\n"
                + "\n".join(placeholders)
            )
            tmpl_node = node(
                template_str,
                name=f"prompt_{callsite_key}",
                trainable=trainable_prompt_template,
                description=f"[prompt_template] {callsite.get('function')}:{callsite.get('lineno')}",
                constraint=constraint,
                info={"callsite": callsite, "placeholders": placeholders},
            )
            self.prompt_templates[callsite_key] = tmpl_node

        # Slot resolution + dependency nodes
        values: Dict[str, str] = {}
        deps: Dict[str, Node] = {}
        for s in slots:
            src = self.msg_registry.get(s.trace_id)
            if src is None:
                values[s.placeholder] = ""
                continue
            values[s.placeholder] = src.clean
            if src.node is not None:
                deps[s.trace_id] = src.node

        # Render prompt LLM sees
        rendered = tmpl_node.data
        for ph, rep in values.items():
            rendered = rendered.replace(ph, rep)
        rendered = strip_trace_tags(rendered)

        # Call backend
        response = self._backend(system_prompt, rendered)

        # Create LLM node in the trace graph
        inputs: Dict[str, Node] = {"prompt_template": tmpl_node}
        for i, dep_node in enumerate(deps.values()):
            inputs[f"dep_{i}"] = dep_node

        # Node naming: stable "llm" plus optional explicit tag
        name = call_tag or "llm"
        desc = f"[llm] {name} @ {callsite.get('function')}:{callsite.get('lineno')}"

        info: Dict[str, Any] = {
            "callsite": callsite,
            "prompt_tagged": prompt_tagged,
            "prompt_template": tmpl_node,
            "prompt_rendered": rendered,
            "system_prompt": system_prompt,
            "trace_deps": list(deps.keys()),
        }
        if extra_info:
            info.update(extra_info)

        llm_node = MessageNode(response, inputs=inputs, description=desc, name="llm", info=info)
        self.llm_nodes.append(llm_node)

        # Wrap output as Msg
        out = Msg(response, trace_id=str(uuid.uuid4()).lower(), node=llm_node)
        self.msg_registry[out.trace_id] = out
        return out

    def to_ir(self, *, max_prompt_chars: int = 500, max_out_chars: int = 200) -> Dict[str, Any]:
        def trunc(s: str, n: int) -> str:
            return s if len(s) <= n else s[:n] + " ...<truncated>"

        nodes = []
        edges = []
        for n in self.llm_nodes:
            cs = n.info.get("callsite") or {}
            tmpl = n.info.get("prompt_template")
            nodes.append({
                "id": n.name,
                "level": getattr(n, "level", None),
                "callsite": {
                    "function": cs.get("function"),
                    "lineno": cs.get("lineno"),
                    "filename": cs.get("filename"),
                },
                "template": {
                    "name": getattr(tmpl, "name", None),
                    "placeholders": (tmpl.info.get("placeholders") if tmpl is not None else None),
                },
                "prompt_rendered": trunc(n.info.get("prompt_rendered", ""), max_prompt_chars),
                "output_preview": trunc(str(n.data), max_out_chars),
                "parents": [p.name for p in getattr(n, "parents", [])],
            })
            for p in getattr(n, "parents", []):
                edges.append({"src": p.name, "dst": n.name})

        return {"nodes": nodes, "edges": edges}



# -----------------------------
# Convenience functions for user code
# -----------------------------
def active_tracer() -> Optional[RuntimeTracer]:
    return _ACTIVE_TRACER.get()


def msg(value: str, *, name: str = "input", info: Optional[Dict[str, Any]] = None) -> Msg:
    tr = active_tracer()
    if tr is None:
        # No tracing context: still return a Msg (so tags propagate), but no node linkage.
        return Msg(value, trace_id=str(uuid.uuid4()).lower(), node=None)
    return tr.msg(value, name=name, info=info)


def llm(
    prompt: Union[str, Msg],
    *,
    system_prompt: Optional[str] = None,
    call_tag: Optional[str] = None,
    trainable_prompt_template: Optional[bool] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Union[str, Msg]:
    tr = active_tracer()
    if tr is None:
        return _default_backend(system_prompt, strip_trace_tags(str(prompt)))
    return tr.llm(
        prompt,
        system_prompt=system_prompt,
        call_tag=call_tag,
        trainable_prompt_template=trainable_prompt_template,
        extra_info=extra_info,
    )

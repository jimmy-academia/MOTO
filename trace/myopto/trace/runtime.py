# trace/myopto/trace/runtime.py
from __future__ import annotations

import contextvars
import hashlib
import inspect
import re
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from myopto.trace.nodes import GRAPH, MessageNode, Node, ParameterNode, node
from myopto.utils.usage import compute_cost_usd, extract_model_name, extract_usage


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


def _extract_text(resp: Any) -> str:
    """Best-effort extraction of assistant text from an OpenAI/LiteLLM style response."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp

    # dict-style responses
    if isinstance(resp, dict):
        try:
            return str(resp["choices"][0]["message"]["content"] or "")
        except Exception:
            pass
        try:
            return str(resp["choices"][0].get("text") or "")
        except Exception:
            pass
        return str(resp)

    # object-style responses (OpenAI python / LiteLLM compatible)
    try:
        choices = getattr(resp, "choices", None)
        if choices:
            ch0 = choices[0]
            msg = getattr(ch0, "message", None) if not isinstance(ch0, dict) else ch0.get("message")
            if msg is not None:
                content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
                if content is not None:
                    return str(content)
            text = getattr(ch0, "text", None) if not isinstance(ch0, dict) else ch0.get("text")
            if text is not None:
                return str(text)
    except Exception:
        pass

    return str(resp)


def _usage_dict_from_raw(
    raw: Any,
    *,
    model_fallback: Optional[str] = None,
    role_fallback: Optional[str] = None,
    backend_fallback: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute per-call usage/cost WITHOUT mutating ContextVar aggregators.

    IMPORTANT:
    - Do NOT call track_response() here. Your AbstractModel.__call__ may already
      do request-level accounting, so calling track_response again would double-count.
    """
    try:
        model = extract_model_name(raw) or model_fallback
        pt, ct = extract_usage(raw)
        usd = compute_cost_usd(pt, ct, model)
        return {
            "model": model,
            "role": role_fallback,
            "backend": backend_fallback,
            "prompt_tokens": int(pt or 0),
            "completion_tokens": int(ct or 0),
            "total_tokens": int(pt or 0) + int(ct or 0),
            "usd": float(usd or 0.0),
        }
    except Exception:
        return {
            "model": model_fallback,
            "role": role_fallback,
            "backend": backend_fallback,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "usd": 0.0,
        }


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


def _normalize_call_tag(call_tag: str) -> str:
    """
    Turn an arbitrary tag into a stable, name-safe identifier.
    We keep some readability + add a hash suffix to avoid collisions.
    """
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", call_tag).strip("_")
    if not safe:
        safe = "tag"
    if safe[0].isdigit():
        safe = f"tag_{safe}"
    h = hashlib.sha1(call_tag.encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{h}"


def _template_key(callsite_key: str, call_tag: Optional[str]) -> str:
    return _normalize_call_tag(call_tag) if call_tag else callsite_key


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


def _default_backend_raw(system_prompt: Optional[str], user_prompt: str, llm: Any = None) -> Any:
    """
    Default backend mirrors the existing operator style that builds messages,
    but returns the *raw* provider response so tracer can read usage/model.
    """
    from myopto.utils.llm import LLM

    llm = llm or LLM(role="executor")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return llm(messages=messages)


def _default_backend(system_prompt: Optional[str], user_prompt: str, llm: Any = None) -> str:
    """Compatibility wrapper: return assistant text (string)."""
    return _extract_text(_default_backend_raw(system_prompt, user_prompt, llm=llm))


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
        executor_llm: Optional[Any] = None,
        backend: Optional[Callable[[Optional[str], str], Any]] = None,
        clear_graph_on_enter: bool = True,
        trainable_prompt_templates: bool = True,
    ):
        # If backend isn't provided, default to using executor_llm (NOT the optimizer model).
        if backend is None:
            if executor_llm is None:
                from myopto.utils.llm import LLM

                executor_llm = LLM(role="executor")
            self._backend = lambda sys_p, user_p: _default_backend_raw(sys_p, user_p, llm=executor_llm)
        else:
            self._backend = backend

        self.executor_llm = executor_llm
        self.clear_graph_on_enter = clear_graph_on_enter
        self.trainable_prompt_templates = trainable_prompt_templates

        # Registry for dependency resolution
        self.msg_registry: Dict[str, Msg] = {}

        # Callsite identity -> prompt template ParameterNode
        self.prompt_templates: Dict[str, ParameterNode] = {}

        # LLM nodes in execution order
        self.llm_nodes: List[MessageNode] = []

        self._token = None

    def parameters(self) -> List[ParameterNode]:
        """Trainable prompt template parameters collected so far."""
        return list(self.prompt_templates.values())

    @property
    def output_node(self) -> Optional[MessageNode]:
        """Heuristic: last LLM node executed in this tracer."""
        return self.llm_nodes[-1] if self.llm_nodes else None

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
        """Wrap arbitrary text as a traced root Msg (useful for function inputs)."""
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
            raw = self._backend(system_prompt, strip_trace_tags(str(prompt)))
            return _extract_text(raw)

        callsite = _get_callsite()
        callsite_key = _hash_callsite(callsite)
        key = _template_key(callsite_key, call_tag)

        # Parse dependencies from tags + build template representation
        prompt_tagged = str(prompt)
        template_str, slots = _prompt_to_template(prompt_tagged)

        # Trainable template node (identity = call_tag if provided, else callsite)
        if trainable_prompt_template is None:
            trainable_prompt_template = self.trainable_prompt_templates

        tmpl_node = self.prompt_templates.get(key)
        if tmpl_node is None:
            placeholders = [s.placeholder for s in slots]
            constraint = (
                "You are editing a prompt template.\n"
                "Do NOT remove or alter these placeholder tokens (keep EXACT text):\n"
                + "\n".join(placeholders)
            )
            tmpl_node = node(
                template_str,
                name=f"prompt_{key}",
                trainable=trainable_prompt_template,
                description=f"[prompt_template] {callsite.get('function')}:{callsite.get('lineno')}"
                + (f" tag={call_tag}" if call_tag else ""),
                constraint=constraint,
                info={"callsite": callsite, "placeholders": placeholders, "call_tag": call_tag, "template_key": key},
            )
            self.prompt_templates[key] = tmpl_node
        else:
            # If placeholder shape changed (rare, but can happen with rewrites), keep things runnable.
            new_placeholders = [s.placeholder for s in slots]
            old_placeholders = (tmpl_node.info.get("placeholders") if isinstance(tmpl_node.info, dict) else None) or []

            if old_placeholders != new_placeholders:
                if len(old_placeholders) == len(new_placeholders) and len(old_placeholders) > 0:
                    # preserve learned wording, just rename tokens by position
                    rewritten = str(tmpl_node.data)
                    for old_ph, new_ph in zip(old_placeholders, new_placeholders):
                        rewritten = rewritten.replace(old_ph, new_ph)
                    tmpl_node._data = rewritten  # internal, but ok for ParameterNode
                else:
                    # fallback: reset to freshly derived template_str (warn because learned text may be lost)
                    warnings.warn(
                        f"[RuntimeTracer] Placeholder shape changed for template_key={key}. "
                        f"Resetting template to current render shape.",
                        RuntimeWarning,
                    )
                    tmpl_node._data = template_str

                tmpl_node.info["placeholders"] = new_placeholders
                tmpl_node._constraint = (
                    "You are editing a prompt template.\n"
                    "Do NOT remove or alter these placeholder tokens (keep EXACT text):\n"
                    + "\n".join(new_placeholders)
                )

            # Allow callers to flip trainability at runtime.
            tmpl_node.trainable = bool(trainable_prompt_template)

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
        rendered = str(tmpl_node.data)
        for ph, rep in values.items():
            rendered = rendered.replace(ph, rep)
        rendered = strip_trace_tags(rendered)

        # Call backend (raw) + measure latency
        t0 = time.perf_counter()
        raw = self._backend(system_prompt, rendered)
        latency_s = time.perf_counter() - t0

        response_text = _extract_text(raw)

        # Usage/cost (per-call), without mutating ContextVar aggregators
        model_fallback = None
        role_fallback = "executor"
        backend_fallback = None

        if self.executor_llm is not None:
            model_fallback = getattr(self.executor_llm, "model_name", None) or getattr(self.executor_llm, "model", None)
            role_fallback = getattr(self.executor_llm, "role", None) or role_fallback
            backend_fallback = self.executor_llm.__class__.__name__

        usage = _usage_dict_from_raw(
            raw,
            model_fallback=model_fallback,
            role_fallback=role_fallback,
            backend_fallback=backend_fallback,
        )

        # Create LLM node in the trace graph
        inputs: Dict[str, Node] = {"prompt_template": tmpl_node}
        for i, dep_node in enumerate(deps.values()):
            inputs[f"dep_{i}"] = dep_node

        # Node naming: if call_tag exists, name becomes llm_<taghash> so identity survives line moves.
        llm_base_name = f"llm_{key}" if call_tag else "llm"
        desc_tag = call_tag or "llm"
        desc = f"[llm] {desc_tag} @ {callsite.get('function')}:{callsite.get('lineno')}"

        info: Dict[str, Any] = {
            "callsite": callsite,
            "prompt_tagged": prompt_tagged,
            "prompt_template": tmpl_node,
            "prompt_rendered": rendered,
            "system_prompt": system_prompt,
            "trace_deps": list(deps.keys()),
            "call_tag": call_tag,
            "template_key": key,
            "usage": usage,
            "latency_s": latency_s,
            "response_type": type(raw).__name__,
        }
        if extra_info:
            info.update(extra_info)

        llm_node = MessageNode(response_text, inputs=inputs, description=desc, name=llm_base_name, info=info)
        self.llm_nodes.append(llm_node)

        # Wrap output as Msg
        out = Msg(response_text, trace_id=str(uuid.uuid4()).lower(), node=llm_node)
        self.msg_registry[out.trace_id] = out
        return out

    def get_cost_summary(self) -> Dict[str, Any]:
        total_usd = 0.0
        total_pt = 0
        total_ct = 0
        by_model: Dict[str, float] = {}
        by_role: Dict[str, float] = {}

        for n in self.llm_nodes:
            info = n.info if isinstance(n.info, dict) else {}
            u = info.get("usage") or {}

            usd = float(u.get("usd") or 0.0)
            pt = int(u.get("prompt_tokens") or 0)
            ct = int(u.get("completion_tokens") or 0)

            model = u.get("model") or "<unknown>"
            role = u.get("role") or "<unknown>"

            total_usd += usd
            total_pt += pt
            total_ct += ct

            by_model[model] = float(by_model.get(model, 0.0)) + usd
            by_role[role] = float(by_role.get(role, 0.0)) + usd

        return {
            "num_calls": len(self.llm_nodes),
            "total_usd": total_usd,
            "prompt_tokens": total_pt,
            "completion_tokens": total_ct,
            "total_tokens": total_pt + total_ct,
            "by_model_usd": by_model,
            "by_role_usd": by_role,
        }

    def to_ir(self, *, max_prompt_chars: int = 500, max_out_chars: int = 200) -> Dict[str, Any]:
        def trunc(s: str, n: int) -> str:
            return s if len(s) <= n else s[:n] + " ...<truncated>"

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        for n in self.llm_nodes:
            info = n.info if isinstance(n.info, dict) else {}
            cs = info.get("callsite") or {}
            tmpl = info.get("prompt_template")
            call_tag = info.get("call_tag")
            template_key = info.get("template_key")

            u = info.get("usage") or {}
            latency_s = info.get("latency_s")

            # Stable schema: always include usage keys
            usage_obj = {
                "model": u.get("model"),
                "role": u.get("role"),
                "backend": u.get("backend"),
                "prompt_tokens": int(u.get("prompt_tokens") or 0),
                "completion_tokens": int(u.get("completion_tokens") or 0),
                "total_tokens": int(u.get("total_tokens") or 0),
                "usd": float(u.get("usd") or 0.0),
                "latency_s": float(latency_s) if latency_s is not None else None,
            }

            nodes.append(
                {
                    "id": n.name,
                    "level": getattr(n, "level", None),
                    "callsite": {
                        "function": cs.get("function"),
                        "lineno": cs.get("lineno"),
                        "filename": cs.get("filename"),
                    },
                    "call_tag": call_tag,
                    "template_key": template_key,
                    "template": {
                        "name": getattr(tmpl, "name", None),
                        "placeholders": (tmpl.info.get("placeholders") if tmpl is not None else None),
                    },
                    "prompt_rendered": trunc(str(info.get("prompt_rendered", "")), max_prompt_chars),
                    "output_preview": trunc(str(n.data), max_out_chars),
                    "usage": usage_obj,
                    "parents": [p.name for p in getattr(n, "parents", [])],
                }
            )

            for p in getattr(n, "parents", []):
                edges.append({"src": p.name, "dst": n.name})

        return {"nodes": nodes, "edges": edges, "summary": self.get_cost_summary()}


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

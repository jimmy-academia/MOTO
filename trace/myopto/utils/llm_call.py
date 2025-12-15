# trace/myopto/utils/llm_call.py
"""
Convenience functions for making LLM calls using Trace backends.

- llm_chat(messages, role=...)
- llm_text(prompt, role=..., system_prompt=...)
- llm_prep(role=...) -> callable(prompt)

And an alias:
    _llm_prep_function = llm_prep
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from myopto.utils.llm_router import get_llm

Message = Dict[str, str]


def _ensure_messages(
    prompt: Union[str, Sequence[Message]],
    *,
    system_prompt: Optional[str] = None,
) -> List[Message]:
    if isinstance(prompt, str):
        msgs: List[Message] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    msgs2 = [dict(m) for m in prompt]
    if system_prompt:
        if not msgs2 or msgs2[0].get("role") != "system":
            msgs2.insert(0, {"role": "system", "content": system_prompt})
    return msgs2


def extract_text(resp: Any) -> str:
    """Best-effort extraction of assistant text from an OpenAI/LiteLLM style response."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp

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


def llm_chat(
    messages: Sequence[Message],
    *,
    role: str = "executor",
    llm: Optional[Any] = None,
    **kwargs,
) -> Any:
    """Make a chat completion call (returns raw provider response)."""
    model = llm or get_llm(role)
    return model(messages=list(messages), **kwargs)


def llm_text(
    prompt: str,
    *,
    role: str = "executor",
    system_prompt: Optional[str] = None,
    llm: Optional[Any] = None,
    strip: bool = True,
    **kwargs,
) -> str:
    """Make a single-turn call and return the assistant text."""
    msgs = _ensure_messages(prompt, system_prompt=system_prompt)
    resp = llm_chat(msgs, role=role, llm=llm, **kwargs)
    out = extract_text(resp)
    return out.strip() if strip else out


def llm_prep(
    *,
    role: str = "executor",
    system_prompt: Optional[str] = None,
    llm: Optional[Any] = None,
    strip: bool = True,
    **default_kwargs,
) -> Callable[[str], str]:
    """
    Pre-build a callable for a given role:

        llm = llm_prep(role="executor")
        out = llm("hello")
    """
    frozen_llm = llm or get_llm(role)

    def _call(prompt: str, *, system_prompt_override: Optional[str] = None, **kwargs) -> str:
        sp = system_prompt_override if system_prompt_override is not None else system_prompt
        merged = dict(default_kwargs)
        merged.update(kwargs)
        return llm_text(
            prompt,
            role=role,
            system_prompt=sp,
            llm=frozen_llm,
            strip=strip,
            **merged,
        )

    return _call


# Alias (matches your requested usage)
_llm_prep_function = llm_prep

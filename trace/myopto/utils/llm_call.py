# trace/myopto/utils/llm_call.py
"""Minimal LLM call helpers - only llm_json is actively used."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Union

from myopto.utils.llm_router import get_llm

Message = Dict[str, str]


# --------------------------------------------------
# Internal Helpers
# --------------------------------------------------
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
            msg_obj = getattr(ch0, "message", None) if not isinstance(ch0, dict) else ch0.get("message")
            if msg_obj is not None:
                content = getattr(msg_obj, "content", None) if not isinstance(msg_obj, dict) else msg_obj.get("content")
                if content is not None:
                    return str(content)
            text = getattr(ch0, "text", None) if not isinstance(ch0, dict) else ch0.get("text")
            if text is not None:
                return str(text)
    except Exception:
        pass

    return str(resp)


def _strip_code_fences(s: str) -> str:
    s2 = (s or "").strip()
    if not s2.startswith("```"):
        return s2
    lines = s2.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_substring(text: str) -> str:
    t = _strip_code_fences(text).strip()

    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        return t

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = t.find(open_ch)
        end = t.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            return t[start : end + 1].strip()

    return t


def _try_parse_json(text: str) -> Any:
    cand = _extract_json_substring(text)
    return json.loads(cand)


# --------------------------------------------------
# Public API
# --------------------------------------------------
def llm_json(
    prompt: str,
    *,
    role: str = "executor",
    system_prompt: Optional[str] = None,
    llm: Optional[Any] = None,
    json_schema: Dict[str, Any],
    call_tag: Optional[str] = None,
    max_retries: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """Call an LLM and return a parsed JSON dict.

    - Preferred (OpenAI-compatible) path uses response_format json_schema.
    - Fallback path: strict parse + one retry with an explicit JSON-only instruction.
    """
    if not isinstance(json_schema, dict) or not json_schema:
        raise ValueError("llm_json requires a non-empty json_schema dict")

    msgs = _ensure_messages(prompt, system_prompt=system_prompt)
    model = llm or get_llm(role)

    def _call_model(*, messages: List[Message], response_format: Optional[Dict[str, Any]] = None) -> Any:
        call_kwargs = dict(kwargs)
        if response_format is not None:
            call_kwargs["response_format"] = response_format
        if call_tag is not None:
            call_kwargs["call_tag"] = call_tag
        try:
            return model(messages=list(messages), **call_kwargs)
        except TypeError as e:
            msg = str(e)
            if "call_tag" in msg and "unexpected keyword" in msg:
                call_kwargs.pop("call_tag", None)
                return model(messages=list(messages), **call_kwargs)
            if "response_format" in msg and "unexpected keyword" in msg:
                call_kwargs.pop("response_format", None)
                return model(messages=list(messages), **call_kwargs)
            raise

    last_err: Optional[Exception] = None

    # Attempt 1: schema-enforced output (when supported)
    try:
        resp = _call_model(
            messages=msgs,
            response_format={"type": "json_schema", "json_schema": json_schema},
        )
        obj = _try_parse_json(extract_text(resp))
        if not isinstance(obj, dict):
            raise ValueError(f"llm_json expected dict, got {type(obj).__name__}")
        return obj
    except Exception as e:
        last_err = e

    # Fallback: one controlled retry without response_format
    schema_hint = json.dumps(json_schema.get("schema", json_schema), ensure_ascii=False)
    for _ in range(max_retries):
        retry_prompt = (
            prompt
            + "\n\n"
            + "Return ONLY valid JSON that conforms to this JSON Schema (no markdown, no prose):\n"
            + schema_hint
        )
        msgs2 = _ensure_messages(retry_prompt, system_prompt=system_prompt)
        try:
            resp2 = _call_model(messages=msgs2, response_format=None)
            obj2 = _try_parse_json(extract_text(resp2))
            if not isinstance(obj2, dict):
                raise ValueError(f"llm_json expected dict, got {type(obj2).__name__}")
            return obj2
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"llm_json failed to produce valid JSON: {last_err}")
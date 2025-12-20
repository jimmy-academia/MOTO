# trace/myopto/utils/llm_call.py
"""
LLM call helpers with structured output support.

This module provides utilities for calling LLMs with structured (JSON) outputs.
With modern OpenAI-compatible APIs supporting native json_schema response_format,
most of the parsing logic here is now only needed as a fallback.

Primary function: llm_json() - Call LLM and get parsed JSON dict.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from myopto.utils.llm_router import get_llm

Message = Dict[str, str]


# --------------------------------------------------
# Response Text Extraction
# --------------------------------------------------
def extract_text(resp: Any) -> str:
    """
    Extract text content from LLM response.
    
    Handles:
    - Plain strings
    - OpenAI-style responses (resp.choices[0].message.content)
    - Dict responses
    - LocalSLM dict responses
    """
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    
    # Dict-style responses
    if isinstance(resp, dict):
        try:
            return str(resp["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            pass
        try:
            return str(resp["choices"][0].get("text") or "")
        except (KeyError, IndexError, TypeError):
            pass
        return str(resp)

    # Object-style responses (OpenAI SDK / LiteLLM)
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


# --------------------------------------------------
# Internal Helpers
# --------------------------------------------------
def _ensure_messages(
    prompt: Union[str, Sequence[Message]],
    *,
    system_prompt: Optional[str] = None,
) -> List[Message]:
    """Convert prompt to messages list."""
    if isinstance(prompt, str):
        msgs: List[Message] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    msgs = [dict(m) for m in prompt]
    if system_prompt and (not msgs or msgs[0].get("role") != "system"):
        msgs.insert(0, {"role": "system", "content": system_prompt})
    return msgs


def _strip_code_fences(s: str) -> str:
    """Remove markdown code fences from string."""
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
    """Extract JSON object/array from text."""
    t = _strip_code_fences(text).strip()

    # Already valid JSON boundaries
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        return t

    # Find outermost JSON
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = t.find(open_ch)
        end = t.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            return t[start:end + 1].strip()

    return t


def _try_parse_json(text: str) -> Any:
    """Attempt to parse JSON from text."""
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
    """
    Call an LLM and return a parsed JSON dict.
    
    Uses native json_schema response_format when available (OpenAI, LiteLLM).
    Falls back to prompt-based JSON extraction with retries.
    
    Args:
        prompt: User prompt text
        role: LLM role for routing (executor, optimizer, metaoptimizer)
        system_prompt: Optional system prompt
        llm: Optional LLM instance (uses get_llm(role) if not provided)
        json_schema: JSON schema dict. Can be:
            - {"name": "...", "schema": {...}, "strict": True}  (OpenAI format)
            - {"type": "object", "properties": {...}}  (raw schema)
        call_tag: Optional tag for tracing
        max_retries: Retry count for fallback parsing
        **kwargs: Additional args passed to LLM
    
    Returns:
        Parsed JSON dict
    
    Raises:
        ValueError: If json_schema is invalid
        RuntimeError: If all attempts fail
    
    Example:
        result = llm_json(
            "Extract the name and age from: John is 30 years old.",
            json_schema={
                "name": "person_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"]
                },
                "strict": True
            }
        )
        # result = {"name": "John", "age": 30}
    """
    if not isinstance(json_schema, dict) or not json_schema:
        raise ValueError("llm_json requires a non-empty json_schema dict")

    msgs = _ensure_messages(prompt, system_prompt=system_prompt)
    model = llm or get_llm(role)

    def _call_model(
        *,
        messages: List[Message],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        call_kwargs = dict(kwargs)
        if response_format is not None:
            call_kwargs["response_format"] = response_format
        if call_tag is not None:
            call_kwargs["call_tag"] = call_tag

        try:
            return model(messages=list(messages), **call_kwargs)
        except TypeError as e:
            msg = str(e)
            # Handle backends that don't support certain kwargs
            if "call_tag" in msg and "unexpected keyword" in msg:
                call_kwargs.pop("call_tag", None)
                return model(messages=list(messages), **call_kwargs)
            if "response_format" in msg and "unexpected keyword" in msg:
                call_kwargs.pop("response_format", None)
                return model(messages=list(messages), **call_kwargs)
            raise

    last_err: Optional[Exception] = None

    # Attempt 1: Native JSON schema (OpenAI/LiteLLM)
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

    # Attempt 2: json_object mode (simpler, wider support)
    try:
        resp = _call_model(
            messages=msgs,
            response_format={"type": "json_object"},
        )
        obj = _try_parse_json(extract_text(resp))
        if not isinstance(obj, dict):
            raise ValueError(f"llm_json expected dict, got {type(obj).__name__}")
        return obj
    except Exception as e:
        last_err = e

    # Fallback: Prompt-based JSON with explicit schema hint
    schema_hint = json.dumps(
        json_schema.get("schema", json_schema),
        ensure_ascii=False,
        indent=2,
    )
    
    for _ in range(max_retries):
        retry_prompt = (
            f"{prompt}\n\n"
            "Return ONLY valid JSON that conforms to this schema (no markdown, no prose):\n"
            f"```json\n{schema_hint}\n```"
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


def llm_text(
    prompt: str,
    *,
    role: str = "executor",
    system_prompt: Optional[str] = None,
    llm: Optional[Any] = None,
    call_tag: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Simple text completion helper.
    
    Args:
        prompt: User prompt text
        role: LLM role for routing
        system_prompt: Optional system prompt
        llm: Optional LLM instance
        call_tag: Optional tag for tracing
        **kwargs: Additional args passed to LLM
    
    Returns:
        Response text string
    
    Example:
        text = llm_text("Summarize this article...", role="executor")
    """
    msgs = _ensure_messages(prompt, system_prompt=system_prompt)
    model = llm or get_llm(role)

    call_kwargs = dict(kwargs)
    if call_tag is not None:
        call_kwargs["call_tag"] = call_tag

    try:
        resp = model(messages=msgs, **call_kwargs)
    except TypeError as e:
        if "call_tag" in str(e) and "unexpected keyword" in str(e):
            call_kwargs.pop("call_tag", None)
            resp = model(messages=msgs, **call_kwargs)
        else:
            raise

    return extract_text(resp)
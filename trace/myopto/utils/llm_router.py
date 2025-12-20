# trace/myopto/utils/llm_router.py
"""
LLM Router - Centralized model routing by role.

Provides a clean API for configuring different models for different roles
(executor, optimizer, metaoptimizer) without environment variables.

Includes structured output (JSON) support:
- OpenAI/LiteLLM: Uses native json_schema response_format
- LocalSLM: Injects JSON instruction into prompt (no response_format)

Usage:
    from myopto.utils.llm_router import set_role_models, get_llm
    
    # Configure models for each role
    set_role_models(
        executor="gpt-4o-mini",
        optimizer="gpt-4o",
        metaoptimizer="gpt-4o",
    )
    
    # Get LLM instance
    llm = get_llm("executor")
    response = llm(messages=[{"role": "user", "content": "Hello"}])
    
    # Use JSON schema (OpenAI only)
    response = llm(
        messages=[...],
        response_format={"type": "json_schema", "json_schema": {...}}
    )
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import threading
import sys

from myopto.utils.llm import LiteLLM, CustomLLM, AutoGenLLM

# Lazy import to avoid circular dependency
LocalSLM = None


def _lazy_import_local_slm():
    global LocalSLM
    if LocalSLM is None:
        try:
            from myopto.utils.llm import LocalSLM as _LocalSLM
            LocalSLM = _LocalSLM
        except ImportError:
            LocalSLM = None
    return LocalSLM


def _norm_role(role: Optional[str]) -> str:
    """Normalize role name to standard form."""
    r = (role or "executor").strip().lower()
    if r in {"meta", "meta_optimizer", "meta-optimizer", "metaoptimizer"}:
        return "metaoptimizer"
    if r in {"opt", "optimizer"}:
        return "optimizer"
    if r in {"exec", "executor"}:
        return "executor"
    return r


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
# JSON Parsing Helpers
# --------------------------------------------------
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(s: str) -> str:
    """Remove markdown code fences from string."""
    return _CODE_FENCE_RE.sub("", (s or "").strip()).strip()


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


def try_parse_json(text: str) -> Any:
    """Attempt to parse JSON from text (with fence stripping)."""
    cand = _extract_json_substring(text)
    return json.loads(cand)


# --------------------------------------------------
# Role Configuration
# --------------------------------------------------
@dataclass(frozen=True)
class RoleLLMConfig:
    """Configuration for a role's LLM."""
    role: str
    backend: str = "LiteLLM"  # LiteLLM | CustomLLM | AutoGen | LocalSLM
    model: Optional[str] = None
    base_url: Optional[str] = None  # For CustomLLM
    api_key: Optional[str] = None   # For CustomLLM
    reset_freq: Optional[int] = None
    cache: bool = True
    # LocalSLM options
    use_mlx: bool = False
    device: Optional[str] = None
    max_new_tokens: int = 150
    torch_dtype: str = "float16"


_ROLE_CFG_LOCK = threading.Lock()
_ROLE_CFG: Dict[str, RoleLLMConfig] = {}


def _warn_pause_or_raise(msg: str) -> None:
    """Warn about config override, pause if interactive."""
    print(msg, file=sys.stderr)
    if sys.stdin is not None and sys.stdin.isatty():
        input("[llm_router] Press Enter to continue (Ctrl+C to abort)...")
        return
    raise RuntimeError(msg)


# --------------------------------------------------
# Public Configuration API
# --------------------------------------------------
def set_role_models(
    *,
    executor: Optional[str] = None,
    optimizer: Optional[str] = None,
    metaoptimizer: Optional[str] = None,
    key: Optional[str] = None,
    backend: Optional[str] = None,
) -> None:
    """
    Simple model configuration by role.
    
    Args:
        executor: Model name for executor role
        optimizer: Model name for optimizer role
        metaoptimizer: Model name for metaoptimizer role
        key: Optional API key
        backend: Backend to use (LiteLLM, CustomLLM, AutoGen, LocalSLM)
    
    Example:
        # Cloud models
        set_role_models(executor="gpt-4o-mini", optimizer="gpt-4o")
        
        # Local SLMs
        set_role_models(
            executor="Qwen/Qwen2-0.5B-Instruct",
            optimizer="Qwen/Qwen2-0.5B-Instruct",
            backend="LocalSLM"
        )
    """
    for role, model in (
        ("executor", executor),
        ("optimizer", optimizer),
        ("metaoptimizer", metaoptimizer),
    ):
        if model is None:
            continue
        r = _norm_role(role)
        with _ROLE_CFG_LOCK:
            prev = _ROLE_CFG.get(r)
            if prev is not None and prev.model is not None and prev.model != model:
                _warn_pause_or_raise(
                    f"[llm_router] WARNING: role={r!r} model override: {prev.model!r} -> {model!r}\n"
                    "This likely means you are re-calling set_role_models with different args.\n"
                )
            _ROLE_CFG[r] = RoleLLMConfig(
                role=r,
                backend=backend if backend is not None else (prev.backend if prev else "LiteLLM"),
                model=model,
                api_key=key if key is not None else (prev.api_key if prev else None),
                base_url=prev.base_url if prev else None,
                reset_freq=prev.reset_freq if prev else None,
                cache=prev.cache if prev else True,
                use_mlx=prev.use_mlx if prev else False,
                device=prev.device if prev else None,
                max_new_tokens=prev.max_new_tokens if prev else 150,
                torch_dtype=prev.torch_dtype if prev else "float16",
            )


def set_role_config(
    role: str,
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    reset_freq: Optional[int] = None,
    cache: Optional[bool] = None,
    # LocalSLM options
    use_mlx: Optional[bool] = None,
    device: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    torch_dtype: Optional[str] = None,
) -> None:
    """
    Fine-grained configuration for a specific role.
    
    Args:
        role: Role name (executor, optimizer, metaoptimizer)
        backend: LiteLLM | CustomLLM | AutoGen | LocalSLM
        model: Model name
        base_url: API endpoint (for CustomLLM)
        api_key: API key (for CustomLLM)
        reset_freq: Model refresh interval in seconds
        cache: Enable instance caching
        use_mlx: Use MLX backend (for LocalSLM on Apple Silicon)
        device: Device for LocalSLM (mps, cuda, cpu)
        max_new_tokens: Max generation length for LocalSLM
        torch_dtype: Data type for LocalSLM (float16, float32, bfloat16)
    
    Example:
        # Configure executor to use local SLM with MLX
        set_role_config(
            "executor",
            backend="LocalSLM",
            model="mlx-community/Qwen2-0.5B-Instruct",
            use_mlx=True,
            max_new_tokens=200,
        )
        
        # Configure optimizer to use custom endpoint
        set_role_config(
            "optimizer",
            backend="CustomLLM",
            model="gpt-4o",
            base_url="http://localhost:4000",
            api_key="sk-xxx",
        )
    """
    r = _norm_role(role)
    with _ROLE_CFG_LOCK:
        prev = _ROLE_CFG.get(r, RoleLLMConfig(role=r))
        new_model = model if model is not None else prev.model
        if prev.model is not None and new_model is not None and prev.model != new_model:
            _warn_pause_or_raise(
                f"[llm_router] WARNING: role={r!r} model override: {prev.model!r} -> {new_model!r}\n"
            )
        _ROLE_CFG[r] = RoleLLMConfig(
            role=r,
            backend=backend if backend is not None else prev.backend,
            model=new_model,
            base_url=base_url if base_url is not None else prev.base_url,
            api_key=api_key if api_key is not None else prev.api_key,
            reset_freq=reset_freq if reset_freq is not None else prev.reset_freq,
            cache=bool(cache if cache is not None else prev.cache),
            use_mlx=bool(use_mlx if use_mlx is not None else prev.use_mlx),
            device=device if device is not None else prev.device,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else prev.max_new_tokens,
            torch_dtype=torch_dtype if torch_dtype is not None else prev.torch_dtype,
        )


def clear_role_config(role: Optional[str] = None) -> None:
    """Clear configuration for a role (or all roles if None)."""
    with _ROLE_CFG_LOCK:
        if role is None:
            _ROLE_CFG.clear()
        else:
            _ROLE_CFG.pop(_norm_role(role), None)


def get_role_config(role: str) -> Optional[RoleLLMConfig]:
    """Get current config for a role."""
    r = _norm_role(role)
    with _ROLE_CFG_LOCK:
        return _ROLE_CFG.get(r)


# --------------------------------------------------
# LLM Router Class
# --------------------------------------------------
class LLMRouter:
    """Router that creates and caches LLM instances by role."""

    def __init__(self, *, enable_cache: bool = True) -> None:
        self.enable_cache = enable_cache
        self._cache: Dict[Tuple, Any] = {}
        self._lock = threading.Lock()

    def _resolve(self, role: str, **overrides: Any) -> RoleLLMConfig:
        """Resolve config for a role, applying any call-time overrides."""
        r = _norm_role(role)
        with _ROLE_CFG_LOCK:
            base = _ROLE_CFG.get(r, RoleLLMConfig(role=r))

        return RoleLLMConfig(
            role=r,
            backend=overrides.get("backend", base.backend),
            model=overrides.get("model", base.model),
            base_url=overrides.get("base_url", base.base_url),
            api_key=overrides.get("api_key", base.api_key),
            reset_freq=overrides.get("reset_freq", base.reset_freq),
            cache=overrides.get("cache", base.cache),
            use_mlx=overrides.get("use_mlx", base.use_mlx),
            device=overrides.get("device", base.device),
            max_new_tokens=overrides.get("max_new_tokens", base.max_new_tokens),
            torch_dtype=overrides.get("torch_dtype", base.torch_dtype),
        )

    def get_llm(self, role: str = "executor", **kwargs: Any) -> Any:
        """
        Get an LLM instance for a role.
        
        Args:
            role: Role name (executor, optimizer, metaoptimizer)
            **kwargs: Override any config options
        
        Returns:
            LLM instance (callable)
        
        Raises:
            ValueError: If no model configured for role
        """
        cfg = self._resolve(role, **kwargs)
        if cfg.model is None:
            raise ValueError(
                f"[llm_router] No model configured for role={cfg.role!r}.\n"
                "Call set_role_models(...) or set_role_config(...) first."
            )

        cache_key = (
            cfg.role, cfg.backend, cfg.model, cfg.base_url,
            cfg.api_key, cfg.reset_freq, cfg.cache,
            cfg.use_mlx, cfg.device, cfg.max_new_tokens, cfg.torch_dtype,
        )

        if self.enable_cache:
            with self._lock:
                inst = self._cache.get(cache_key)
                if inst is not None:
                    return inst

        # Create instance based on backend
        if cfg.backend == "LiteLLM":
            inst = LiteLLM(
                model=cfg.model,
                reset_freq=cfg.reset_freq,
                cache=cfg.cache,
                role=cfg.role,
            )
        elif cfg.backend == "CustomLLM":
            inst = CustomLLM(
                model=cfg.model,
                reset_freq=cfg.reset_freq,
                cache=cfg.cache,
                role=cfg.role,
                base_url=cfg.base_url,
                api_key=cfg.api_key,
            )
        elif cfg.backend == "AutoGen":
            inst = AutoGenLLM(
                model=cfg.model,
                reset_freq=cfg.reset_freq,
                role=cfg.role,
            )
        elif cfg.backend == "LocalSLM":
            _lazy_import_local_slm()
            if LocalSLM is None:
                raise ImportError(
                    "LocalSLM backend requires transformers or mlx-lm.\n"
                    "Install with: pip install torch transformers\n"
                    "Or for MLX: pip install mlx mlx-lm"
                )
            inst = LocalSLM(
                model=cfg.model,
                reset_freq=cfg.reset_freq,
                role=cfg.role,
                use_mlx=cfg.use_mlx,
                device=cfg.device,
                max_new_tokens=cfg.max_new_tokens,
                torch_dtype=cfg.torch_dtype,
            )
        else:
            raise ValueError(f"[llm_router] Unknown backend: {cfg.backend!r}")

        # Set role on instance
        try:
            inst.role = cfg.role
        except Exception:
            pass

        if self.enable_cache:
            with self._lock:
                self._cache[cache_key] = inst

        return inst

    def clear_cache(self) -> None:
        """Clear all cached LLM instances."""
        with self._lock:
            self._cache.clear()


# --------------------------------------------------
# Module-level convenience
# --------------------------------------------------
router = LLMRouter(enable_cache=True)


def get_llm(role: str = "executor", **kwargs: Any) -> Any:
    """
    Get an LLM instance for a role.
    
    This is the main entry point for getting LLM instances.
    
    Args:
        role: Role name (executor, optimizer, metaoptimizer)
        **kwargs: Override any config options
    
    Returns:
        LLM instance (callable)
    
    Example:
        llm = get_llm("executor")
        response = llm(messages=[{"role": "user", "content": "Hello"}])
    """
    return router.get_llm(role=role, **kwargs)


# --------------------------------------------------
# High-Level JSON/Text Helpers
# --------------------------------------------------
Message = Dict[str, str]


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


def _is_slm_backend(role: str) -> bool:
    """Check if the role is configured to use LocalSLM backend."""
    cfg = get_role_config(role)
    if cfg is not None:
        return cfg.backend == "LocalSLM"
    return False


def llm_json(
    prompt: str,
    *,
    role: str = "executor",
    system_prompt: Optional[str] = None,
    llm: Optional[Any] = None,
    json_schema: Dict[str, Any],
    call_tag: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Call an LLM and return a parsed JSON dict.
    
    For OpenAI/LiteLLM: Uses native json_schema response_format.
    For LocalSLM: Injects JSON instruction into prompt (no response_format).
    
    Args:
        prompt: User prompt text
        role: LLM role for routing (executor, optimizer, metaoptimizer)
        system_prompt: Optional system prompt
        llm: Optional LLM instance (uses get_llm(role) if not provided)
        json_schema: JSON schema dict. Can be:
            - {"name": "...", "schema": {...}, "strict": True}  (OpenAI format)
            - {"type": "object", "properties": {...}}  (raw schema)
        call_tag: Optional tag for tracing
        **kwargs: Additional args passed to LLM
    
    Returns:
        Parsed JSON dict
    
    Raises:
        ValueError: If json_schema is invalid
        RuntimeError: If parsing fails
    
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

    model = llm or get_llm(role)
    is_slm = _is_slm_backend(role) if llm is None else False
    
    # Build call kwargs
    call_kwargs = dict(kwargs)
    if call_tag is not None:
        call_kwargs["call_tag"] = call_tag

    if is_slm:
        # For SLMs: inject JSON instruction into prompt, no response_format
        schema_hint = json.dumps(
            json_schema.get("schema", json_schema),
            ensure_ascii=False,
            indent=2,
        )
        augmented_prompt = (
            f"{prompt}\n\n"
            "Respond ONLY with valid JSON matching this schema:\n"
            f"```json\n{schema_hint}\n```"
        )
        msgs = _ensure_messages(augmented_prompt, system_prompt=system_prompt)
        
        try:
            resp = model(messages=msgs, **call_kwargs)
        except TypeError as e:
            if "call_tag" in str(e):
                call_kwargs.pop("call_tag", None)
                resp = model(messages=msgs, **call_kwargs)
            else:
                raise
    else:
        # For OpenAI/LiteLLM: use native json_schema response_format
        msgs = _ensure_messages(prompt, system_prompt=system_prompt)
        call_kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        
        try:
            resp = model(messages=msgs, **call_kwargs)
        except TypeError as e:
            msg = str(e)
            # Handle backends that don't support certain kwargs
            if "call_tag" in msg and "unexpected keyword" in msg:
                call_kwargs.pop("call_tag", None)
                resp = model(messages=msgs, **call_kwargs)
            elif "response_format" in msg and "unexpected keyword" in msg:
                call_kwargs.pop("response_format", None)
                resp = model(messages=msgs, **call_kwargs)
            else:
                raise

    # Parse response
    text = extract_text(resp)
    try:
        obj = try_parse_json(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"llm_json failed to parse JSON: {e}\nResponse: {text[:500]}")
    
    if not isinstance(obj, dict):
        raise ValueError(f"llm_json expected dict, got {type(obj).__name__}")
    
    return obj


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
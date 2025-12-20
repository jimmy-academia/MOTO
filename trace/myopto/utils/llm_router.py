# trace/myopto/utils/llm_router.py
"""
LLM Router - Centralized model routing by role.

Provides a clean API for configuring different models for different roles
(executor, optimizer, metaoptimizer) without environment variables.

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
        executor: Model for executor role
        optimizer: Model for optimizer role
        metaoptimizer: Model for metaoptimizer role
        key: API key (for CustomLLM)
        backend: Backend to use (LiteLLM, CustomLLM, LocalSLM, AutoGen)
    
    Example:
        set_role_models(
            executor="gpt-4o-mini",
            optimizer="gpt-4o",
            metaoptimizer="gpt-4o",
        )
    """
    for role_name, model in [("executor", executor), ("optimizer", optimizer), ("metaoptimizer", metaoptimizer)]:
        if model is not None:
            set_role_config(
                role_name,
                model=model,
                api_key=key,
                backend=backend,
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
    use_mlx: Optional[bool] = None,
    device: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    torch_dtype: Optional[str] = None,
) -> None:
    """
    Fine-grained configuration for a role.
    
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

        # Build cache key
        cache_key = (
            cfg.role,
            cfg.backend,
            cfg.model,
            cfg.base_url,
            cfg.api_key,
            cfg.use_mlx,
            cfg.device,
            cfg.max_new_tokens,
            cfg.torch_dtype,
        )

        if self.enable_cache and cfg.cache:
            with self._lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]

        # Create LLM instance
        llm = self._create_llm(cfg)

        if self.enable_cache and cfg.cache:
            with self._lock:
                self._cache[cache_key] = llm

        return llm

    def _create_llm(self, cfg: RoleLLMConfig) -> Any:
        """Create LLM instance based on config."""
        backend = cfg.backend.lower()

        if backend == "localslm":
            _lazy_import_local_slm()
            if LocalSLM is None:
                raise ImportError(
                    "LocalSLM requires transformers or mlx. "
                    "Install with: pip install torch transformers accelerate"
                )
            return LocalSLM(
                model_name=cfg.model,
                use_mlx=cfg.use_mlx,
                device=cfg.device,
                max_new_tokens=cfg.max_new_tokens,
                torch_dtype=cfg.torch_dtype,
            )

        if backend == "customllm":
            return CustomLLM(
                model=cfg.model,
                base_url=cfg.base_url,
                api_key=cfg.api_key,
            )

        if backend == "autogen":
            return AutoGenLLM(
                model=cfg.model,
                api_key=cfg.api_key,
            )

        # Default: LiteLLM
        return LiteLLM(
            model=cfg.model,
            api_key=cfg.api_key,
            role=cfg.role,
        )

    def clear_cache(self) -> None:
        """Clear the LLM instance cache."""
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
# High-Level Text Helper
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
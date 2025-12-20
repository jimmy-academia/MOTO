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
    
    # Or use local SLMs
    set_role_config("executor", backend="LocalSLM", model="Qwen/Qwen2-0.5B-Instruct")
    
    # Get LLM instance
    llm = get_llm("executor")
    response = llm(messages=[{"role": "user", "content": "Hello"}])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
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
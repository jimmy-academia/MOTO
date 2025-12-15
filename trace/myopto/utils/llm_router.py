# trace/myopto/utils/llm_router.py
"""
Role-aware LLM factory / router.

This is intentionally small and provider-agnostic:
- You can configure 1, 2, or 3 different models for (metaoptimizer, optimizer, executor).
- You can optionally configure different backends per role.
- It returns instances of Trace's LLM backends (LiteLLM / CustomLLM / AutoGenLLM).

Env vars (optional):
    TRACE_DEFAULT_LLM_BACKEND=LiteLLM|CustomLLM|AutoGen
    TRACE_LLM_BACKEND_EXECUTOR / OPTIMIZER / METAOPTIMIZER

Model per role (generic override):
    TRACE_LLM_MODEL_EXECUTOR / OPTIMIZER / METAOPTIMIZER

Backend-specific:
    TRACE_LITELLM_MODEL_EXECUTOR / OPTIMIZER / METAOPTIMIZER
    TRACE_CUSTOMLLM_MODEL_EXECUTOR / OPTIMIZER / METAOPTIMIZER
    TRACE_AUTOGEN_MODEL_EXECUTOR / OPTIMIZER / METAOPTIMIZER

CustomLLM endpoint vars:
    TRACE_CUSTOMLLM_URL_EXECUTOR / OPTIMIZER / METAOPTIMIZER
    TRACE_CUSTOMLLM_API_KEY_EXECUTOR / OPTIMIZER / METAOPTIMIZER
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os
import threading

from myopto.utils.llm import LiteLLM, CustomLLM, AutoGenLLM


def _norm_role(role: Optional[str]) -> str:
    if role is None:
        return "executor"
    r = str(role).strip().lower()
    if r in {"meta", "meta_optimizer", "meta-optimizer"}:
        return "metaoptimizer"
    if r in {"opt", "optimizer"}:
        return "optimizer"
    if r in {"exec", "executor"}:
        return "executor"
    return r


def _env_role_suffix(role: str) -> str:
    return _norm_role(role).upper()


def _norm_backend(backend: Optional[str]) -> str:
    b = (backend or os.getenv("TRACE_DEFAULT_LLM_BACKEND", "LiteLLM")).strip()
    if b.lower() == "autogen":
        return "AutoGen"
    if b.lower() == "litellm":
        return "LiteLLM"
    if b.lower() == "customllm":
        return "CustomLLM"
    return b


@dataclass(frozen=True)
class RoleLLMConfig:
    role: str
    backend: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    reset_freq: Optional[int] = None
    cache: bool = True


_ROLE_CFG_LOCK = threading.Lock()
_ROLE_CFG: Dict[str, RoleLLMConfig] = {}


def set_role_config(
    role: str,
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    reset_freq: Optional[int] = None,
    cache: Optional[bool] = None,
) -> None:
    """Programmatic override (takes priority over env)."""
    r = _norm_role(role)
    with _ROLE_CFG_LOCK:
        prev = _ROLE_CFG.get(r)
        _ROLE_CFG[r] = RoleLLMConfig(
            role=r,
            backend=_norm_backend(backend or (prev.backend if prev else None)),
            model=model if model is not None else (prev.model if prev else None),
            base_url=base_url if base_url is not None else (prev.base_url if prev else None),
            api_key=api_key if api_key is not None else (prev.api_key if prev else None),
            reset_freq=reset_freq if reset_freq is not None else (prev.reset_freq if prev else None),
            cache=bool(cache if cache is not None else (prev.cache if prev else True)),
        )


def set_role_models(
    *,
    metaoptimizer: Optional[str] = None,
    optimizer: Optional[str] = None,
    executor: Optional[str] = None,
) -> None:
    if metaoptimizer is not None:
        set_role_config("metaoptimizer", model=metaoptimizer)
    if optimizer is not None:
        set_role_config("optimizer", model=optimizer)
    if executor is not None:
        set_role_config("executor", model=executor)


def clear_role_config(role: Optional[str] = None) -> None:
    with _ROLE_CFG_LOCK:
        if role is None:
            _ROLE_CFG.clear()
        else:
            _ROLE_CFG.pop(_norm_role(role), None)


def _resolve_model(role: str, backend: str, override: Optional[str]) -> Optional[str]:
    """
    Resolve model deterministically so caching is safe:
    programmatic override -> TRACE_LLM_MODEL_<ROLE> -> backend-specific env -> backend default env.
    """
    if override:
        return override

    suffix = _env_role_suffix(role)

    # Generic override (backend-agnostic)
    m = os.getenv(f"TRACE_LLM_MODEL_{suffix}")
    if m:
        return m.strip()

    if backend == "LiteLLM":
        return (
            os.getenv(f"TRACE_LITELLM_MODEL_{suffix}")
            or os.getenv("TRACE_LITELLM_MODEL")
            or os.getenv("DEFAULT_LITELLM_MODEL", "gpt-4o")
        )
    if backend == "CustomLLM":
        return os.getenv(f"TRACE_CUSTOMLLM_MODEL_{suffix}") or os.getenv("TRACE_CUSTOMLLM_MODEL", "gpt-4o")
    if backend == "AutoGen":
        return os.getenv(f"TRACE_AUTOGEN_MODEL_{suffix}") or os.getenv("TRACE_AUTOGEN_MODEL")
    return None


def _resolve_backend(role: str, override: Optional[str]) -> str:
    if override:
        return _norm_backend(override)
    suffix = _env_role_suffix(role)
    env_b = os.getenv(f"TRACE_LLM_BACKEND_{suffix}")
    if env_b:
        return _norm_backend(env_b)
    return _norm_backend(None)


def _resolve_customllm_endpoint(
    role: str, base_url: Optional[str], api_key: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    suffix = _env_role_suffix(role)
    url = base_url or os.getenv(f"TRACE_CUSTOMLLM_URL_{suffix}") or os.getenv("TRACE_CUSTOMLLM_URL")
    key = api_key or os.getenv(f"TRACE_CUSTOMLLM_API_KEY_{suffix}") or os.getenv("TRACE_CUSTOMLLM_API_KEY")
    return url, key


class LLMRouter:
    def __init__(self, *, enable_cache: bool = True) -> None:
        self.enable_cache = enable_cache
        self._cache: Dict[Tuple, Any] = {}
        self._lock = threading.Lock()

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def resolve_config(
        self,
        role: str,
        *,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reset_freq: Optional[int] = None,
        cache: Optional[bool] = None,
    ) -> RoleLLMConfig:
        r = _norm_role(role)

        with _ROLE_CFG_LOCK:
            prog = _ROLE_CFG.get(r)

        resolved_backend = _resolve_backend(r, backend or (prog.backend if prog else None))
        resolved_model = _resolve_model(r, resolved_backend, model or (prog.model if prog else None))

        resolved_base_url = base_url if base_url is not None else (prog.base_url if prog else None)
        resolved_api_key = api_key if api_key is not None else (prog.api_key if prog else None)

        if resolved_backend == "CustomLLM":
            resolved_base_url, resolved_api_key = _resolve_customllm_endpoint(
                r, resolved_base_url, resolved_api_key
            )

        resolved_reset_freq = reset_freq if reset_freq is not None else (prog.reset_freq if prog else None)
        resolved_cache = bool(cache if cache is not None else (prog.cache if prog else True))

        return RoleLLMConfig(
            role=r,
            backend=resolved_backend,
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            reset_freq=resolved_reset_freq,
            cache=resolved_cache,
        )

    def get_llm(
        self,
        role: str = "executor",
        *,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reset_freq: Optional[int] = None,
        cache: Optional[bool] = None,
    ) -> Any:
        cfg = self.resolve_config(
            role,
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            reset_freq=reset_freq,
            cache=cache,
        )

        cache_key = (
            cfg.role,
            cfg.backend,
            cfg.model,
            cfg.base_url,
            cfg.api_key,
            cfg.reset_freq,
            cfg.cache,
        )

        if self.enable_cache:
            with self._lock:
                inst = self._cache.get(cache_key)
                if inst is not None:
                    return inst

        if cfg.backend == "LiteLLM":
            inst = LiteLLM(model=cfg.model, reset_freq=cfg.reset_freq, cache=cfg.cache, role=cfg.role)
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
            inst = AutoGenLLM(model=cfg.model, reset_freq=cfg.reset_freq, role=cfg.role)
        else:
            raise ValueError(f"Unknown LLM backend: {cfg.backend}")

        # Ensure role exists even if backend doesn't store it.
        try:
            inst.role = cfg.role
        except Exception:
            pass

        if self.enable_cache:
            with self._lock:
                self._cache[cache_key] = inst

        return inst


router = LLMRouter(enable_cache=True)


def get_llm(role: str = "executor", **kwargs) -> Any:
    return router.get_llm(role=role, **kwargs)

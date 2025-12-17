# trace/myopto/utils/llm_router.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import threading
import sys

from myopto.utils.llm import LiteLLM, CustomLLM, AutoGenLLM


def _norm_role(role: Optional[str]) -> str:
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
    role: str
    backend: str = "LiteLLM"         # "LiteLLM" | "CustomLLM" | "AutoGen"
    model: Optional[str] = None
    base_url: Optional[str] = None   # only for CustomLLM
    api_key: Optional[str] = None    # only for CustomLLM
    reset_freq: Optional[int] = None
    cache: bool = True


_ROLE_CFG_LOCK = threading.Lock()
_ROLE_CFG: Dict[str, RoleLLMConfig] = {}


def _warn_pause_or_raise(msg: str) -> None:
    print(msg, file=sys.stderr)
    if sys.stdin is not None and sys.stdin.isatty():
        input("[llm_router] Press Enter to continue (Ctrl+C to abort)...")
        return
    raise RuntimeError(msg)


def set_role_models(*, executor: Optional[str] = None, optimizer: Optional[str] = None, metaoptimizer: Optional[str] = None, key: Optional[str] = None) -> None:
    """
    Programmatic model selection. No environment variables.

    - executor/optimizer/metaoptimizer: model names (provider-qualified if needed by LiteLLM)
    - key: optional API key; stored and passed to backends that accept it (primarily CustomLLM).
      (If LiteLLM in your repo uses env vars only, this key will simply be ignored safely.)
    """
    for role, model in (("executor", executor), ("optimizer", optimizer), ("metaoptimizer", metaoptimizer)):
        if model is None:
            continue
        r = _norm_role(role)
        with _ROLE_CFG_LOCK:
            prev = _ROLE_CFG.get(r)
            if prev is not None and prev.model is not None and prev.model != model:
                _warn_pause_or_raise(
                    f"[llm_router] WARNING: role={r!r} model override: {prev.model!r} -> {model!r}\n"
                    "This likely means you are re-calling set_role_models with different CLI args.\n"
                )
            _ROLE_CFG[r] = RoleLLMConfig(role=r, backend=(prev.backend if prev else "LiteLLM"), model=model, api_key=(key if key is not None else (prev.api_key if prev else None)), base_url=(prev.base_url if prev else None), reset_freq=(prev.reset_freq if prev else None), cache=(prev.cache if prev else True))


def set_role_config(role: str, *, backend: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None, reset_freq: Optional[int] = None, cache: Optional[bool] = None) -> None:
    """
    Optional fine-grained config (still no env vars).
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
        )


def clear_role_config(role: Optional[str] = None) -> None:
    with _ROLE_CFG_LOCK:
        if role is None:
            _ROLE_CFG.clear()
        else:
            _ROLE_CFG.pop(_norm_role(role), None)


class LLMRouter:
    def __init__(self, *, enable_cache: bool = True) -> None:
        self.enable_cache = enable_cache
        self._cache: Dict[Tuple, Any] = {}
        self._lock = threading.Lock()

    def _resolve(self, role: str, **overrides: Any) -> RoleLLMConfig:
        r = _norm_role(role)
        with _ROLE_CFG_LOCK:
            base = _ROLE_CFG.get(r, RoleLLMConfig(role=r))

        # Apply call-time overrides (optional)
        return RoleLLMConfig(
            role=r,
            backend=overrides.get("backend", base.backend),
            model=overrides.get("model", base.model),
            base_url=overrides.get("base_url", base.base_url),
            api_key=overrides.get("api_key", base.api_key),
            reset_freq=overrides.get("reset_freq", base.reset_freq),
            cache=overrides.get("cache", base.cache),
        )

    def get_llm(self, role: str = "executor", **kwargs: Any) -> Any:
        cfg = self._resolve(role, **kwargs)
        if cfg.model is None:
            raise ValueError(f"[llm_router] No model configured for role={cfg.role!r}. Call set_role_models(...) first.")

        cache_key = (cfg.role, cfg.backend, cfg.model, cfg.base_url, cfg.api_key, cfg.reset_freq, cfg.cache)
        if self.enable_cache:
            with self._lock:
                inst = self._cache.get(cache_key)
                if inst is not None:
                    return inst

        if cfg.backend == "LiteLLM":
            inst = LiteLLM(model=cfg.model, reset_freq=cfg.reset_freq, cache=cfg.cache, role=cfg.role)
        elif cfg.backend == "CustomLLM":
            inst = CustomLLM(model=cfg.model, reset_freq=cfg.reset_freq, cache=cfg.cache, role=cfg.role, base_url=cfg.base_url, api_key=cfg.api_key)
        elif cfg.backend == "AutoGen":
            inst = AutoGenLLM(model=cfg.model, reset_freq=cfg.reset_freq, role=cfg.role)
        else:
            raise ValueError(f"[llm_router] Unknown backend: {cfg.backend!r}")

        try:
            inst.role = cfg.role
        except Exception:
            pass

        if self.enable_cache:
            with self._lock:
                self._cache[cache_key] = inst
        return inst


router = LLMRouter(enable_cache=True)

def get_llm(role: str = "executor", **kwargs: Any) -> Any:
    return router.get_llm(role=role, **kwargs)

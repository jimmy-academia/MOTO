# trace/myopto/utils/usage.py
"""
Usage / token-cost accounting utilities for Trace's LLM backends.

Design goals:
- No provider lock-in: works with LiteLLM, OpenAI-compatible, AutoGen, LocalSLM.
- Safe defaults: enabled only when you explicitly turn it on.
- Context-local aggregation via ContextVar (plays nicely with async + threading).
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import re
import time


# --------------------------------------------------
# Global switches (explicit opt-in)
# --------------------------------------------------
_TRACK_USAGE: bool = False


def configure_usage(enabled: bool = True) -> None:
    """Enable/disable aggregation into ContextVars."""
    global _TRACK_USAGE
    _TRACK_USAGE = bool(enabled)


def usage_enabled() -> bool:
    return _TRACK_USAGE


# --------------------------------------------------
# Context-local accumulators
# --------------------------------------------------
request_cost: ContextVar[float] = ContextVar("request_cost", default=0.0)
request_cost_by_role: ContextVar[Optional[Dict[str, float]]] = ContextVar(
    "request_cost_by_role", default=None
)
request_tokens_by_role: ContextVar[Optional[Dict[str, int]]] = ContextVar(
    "request_tokens_by_role", default=None
)


def reset_usage() -> None:
    """Reset all context-local aggregates."""
    request_cost.set(0.0)
    request_cost_by_role.set({})
    request_tokens_by_role.set({})


def get_total_cost() -> float:
    return float(request_cost.get())


def get_cost_by_role() -> Dict[str, float]:
    return dict(request_cost_by_role.get() or {})


def get_tokens_by_role() -> Dict[str, int]:
    return dict(request_tokens_by_role.get() or {})


# --------------------------------------------------
# Pricing (USD per token)
# --------------------------------------------------
# Keep this table small and explicit; unknown models fall back to cost=0.
PRICING_PER_TOKEN: Dict[str, Dict[str, float]] = {
    # OpenAI GPT-5 series
    "gpt-5": {"in": 1.25 / 1_000_000, "out": 10.00 / 1_000_000},
    "gpt-5-mini": {"in": 0.25 / 1_000_000, "out": 2.00 / 1_000_000},
    "gpt-5-nano": {"in": 0.05 / 1_000_000, "out": 0.40 / 1_000_000},
    # OpenAI GPT-4.1 series
    "gpt-4.1": {"in": 5.00 / 1_000_000, "out": 15.00 / 1_000_000},
    "gpt-4.1-mini": {"in": 0.40 / 1_000_000, "out": 1.60 / 1_000_000},
    # OpenAI GPT-4o series
    "gpt-4o": {"in": 5.00 / 1_000_000, "out": 15.00 / 1_000_000},
    "gpt-4o-mini": {"in": 0.50 / 1_000_000, "out": 1.50 / 1_000_000},
    "gpt-4-turbo": {"in": 10.00 / 1_000_000, "out": 30.00 / 1_000_000},
    # Embeddings
    "text-embedding-3-small": {"in": 0.02 / 1_000_000, "out": 0.0},
    "text-embedding-3-large": {"in": 0.13 / 1_000_000, "out": 0.0},
    # Anthropic Claude
    "claude-3-5-sonnet": {"in": 3.00 / 1_000_000, "out": 15.00 / 1_000_000},
    "claude-3-opus": {"in": 15.00 / 1_000_000, "out": 75.00 / 1_000_000},
    "claude-3-haiku": {"in": 0.25 / 1_000_000, "out": 1.25 / 1_000_000},
    # Local SLMs (free)
    "qwen2-0.5b": {"in": 0.0, "out": 0.0},
    "qwen2": {"in": 0.0, "out": 0.0},
    "tinyllama": {"in": 0.0, "out": 0.0},
    "phi-1.5": {"in": 0.0, "out": 0.0},
    "phi": {"in": 0.0, "out": 0.0},
    "local": {"in": 0.0, "out": 0.0},
}


def _pricing_key(model: Optional[str]) -> Optional[str]:
    """Map a model string to a pricing-table key (best-effort)."""
    if not model:
        return None

    m = str(model).lower()

    # Direct match
    if m in PRICING_PER_TOKEN:
        return m

    # OpenAI GPT patterns
    if "gpt-4o-mini" in m:
        return "gpt-4o-mini"
    if re.search(r"\bgpt-4o\b", m):
        return "gpt-4o"
    if "gpt-4.1-mini" in m:
        return "gpt-4.1-mini"
    if re.search(r"\bgpt-4\.1\b", m):
        return "gpt-4.1"
    if "gpt-4-turbo" in m:
        return "gpt-4-turbo"
    if "gpt-5-nano" in m:
        return "gpt-5-nano"
    if "gpt-5-mini" in m:
        return "gpt-5-mini"
    if re.search(r"\bgpt-5\b", m):
        return "gpt-5"

    # Claude patterns
    if "claude-3-5-sonnet" in m or "claude-3.5-sonnet" in m:
        return "claude-3-5-sonnet"
    if "claude-3-opus" in m:
        return "claude-3-opus"
    if "claude-3-haiku" in m:
        return "claude-3-haiku"

    # Local SLM patterns (all free)
    if "qwen2-0.5b" in m or "qwen/qwen2-0.5b" in m:
        return "qwen2-0.5b"
    if "qwen" in m:
        return "qwen2"
    if "tinyllama" in m:
        return "tinyllama"
    if "phi-1" in m or "phi-1.5" in m:
        return "phi-1.5"
    if "phi" in m:
        return "phi"

    # Generic local model detection
    if any(x in m for x in ["local", "mlx", "/home/", "localhost"]):
        return "local"

    return None


# --------------------------------------------------
# Extraction helpers (provider-agnostic)
# --------------------------------------------------
def _get_attr_or_key(obj: Any, key: str) -> Any:
    """Get attribute or dict key."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_model_name(resp: Any) -> Optional[str]:
    """Best-effort model name extraction from response."""
    m = _get_attr_or_key(resp, "model")
    if isinstance(m, str) and m.strip():
        return m.strip()

    # Sometimes nested
    m = _get_attr_or_key(_get_attr_or_key(resp, "response"), "model")
    if isinstance(m, str) and m.strip():
        return m.strip()

    return None


def extract_usage(resp: Any) -> Tuple[int, int]:
    """
    Return (prompt_tokens, completion_tokens) if available; else (0, 0).
    
    Supports:
    - OpenAI python responses (resp.usage.prompt_tokens / completion_tokens)
    - LiteLLM responses (similar shape)
    - Plain dict responses with "usage"
    - LocalSLM approximate token counts
    """
    usage = _get_attr_or_key(resp, "usage")
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")

    pt = _get_attr_or_key(usage, "prompt_tokens")
    ct = _get_attr_or_key(usage, "completion_tokens")

    # Some providers use input_tokens/output_tokens naming
    if pt is None:
        pt = _get_attr_or_key(usage, "input_tokens")
    if ct is None:
        ct = _get_attr_or_key(usage, "output_tokens")

    # Handle total_tokens when individual counts missing
    if pt is None and ct is None:
        tt = _get_attr_or_key(usage, "total_tokens")
        if tt is not None:
            # Rough split when only total available
            try:
                total = int(tt)
                pt = total // 2
                ct = total - pt
            except Exception:
                pass

    try:
        prompt_tokens = int(pt or 0)
    except Exception:
        prompt_tokens = 0
    try:
        completion_tokens = int(ct or 0)
    except Exception:
        completion_tokens = 0

    return prompt_tokens, completion_tokens


def compute_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model: Optional[str],
) -> float:
    """Compute USD cost for given token counts and model."""
    key = _pricing_key(model)
    if not key:
        return 0.0
    pricing = PRICING_PER_TOKEN.get(key)
    if not pricing:
        return 0.0
    return (
        float(prompt_tokens) * float(pricing.get("in", 0.0)) +
        float(completion_tokens) * float(pricing.get("out", 0.0))
    )


# --------------------------------------------------
# Usage Record
# --------------------------------------------------
@dataclass(frozen=True)
class UsageRecord:
    """Immutable record of a single LLM call's usage."""
    model: Optional[str]
    role: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    usd: float
    backend: Optional[str] = None
    ts: float = 0.0


def track_response(
    resp: Any,
    *,
    model: Optional[str] = None,
    role: Optional[str] = None,
    backend: Optional[str] = None,
) -> UsageRecord:
    """
    Extract usage from response, compute USD, and accumulate into ContextVars.
    
    Args:
        resp: LLM response object (OpenAI, LiteLLM, dict, etc.)
        model: Override model name if not in response
        role: Role label (executor, optimizer, metaoptimizer)
        backend: Backend name (LiteLLM, CustomLLM, LocalSLM, etc.)
    
    Returns:
        UsageRecord with extracted/computed values
    """
    resp_model = extract_model_name(resp) or model
    pt, ct = extract_usage(resp)
    usd = compute_cost_usd(pt, ct, resp_model)

    rec = UsageRecord(
        model=resp_model,
        role=role,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=pt + ct,
        usd=usd,
        backend=backend,
        ts=time.time(),
    )

    if not _TRACK_USAGE:
        return rec

    # Accumulate total cost
    request_cost.set(request_cost.get() + usd)

    # Accumulate by role
    if role:
        cur = request_cost_by_role.get() or {}
        nxt = dict(cur)
        nxt[role] = float(nxt.get(role, 0.0)) + float(usd)
        request_cost_by_role.set(nxt)

        cur_t = request_tokens_by_role.get() or {}
        nxt_t = dict(cur_t)
        nxt_t[role] = int(nxt_t.get(role, 0)) + int(pt + ct)
        request_tokens_by_role.set(nxt_t)

    return rec


# --------------------------------------------------
# Summary utilities
# --------------------------------------------------
def get_usage_summary() -> Dict[str, Any]:
    """Get current usage summary."""
    return {
        "total_cost_usd": get_total_cost(),
        "cost_by_role": get_cost_by_role(),
        "tokens_by_role": get_tokens_by_role(),
    }


def print_usage_summary() -> None:
    """Print formatted usage summary."""
    summary = get_usage_summary()
    print("\n=== Usage Summary ===")
    print(f"Total Cost: ${summary['total_cost_usd']:.6f}")
    
    if summary['cost_by_role']:
        print("\nCost by Role:")
        for role, cost in sorted(summary['cost_by_role'].items()):
            tokens = summary['tokens_by_role'].get(role, 0)
            print(f"  {role}: ${cost:.6f} ({tokens:,} tokens)")
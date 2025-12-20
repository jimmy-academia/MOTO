# trace/myopto/utils/usage.py
"""
Usage Tracking

Track token usage and costs for LLM calls.
"""

from typing import Dict, Any
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class UsageStats:
    """Usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    cost: float = 0.0


# --------------------------------------------------
# Global Usage Tracker
# --------------------------------------------------

_usage = UsageStats()
_usage_lock = Lock()


# --------------------------------------------------
# Model Name Mapping (short name -> HuggingFace name)
# --------------------------------------------------

MODEL_REGISTRY = {
    # Local SLMs
    "qwen2-0.5b": "Qwen/Qwen2-0.5B-Instruct",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2b": "google/gemma-2b-it",
    "stablelm-2": "stabilityai/stablelm-2-zephyr-1_6b",
    # Cloud models (pass through)
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}


def resolve_model_name(short_name: str) -> str:
    """Resolve short model name to full name."""
    return MODEL_REGISTRY.get(short_name, short_name)


# --------------------------------------------------
# Pricing per token
# --------------------------------------------------

PRICING_PER_TOKEN = {
    # Local SLMs (free)
    "qwen2-0.5b": {"prompt": 0.0, "completion": 0.0},
    "qwen2-1.5b": {"prompt": 0.0, "completion": 0.0},
    "qwen2-7b": {"prompt": 0.0, "completion": 0.0},
    "tinyllama": {"prompt": 0.0, "completion": 0.0},
    "phi3-mini": {"prompt": 0.0, "completion": 0.0},
    "gemma-2b": {"prompt": 0.0, "completion": 0.0},
    "stablelm-2": {"prompt": 0.0, "completion": 0.0},
    # Cloud models (per token)
    "gpt-4o": {"prompt": 5e-6, "completion": 15e-6},
    "gpt-4o-mini": {"prompt": 0.15e-6, "completion": 0.6e-6},
    "gpt-4-turbo": {"prompt": 10e-6, "completion": 30e-6},
    "claude-3-opus": {"prompt": 15e-6, "completion": 75e-6},
    "claude-3-sonnet": {"prompt": 3e-6, "completion": 15e-6},
    "claude-3-haiku": {"prompt": 0.25e-6, "completion": 1.25e-6},
}


def record_usage(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini"):
    """Record token usage."""
    global _usage
    
    # Find pricing (default to free for unknown models)
    pricing = PRICING_PER_TOKEN.get(model, {"prompt": 0.0, "completion": 0.0})
    cost = prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]
    
    with _usage_lock:
        _usage.prompt_tokens += prompt_tokens
        _usage.completion_tokens += completion_tokens
        _usage.total_tokens += prompt_tokens + completion_tokens
        _usage.calls += 1
        _usage.cost += cost


def get_usage() -> Dict[str, Any]:
    """Get current usage statistics."""
    with _usage_lock:
        return {
            "prompt_tokens": _usage.prompt_tokens,
            "completion_tokens": _usage.completion_tokens,
            "total_tokens": _usage.total_tokens,
            "calls": _usage.calls,
            "cost": _usage.cost,
        }


def get_total_cost() -> float:
    """Get total cost."""
    with _usage_lock:
        return _usage.cost


def reset_usage():
    """Reset usage statistics."""
    global _usage
    with _usage_lock:
        _usage = UsageStats()
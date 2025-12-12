# llm.py

import asyncio
import threading
from typing import Optional
from openai import OpenAI, AsyncOpenAI

# Global mode: "sync" for training, "async" for testing
_track_usage: bool = False

# Global total cost (USD) across all calls
from contextvars import ContextVar
request_cost: ContextVar[float] = ContextVar("request_cost", default=0.0)

PRICING_PER_TOKEN = {
    "gpt-5":            {"in": 1.25/1_000_000, "out": 10.00/1_000_000},
    "gpt-5-mini":       {"in": 0.25/1_000_000, "out":  2.00/1_000_000},
    "gpt-5-nano":       {"in": 0.05/1_000_000, "out":  0.40/1_000_000},
    "gpt-4.1":          {"in": 5.00/1_000_000, "out": 15.00/1_000_000},
    "gpt-4.1-mini":     {"in": 0.40/1_000_000, "out":  1.60/1_000_000},
    "gpt-4o":           {"in": 5.00/1_000_000, "out": 15.00/1_000_000},
    "gpt-4o-mini":      {"in": 0.50/1_000_000, "out":  1.50/1_000_000},
    "gpt-4-turbo":      {"in": 10.00/1_000_000, "out": 30.00/1_000_000},
    "text-embedding-3-small": {"in": 0.02/1_000_000, "out": 0.0},
    "text-embedding-3-large": {"in": 0.13/1_000_000, "out": 0.0},
}


def get_key():
    with open("../.openaiapi", "r") as f:
        return f.read().strip()


def _compute_and_update_cost(resp, model: str):
    """
    Compute token + $ cost and update the per-context `request_cost`
    if `_track_usage` is True.
    Returns (prompt_tokens, completion_tokens, usd).
    """
    global _track_usage

    try:
        usage = getattr(resp, "usage", None)
        if not usage:
            return 0, 0, 0.0

        pricing = PRICING_PER_TOKEN.get(model)
        if not pricing and "gpt-4o-mini" in model:
            pricing = PRICING_PER_TOKEN["gpt-4o-mini"]

        if not pricing:
            return usage.prompt_tokens, usage.completion_tokens, 0.0

        cost_in = usage.prompt_tokens * pricing.get("in", 0.0)
        cost_out = usage.completion_tokens * pricing.get("out", 0.0)
        total_cost = cost_in + cost_out

        if _track_usage:
            current = request_cost.get()
            request_cost.set(current + total_cost)

        return usage.prompt_tokens, usage.completion_tokens, total_cost
    except Exception as e:
        print(f"Warning: Cost calculation failed: {e}")
        return 0, 0, 0.0



class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 2200):
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = get_key()
        self.client = OpenAI(api_key=self.api_key)
        
    def _build_messages(self, prompt: str, system_prompt: Optional[str]):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def answer_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        kwargs = dict(
            model=self.model,
            messages=self._build_messages(prompt, system_prompt)
        )
        if 'gpt-5' in self.model:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens

        resp = self.client.chat.completions.create(**kwargs)

        _compute_and_update_cost(resp, self.model)
        return resp.choices[0].message.content.strip()
        
_GLOBAL_CLIENT = LLMClient(model="gpt-4o-mini")

def configure_llm(
    # mode: str = "sync",
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    usage: Optional[bool] = None,
):
    """
    Configure global LLM behavior.

    mode: "sync" or "async"
    model: optional new model name
    usage: if not None, turn global cost tracking on/off
    """
    global _GLOBAL_CLIENT, _track_usage

    if usage is not None:
        _track_usage = usage

    if model is not None or max_tokens is not None:
        _GLOBAL_CLIENT = LLMClient(
            model=model or _GLOBAL_CLIENT.model,
            max_tokens=max_tokens or _GLOBAL_CLIENT.max_tokens,
        )

def global_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    return _GLOBAL_CLIENT.answer_sync(prompt, system_prompt)
    raise RuntimeError(f"Unknown LLM mode: {mode}")



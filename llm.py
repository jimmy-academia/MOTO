# llm.py

import asyncio
import threading
from typing import Optional
from openai import OpenAI, AsyncOpenAI

# Global mode: "sync" for training, "async" for testing
_llm_mode: str = "sync"
_mode_lock = threading.Lock()
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
        self._async_client: Optional[AsyncOpenAI] = None

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key)
        return self._async_client

    def _build_messages(self, prompt: str, system_prompt: Optional[str]):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def answer_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system_prompt),
            max_tokens=self.max_tokens,
        )
        _compute_and_update_cost(resp, self.model)
        return resp.choices[0].message.content.strip()
        
    async def answer_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        client = self._get_async_client()
        resp = await client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system_prompt),
            max_tokens=self.max_tokens,
        )
        _compute_and_update_cost(resp, self.model)
        return resp.choices[0].message.content.strip()


_GLOBAL_CLIENT = LLMClient(model="gpt-4o-mini")


def configure_llm(
    mode: str = "sync",
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
    global _llm_mode, _GLOBAL_CLIENT, _track_usage

    if mode not in ("sync", "async"):
        raise ValueError(f"Invalid mode: {mode}")

    with _mode_lock:
        _llm_mode = mode

        if usage is not None:
            _track_usage = usage

        if model is not None or max_tokens is not None:
            _GLOBAL_CLIENT = LLMClient(
                model=model or _GLOBAL_CLIENT.model,
                max_tokens=max_tokens or _GLOBAL_CLIENT.max_tokens,
            )



def global_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Global wrapper used inside solution_workflow.

    - If configured sync: blocks.
    - If configured async: uses AsyncOpenAI under the hood via asyncio.run().
      Must be called from a context with no running event loop
      (i.e., from worker threads via asyncio.to_thread).
    """
    mode = _llm_mode   # read under the assumption config is set before use

    if mode == "sync":
        return _GLOBAL_CLIENT.answer_sync(prompt, system_prompt)

    if mode == "async":
        # Called from a worker thread during inference, so nested loop is OK.
        return asyncio.run(_GLOBAL_CLIENT.answer_async(prompt, system_prompt))

    raise RuntimeError(f"Unknown LLM mode: {mode}")

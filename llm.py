# llm.py

import asyncio
from contextvars import ContextVar
from typing import Optional
from openai import OpenAI, AsyncOpenAI

# --- Global Contexts ---

# "sync" during training, "async" during testing
llm_mode = ContextVar("llm_mode", default="sync")

# Global total cost (if you care); you can ignore it if not needed.
request_cost = ContextVar("request_cost", default=0.0)

PRICING_PER_TOKEN = {
    # GPT-5 family
    "gpt-5":            {"in": 1.25/1_000_000, "out": 10.00/1_000_000},
    "gpt-5-mini":       {"in": 0.25/1_000_000, "out":  2.00/1_000_000},
    "gpt-5-nano":       {"in": 0.05/1_000_000, "out":  0.40/1_000_000},

    # GPT-4 family
    "gpt-4.1":          {"in": 5.00/1_000_000, "out": 15.00/1_000_000},
    "gpt-4.1-mini":     {"in": 0.40/1_000_000, "out":  1.60/1_000_000},
    "gpt-4o":           {"in": 5.00/1_000_000, "out": 15.00/1_000_000},
    "gpt-4o-mini":      {"in": 0.50/1_000_000, "out":  1.50/1_000_000},
    "gpt-4-turbo":      {"in": 10.00/1_000_000, "out": 30.00/1_000_000},

    # Embedding
    "text-embedding-3-small": {"in": 0.02/1_000_000, "out": 0.0},
    "text-embedding-3-large": {"in": 0.13/1_000_000, "out": 0.0},
}


def get_key():
    try:
        with open("../.openaiapi", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "MISSING_KEY"


def _compute_and_update_cost(resp, model: str, track_usage: bool):
    """
    Compute token + $ cost and optionally update the global ContextVar.
    Returns (prompt_tokens, completion_tokens, usd) or (0,0,0) on failure.
    """
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

        if track_usage:
            current = request_cost.get()
            request_cost.set(current + total_cost)

        return usage.prompt_tokens, usage.completion_tokens, total_cost
    except Exception as e:
        print(f"Warning: Cost calculation failed: {e}")
        return 0, 0, 0.0


class LLMClient:
    """
    One client that supports both sync and async access.
    Mode is chosen by the global llm_mode ContextVar via the llm() wrapper.
    """

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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # ---- sync core ----
    def answer_sync(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = self._build_messages(prompt, system_prompt)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        _compute_and_update_cost(resp, self.model, track_usage=True)
        return resp.choices[0].message.content.strip()

    # ---- async core ----
    async def answer_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        client = self._get_async_client()
        messages = self._build_messages(prompt, system_prompt)
        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        _compute_and_update_cost(resp, self.model, track_usage=True)
        return resp.choices[0].message.content.strip()


# --- Global singleton & wrapper ---

GLOBAL_CLIENT = LLMClient(model="gpt-4o-mini")


def set_llm_mode(mode: str):
    """
    mode: "sync" or "async"
    TRAIN: set_llm_mode("sync")
    TEST:  set_llm_mode("async")
    """
    if mode not in ("sync", "async"):
        raise ValueError(f"Invalid llm_mode: {mode}")
    llm_mode.set(mode)


def llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    The universal wrapper.

    - In TRAIN (llm_mode="sync"):
        called directly from sync Trace code.
    - In TEST (llm_mode="async"):
        called from worker threads (via asyncio.to_thread),
        where we can safely use asyncio.run for the async client.
    """
    mode = llm_mode.get()

    if mode == "sync":
        return GLOBAL_CLIENT.answer_sync(prompt, system_prompt)

    if mode == "async":
        # This must be called from a context *without* an active event loop.
        # In our design, that means: inside worker threads during inference.
        return asyncio.run(GLOBAL_CLIENT.answer_async(prompt, system_prompt))

    raise RuntimeError(f"Unknown llm_mode: {mode}")

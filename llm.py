import asyncio
from contextvars import ContextVar
from typing import Optional, Any
from openai import OpenAI, AsyncOpenAI

# 1. Global Context & Configuration
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
    # Adjust path as necessary for your environment
    try:
        with open("../.openaiapi", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback or explicit error handling
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
        # Fallback to gpt-4o-mini pricing if model not found
        if not pricing and "gpt-4o-mini" in model:
            pricing = PRICING_PER_TOKEN["gpt-4o-mini"]

        if not pricing:
            return usage.prompt_tokens, usage.completion_tokens, 0.0

        cost_in = usage.prompt_tokens * pricing.get("in", 0.0)
        cost_out = usage.completion_tokens * pricing.get("out", 0.0)
        total_cost = cost_in + cost_out

        if track_usage:
            current_cost = request_cost.get()
            request_cost.set(current_cost + total_cost)

        return usage.prompt_tokens, usage.completion_tokens, total_cost

    except Exception as e:
        print(f"Warning: Cost calculation failed: {e}")
        return 0, 0, 0.0


# 2. The Class Definition (Must be defined BEFORE the wrappers use it)
class LLMClient:
    """
    Unified sync/async LLM client with optional usage tracking.

    - mode="sync": use OpenAI (blocking).
    - mode="async": use AsyncOpenAI under the hood; `answer()` is still sync
      (via asyncio.run), and `answer_async()` is the true async API.
    """

    def __init__(
        self,
        model: str = "gpt-5-nano",
        max_tokens: int = 2200,
        mode: str = "sync",          # "sync" or "async"
        track_usage: bool = True,    # whether to update request_cost
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.mode = mode
        self.track_usage = track_usage

        api_key = get_key()
        # Sync client
        self.client = OpenAI(api_key=api_key)
        # Async client (lazy init in case you never use async)
        self._async_client: AsyncOpenAI | None = None

    def _build_messages(self, prompt: str, system_prompt: str | None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # ---------- SYNC PATH (blocking) ----------

    def _answer_sync_core(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ):
        messages = self._build_messages(prompt, system_prompt)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        pt, ct, usd = _compute_and_update_cost(resp, self.model, self.track_usage)
        content = resp.choices[0].message.content.strip()
        return content, pt, ct, usd

    # ---------- ASYNC PATH (true async) ----------

    async def _ensure_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=get_key())
        return self._async_client

    async def _answer_async_core(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ):
        client = await self._ensure_async_client()
        messages = self._build_messages(prompt, system_prompt)
        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        pt, ct, usd = _compute_and_update_cost(resp, self.model, self.track_usage)
        content = resp.choices[0].message.content.strip()
        return content, pt, ct, usd

    # ---------- PUBLIC APIS ----------

    def answer(
        self,
        prompt: str,
        system_prompt: str | None = None,
        return_usage: bool = False,
    ):
        """
        Synchronous API used by wrapper / normal code.
        If mode="async", this calls asyncio.run(), so this method
        MUST be called from a thread where there is no running event loop.
        """
        if self.mode == "sync":
            content, pt, ct, usd = self._answer_sync_core(prompt, system_prompt)
        else:
            # async under the hood, still sync to the caller
            content, pt, ct, usd = asyncio.run(
                self._answer_async_core(prompt, system_prompt)
            )

        if return_usage:
            return content, pt, ct, usd
        return content

    async def answer_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        return_usage: bool = False,
    ):
        """
        True async API.
        """
        if self.mode == "sync":
            content, pt, ct, usd = self._answer_sync_core(prompt, system_prompt)
        else:
            content, pt, ct, usd = await self._answer_async_core(
                prompt, system_prompt
            )

        if return_usage:
            return content, pt, ct, usd
        return content


# 3. Global State & Wrapper Functions (Defined AFTER class)

_ACTIVE_CLIENT: Optional[LLMClient] = None

def _get_client() -> LLMClient:
    """Lazy initialization of the global client."""
    global _ACTIVE_CLIENT
    if _ACTIVE_CLIENT is None:
        # Default to safe sync mode
        _ACTIVE_CLIENT = LLMClient(mode="sync")
    return _ACTIVE_CLIENT

def configure_llm(mode: str = "sync", model: str = "gpt-5-nano"):
    """
    Global Phase Change.
    - "sync": Normal blocking scripts.
    - "async": Use when running solution_workflow inside threads in an async app.
    """
    global _ACTIVE_CLIENT
    print(f"--- Switching LLM Phase to: {mode.upper()} ({model}) ---")
    _ACTIVE_CLIENT = LLMClient(mode=mode, model=model)

def llm(
    prompt: str, 
    system_prompt: str = None, 
    return_usage: bool = False
) -> str | tuple:
    """
    The single function your business logic should call.
    """
    client = _get_client()
    return client.answer(
        prompt=prompt, 
        system_prompt=system_prompt, 
        return_usage=return_usage
    )
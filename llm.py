#lwt.py, fixed
from contextvars import ContextVar
from openai import OpenAI

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
    with open('../.openaiapi', "r") as f:
        api_key = f.read().strip()
    return api_key

class LLMClient:
    def __init__(self, model: str = "gpt-5-nano", max_tokens=2200):
        
        self.client = OpenAI(api_key=get_key())
        self.model = model
        self.max_tokens = max_tokens

    def answer(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Minimal chat-completions call.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        # Calculate Cost
        try:
            usage = resp.usage
            pricing = PRICING_PER_TOKEN.get(self.model)
            # Fallback to gpt-4o-mini pricing if model not found, or handle as 0
            if not pricing and "gpt-4o-mini" in self.model: 
                pricing = PRICING_PER_TOKEN["gpt-4o-mini"]
            
            if pricing and usage:
                cost_in = usage.prompt_tokens * pricing.get("in", 0)
                cost_out = usage.completion_tokens * pricing.get("out", 0)
                total_cost = cost_in + cost_out
                
                # Update the context variable safely
                current_cost = request_cost.get()
                request_cost.set(current_cost + total_cost)
        except Exception as e:
            print(f"Warning: Cost calculation failed: {e}")
            
        return resp.choices[0].message.content.strip()

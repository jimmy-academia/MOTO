#lwt.py, fixed

import os
import re
import ast
import time
from functools import partial

from openai import OpenAI

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
        return resp.choices[0].message.content.strip()

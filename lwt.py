#lwt.py, fixed

import os
import re
import ast
import time
from functools import partial

from openai import OpenAI

def get_key():
    with open('../../.openaiapi', "r") as f:
        api_key = f.read().strip()
    return api_key

class LLMClient:
    def __init__(self, model: str = "gpt-5-nano"):
        
        self.client = OpenAI(api_key=get_key())
        self.model = model

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
        )
        return resp.choices[0].message.content.strip()


# ============================================================
# 2. Minimal LWT interpreter
# ============================================================

def _sub(match, query, cache):
    var_name = match.group(1)
    index_str = match.group(2)  # None if no index

    if var_name == "input":
        try:
            base_value = ast.literal_eval(query)
        except (SyntaxError, ValueError):
            base_value = query
    else:
        base_value = cache.get(var_name, "")

    # Optional indexing
    if index_str is not None and isinstance(base_value, (list, tuple)) and base_value:
        idx = int(index_str)
        if 0 <= idx < len(base_value):
            return str(base_value[idx])
        return ""

    # Ensure string output
    if isinstance(base_value, (list, tuple)):
        import json
        return json.dumps(base_value)
    return str(base_value)


class LWTExecutor:
    def __init__(self, llm_client: LLMClient, system_prompt: str | None = None):
        self.llm_client = llm_client
        self.system_prompt = system_prompt or (
            "You are a careful medical triage assistant. "
            "Follow the instructions exactly and keep answers concise."
        )

    def llm_answer(self, prompt: str) -> str:
        return self.llm_client.answer(prompt, system_prompt=self.system_prompt)

    def solve_query(self, script: str, query: str) -> str:
        """
        Execute an LWT script of the form:
            (0)=LLM("...")
            (1)=LLM("... {(0)} ... {(input)} ...")
        and return the final LLM output (last step).
        """
        cache: dict[str, object] = {}
        output: str = ""

        for raw_step in script.split("\n"):
            step = raw_step.strip()
            if not step or "=LLM(" not in step:
                continue

            idx_match = re.search(r"\((\d+)\)=LLM", step)
            instr_match = re.search(r'LLM\("(.*)"\)', step)
            if not idx_match or not instr_match:
                continue

            index = idx_match.group(1)
            instruction = instr_match.group(1)

            # Substitute {(input)}, {(0)}, {(1)}[0], etc.
            _sub_with_args = partial(_sub, query=query, cache=cache)
            instruction = re.sub(r"\{\((\w+)\)\}(?:\[(\d+)\])?", _sub_with_args, instruction)

            start = time.time()
            output = self.llm_answer(instruction)
            _ = time.time() - start  # you can log if needed

            # Try to store as Python object if possible
            try:
                cache[index] = ast.literal_eval(output)
            except Exception:
                cache[index] = output

        return str(output), cache


# ============================================================
# 3. Small healthcare triage LWT + 5 examples
# ============================================================

HEALTHCARE_LWT_SCRIPT = r'''
(0)=LLM("Read the patient case: {(input)}. Extract oxygen saturation (SpO2, in %), body temperature in Celsius, and age in years. Return a JSON object with keys 'spo2', 'temp_c', and 'age'.")
(1)=LLM("Given the structured data {(0)}, decide if there is a life-threatening issue. If spo2 < 92 OR temp_c >= 40, output exactly 'ER referral'. Otherwise output 'no ER'.")
(2)=LLM("Given {(0)}, if 'no ER', check if spo2 < 95 OR temp_c >= 38 OR age >= 70. If any of these is true, output exactly 'Urgent clinical evaluation'. Otherwise output 'Home care'. Use only the JSON in {(0)} as evidence.")
(3)=LLM("Combine {(1)} and {(2)}. If {(1)} is 'ER referral', final answer is 'ER referral'. If {(1)} is 'no ER', final answer is the value from {(2)}. Output exactly one of: 'ER referral', 'Urgent clinical evaluation', or 'Home care'.")
'''

HEALTHCARE_EXAMPLES = [
    {
        "name": "Young mild fever",
        "query": "A 25-year-old with sore throat, SpO2 98%, temperature 37.8°C, otherwise healthy.",
        "label": "Home care",
    },
    {
        "name": "High fever, normal oxygen",
        "query": "A 40-year-old with cough and fatigue, SpO2 97%, temperature 38.5°C, no comorbidities.",
        "label": "Urgent clinical evaluation",
    },
    {
        "name": "Low oxygen, elderly",
        "query": "A 78-year-old with COPD, SpO2 90%, temperature 37.0°C, mild shortness of breath.",
        "label": "ER referral",
    },
    {
        "name": "Borderline elderly, mild symptoms",
        "query": "A 69-year-old with hypertension, SpO2 96%, temperature 37.3°C, mild cough.",
        "label": "Home care",
    },
    {
        "name": "Elderly with moderate fever",
        "query": "A 72-year-old with diabetes, SpO2 95%, temperature 38.2°C, some weakness.",
        "label": "Urgent clinical evaluation",
    },
]


def run_healthcare_demo():
    llm_client = LLMClient(model="gpt-5-nano")
    executor = LWTExecutor(llm_client)

    script = HEALTHCARE_LWT_SCRIPT

    # for ex in HEALTHCARE_EXAMPLES:
    ex = {
        "name": "Elderly with moderate fever",
        "query": "A 72-year-old with diabetes, SpO2 95%, temperature 38.2°C, some weakness.",
        "label": "Urgent clinical evaluation",
    },

    print("=" * 80)
    print(f"Case: {ex['name']}")
    print(f"Input: {ex['query']}")
    print(f"Expected: {ex['label']}")
    try:
        pred, cache = executor.solve_query(script, ex["query"])
    except Exception as e:
        pred = f"[ERROR] {e}"
    print(f"Predicted: {pred}")
    print()


if __name__ == "__main__":
    run_healthcare_demo()

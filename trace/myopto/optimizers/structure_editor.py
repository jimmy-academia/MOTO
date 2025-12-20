# schemes/structure_editor.py
from __future__ import annotations

import ast
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from myopto.utils.llm_router import get_llm


# -----------------------------
# Utilities
# -----------------------------
_CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*|\s*```$", re.MULTILINE)

def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s).strip()

def _extract_json_object(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(text[i : j + 1])
        except Exception:
            return None
    return None

def _callsite_brief(cs: Dict[str, Any]) -> str:
    return f"{cs.get('function','?')}:{cs.get('lineno','?')}"


# -----------------------------
# Validation Helpers
# -----------------------------
def _find_function_def(tree: ast.AST, func_name: str):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func_name:
            return n
    return None

def _get_signature_str(func) -> str:
    try:
        return str(inspect.signature(func))
    except Exception:
        return "(unknown)"

def _extract_llm_calls(fn: ast.FunctionDef) -> List[ast.Call]:
    calls: List[ast.Call] = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id == "llm":
                calls.append(n)
            if isinstance(n.func, ast.Attribute) and n.func.attr == "llm":
                calls.append(n)
    return calls

def _call_has_call_tag(call: ast.Call) -> bool:
    for kw in call.keywords or []:
        if kw.arg == "call_tag":
            return True
    return False

def _call_tag_value(call: ast.Call) -> Optional[str]:
    for kw in call.keywords or []:
        if kw.arg == "call_tag":
            v = kw.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return v.value
    return None

def _contains_name(fn: ast.FunctionDef, name: str) -> bool:
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id == name:
            return True
    return False


@dataclass
class StructureEditResult:
    ok: bool
    code: str
    reasoning: str
    raw_response: str
    errors: List[str]


# -----------------------------
# Structure Editor Class
# -----------------------------
class StructureEditor:
    def __init__(
        self,
        llm: Optional[Any] = None,
        *,
        max_tokens: int = 12000,
        require_call_tag: bool = True,
        forbid_strip_trace_tags: bool = True,
        forbid_imports: bool = False,
        verbose: bool = False,
    ):
        self.llm = llm or get_llm(role="optimizer")
        self.max_tokens = max_tokens
        self.require_call_tag = require_call_tag
        self.forbid_strip_trace_tags = forbid_strip_trace_tags
        self.forbid_imports = forbid_imports
        self.verbose = verbose

    def build_prompt(
        self,
        *,
        func_name: str,
        signature_str: str,
        code: str,
        ir: Dict[str, Any],
        feedback: str,
        required_call_tags: Sequence[str],
    ) -> Tuple[str, str]:
        
        # Summarize IR
        nodes_txt = []
        for n in ir.get("nodes", []):
            nodes_txt.append(
                "\n".join(
                    [
                        f"- id: {n.get('id')}",
                        f"  call_tag: {n.get('call_tag')}",
                        f"  callsite: {_callsite_brief(n.get('callsite', {}))}",
                        f"  prompt:\n{n.get('prompt_rendered','')}",
                        f"  output_preview: {n.get('output_preview','')}",
                        f"  parents: {n.get('parents', [])}",
                    ]
                )
            )
        ir_block = "\n\n".join(nodes_txt) if nodes_txt else "(no IR nodes)"

        system_prompt = (
            "You are an expert AI Systems Engineer.\n"
            "You will rewrite a Python function that implements an LLM workflow.\n"
            "Your goal is to improve correctness on the provided failure case.\n\n"
            "Output MUST be valid JSON with keys:\n"
            '  {"reasoning": "...", "code": "..."}\n'
            "The 'code' value MUST be ONLY the full updated Python source of the function.\n"
            "No markdown. No backticks."
        )

        constraints = [
            f"- Keep the function name EXACTLY: {func_name}",
            f"- Keep the function signature EXACTLY: def {func_name}{signature_str}:",
            "- The function MUST return a Msg from the final llm() call.",
            "- Every llm(...) call MUST include a call_tag keyword argument.",
            f"- Preserve these existing call_tag values: {list(required_call_tags)}",
            "- Keep the workflow pure: no file I/O, no network calls.",
        ]
        if self.forbid_imports:
            constraints.append("- Do not add any import statements.")

        user_prompt = (
            "Rewrite the function code below to improve the output according to the feedback.\n\n"
            "=== CURRENT FUNCTION CODE ===\n"
            f"{code}\n\n"
            "=== RUNTIME TRACE (IR) FROM A FAILED RUN ===\n"
            f"{ir_block}\n\n"
            "=== FEEDBACK ===\n"
            f"{feedback}\n\n"
            "=== HARD CONSTRAINTS ===\n"
            + "\n".join(constraints)
            + "\n\n"
            "=== OUTPUT REQUIREMENT ===\n"
            "Return JSON ONLY: {\"reasoning\": \"...\", \"code\": \"...\"}"
        )
        return system_prompt, user_prompt

    def call_llm(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        max_tokens = max_tokens or self.max_tokens
        response = None
        
        # Attempt 1: JSON mode
        try:
            response = self.llm(
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
            )
        except Exception as e:
            if self.verbose:
                print(f"[StructureEditor] Attempt 1 failed: {e!r}")

        # Attempt 2: Fallback
        if response is None:
            try:
                response = self.llm(messages=messages, max_tokens=max_tokens)
            except Exception as e:
                if self.verbose:
                    print(f"[StructureEditor] Attempt 2 failed: {e!r}")
                return ""

        try:
            content = response.choices[0].message.content or ""
            return content
        except Exception:
            return ""

    def _extract_reasoning_and_code(self, raw: str) -> Tuple[str, str]:
        obj = _extract_json_object(raw)
        if isinstance(obj, dict):
            reasoning = str(obj.get("reasoning", "")).strip()
            if "code" in obj and isinstance(obj["code"], str):
                return reasoning, _strip_code_fences(obj["code"]).strip()
            if "answer" in obj and "def " in str(obj["answer"]):
                return reasoning, _strip_code_fences(obj["answer"]).strip()
        return "", _strip_code_fences(raw).strip()

    def validate_code(
        self,
        *,
        code: str,
        func_name: str,
        signature_str: str,
        required_call_tags: Sequence[str],
    ) -> List[str]:
        errs: List[str] = []
        if not code.strip().startswith(f"def {func_name}"):
            errs.append(f"Code must start with 'def {func_name}...'")

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"SyntaxError: {e}"]

        fn = _find_function_def(tree, func_name)
        if fn is None:
            return [f"Missing function def {func_name!r}."]

        if self.forbid_strip_trace_tags and _contains_name(fn, "strip_trace_tags"):
            errs.append("Do not call strip_trace_tags() inside the workflow.")

        if self.forbid_imports:
            for n in ast.walk(tree):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    errs.append("Imports are forbidden.")
                    break

        llm_calls = _extract_llm_calls(fn)
        if not llm_calls:
            errs.append("No llm(...) calls found.")

        if self.require_call_tag:
            missing = [c for c in llm_calls if not _call_has_call_tag(c)]
            if missing:
                errs.append(f"{len(missing)} llm(...) calls are missing call_tag.")

        if required_call_tags:
            present = set([t for t in (_call_tag_value(c) for c in llm_calls) if t])
            for t in required_call_tags:
                if t not in present:
                    errs.append(f"Required call_tag {t!r} is missing.")

        return errs

    def rewrite_function(
        self,
        func_or_code: Union[Any, str],
        *,
        ir: Dict[str, Any],
        feedback: str,
        func_name: Optional[str] = None,
        required_call_tags: Optional[Sequence[str]] = None,
        max_retries: int = 2,
    ) -> StructureEditResult:
        
        # FIX: Handle string input vs function object
        if isinstance(func_or_code, str):
            src = textwrap.dedent(func_or_code).strip()
            # Try to guess name from "def name"
            if not func_name:
                m = re.match(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", src)
                func_name = m.group(1) if m else "workflow"
            signature_str = "(...)" # Approximate if raw code
        else:
            func_name = func_name or getattr(func_or_code, "__name__", "workflow")
            signature_str = _get_signature_str(func_or_code)
            src = textwrap.dedent(inspect.getsource(func_or_code)).strip()

        if required_call_tags is None:
            tags = []
            for n in ir.get("nodes", []):
                t = n.get("call_tag")
                if t and t not in tags:
                    tags.append(t)
            required_call_tags = tags

        last_raw = ""
        last_code = src
        last_reasoning = ""
        last_errs: List[str] = []

        for attempt in range(max_retries + 1):
            fb = feedback
            if last_errs:
                fb = fb + "\n\nVALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in last_errs)

            system_prompt, user_prompt = self.build_prompt(
                func_name=func_name,
                signature_str=signature_str,
                code=src,
                ir=ir,
                feedback=fb,
                required_call_tags=required_call_tags,
            )

            if self.verbose:
                print(f"[StructureEditor] Attempt {attempt+1}/{max_retries+1}")

            raw = self.call_llm(system_prompt, user_prompt)
            reasoning, code = self._extract_reasoning_and_code(raw)
            
            errs = self.validate_code(
                code=code,
                func_name=func_name,
                signature_str=signature_str,
                required_call_tags=required_call_tags,
            )

            last_raw, last_code, last_reasoning, last_errs = raw, code, reasoning, errs

            if not errs:
                if self.verbose:
                    print(f"[StructureEditor] Success! Reasoning: {reasoning}")
                return StructureEditResult(ok=True, code=code, reasoning=reasoning, raw_response=raw, errors=[])
            
            if self.verbose:
                print("[StructureEditor] Validation errors:", errs)

        return StructureEditResult(ok=False, code=last_code, reasoning=last_reasoning, raw_response=last_raw, errors=last_errs)

    def load_function(
        self,
        code: str,
        func_name: Optional[str] = None,
        *,
        extra_globals: Optional[Dict[str, Any]] = None,
    ):
        g: Dict[str, Any] = {}
        if extra_globals:
            g.update(extra_globals)

        # Infer name if not given
        if not func_name:
            m = re.match(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
            func_name = m.group(1) if m else "workflow"

        compiled = compile(code, filename="<structure_edit>", mode="exec")
        exec(compiled, g, g)

        if func_name not in g:
            raise ValueError(f"Function {func_name!r} not found in executed code.")
        return g[func_name]
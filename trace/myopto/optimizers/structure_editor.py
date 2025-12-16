# trace/myopto/optimizers/structure_editor.py
from __future__ import annotations

import ast
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from myopto.utils.llm import LLM


# -----------------------------
# Utilities
# -----------------------------
_CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s).strip()


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Robust-ish JSON extraction:
    - Try full parse
    - Else try parsing substring from first '{' to last '}'
    """
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
# Validation
# -----------------------------

def _find_function_def(tree: ast.AST, func_name: str):
    # Find both sync + async defs anywhere in the module (including inside if/try)
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
            # llm(...)
            if isinstance(n.func, ast.Name) and n.func.id == "llm":
                calls.append(n)
            # runtime.llm(...) (optional)
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
# Structure editor
# -----------------------------
class StructureEditor:
    """
    Outer-loop structure editor that rewrites a Python workflow function.

    This is intentionally NOT "differentiable".
    It’s an explicit program-rewrite step informed by IR + feedback.

    The *inner loop* (OptoPrimeLocal) still tunes prompt templates afterward.
    """

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
        self.llm = llm or LLM(role="optimizer")
        self.max_tokens = max_tokens
        self.require_call_tag = require_call_tag
        self.forbid_strip_trace_tags = forbid_strip_trace_tags
        self.forbid_imports = forbid_imports
        self.verbose = verbose

    # ---------- prompting ----------
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
        """
        Build (system_prompt, user_prompt) for a structure rewrite.
        """
        # Summarize IR as readable text
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
            "- The function MUST return a Msg from the final llm() call (do NOT strip trace tags inside).",
            "- Every llm(...) call MUST include a call_tag keyword argument.",
            f"- Preserve these existing call_tag values (do not rename/remove unless you delete that stage): {list(required_call_tags)}",
            "- If you add new llm calls, assign NEW call_tag names that describe their role (e.g., plan, derive, verify, fix, extract).",
            "- Keep the workflow pure: do not read/write files, do not use network calls, do not use subprocess.",
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
            "Return JSON ONLY:\n"
            '{"reasoning": <string>, "code": <string with full function source>}'  # keep simple
        )

        return system_prompt, user_prompt

    # ---------- calling ----------
    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ):
        """Call the LLM with a prompt and return the response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = None
        last_err = None
        if max_tokens is None:
            max_tokens = self.max_tokens

        # Attempt 1: force json_object
        try:
            response = self.llm(
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
            )
        except Exception as e:
            last_err = e
            if self.verbose:
                print(f"[StructureEditor--call_llm] Attempt 1: LL!M ERROR: {last_err!r}")

        # Attempt 2: fallback without response_format
        if response is None:
            try:
                response = self.llm(messages=messages, max_tokens=max_tokens)
            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"[StructureEditor--call_llm] Attempt 2: LL!M ERROR: {last_err!r}")

        # Extract content safely
        try:
            content = response.choices[0].message.content
            if content is None:
                content = ""
        except Exception as e:
            if self.verbose:
                print(f"[StructureEditor--call_llm] LLM RESPONSE PARS!E ERROR: {e!r}")
                print("Raw response repr:", repr(response))
                # print("Prompt (debug)\n", system_prompt + user_prompt)
            return ""

        if self.verbose:
            # If content is empty, dump prompt too (so you can reproduce)
            if content.strip() == "":
                print(f"[StructureEditor--call_llm] content is empty!")
            else:
                print("[StructureEditor--call_llm] LLM response:\n", content)

        # from debug import check
        # check()
        return content

    # ---------- extraction + validation ----------
    def _extract_reasoning_and_code(self, raw: str) -> Tuple[str, str]:
        obj = _extract_json_object(raw)
        if isinstance(obj, dict):
            reasoning = str(obj.get("reasoning", "")).strip()

            # Preferred key
            if "code" in obj and isinstance(obj["code"], str):
                code = _strip_code_fences(obj["code"]).strip()
                return reasoning, code

            # Common deviation: model puts code in "answer"
            if "answer" in obj and isinstance(obj["answer"], str) and "def " in obj["answer"]:
                code = _strip_code_fences(obj["answer"]).strip()
                return reasoning, code

        # fallback: treat raw as code
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
            all_defs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

        except SyntaxError as e:
            return [f"SyntaxError: {e}"]

        fn = _find_function_def(tree, func_name)
        if fn is None:
            return [f"Missing function def {func_name!r}. Found defs: {all_defs}"]

        # Signature check (basic): compare arg names/count only
        # We enforce exact signature string in prompt, but validate roughly here.
        # Example signature_str: "(problem_text: str) -> str"
        # We'll just check number of args matches.
        want_args = signature_str.split(")")[0].lstrip("(").strip()
        want_n = 0 if want_args == "" else len([x for x in want_args.split(",")])
        got_n = len(fn.args.args)
        if got_n != want_n:
            errs.append(f"Signature mismatch: expected {want_n} args, got {got_n} args.")

        if self.forbid_strip_trace_tags and _contains_name(fn, "strip_trace_tags"):
            errs.append("Do not call strip_trace_tags() inside the workflow; handle it outside.")

        if self.forbid_imports:
            for n in ast.walk(tree):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    errs.append("Imports are forbidden in structure-edited code.")
                    break

        llm_calls = _extract_llm_calls(fn)
        if not llm_calls:
            errs.append("No llm(...) calls found — this is supposed to be an LLM workflow.")

        if self.require_call_tag:
            missing = [c for c in llm_calls if not _call_has_call_tag(c)]
            if missing:
                errs.append(f"{len(missing)} llm(...) calls are missing call_tag=...")

        # Required call_tag presence
        if required_call_tags:
            present_tags = set([t for t in (_call_tag_value(c) for c in llm_calls) if t])
            for t in required_call_tags:
                if t not in present_tags:
                    errs.append(f"Required call_tag {t!r} is missing from rewritten code.")

        return errs

    # ---------- main API ----------
    def rewrite_function(
        self,
        func,
        *,
        ir: Dict[str, Any],
        feedback: str,
        func_name: Optional[str] = None,
        required_call_tags: Optional[Sequence[str]] = None,
        max_retries: int = 2,
    ) -> StructureEditResult:
        """
        Rewrite the function definition using LLM.
        Returns StructureEditResult with ok/code/errors.

        You can then exec it via load_function().
        """
        func_name = func_name or getattr(func, "__name__", "workflow")
        signature_str = _get_signature_str(func)

        src = inspect.getsource(func)
        src = textwrap.dedent(src).strip()

        # derive required tags from IR if not provided
        if required_call_tags is None:
            tags = []
            for n in ir.get("nodes", []):
                t = n.get("call_tag")
                if t and t not in tags:
                    tags.append(t)
            required_call_tags = tags

        last_raw = ""
        last_code = ""
        last_reasoning = ""
        last_errs: List[str] = []

        for attempt in range(max_retries + 1):
            # Attach validation errors to feedback so the model repairs itself
            fb = feedback
            if last_errs:
                fb = fb + "\n\nVALIDATION ERRORS FROM LAST ATTEMPT:\n" + "\n".join(f"- {e}" for e in last_errs)

            system_prompt, user_prompt = self.build_prompt(
                func_name=func_name,
                signature_str=signature_str,
                code=src,
                ir=ir,
                feedback=fb,
                required_call_tags=required_call_tags,
            )
            if self.verbose:
                print("\n[StructureEditor] --- BUILT PROMPTS ---")
                print(":"*40+"::system_prompt::"+":"*40)
                print(system_prompt)
                print(":"*40+"::user_prompt::"+":"*40)
                print(user_prompt)
                print(":"*98)
            raw = self.call_llm(system_prompt, user_prompt)
            reasoning, code = self._extract_reasoning_and_code(raw)

            errs = self.validate_code(
                code=code,
                func_name=func_name,
                signature_str=signature_str,
                required_call_tags=required_call_tags,
            )

            if self.verbose:
                print("\n[StructureEditor] --- RAW RESPONSE (repr head) ---")
                print(repr(raw[:800]))
                print("\n[StructureEditor] --- EXTRACTED CODE (first 40 lines) ---")
                code_head = "\n".join(code.splitlines()[:40])
                print(code_head)
                print("\n[StructureEditor] --- EXTRACTED CODE STARTS WITH ---")
                print(repr(code[:80]))

                print(f"\n[StructureEditor] attempt={attempt} ok={not errs}")
                if errs:
                    print("[StructureEditor] errors:")
                    for e in errs:
                        print(" -", e)

            last_raw, last_code, last_reasoning, last_errs = raw, code, reasoning, errs

            if not errs:
                return StructureEditResult(ok=True, code=code, reasoning=reasoning, raw_response=raw, errors=[])

        return StructureEditResult(ok=False, code=last_code, reasoning=last_reasoning, raw_response=last_raw, errors=last_errs)

    def load_function(
        self,
        code: str,
        *,
        func_name: str,
        extra_globals: Optional[Dict[str, Any]] = None,
    ):
        """
        Compile + exec the rewritten function and return it.
        You should pass llm/msg from myopto.trace.runtime in extra_globals.
        """
        g: Dict[str, Any] = {}
        if extra_globals:
            g.update(extra_globals)

        compiled = compile(code, filename="<structure_edit>", mode="exec")
        exec(compiled, g, g)

        if func_name not in g:
            raise ValueError(f"Function {func_name!r} not found after exec(). Keys: {list(g.keys())[:20]}")
        return g[func_name]

import inspect
import re
from math import isclose
from typing import Any, Callable, List, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from utils.logs import logger

class MATHBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_model_answer(self, text: str) -> str:
        """
        (updated)
        Robustly extracts content from \boxed{...} allowing for nested braces.
        """
        # 1. Try finding the last \boxed{
        idx = text.rfind("\\boxed{")
        if idx < 0:
            # Fallback: strict last number/sentence logic
            # (You might want to improve this to look for "The answer is: ...")
            sentence_end_pattern = r"(?<!\d)[.!?]\s+"
            sentences = re.split(sentence_end_pattern, text)
            return sentences[-1].strip() if sentences else ""

        # 2. Extract balanced braces starting after \boxed{
        idx += len("\\boxed{")
        count = 1
        for i in range(idx, len(text)):
            if text[i] == "{":
                count += 1
            elif text[i] == "}":
                count -= 1
            
            if count == 0:
                return text[idx:i].strip()
        
        # 3. If braces are unbalanced, return the rest of the string
        return text[idx:].strip()

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)
        cleaned_result = (expected_answer, predicted_answer)
        if self.math_equal(predicted_answer, expected_answer):
            return 1, cleaned_result
        else:
            return 0, cleaned_result

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        # (updated)
        # 1. Exact string match (covers "MAKE" == "MAKE")
        if str(prediction) == str(reference):
            return True

        # 2. Normalize strings (remove LaTeX commands like \text{})
        # Removes \text{...} but keeps the content inside
        # e.g., "\text{Devon}" becomes "Devon"
        # e.g., "5.4 \text{cents}" becomes "5.4 cents"
        def clean_latex(s):
            s = str(s).strip()
            # Remove \text{...} wrapper
            s = re.sub(r"\\text\{([^{}]+)\}", r"\1", s)
            # Remove \$ signs
            s = s.replace(r"\$", "") 
            return s.strip()

        pred_clean = clean_latex(prediction)
        ref_clean = clean_latex(reference)

        # Check equality on cleaned strings
        if pred_clean == ref_clean:
            return True

        # 3. Numeric Comparison (for 5.4 cents vs 5.4)
        # Try to extract just the float value if possible
        try:
            if self.is_digit(pred_clean) and self.is_digit(ref_clean):
                p_val = self.parse_digits(pred_clean)
                r_val = self.parse_digits(ref_clean)
                return isclose(p_val, r_val, abs_tol=1e-3)
        except:
            pass

        # 4. Symbolic Comparison (SymPy)
        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False
        
    def is_digit(self, num):
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def symbolic_equal(self, a, b):
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False

    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, int, float]:
        input_text = problem["problem"]
        expected_output = problem["solution"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            uni_score, extracted_output = self.calculate_score(expected_output, output)

            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=self.get_function_code(self.extract_model_answer),
                )

            return input_text, output, expected_output, uni_score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]

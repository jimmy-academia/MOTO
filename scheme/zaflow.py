"""
Minimal AFLOW & A2FLOW-style baselines.

- AFLOWBaseline:
    * Fixed operator inventory (Generate, Review, Verify, Ensemble, etc.) 
    * Soft-mixed selection over workflows (Eq. 3 in AFLOW). 
    * LLM-based workflow expansion.
    * Execution + evaluation on a validation set.

- A2FlowBaseline:
    * Learns operators from expert traces via 3-stage pipeline: 
        1) Case-based initial operator generation
        2) Operator clustering + preliminary abstraction
        3) Deep extraction for abstract execution operators
    * Adds operator memory Mk as in Eq. (6)–(7). 
    * Uses the same AFLOW-style search loop with learned operators.

You must provide:
    - optimizer_llm(prompt: str) -> str      # for workflow/operator generation
    - executor_llm(prompt: str) -> str       # for running workflows on data
    - expert_data: list of {"question", "solution"} for A2Flow operator learning
    - val_data:    list of {"question", "answer"} for evaluation

This file deliberately avoids any dependency on MetaGPT / AFlow repos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import math
import random
import json

# Type alias for a synchronous LLM call
LLMFn = Callable[[str], str]


# -------------------------------------------------------------------------
# Shared utilities
# -------------------------------------------------------------------------


def normalize_answer(ans: str) -> str:
    """Very simple normalization for math / QA answers."""
    ans = ans.strip()
    if ans.endswith("."):
        ans = ans[:-1]
    ans = " ".join(ans.split())
    return ans


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the last 'Final answer: <...>' from text.
    This matches how AFLOW-style workflows typically report answers.
    """
    lines = text.splitlines()
    for line in reversed(lines):
        if "final answer" in line.lower():
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue
            return normalize_answer(parts[1])
    return None


# -------------------------------------------------------------------------
# Operator abstractions
# -------------------------------------------------------------------------


@dataclass
class Operator:
    """
    Basic operator: a reusable reasoning/action step.

    In AFLOW this is a *predefined* set like:
    Generate, Format, Review & Revise, Ensemble, Test, Programmer, Custom. 
    """
    name: str
    description: str

    def build_prompt(
        self,
        question: str,
        history: str,
        memory: Optional[str] = None,
    ) -> str:
        """
        Build a prompt for this operator.

        For AFLOW we ignore memory (no operator memory).
        For A2Flow we’ll override to include memory.
        """
        mem_part = ""
        if memory:
            mem_part = (
                "\n\n[Operator memory:]\n"
                + memory.strip()
                + "\n"
            )

        return f"""
You are executing step [{self.name}] in a reasoning workflow.

[Operator description]
{self.description}

[Problem]
{question}

[Current reasoning history]
{history.strip() if history.strip() else "(empty)"}
{mem_part}

Follow the operator description carefully. Show your reasoning if relevant.
If this operator is supposed to produce a final answer, end with a line:
Final answer: <answer>
""".strip()


@dataclass
class LearnedOperator(Operator):
    """
    Operator learned by A2Flow from expert data.

    `origin` can store metadata (cluster info, examples, etc.).
    """
    origin: str = ""


# -------------------------------------------------------------------------
# Workflow representation
# -------------------------------------------------------------------------


@dataclass
class WorkflowStep:
    operator_name: str


@dataclass
class Workflow:
    steps: List[WorkflowStep]

    def to_text(self, operator_library: Dict[str, Operator]) -> str:
        """Human-readable / LLM-readable representation of this workflow."""
        lines = []
        for i, step in enumerate(self.steps, start=1):
            op = operator_library[step.operator_name]
            lines.append(f"{i}. [{op.name}] {op.description}")
        return "\n".join(lines)

    @classmethod
    def from_text(cls, text: str, operator_library: Dict[str, Operator]) -> "Workflow":
        """
        Parse a workflow text like:

            1. [Generate] ...
            2. [Review] ...
            3. [Verify] ...

        Only keeps steps whose operator is in operator_library (fixed operator set).
        """
        steps: List[WorkflowStep] = []
        known_names = set(operator_library.keys())
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Look for pattern: [OperatorName]
            if "[" in line and "]" in line:
                inside = line.split("[", 1)[1].split("]", 1)[0].strip()
                if inside in known_names:
                    steps.append(WorkflowStep(operator_name=inside))

        # Fallback: if nothing parsed, keep a trivial one-step workflow
        if not steps:
            # Choose a default operator if present
            default = None
            for candidate in ("Generate", "Custom"):
                if candidate in known_names:
                    default = candidate
                    break
            if default is None:
                default = next(iter(known_names))
            steps = [WorkflowStep(default)]

        return cls(steps=steps)


@dataclass
class WorkflowRecord:
    workflow: Workflow
    score: float
    eval_history: List[float] = field(default_factory=list)
    parent_idx: Optional[int] = None


# -------------------------------------------------------------------------
# AFLOW baseline (fixed operators)
# -------------------------------------------------------------------------


class AFlowBaseline:
    """
    Minimal AFLOW-like workflow search baseline.

    Key ideas from AFLOW: 
      - Represent workflows as sequences of operators.
      - Use a fixed operator set O (Generate, Review&Revise, Ensemble, etc.).
      - Use a variant of MCTS where each tree node is a whole workflow.
      - Selection: soft-mixed probability over workflow scores.
      - Expansion: LLM-based workflow modification.
      - Evaluation: run workflow on validation set multiple times, average score.
    """

    def __init__(
        self,
        *,
        optimizer_llm: LLMFn,
        executor_llm: LLMFn,
        val_data: Sequence[Dict[str, Any]],
        fixed_operators: Optional[Dict[str, Operator]] = None,
        max_iterations: int = 20,
        max_val_examples: int = 32,
        eval_repeats: int = 1,
        lambda_mixed: float = 0.2,
        alpha_softmax: float = 0.4,
        patience: int = 5,
        random_seed: int = 42,
        task_description: str = "math word problems",
    ) -> None:
        self.optimizer_llm = optimizer_llm
        self.executor_llm = executor_llm
        self.val_data = list(val_data)
        if not self.val_data:
            raise ValueError("val_data must be non-empty.")

        self.max_iterations = max_iterations
        self.max_val_examples = max_val_examples
        self.eval_repeats = eval_repeats
        self.lambda_mixed = lambda_mixed
        self.alpha_softmax = alpha_softmax
        self.patience = patience
        self.rng = random.Random(random_seed)
        self.task_description = task_description

        self.operators = fixed_operators or self._default_fixed_operators()

    # -------------------- public API --------------------

    def search(self, initial_workflow: Optional[Workflow] = None) -> Tuple[WorkflowRecord, List[WorkflowRecord]]:
        """
        Run AFLOW-style search and return (best_workflow_record, all_records).
        """
        if initial_workflow is None:
            initial_workflow = self._default_initial_workflow()

        root_score = self._evaluate_workflow(initial_workflow)
        root = WorkflowRecord(workflow=initial_workflow, score=root_score, eval_history=[root_score])
        records: List[WorkflowRecord] = [root]

        best = root
        best_score = root_score
        no_improve = 0

        for it in range(1, self.max_iterations + 1):
            idx = self._select_index(records)
            parent = records[idx]

            child_wf = self._expand_workflow(parent.workflow, parent.score)
            child_score = self._evaluate_workflow(child_wf)

            child = WorkflowRecord(
                workflow=child_wf,
                score=child_score,
                eval_history=[child_score],
                parent_idx=idx,
            )
            records.append(child)

            if child_score > best_score:
                best_score = child_score
                best = child
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"[AFLOW-mini] iter={it} parent_idx={idx} "
                f"parent_score={parent.score:.3f} child_score={child_score:.3f} "
                f"best_score={best_score:.3f}"
            )

            if no_improve >= self.patience:
                print(f"[AFLOW-mini] Early stop: no improvement for {self.patience} iterations.")
                break

        return best, records

    # -------------------- operator inventory --------------------

    def _default_fixed_operators(self) -> Dict[str, Operator]:
        """
        Approximate AFLOW's fixed operator set: 
          (1) Generate, (2) Format, (3) Review&Revise,
          (4) Ensemble, (5) Test, (6) Programmer, (7) Custom.
        """
        ops = {
            "Generate": Operator(
                name="Generate",
                description="Generate an initial solution with detailed reasoning.",
            ),
            "Format": Operator(
                name="Format",
                description="Clean up and format the solution clearly.",
            ),
            "Review&Revise": Operator(
                name="Review&Revise",
                description="Critically review the reasoning and revise any errors.",
            ),
            "Ensemble": Operator(
                name="Ensemble",
                description="Combine multiple candidate solutions and pick the most consistent one.",
            ),
            "Test": Operator(
                name="Test",
                description="Test the solution against the problem statement and constraints.",
            ),
            "Programmer": Operator(
                name="Programmer",
                description="If useful, write and mentally execute pseudo-code or code to verify the solution.",
            ),
            "Custom": Operator(
                name="Custom",
                description="Perform any basic reasoning step that advances the solution.",
            ),
        }
        return ops

    def _default_initial_workflow(self) -> Workflow:
        """
        Simple seed workflow akin to manual math workflows AFLOW starts from. 
        """
        steps = [
            WorkflowStep("Generate"),
            WorkflowStep("Review&Revise"),
            WorkflowStep("Test"),
        ]
        return Workflow(steps=steps)

    # -------------------- selection / expansion / evaluation --------------------

    def _select_index(self, records: List[WorkflowRecord]) -> int:
        """
        Soft-mixed probability selection from AFLOW Eq. (3): 

            P(i) = λ * 1/n + (1-λ) * softmax_i(α * (s_i - s_max))
        """
        n = len(records)
        if n == 1:
            return 0

        scores = [rec.score for rec in records]
        s_max = max(scores)

        exp_vals = [math.exp(self.alpha_softmax * (s - s_max)) for s in scores]
        denom = sum(exp_vals) or 1.0
        probs_score = [v / denom for v in exp_vals]

        probs = [
            self.lambda_mixed * (1.0 / n) + (1.0 - self.lambda_mixed) * ps
            for ps in probs_score
        ]

        idxs = list(range(n))
        return self.rng.choices(idxs, weights=probs, k=1)[0]

    def _expand_workflow(self, workflow: Workflow, parent_score: float) -> Workflow:
        """
        LLM-based workflow expansion (optimizer). 

        We show the current workflow and its score, plus the fixed operator set,
        and ask the optimizer to output a new workflow in our text format.
        """
        wf_text = workflow.to_text(self.operators)
        ops_desc = "\n".join(
            f"- [{op.name}]: {op.description}" for op in self.operators.values()
        )

        prompt = f"""
You are an expert workflow optimizer for LLM-based reasoning.

Task type: {self.task_description}

You have a FIXED operator inventory:
{ops_desc}

Current workflow (score on validation set ≈ {parent_score:.3f}):

\"\"\"workflow
{wf_text}
\"\"\"

Goal:
- Propose ONE improved workflow using ONLY the operators from the inventory.
- You may add, remove, or reorder steps.
- You must output numbered steps, each in the form:
  k. [OperatorName] short description

Constraints:
- OperatorName MUST be exactly one of: {", ".join(self.operators.keys())}
- Do NOT introduce new operator names.
- The last step must ensure the final answer is expressed as:
  Final answer: <answer>

Output ONLY the new workflow steps, nothing else.
"""
        raw = self.optimizer_llm(prompt)
        return Workflow.from_text(raw, self.operators)

    def _evaluate_workflow(self, workflow: Workflow) -> float:
        """
        Execute workflow on a subset of validation data and return mean accuracy in [0,1].
        """
        if len(self.val_data) <= self.max_val_examples:
            eval_set = list(self.val_data)
        else:
            eval_set = self.rng.sample(self.val_data, self.max_val_examples)

        scores = []
        for _ in range(self.eval_repeats):
            correct = 0.0
            for ex in eval_set:
                q = str(ex["question"])
                gold = normalize_answer(str(ex["answer"]))

                pred = self._run_workflow_on_example(workflow, q)
                if pred is None:
                    s = 0.0
                else:
                    s = 1.0 if gold and gold in pred else 0.0
                correct += s
            scores.append(correct / len(eval_set))
        return sum(scores) / len(scores)

    def _run_workflow_on_example(self, workflow: Workflow, question: str) -> Optional[str]:
        """
        Sequentially execute each operator in the workflow with the executor LLM.

        AFLOW baseline has no operator memory; we just pass the accumulated history.
        """
        history = ""
        for step in workflow.steps:
            op = self.operators[step.operator_name]
            prompt = op.build_prompt(question=question, history=history, memory=None)
            out = self.executor_llm(prompt)
            history += f"\n\n[{op.name}] OUTPUT:\n{out}"

        return extract_final_answer(history)


# -------------------------------------------------------------------------
# A2FLOW: learn operators, add memory, then AFLOW-style search
# -------------------------------------------------------------------------


class A2FlowOperatorLearner:
    """
    Minimal A2Flow-style operator learner. 

    Pipeline:
      1) Case-based initial operator generation (Eq. 3). 
      2) Operator clustering & preliminary abstraction (Eq. 4). 
      3) Deep extraction for abstract execution operators with multi-path CoT (Eq. 5). 

    All steps rely on an LLM and JSON-formatted outputs for parsing.
    """

    def __init__(
        self,
        llm: LLMFn,
        task_description: str,
        random_seed: int = 42,
        n_cases: int = 32,
        n_paths: int = 3,
    ) -> None:
        self.llm = llm
        self.task_description = task_description
        self.rng = random.Random(random_seed)
        self.n_cases = n_cases
        self.n_paths = n_paths

    def learn_operators(
        self,
        expert_data: Sequence[Dict[str, Any]],
    ) -> Dict[str, LearnedOperator]:
        """
        Main entry: returns a dict[name, LearnedOperator].
        expert_data items should at least have:
           {"question": ..., "solution": ...} where solution includes reasoning.
        """
        if not expert_data:
            raise ValueError("expert_data must be non-empty")

        chosen = list(expert_data)
        if len(chosen) > self.n_cases:
            chosen = self.rng.sample(chosen, self.n_cases)

        raw_ops = self._case_based_initial_generation(chosen)
        prelim = self._cluster_and_abstract(raw_ops)
        learned = self._deep_extract_abstract_operators(prelim)
        return learned

    # -------- Stage 1: Case-based initial operator generation --------

    def _case_based_initial_generation(
        self,
        cases: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate initial case-aware operators O^(e) as in Eq. (3). 

        We ask the LLM to extract a *list of operators per case* and return them as JSON:

            [
              {"name": "...", "description": "...", "example_case_id": "..."},
              ...
            ]
        """
        all_ops: List[Dict[str, Any]] = []

        for idx, ex in enumerate(cases):
            q = str(ex.get("question", ""))
            sol = str(ex.get("solution", ""))

            prompt = f"""
You are analyzing an expert solution for a {self.task_description} task.

[Problem]
{q}

[Expert solution with reasoning]
{sol}

Task:
- Decompose this solution into a sequence of reusable *operators*.
- Each operator should be an atomic reasoning or action step that could be reused in other workflows.
- For each operator, produce:
    - "name": short name (few words).
    - "description": detailed description of what this operator does.
- Do NOT include case-specific numbers; keep descriptions general.

Output STRICTLY as JSON list, e.g.:

[
  {{"name": "IdentifyKnowns", "description": "Identify all known quantities from the problem."}},
  {{"name": "SetUpEquation", "description": "Formulate the key equation based on problem relationships."}}
]
"""
            raw = self.llm(prompt)
            try:
                operators = json.loads(raw)
                if isinstance(operators, list):
                    for op in operators:
                        op["example_case_id"] = idx
                        all_ops.append(op)
            except Exception:
                # If parsing fails, skip this case.
                continue

        return all_ops

    # -------- Stage 2: Clustering & preliminary abstraction --------

    def _cluster_and_abstract(
        self,
        raw_ops: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Cluster similar operators and produce preliminary abstractions O^(a). 

        Returns list of dicts:
            {"name": str, "description": str, "member_ops": [raw_name1, ...]}
        """
        if not raw_ops:
            return []

        # Build a concise summary list for the LLM
        ops_summary = [
            f"- {op.get('name', f'op_{i}')} : {op.get('description','')}"
            for i, op in enumerate(raw_ops)
        ]
        summary_text = "\n".join(ops_summary)

        prompt = f"""
You are clustering operator descriptions into functional groups.

[Task description]
{self.task_description}

[Raw operators]
{summary_text}

Goal:
- Group similar operators into clusters based on functionality.
- For each cluster, define:
    - "name": short abstract operator name
    - "description": what it does in general
    - "members": list of raw operator names in this cluster

Output STRICTLY as JSON list, e.g.:

[
  {{
    "name": "VerifyConsistency",
    "description": "Check logical and numerical consistency of the current solution.",
    "members": ["CheckContradictions", "CheckEquation", "CheckNumericConsistency"]
  }},
  ...
]
"""
        raw = self.llm(prompt)
        prelim: List[Dict[str, Any]] = []
        try:
            prelim = json.loads(raw)
            if not isinstance(prelim, list):
                prelim = []
        except Exception:
            prelim = []

        return prelim

    # -------- Stage 3: Deep extraction for abstract execution operators --------

    def _deep_extract_abstract_operators(
        self,
        prelim_ops: List[Dict[str, Any]],
    ) -> Dict[str, LearnedOperator]:
        """
        Multi-path Long CoT refinement to get final operators O^(t). 

        For each preliminary operator:
          - Run m paths of CoT refinements (o1 -> o2 -> o3).
          - Aggregate final o3's into a single abstract operator with LLM.
        """
        learned: Dict[str, LearnedOperator] = {}

        for op in prelim_ops:
            base_name = op.get("name", "Operator")
            base_desc = op.get("description", "")

            candidates: List[str] = []  # final o3's (strings)

            for path_id in range(self.n_paths):
                # Step 1: initial abstract operator
                prompt1 = f"""
You are refining an abstract operator for {self.task_description}.

[Preliminary operator]
Name: {base_name}
Description: {base_desc}

Step 1:
- Propose a clearer and more precise description for this operator.
- Focus on what inputs it takes and what outputs it produces.
Output ONLY the new description as plain text.
"""
                o1 = self.llm(prompt1).strip()

                # Step 2: CoT refinement
                prompt2 = f"""
You are further refining an abstract operator description.

[Task]
{self.task_description}

[Previous draft o1]
{o1}

Step 2:
- Think step by step (chain of thought) about how this operator will be executed as a single LLM call.
- Improve the description to make inputs/outputs and behavior explicit.
Output ONLY the improved description o2, as plain text.
"""
                o2 = self.llm(prompt2).strip()

                # Step 3: final refinement
                prompt3 = f"""
You are finalizing an abstract operator description.

[Task]
{self.task_description}

[Preliminary]
{base_desc}

[Refinement o1]
{o1}

[Refinement o2]
{o2}

Step 3:
- Combine all information and produce a final, concise but precise description.
- The operator must be usable as a single LLM call, with clear inputs and outputs.

Output ONLY the final description o3, as plain text.
"""
                o3 = self.llm(prompt3).strip()
                candidates.append(o3)

            # Aggregate all o3 into one final operator via LLM
            agg_prompt = f"""
You are aggregating multiple candidate abstract operator descriptions into ONE final operator.

[Task]
{self.task_description}

[Candidates]
{json.dumps(candidates, indent=2)}

Goal:
- Produce a single operator description that captures the common core behavior.
- It must:
    - Specify what input it takes (e.g., current state, memory).
    - Specify what output it produces (updated state).
    - Be generic enough for many problems in this task.

Output STRICTLY as JSON:
{{
  "name": "{base_name}",
  "description": "<final operator description>"
}}
"""
            raw = self.llm(agg_prompt)
            try:
                obj = json.loads(raw)
                name = obj.get("name", base_name)
                desc = obj.get("description", base_desc)
            except Exception:
                name, desc = base_name, base_desc

            learned[name] = LearnedOperator(
                name=name,
                description=desc,
                origin=f"prelim:{base_name}",
            )

        return learned


# -------------------------------------------------------------------------
# A2Flow baseline: AFLOW search + learned operators + memory
# -------------------------------------------------------------------------


class A2FlowBaseline(AFlowBaseline):
    """
    Minimal A2Flow-style baseline:

      - First learn operators from expert data via A2FlowOperatorLearner. 
      - Then run AFLOW-style search over workflows that use these operators.
      - During execution, maintain an operator memory Mk that accumulates all past outputs. 
    """

    def __init__(
        self,
        *,
        optimizer_llm: LLMFn,
        executor_llm: LLMFn,
        val_data: Sequence[Dict[str, Any]],
        expert_data: Sequence[Dict[str, Any]],
        task_description: str = "math word problems",
        random_seed: int = 42,
        # AFlow params
        max_iterations: int = 20,
        max_val_examples: int = 32,
        eval_repeats: int = 1,
        lambda_mixed: float = 0.2,
        alpha_softmax: float = 0.4,
        patience: int = 5,
        # A2Flow operator learning params
        n_cases: int = 32,
        n_paths: int = 3,
    ) -> None:
        # Learn operator inventory first
        learner = A2FlowOperatorLearner(
            llm=optimizer_llm,
            task_description=task_description,
            random_seed=random_seed,
            n_cases=n_cases,
            n_paths=n_paths,
        )
        learned_ops = learner.learn_operators(expert_data)

        super().__init__(
            optimizer_llm=optimizer_llm,
            executor_llm=executor_llm,
            val_data=val_data,
            fixed_operators=learned_ops,  # replace fixed set with learned operators
            max_iterations=max_iterations,
            max_val_examples=max_val_examples,
            eval_repeats=eval_repeats,
            lambda_mixed=lambda_mixed,
            alpha_softmax=alpha_softmax,
            patience=patience,
            random_seed=random_seed,
            task_description=task_description,
        )

    # Override execution to add operator memory Mk as in Eq. (6)-(7). 
    def _run_workflow_on_example(self, workflow: Workflow, question: str) -> Optional[str]:
        """
        Execute with operator memory:

            ok = fk(input_k, Pk, M_{k-1})
            M_k = M_{k-1} ∪ {ok}
        """
        history = ""
        memory_items: List[str] = []

        for step in workflow.steps:
            op = self.operators[step.operator_name]
            memory_text = "\n".join(memory_items) if memory_items else ""

            prompt = op.build_prompt(
                question=question,
                history=history,
                memory=memory_text or None,
            )
            out = self.executor_llm(prompt)

            # Update history & memory
            history += f"\n\n[{op.name}] OUTPUT:\n{out}"
            memory_items.append(f"[{op.name}] {out}")

        return extract_final_answer(history)


# -------------------------------------------------------------------------
# Example usage (replace dummy_llm with a real client)
# -------------------------------------------------------------------------


if __name__ == "__main__":
    # Dummy LLM for testing wiring only (always answers "0").
    def dummy_llm(prompt: str) -> str:
        return "Reasoning...\nFinal answer: 0"

    # Tiny fake expert data for A2Flow operator learning
    expert_data = [
        {
            "question": "What is 2 + 2?",
            "solution": "First identify the numbers (2 and 2). Then add them to get 4. Finally, check that 4 is correct."
        },
        {
            "question": "Tom has 3 apples and buys 2 more. How many apples?",
            "solution": "Identify Tom's initial apples (3). Identify the apples he buys (2). Add 3 and 2 to get 5. Conclude 5."
        },
    ]

    # Tiny validation data
    val_data = [
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "3 + 2 = ?", "answer": "5"},
    ]

    print("=== AFLOW baseline (fixed operators) ===")
    aflow = AFlowBaseline(
        optimizer_llm=dummy_llm,
        executor_llm=dummy_llm,
        val_data=val_data,
        max_iterations=3,
        max_val_examples=2,
    )
    best_af, _ = aflow.search()
    print("Best AFLOW score:", best_af.score)
    print("Best AFLOW workflow:\n", best_af.workflow.to_text(aflow.operators))

    print("\n=== A2FLOW baseline (learned operators) ===")
    a2flow = A2FlowBaseline(
        optimizer_llm=dummy_llm,
        executor_llm=dummy_llm,
        val_data=val_data,
        expert_data=expert_data,
        max_iterations=3,
        max_val_examples=2,
    )
    best_a2, _ = a2flow.search()
    print("Best A2FLOW score:", best_a2.score)
    print("Best A2FLOW workflow:\n", best_a2.workflow.to_text(a2flow.operators))

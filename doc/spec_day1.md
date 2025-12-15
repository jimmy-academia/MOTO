# Day 1 Spec — Meta-Optimizer for Code-based LLM Workflows (No-GT Test-Time)
Date: 2025-12-15
Version: v0.1 (locked for Day1–Day3)

## 0) One-sentence Goal
We meta-learn a **workflow optimizer** that can **repair/optimize an executable Python LLM-workflow at test time** using only **deployment-available structured verifiers + execution traces (no ground truth)**, under a strict cost/budget constraint (edge/weak optimizer).

---

## 1) Problem Motivation (the “why this is hard”)
Most prompt/workflow optimization work either:
1) optimizes a fixed prompt/workflow **over a whole dataset** (dataset-level optimum), or
2) does test-time adaptation in an **open-loop** way (input-only semantic specialization), or
3) relies on a strong optimizer LLM (expensive, not deployable on edge/on-prem).

In real deployments (corporate/contract/request-fulfillment), we often have:
- **No ground truth** at test time,
- Only **structured compliance/consistency/evidence checks** + runtime traces,
- Strict constraints: **cost, latency, privacy**, often requiring **small/edge models**.

We therefore study **closed-loop test-time workflow repair** driven by verifiers and traces, not by ground truth.

---

## 2) Objects & Setting
### 2.1 Task Distribution
Tasks are sampled from a distribution p(T). Each task instance provides input x.

**MVP (Day1–Day10)**: coding tasks with unit tests (HumanEval or MBPP).
- x = problem statement (+ hidden tests for evaluation only).
- Deployment-available signals: compile/runtime errors, unit-test results (pass/fail per test).

### 2.2 Workflow Program (what we optimize)
A workflow is an **executable Python function** π that orchestrates multiple LLM calls:

Example (illustrative):
```python
def solve(x):
    plan = llm(f"Plan: {x}")
    code = llm(f"Write code based on plan:\n{plan}")
    return code
```

Execution produces:
- output y (e.g., code)
- trace τ (LLM call logs, token/cost, errors, tool outputs)

### 2.3 Deployment Feedback (no GT)
At test time, we do NOT have ground truth y*.
We have a verifier/feedback function F that returns structured feedback:

`b = F(x, π, τ, y)`

b includes:
- compile_success / runtime_error_type
- passed_tests, failed_tests (list)
- constraint violations (e.g., format/schema)
- scalar score S_F for selection

**Important**: F must be deployable (no GT access).

---

## 3) Method Overview (Bilevel Meta-Optimization)

We learn an optimizer O that can optimize workflows at test time without GT.

### 3.1 Inner loop (deployable, no GT)

Given x and a seed workflow π0:

Repeat k=0..K-1:

1. Execute πk → (y_k, τ_k)
2. Verifier F → b_k (structured + scalar score S_k)
3. Weak optimizer (edge model) proposes candidate patches:
{Δπ_k^1, …, Δπ_k^m}
4. Apply patches → candidates {π_k^1, …, π_k^m}
5. Select best candidate using verifier score S_F (or comparator) → π_{k+1}. Stop if verifier passes or budget exhausted.

We support two edit types:
- Inner edits (prompt edits): rewrite prompts inside existing LLM calls.
- Outer edits (structure edits): add/remove/reorder steps; add conditional/loop; add verification step.

### 3.2 Outer loop (train-time only, uses GT for meta-learning)

Outer loop uses hidden ground-truth evaluation U(π; x) to meta-learn:
- seed workflow template (π0 prior),
- meta prompt / knowledge base for proposing patches,
- verifier aggregation / weighting (to increase selection reliability),
- patch operator set / constraints (to make weak optimizer succeed).

Outer objective:
maximize E[ U(π_K; x) ] while inner loop uses only F (no GT).

**Implementation in Day1–Day10:**
Outer loop will be implemented as black-box meta-search over discrete configs
(seed template choice, patch-template prompts, verifier weights),
not a full differentiable bilevel system.

---

## 4) Edge / Weak Optimizer Constraint (Core Novelty Axis)

Inner-loop optimizer must run on a weak model (edge/on-prem, e.g., 7B).
Therefore:
- feedback must be structured and localized (step-level hints, failing checks),
- patch space must be constrained (operator set), to reduce search.

We will simulate edge in early experiments by:
- restricting optimizer context length,
- restricting patch operators,
- (optionally) using a smaller model / cheaper endpoint.

---

## 5) Verifier Design (MVP)
### 5.1 Structured checks

For coding tasks:

- compile check (syntax errors)
- unit tests (passed/failed list)
- runtime error type (exception category)
- optional: lint/format check (later)

### 5.2 Scalar score for selection (S_F)

We define:
S_F = passed_ratio - α * syntax_error - β * runtime_error - λ * cost

This is used only for selecting between candidate patches.

---

## 6) Mini-Theory Contribution (γ + κ)

We characterize when no-GT test-time optimization is feasible.

### 6.1 Verifier alignment γ (pairwise ranking advantage)

Given two candidate workflows πa, πb on x:

γ = P[ sign(S_F(πa)-S_F(πb)) = sign(U(πa)-U(πb)) ] - 1/2

If γ ≈ 0, no verifier-guided selection can reliably improve without GT.
We expect required comparisons/ensembling to scale ~ 1/γ^2 to achieve fixed selection reliability.

### 6.2 Proposal quality κ (weak optimizer coverage)

In each iteration, the weak optimizer proposes m candidates.
κ = P[ ∃ candidate π' with U(π') ≥ U(πk) + Δ ]

Meta-learning should increase both:

- κ (weak optimizer proposes useful candidates),
- γ (verifier selection becomes more aligned).

---

## 7) Evaluation & Metrics (MVP)

Primary:
- pass@1 (or solve rate) on test split
- success under budget B (max LLM calls / token cost)
- average #iterations / #LLM calls / token cost per solved instance

Secondary:
- γ measured on validation pairs (verifier alignment)
- κ measured by “best-of-m candidates improves GT utility”
- patch size / stability (AST diff size; compilation success rate)

---

## 8) Baselines (Day1–Day10)

- B0: single-shot LLM code generation
- B1: fixed workflow: generate → run tests → single repair prompt → run tests
- B2: prompt-only refinement loop (no structure edits)
- B3 (optional if time): candidate-sampling + verifier selection (self-consistency)

We will compare:

- prompt-only vs structure+prompt
- meta-initialized vs non-meta initialization

# `paper/outline.md`

```md
# Paper Outline — Meta-Optimizer for Code-based LLM Workflows (No-GT Test-Time)

## 1. Introduction
- Deployment reality: test-time has no ground truth; only verifiers/traces + strict budgets (cost/privacy/edge).
- Goal: meta-learn a workflow optimizer that can perform closed-loop test-time workflow repair.
- Contributions: (i) problem setting, (ii) meta-optimizer method, (iii) γ+κ analysis and scaling, (iv) empirical cost–accuracy gains.

## 2. Problem Setup
- Task distribution p(T), instance input x.
- Workflow program π (executable Python; multi-LLM calls; control flow).
- Trace τ and deployable verifier F (no GT).
- Objective under budget B; GT only used for offline evaluation/meta-training.

## 3. Method
- Inner loop: weak optimizer proposes patches; verifier selects; closed-loop iterations.
- Outer loop: black-box bilevel meta-search for seed templates, patch templates, verifier weights.
- Edge constraint: optimizer runs on weak model; feedback/patch space designed accordingly.

## 4. Mini-Theory / Analysis
- Define verifier alignment γ; necessity of γ>0.
- Define proposal quality κ; necessity of κ>0.
- Cost scaling intuition: selection reliability ~ 1/γ^2; overall improvement requires both γ and κ.
- Predictions to validate experimentally.

## 5. Experiments (MVP: coding with tests)
- Dataset: HumanEval (or MBPP) split train/val/test.
- Verifiers: unit tests + error diagnostics + cost.
- Baselines: single-shot; fixed workflow; prompt-only refinement; candidate sampling.
- Metrics: pass@1, cost, iterations; measure γ and κ; ablations.

## 6. Related Work (must mention explicitly)
- Workflow meta-learning / bilevel workflow optimization: AdaptFlow (test-time semantic specialization; strong optimizer).
- Inference-time optimization without GT: metaTextGrad (solution/prompt-level; evaluator-based).
- Prompt refinement without GT: ProRefine (prompt-level).
- Code-based workflow search: AFLOW (per-task search heavy).
- Memory-based test-time learning: Evo-Memory (memory evolution, not workflow program repair).

## 7. Discussion / Limitations
- Verifier design assumptions; failure when γ~0.
- Operator set limitations; safety constraints for executable code.
- Extension to corporate/contract/request-fulfillment workflows.

## 8. Conclusion
- Summary + takeaways: closed-loop verifier-driven test-time workflow repair is feasible under weak optimizer when meta-learning improves γ and κ.

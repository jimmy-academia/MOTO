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

# Master Plan — Clover (Context-Conditioned Workflow Repair under No-GT Test-Time)

Last updated: 2025-12-18 (Day 4 / Day 10)

## 0) North Star (one sentence)
Current LLM system are brittle because they 1) (prompt schemes) set and forget, or 2) (prompt optimization schemes) assume the training distribution (General) matches the deployment environment (Local). However, relying on 'Generalist' capabilities fails when edge cases drift from training data, and we cannot wait for cloud-based fine-tuning with ground truth (which requires data collection process) to fix it. Clover solves this by proving we can optimize workflows locally using only intrinsic proxy signals, challenging the belief that Ground Truth is required for reliable adaptation.

Clover is a workflow-optimizer system that can adapt an executable LLM workflow to a *new local context* at test time **without ground truth**, using only deployable feedback signals (context + traces), under strict budget and edge-model constraints; and meta-optimize the adaptation tools offline.

## 1) System Setup
### Methodological Framework
> CLOVER = CLOsed-loop VERifier-driven workflow optimization = MOTO + ECWO
- Innerloop -- ECWO (Edge-Cloud Workflow Optimzation):
  - Given: `initial program \pi_0, optimizer meta prompt M, evaluation program for verifier \epsilon`
  - (Optimize + Verifier) optimize solution program without ground truth
  
- Outerloop -- MOTO (Meta Optimizer-Tuning Operator)
  - Optimize parameters (`initial program \pi_0, optimizer meta prompt M, evaluation program for verifier \epsilon`) for the Innerloop with ground truth


### LLM roles
- Executor 
  - on edge device 
  - operates solution program \pi using input x; returns prediction y; the process produces trace \tau

- Verifier
  - on edge device
  - operates evaluation program \epsilon using \pi, x, y, \tau, c; returns feedback b; the proces produces verifier trace \nu 
  - (verification goal: structure alignment with context, local blames)

- Optimizer
  - on edge device
  - operates optimization process \omega(meta prompt M) using (multiple examples of) \pi, x, y, \tau, c, b; returns updated solution program \pi'; the process optimizer trace \mu
  - (optimization goal: performance, token cost)
  - Current approach: beam search + (aiming for local structural edit)

- Meta-Optimizer
  - on cloud
  - operates meta-optimization process \zeta using (multiple examples of) \pi, x, y, \tau, c, b, \nu, \mu, and ground truth y*; returns updated `initial program \pi_0, optimizer meta prompt M, evaluation program for verifier \epsilon`

> future extention: solution program \pi and optimization process involves multiple edge+cloud executors and optimizers with different privacy constraint.

## 2) Methodology

### Current Designs

- Inner Loop (deployable optimization process)
  - Candidate representation = {workflow_code, id, parent_id, mutation_type, score, feedback, trace_ir, cost, crash_flag}
  - Beam inner-loop algorithm (concept below)
  - Stop condition
    - Pass condition: constraint_flags satisfied OR score >= threshold
    - Budget: max iterations, max token/cost, max wall-time

#### Beam algorithm
Input: (c, x), budget B, beam size K, iterations T
1) Initialize beam with seed workflow(s) (+ retrieved templates from L)
2) For t in 1..T:
   a) Execute each candidate -> (y, τ)
   b) Compute feedback b = F(c, x, W, y, τ)
   c) Score candidate for selection; hard-filter crashes if needed
   d) Prune to top-K
   e) Expand survivors: propose edits via operators guided by (b, P_meta, retrieved cases)
   f) Stop if pass-condition or budget exhausted

### Future/Working ideas

- Feedback b is structured and must be produced without GT:
  - observations (what happened)
  - constraint_flags (checkable constraints derived from c and runtime)
  - blame (which step/operator likely responsible)
  - suggestion (what operator to try next)
  - score (scalar for selection/pruning)
- Feedback contains structural diagnosis (structure vs prompt error) +
score for theoretical analysis
- Initial program \pi_0 define module portions of "workflow"
- Meta prompt: 
  - Defines allowed operators, constraints, "how to write solution program" and "how to patch"
  - Evolving memory of Case library
- Optimization process:
  - versioning for backtracking

#### Other notes
```
## 6) Feedback Function Design (critical to avoid drift)
Principles:
- Separate **scoring** (conservative, anchored in checkable signals) from **suggestions** (creative, LLM-based).
- Always output a structured object; never rely on free-form critique alone.

Template fields:
- observations: runtime errors, violated constraints, suspicious patterns
- constraint_flags: derived from context rules + format/schema checks + trace checks
- blame: step/module/operator class
- suggestion: next operator + rationale
- score: scalar combining constraint satisfaction + cost penalties + stability penalties

## 7) Edit Language (operators) and “locality”
### 7.1 Why locality exists here
Locality is not “the workflow is a tree.” Locality means:
- edits have **sparse support**: touch few steps/nodes
- edits are attributable and budgetable

### 7.2 Operator set (initial MVP)
Prompt-level:
- EditPrompt(step_i)
Structural:
- InsertCheck(after step_i)
- InsertRetryGuard(step_i, condition, max_tries)
- SplitStep(step_i -> step_i_a + step_i_b)
- RewireIO(step_i uses structured state instead of raw text)

Budgets:
- max steps changed per iteration
- max lines/AST nodes changed
- max new LLM calls inserted

---

## 8) Outer Loop (textual meta-learning)
Outer loop uses GT only to detect systematic mismatch between:
- deployable score/critic opinions vs true utility

Update targets:
A) Critic prompt (fix false positives / false negatives)
B) Meta-prompt constitution (fix stagnation / repeated failure modes)
C) Seed/library (distill slow-success workflows into reusable templates)

Triggers:
- False positive: high score but wrong by GT -> tighten critic/checks
- Stagnation: no improvement after N iters -> add rule / restrict bad edits
- Success-but-slow: solved with many steps -> distill into better seed/template
```


## 3) Theory 

Selection alignment: γ
- How often the selection signal ranks better candidates higher (measured offline using GT)

Proposal quality: κ
- Probability that among m proposed edits, at least one improves true utility by Δ

Optional: blame accuracy β
- Probability feedback points to the right step/operator class (measured via targeted-vs-random edits)
=> conterfactual edits

Predictions:
- Outer loop increases γ and κ
- Structural operators improve κ more for weak/edge models than strong models


### arguments:

> It is considerred that "Self-Correction without Ground Truth is just hallucination." (ref: The Large Language Models Cannot Self-Correct Reasoning Yet). However, while unconstrained self-correction degrades into hallucination, structured workflow repair with context target is different. 

> "The community correctly identifies that self-correction without Ground Truth is unstable because the model lacks a reference signal—it is effectively trying to pull itself up by its own bootstraps (hallucination).
> However, Clover challenges the assumption that only Ground Truth can serve as this reference. In Enterprise/Edge settings, the Local Context (constraints, schema, privacy rules) acts as a Concrete Target State.
> Clover does not ask the LLM to 'make it better' (unbounded); it asks the LLM to 'minimize the distance to the Target' (bounded). By defining optimization as distance-minimization to a context-derived multi-objective, we convert an unstable generation problem into a stable search problem, and achieve positive $\gamma$ (selection alignment) even without GT."



---

## 4) Focus plan:

### 4.1 Experimental axes (must have)
- Prompt-only vs structure+prompt
- With vs without outer-loop updates
- Weak vs strong model (optimizer and/or executor)

### 4.2 Metrics
- Success rate under budget
- Mean iterations / cost per success
- γ and κ estimated from trajectories
- Edit locality stats (#steps touched, operator usage)

### 4.3 Candidate task families (choose one primary)
Primary should have:
- clear evaluation offline
- deployable feedback at test time (even if imperfect)
- context-conditioned rules matter


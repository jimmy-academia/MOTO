# method

Here’s the methodology as a clean **3-role architecture** (outer-loop meta-optimizer + deployable optimizer + executor), matching your “no-GT test-time, trace+verifier driven” setting.

## 1) Meta-Optimizer (Outer Loop, training-time only)

**Goal:** Prepare an optimizer that can reliably improve workflows at test time **without ground truth**, under a budget.

**Inputs (training tasks):**

* A set of contexts/policies (c), inputs (x), and *hidden* ground truth (y^*) (used only for offline scoring).
* A deployable verifier (F(\cdot)) that produces structured feedback without (y^*).

**What it learns/tunes (typical outputs):**

* **Optimizer initialization**: patch prompt templates, edit heuristics, operator priors.
* **Seed workflow templates** (\pi_0(c)): “starting programs” the optimizer can repair efficiently.
* **Verifier aggregation / selection policy**: how to convert multi-field feedback into a scalar or preference used for choosing patches.
* (Optionally) **candidate-generation strategy**: how many candidates per round, diversity controls, stop rules.

**Objective:**
Maximize true utility (U(\pi; x, c)) (computed with hidden (y^*) offline) while minimizing cost (tokens/latency/iterations), but ensuring the inner loop uses only deployable signals.
---

## 2) Optimizer (Inner Loop, test-time deployable; typically weaker/edge model)

**Goal:** Given a *new context* (c) (unseen policy), adapt/repair an executable workflow program using only **execution traces + verifier feedback**.

**Inputs (per request):**

* Context/policy (c)
* Input (x)
* Current workflow (\pi_k)
* Execution trace (\tau_k) from running (\pi_k)
* Structured verifier feedback (b_k = F(x, c, \pi_k, \tau_k, y_k))
* Budget (B) (iterations / tokens / latency)

**Core loop (conceptually):**

1. Run executor with (\pi_k) → get output (y_k) and trace (\tau_k)
2. Run verifier → get structured feedback (b_k) and a selection score (or preferences)
3. Propose patches (\Delta \pi_k) (prompt edits + limited structure edits)
4. Select best candidate via verifier-derived score (no GT), update (\pi_{k+1})
5. Stop when verifier passes or budget is exhausted

**Patch types (typical):**

* Prompt edits (instruction tightening, error-driven repair prompts)
* Structure edits (add retry / self-consistency / re-run tests loop; reorder steps; add tool call; adjust temperature/candidate count)

---

## 3) Executor (Runs the workflow program; can be a different model)

**Goal:** Execute the current workflow (\pi_k) on ((c, x)) to produce output (y_k) plus an auditable trace.

**Responsibilities:**

* Run the Python workflow program (multi-step LLM/tool calls + control flow)
* Log a **trace** per step: prompts, summaries of inputs/outputs, errors, branch path, and **usage/cost/latency**
* Return ((y_k, \tau_k)) for verifier + optimizer

**Key design choice:** Executor can be the “production-capable” model, while optimizer can be smaller/cheaper—this is where your **edge constraint** becomes a first-class contribution.

---

## Put together (one-line pseudocode)

[
\pi_{k+1} \leftarrow \text{Select}\big({\text{Patch}(\pi_k; c, x, \tau_k, b_k)}\ \text{using}\ F\big),\quad
(y_k,\tau_k) \leftarrow \text{Execute}(\pi_k; c,x),\quad
b_k \leftarrow F(c,x,\pi_k,\tau_k,y_k)
]

If you want, I can compress this into a **paper-ready paragraph** (Method overview) plus a **figure caption** describing the three roles and signal flow.

Here is the comprehensive summary of the **Clover Framework** as we have defined it. This specification unifies the **Generative Beam Search** (Inner Loop) with the **Textual Meta-Learning** (Outer Loop).

### Part 1: The Grand Architecture

**The Problem:** We need to deploy an AI workflow to an edge device that must adapt to a specific local context (e.g., specific enterprise policy) without ground truth supervision.

**The Solution:** A Bilevel Optimization framework where the **Outer Loop** (running on a server with ground truth) trains the "adaptation tools" that the **Inner Loop** (running on the edge) uses to solve the problem.

---

### Part 2: The Inner Loop (Test-Time Adaptation)

**Engine:** `BeamInnerLoopEngine`
**Method:** **Generative Beam Search** (Generative Optimization)

Instead of evolving a single workflow, the Inner Loop maintains a **Population (Beam)** of K candidate workflows.

1. **Initialization:**
* The beam starts with a `seed_workflow` (provided by Outer Loop).
* *Library Integration:* The engine can also retrieve N relevant templates from the **Case Library** based on the current context and add them to the initial beam.


2. **Execution & Evaluation:**
* Run all K candidates on the input (C_{edge}, q).
* **The Learned Critic (Feedback Fn):** Analyzes the trace and output. It returns a **Rich Feedback Object**:
```json
{
  "score": 0.85,  // Used for Pruning (Selection)
  "critique": "The logic is sound but you failed to mask the user's ID as per policy.",
  "blame": "Step 3 (Prompt)" // Used for Blame Assignment
}

```




3. **Selection (Pruning):**
* Rank candidates by `score`. Keep the top K.
* *Hard Filter:* Immediately discard candidates with runtime crashes (unless the beam is empty).


4. **Expansion (Generative Optimization):**
* The survivors are passed to the **Optimizer (StructureEditor/OptoPrime)**.
* The Optimizer uses the `critique` from the feedback to generate reasoned improvements (e.g., "Add a masking step").
* *Instruction:* The Optimizer is guided by the **Meta-Prompt** (which contains the "Wisdom" from the Outer Loop).



---

### Part 3: The Outer Loop (Meta-Optimization)

**Engine:** `MetaOptimizer`
**Method:** **Textual Gradient Descent (Reflexion)**

The Meta-Optimizer does not tune weights. It rewrites the **Text Artifacts** that control the Inner Loop. It runs "offline" using a training set of (C, q, a_{ground\_truth}).

It compares the Inner Loop's result (\hat{y}) and the Critic's opinion against the Ground Truth (a). Based on the discrepancy, it updates one of three targets:

#### Target A: The Learned Critic (Feedback Function)

* **Role:** The compass.
* **Trigger:** **False Positive.** The Critic gave a high score, but the prediction \hat{y} was wrong (based on a).
* **Optimization:**
* *Input:* Trace, Critic's Feedback (Score 0.9), Ground Truth (Score 0.0).
* *Meta-Reasoning:* "The Critic failed to notice that the output was hallucinated. It needs to check for factual consistency."
* *Action:* Rewrite the Critic's Prompt.
* *New Prompt:* "...Verify that every claim in the output is supported by the context..."



#### Target B: The Meta-Prompt (The Optimizer's Brain)

* **Role:** The instructions and guidelines for `StructureEditor`.
* **Trigger:** **Stagnation.** The Inner Loop failed to find a solution after N steps, or kept making the same syntax error.
* **Optimization:**
* *Input:* Trajectory of failed edits.
* *Meta-Reasoning:* "The StructureEditor keeps trying to use `json.loads` on markdown text. It needs a strict rule about parsing."
* *Action:* Append a rule to the **Constitution** inside the Meta-Prompt.
* *New Rule:* "Rule #5: Always strip markdown code fences before parsing JSON."



#### Target C: The Seed & Library (The Initialization)

* **Role:** The starting point.
* **Trigger:** **Success.** The Inner Loop found a great solution W^*, but it took many steps to get there.
* **Optimization:**
* *Input:* The final successful workflow W^*.
* *Meta-Reasoning:* "The pattern `Step -> Verify -> Retry` was the key to solving this. We should generalize this."
* *Action:* Distill W^* into a **Template** and add it to the **Case Library**.
* *Usage:* Future Inner Loop runs will retrieve this template via the Meta-Prompt context or directly into the Beam.



---

### Part 4: Detailed Mechanics of Your Questions

#### 1. How the Library is used in the Meta-Prompt

The **Case Library** is a collection of `(Context_Embedding, Solution_Summary, Code_Snippet)` tuples.

When the Inner Loop starts:

1. **Retrieval:** The Engine embeds the current Edge Context C.
2. **Query:** Finds the top 3 most relevant cases from the Library.
3. **Prompt Construction:** The `StructureEditor`'s system prompt is dynamically assembled:
```text
You are an expert AI engineer...

[OPTIMIZATION CONSTITUTION]
1. Always validate inputs...
2. ...

[RELEVANT PAST CASES]
Here are examples of how similar policies were handled:
- Case A (Finance Policy): Used a "checking loop" to ensure compliance.
  Code: ...
- Case B (Privacy Policy): Used a "masking filter" before the LLM call.
  Code: ...

Use these patterns if they apply to the current failure.

```


4. **Effect:** The LLM sees "Oh, a masking filter worked for a similar privacy policy before? I'll try adding that to fix this privacy violation."

#### 2. How the Meta-Optimizer Works (Textual Gradient)

It is a **Diff-based Update**.

```python
def meta_step(trajectory, ground_truth):
    # 1. Evaluate
    pred = trajectory.final_prediction
    critic_score = trajectory.final_critic_score
    real_score = evaluator(pred, ground_truth)
    
    # 2. Compute "Gradient" (Textual Critique)
    discrepancy = critic_score - real_score
    
    if discrepancy > 0.5: # False Positive (Critic is too loose)
        prompt_diff = llm(
            role="meta-optimizer",
            task=f"The Critic approved a bad output. \n"
                 f"Critic Prompt: {current_critic_prompt}\n"
                 f"Bad Output: {pred}\n"
                 f"Reason it's bad: {ground_truth.reason}\n"
                 f"REWRITE the Critic Prompt to catch this error."
        )
        return update_artifact("critic", prompt_diff)
        
    elif real_score < 0.5: # Generator Failure (Optimizer isn't smart enough)
        # Check trajectory for repeated mistakes
        rule_update = llm(
            role="meta-optimizer",
            task=f"The Optimizer failed to solve the problem.\n"
                 f"Trajectory: {trajectory.summary}\n"
                 f"Identify the recurring strategic error and draft a new heuristic Rule."
        )
        return update_artifact("meta-prompt", rule_update)

```

Does this comprehensive summary accurately reflect the system we want to build? If yes, we are ready to code the `BeamInnerLoopEngine`.
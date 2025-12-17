# MyOpto tracing refactor plan

## Architecture decision
- Option A: Namespaced global GRAPH keyed by `run_id`/`domain` while keeping singleton structure; `RuntimeTracer` uses namespaced views, `__enter__` clears only matching namespace. Minimal churn but still couples concurrent runs through shared object.
- Option B: Per-tracer graph instance (preferred). Each `RuntimeTracer` owns its own graph object with nodes/edges, stored on the tracer and selected via the active ContextVar. Avoids cross-run clobbering and nested clearing issues; requires routing helpers to use tracer-local graph but no global mutation.

## API changes
- `RuntimeTracer(run_id=None, domain=None, clear_scope="current", parent_run_id=None, sample_id=None, iteration=None, workflow_level=None)`; defaults preserve current behavior with `run_id` auto-generated once per tracer.
- `__enter__` respects `clear_scope`: `none` (no clear), `current` (clear tracer-local graph), `all` (legacy global clear if enabled). No global GRAPH mutation when using per-tracer graph.
- `msg/llm` fetch active tracer via ContextVar and append nodes into that tracer's graph. Namespace comes from tracer metadata (run_id/domain). If no active tracer, fall back to legacy global GRAPH behavior for backward compatibility.
- `to_ir(run_id=None, domain=None, level=None)` reads from tracer-local graph; filters by tracer metadata or explicit args. Level filters nodes by `workflow_level` (1 workflow / 2 feedback) to export specific slices.

## Data model
- Each node stores: `run_id`, `domain` (workflow/feedback), `parent_run_id` (outer tracer), `workflow_level` (1 workflow seed_workflow, 2 feedback_workflow), `sample_id` (per training sample), `iteration` (optimizer step), existing `call_tag/template_key/usage/latency`, and `extra_info`.
- Graph edges remain parent-child message links. `tracer.prompt_templates`, `llm_nodes`, `output_node` kept per tracer.
- IR schema adds `run_id`, `domain`, `workflow_level`, `parent_run_id`, `sample_id`, `iteration` at top-level and per-node fields; omit suppressed levels based on export filter.

## Migration plan (commits)
1) **Introduce per-tracer graph**: add graph container to RuntimeTracer; ContextVar routing uses tracer.graph instead of global GRAPH. Done when existing tests pass and `to_ir` outputs unchanged for single tracer. Tests: current tracing unit/integration, manual `example.py` run.
2) **Clear semantics**: add `clear_scope` flag; ensure `__enter__` respects tracer-local clear. Done when nested tracer with `clear_scope="none"` preserves outer graph. Test: unit covering nested tracer append and IR merge.
3) **Metadata plumbing**: add run_id/domain/workflow_level/parent_run_id/sample_id/iteration fields on nodes and IR. Backward compatibility: default None; old callers unchanged. Done when IR contains metadata for new tracer; old snapshots still parse. Test: snapshot IR for default tracer, new tracer with custom metadata.
4) **Level-aware to_ir export**: add filtering by workflow_level and domain. Done when workflow trace unaffected by feedback tracer; feedback export isolates its nodes. Test: simulate workflow tracer then feedback tracer; ensure workflow IR excludes feedback nodes.
5) **Optimizer integration**: update OptoPrimeLocal/feedback/meta-optimizer call sites to create tracers with appropriate `workflow_level` (1 vs 2) and `domain` labels. Ensure `clear_scope="current"` so feedback tracing does not clear workflow graph. Done when optimizer run produces separate workflow/feedback IR slices. Test: minimal optimizer + feedback run asserting two IR exports.
6) **Concurrency prep (optional)**: expose `sample_id` and `iteration` to allow per-sample tracers and safe aggregation; document async/multiprocess guidance. Done when parallel sample tracers can merge IR by run_id without collisions. Test: spawn parallel tracers (threads/async) capturing sample_id and merging IR.

## Concurrency plan
- Use separate RuntimeTracer per sample (run_id per tracer, parent_run_id referencing outer workflow run). No shared GRAPH mutations; ContextVar per task ensures isolation in asyncio/threads. Multiprocessing requires serializing tracer outputs; prefer asyncio/threads first.
- Aggregation: collect per-sample `to_ir(level=1)` outputs, merge by `parent_run_id` before meta-optimizer step; meta-optimizer runs synchronously with its own tracer (`workflow_level=2`).

## Edge cases
- Feedback tracing must use `clear_scope="none"` or tracer-local clear to avoid wiping workflow trace; per-tracer graphs prevent cross-run clear.
- Nested tracers: parent_run_id links maintain hierarchy; child tracer does not clear parent. Nodes carry level to keep exports separable.
- Workflows with no llm calls: output_node may be None; IR should emit run metadata with empty nodes and mark as feedback/error signal for caller.
- `clear_graph_on_enter` legacy flag maps to `clear_scope="all"` to preserve behavior while encouraging per-tracer clearing.

## Example flow (pseudocode)
```
with RuntimeTracer(run_id="wf1", domain="workflow", workflow_level=1) as tracer:
    tracer.msg(...)
    tracer.llm(...)
    wf_ir = tracer.to_ir(level=1)

with RuntimeTracer(run_id="fb1", domain="feedback", parent_run_id="wf1", workflow_level=2, clear_scope="none") as fb_tracer:
    fb_tracer.msg(...)
    fb_ir = fb_tracer.to_ir(level=2)

# Aggregation for meta-optimizer happens on wf_ir + fb_ir without cross-clearing.
```

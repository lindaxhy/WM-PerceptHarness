# Scene Prompt Contract Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement and review this bounded repair. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the scene prompt unambiguously describe the already-approved spatial evidence contract.

**Architecture:** Retain all existing strict schemas, branch fallbacks, and single-repair behavior. Render a trusted explicit evidence-availability boolean and flat spatial field lists; clarify the prompt without modifying semantic validation or acceptance thresholds.

**Tech Stack:** Existing Python 3.12, Pydantic, prompt renderer, pytest; no new dependencies.

## Global Constraints

- Work on `feat/sam31-evidence-integration`; preserve frozen Qwen/LAS bytes and evaluation mapping.
- The approved Task 11 contract requires empty spatial lists without available CV evidence and flat provenance fields on supported spatial rows.
- A missing or low-confidence track is not proof of absence or occlusion. Unsupported claims resolve to `unknown` or no event.
- Never store raw masks in SQLite, prompts, API task results, logs, or training JSONL.
- Preserve strict rejection, one schema repair, conservative fallback, public result shapes, and all quality gates. No model/provider, temperature, sampling, threshold, or retry changes.
- This repairs ambiguous instructions to match an existing approved design, not a new schema or feature. Use apply_patch and `.venv/bin/python -m pytest`.

## Observed cause

The immutable `609b907` Doubao-only full pipeline completed but degraded its scene branch after two missing/extra-field failures. A one-stage replay reproduced nested `provenance` objects instead of eight required flat fields on three locations and three relations, despite CV being disabled. The prompt calls these fields "provenance" without an exact flat field list and communicates evidence absence only by omitting an appended summary block.

A diagnostic-only append explicitly stating that CV was disabled produced HTTP 200 and structurally valid scene output in 23.266 seconds (15,004 input and 451 output tokens). An earlier attempt encountered a sanitized transport/service error; no status was captured. These measurements are not final acceptance, and no raw model values were persisted.

### Task 1: Explicit scene availability and flat field contract

**Files:**
- Modify: `src/las_repro/pipelines/embodied.py` (scene renderer only)
- Modify: `src/las_repro/prompts/scene_semantics.txt`
- Modify: `tests/test_embodied_pipeline.py`
- Modify: `tests/test_scene_semantics.py` if needed for a focused contract regression
- Modify: `docs/reports/2026-09-05-doubao-feasibility.md`

**Interfaces:**
- Consumes: existing `PromptRenderer.scene_semantics(..., evidence_summary=None, repair=None)` and unchanged `SceneLocation`/`SceneRelation` schemas.
- Produces: the same prompt string interface and unchanged validated result schema.

- [ ] **Step 1: Write focused RED renderer regressions.**

Require a trusted JSON block `[CV_EVIDENCE_AVAILABILITY_JSON]` containing exactly `{"available":false}` without a summary and `{"available":true}` with an available validated summary. Exercise initial and repair prompts, retaining the exact repair issue codes. Existing wrong-type/nonavailable summary rejection must remain.

```python
prompt = PromptRenderer().scene_semantics([], video_duration=2.0)
assert '[CV_EVIDENCE_AVAILABILITY_JSON]\n{"available":false}' in prompt
```

Require `[SCENE_SPATIAL_FIELDS_JSON]` to contain exactly `location_fields` and `relation_fields`, each equal to the corresponding schema's field names in declaration order. Assert neither field list contains a nested `provenance` key. A schema regression must reject nesting these flat fields into a `provenance` object; do not normalize that invalid output.

- [ ] **Step 2: Run RED.**

```bash
.venv/bin/python -m pytest tests/test_embodied_pipeline.py tests/test_scene_semantics.py -q
```

Record the expected missing-block failures, not an unrelated fixture failure.

- [ ] **Step 3: Render the trusted contract and clarify instructions.**

Use existing canonical prompt substitution with these server-owned values:

```python
{
    "CV_EVIDENCE_AVAILABILITY_JSON": {"available": evidence_summary is not None},
    "SCENE_SPATIAL_FIELDS_JSON": {
        "location_fields": list(SceneLocation.model_fields),
        "relation_fields": list(SceneRelation.model_fields),
    },
}
```

Keep `_with_cv_summary` validation and bounded summary inclusion. The prompt must state that the boolean, not a textual mention of a marker, decides availability. When false, require both spatial lists empty while retaining supported objects, endpoint states, outcome, and semantic events. When true, allow only supported spatial claims. All listed fields are siblings in each row; explicitly forbid a nested `provenance` object. Preserve the existing rules for source IDs, exact overlapping segments, original observed clock, uncertainty, and no invented evidence.

- [ ] **Step 4: Verify GREEN and unchanged integration.**

```bash
.venv/bin/python -m pytest tests/test_embodied_pipeline.py tests/test_scene_semantics.py tests/test_hybrid_result.py -q
.venv/bin/python -m pytest -q
git diff --check
```

Record commands/output and existing warnings. Update the feasibility report with diagnostic findings without claiming full semantic or five-demo acceptance. A controller live no-override replay follows the reviewed commit; it is not required to fabricate a success during implementation.

- [ ] **Step 5: Commit and report.**

```bash
git add src/las_repro/pipelines/embodied.py src/las_repro/prompts/scene_semantics.txt tests/test_embodied_pipeline.py tests/test_scene_semantics.py docs/reports/2026-09-05-doubao-feasibility.md
git commit -m "fix: clarify scene evidence prompt contract"
```

Return the normal SDD report with RED/GREEN evidence and self-review. An independent task review is required before final real-video runs.

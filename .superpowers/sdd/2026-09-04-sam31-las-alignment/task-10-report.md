# Task 10 Report: Dedicated Occlusion Adjudication

## Implemented

- Added frozen, extra-forbid occlusion decision, interval, classification, and event schemas.
- Added deterministic validation for exact candidate cardinality/order, target and occluder closure, observed timestamp selection, positive ordered intervals, same-type non-overlap, video bounds, and conservative non-occlusion decisions.
- Added deterministic positive-event projection with candidate, track, overlay-keyframe, fine-segment, model-stage, repair, evidence-mode, and review provenance.
- Added a data-isolated occlusion prompt and renderer containing canonical trusted candidate, entity, timing, and bounded CV-summary sections. Repair input is restricted to `issue_codes`.
- Registered `OcclusionDecisionSet` with the pre-persistence schema registry and closed repair codes.
- Added the reusable `EmbodiedActionPipeline.adjudicate_occlusions` stage. Empty candidates skip model work; exhausted repair or stage failure returns an empty decision set without failing another semantic branch.
- Registered `occlusion_semantics` in the fake backend and Qwen token budgets (`4096`).

## TDD Evidence

### RED

Command:

```text
.venv/bin/python -m pytest tests/test_occlusion_semantics.py -k decision -v
```

Relevant result:

```text
ModuleNotFoundError: No module named 'las_repro.pipelines.occlusion'
ERROR tests/test_occlusion_semantics.py
```

This was expected because the new decision schema and validator module did not exist.

Additional stage-budget RED:

```text
.venv/bin/python -m pytest tests/test_occlusion_semantics.py -k output_registry tests/test_qwen_backend.py -k 'occlusion_semantics_has' -v
KeyError: 'occlusion_semantics'
1 failed, 141 deselected
```

Additional orchestration RED:

```text
.venv/bin/python -m pytest tests/test_occlusion_semantics.py -k 'empty_candidate or invalid_repair' -q
AttributeError: 'EmbodiedActionPipeline' object has no attribute 'adjudicate_occlusions'
2 failed, 12 deselected
```

### GREEN

Focused integration command:

```text
.venv/bin/python -m pytest tests/test_occlusion_semantics.py tests/test_qwen_backend.py tests/test_embodied_pipeline.py -q
250 passed in 3.31s
```

Full-suite command:

```text
.venv/bin/python -m pytest -q
1447 passed, 2 warnings in 23.98s
```

The two warnings are the pre-existing Starlette/httpx and AnyIO deprecations also present in the controller's baseline.

## Files Changed

- `src/las_repro/pipelines/occlusion.py`
- `src/las_repro/prompts/occlusion_semantics.txt`
- `src/las_repro/pipelines/output_validation.py`
- `src/las_repro/pipelines/embodied.py`
- `src/las_repro/models/fake.py`
- `src/las_repro/models/qwen3_vl.py`
- `tests/test_occlusion_semantics.py`
- `tests/test_qwen_backend.py`
- `.superpowers/sdd/2026-09-04-sam31-las-alignment/task-10-report.md`

## Self-Review

- Re-read the task brief and checked each interface and conservative-failure requirement.
- Tightened projection during review so it revalidates decisions against trusted candidates instead of assuming callers always used the registry first.
- Confirmed event projection emits no records for non-positive classifications and uses only candidate-owned entity, track, timestamp, and overlay provenance.
- Confirmed no Doubao/backend-specific behavior was introduced.
- Confirmed controller-owned untracked Doubao documents were not staged.

## Concerns

- None. Main-pipeline CV artifact acquisition and final branch merging are intentionally left to subsequent planned tasks; this task exposes the backend-neutral adjudication stage for that integration.

# Semantic Contract Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve actionable closed spatial errors, clarify occlusion intervals,
and remove an ineffective runtime option in one bounded, testable release.

**Architecture:** Keep the existing validators, pipeline, evidence algorithms and
cache identity. Change one allowed error code, one prompt's contract explanation,
and the exposed Settings/example surface. Tests exercise the actual sanitizer,
rendered JSON examples, candidate validator and configuration constructor.

**Tech Stack:** Existing Python 3.12, Pydantic, pytest and Node viewer tests;
no new dependencies.

## Global Constraints

- Preserve thresholds, reference data and old experiment results unchanged.
- Do not modify `_possible_occluders`, `EvidenceThresholds`, entity normalization, sampling or cache-key computation.
- Do not expose raw validation exceptions, rejected model output, filesystem paths or uncontrolled error codes.
- Invalid spatial output must remain invalid; no automatic spatial repair.
- No model/weight change, performance tuning, new inference cohort or deployment in this implementation task.
- Preserve the user-owned untracked performance follow-up note untouched.
- Work in the existing `feat/sam31-evidence-integration` linked worktree.
- Run routine implementation steps without asking for confirmation again.

---

### Task 1: Deliver the bounded semantic-contract repair

**Files:**
- Modify: `src/las_repro/pipelines/output_validation.py` (allowed scene code).
- Modify: `src/las_repro/prompts/occlusion_semantics.txt` (interval guidance/examples).
- Modify: `src/las_repro/config.py`, `.env.example` (remove obsolete option).
- Modify: `docs/deployment/sam31-runtime.md` (fixed candidate rule and migration).
- Test: `tests/test_hybrid_result.py`, `tests/test_occlusion_semantics.py`,
  `tests/test_cli.py`, `tests/test_repository_policy.py`.

**Interfaces:**
- Consumes: `DEFAULT_OUTPUT_SCHEMAS.sanitize(name, output, context)` and
  `.failure_codes(name, sanitized)`, `PromptRenderer.occlusion_semantics`,
  `OcclusionDecisionSet`, `validate_occlusion_decisions`, and `Settings`.
- Produces: unchanged output schemas, a retained closed spatial error, an
  unambiguous positive/unknown prompt example pair, and no no-op Settings field.
- Reuse `available_result` and `source_segments()` in `test_hybrid_result.py`;
  reuse `_candidate()`, `_positive_raw()` and `_trusted_prompt_inputs()` in
  `test_occlusion_semantics.py`. Do not add production helpers for test setup.

- [x] **Step 1: Add a failing spatial-code regression.**

Add this test in `test_hybrid_result.py`; it catches both the ordering branch
and the deeper provenance-exception branch without weakening validation:

```python
@pytest.mark.parametrize("fault", ["ordering", "provenance"])
def test_scene_spatial_failure_survives_registry(available_result, fault):
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
    from las_repro.pipelines.embodied import _validated_stage_result

    result, summary = available_result
    scene = {key: copy.deepcopy(result[key]) for key in hybrid_result.SCENE_KEYS}
    context = {
        "duration": 1.0, "require_observed_content": True,
        "required_object_ids": ["cup"], "segments": source_segments(),
        "evidence_summary": summary.model_dump(mode="json"),
    }
    assert DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context) == scene
    if fault == "ordering":
        later = copy.deepcopy(scene["locations"][0])
        later["start"] = 0.5
        scene["locations"].insert(0, later)
    else:
        scene["locations"][0]["source_segment_indices"] = [5]
    before = copy.deepcopy(scene)
    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context)
    assert sanitized == {"_schema_validation": {
        "schema_name": "SceneSemantics", "status": "invalid",
        "issue_codes": ["SCENE_SPATIAL_INVALID"],
    }}
    assert scene == before
    _, codes, _ = _validated_stage_result("SceneSemantics", sanitized, context)
    assert list(codes) == ["SCENE_SPATIAL_INVALID"]
```

Run `.venv/bin/python -m pytest tests/test_hybrid_result.py -k spatial_failure -q`.
Expected RED: generic schema error rather than the required spatial code.
Retain existing registry tests of unknown/generic errors and redaction; add a
narrow regression only if those boundaries are not currently exercised.

- [x] **Step 2: Preserve the existing error code and verify GREEN.**

Add `"SCENE_SPATIAL_INVALID",` to `_SCENE_TEMPORAL_CODES` and nothing else in
production validation. Run the preceding RED command, then
`.venv/bin/python -m pytest tests/test_hybrid_result.py tests/test_scene_semantics.py tests/test_workers.py -q`.
Expected: spatial code preserved, original valid output unchanged, all pass.

- [x] **Step 3: Add executable prompt-example and malformed-event tests.**

Render the existing production prompt with `_candidate()`, trusted inputs,
duration `3.0`, frame PTS `(0.0, 1.0, 2.0, 3.0)` and repair `None`.
Parse complete JSON lines beginning with `{"decisions":` after `[output schema]`
and before `[validation repair data]`. Require at least one unknown decision
with empty events and one occlusion decision with a nonempty interval. Bind
only illustrative candidate/target/occluder IDs to `_candidate()`; do not fill
or rewrite event bounds. Validate every complete example against the real
`OcclusionDecisionSet` and `validate_occlusion_decisions(..., duration=3.0)`.
The positive example below uses `1.0` and `2.0`, already legal fixture bounds.
Run `.venv/bin/python -m pytest tests/test_occlusion_semantics.py -k example -q`.
Expected RED: current prompt has no positive interval example.

Add parametrized malformed events to `_positive_raw(candidate)`:

```python
bad_events = [
    {"event_type": "occluded", "timestamp": 1.0},
    {"event_type": "occluded", "start": 1.0},
    {"event_type": "occluded", "end": 2.0},
    {"event_type": "occluded", "start": 1.0, "end": 2.0, "timestamp": 1.0},
]
```

Assert the real sanitizer emits only the expected closed missing/extra-field
codes and no original event data. Existing legal, unknown, foreign-time and
non-occlusion-event tests must remain passing. These safety checks may already
pass before the prompt edit; the missing example is the RED for this change.

- [x] **Step 4: Clarify the prompt and validate its examples.**

Keep the original unknown JSON example. Add a second complete illustrative
example with one decision and this interval shape:

```json
{"decisions":[{"candidate_id":"occ_example_0001","classification":"occlusion","target_entity_id":"object","occluder_entity_id":"panel","events":[{"event_type":"occluded","start":1.0,"end":2.0}],"visual_evidence":"object is visibly hidden behind the panel during this interval","confidence":0.8}]}
```

Explain that these are alternative shape examples, not outputs or default times
to copy. Actual candidates determine identities, allowed start/end values and
decision count. Require exactly `event_type`, `start`, `end` in each event,
numeric bounds, positive duration, and prohibit `timestamp` substitution.
Retain uncertainty/visible-evidence rules and do not require all event types.
For missing/extra-field repair, regenerate the full set without inferring
bounds from a timestamp. Version this modified prompt locally as
`0906-occlusion-contract-v2`; do not change other prompt identifiers.
Run all `tests/test_occlusion_semantics.py` and `tests/test_embodied_pipeline.py`.

- [x] **Step 5: Fail on obsolete Settings keyword, then remove it.**

Add a test in `test_cli.py`:

```python
def test_removed_overlap_option_is_not_a_runtime_control():
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, cv_overlap_threshold=0.1)
    assert any(item["loc"] == ("cv_overlap_threshold",)
               and item["type"] == "extra_forbidden"
               for item in error.value.errors())
```

Use the existing Pydantic import or add it. Run this test and confirm RED
(no exception before removal). Then remove the field and example environment
line, remove old default/invalid-range expectations for this field in CLI and
repository-policy tests. Do not remove actual threshold validation cases.
Document the fixed positive-overlap candidate rule and obsolete constructor
keyword removal; existing process-environment handling is not redesigned.
Run `.venv/bin/python -m pytest tests/test_cli.py tests/test_repository_policy.py tests/test_cv_summary.py tests/test_cv_artifacts.py -q`.

- [x] **Step 6: Verify, self-review and commit only the implementation.**

Run `.venv/bin/python -m pytest -q --cov=las_repro --cov-report=term-missing`;
expected all passing, configured coverage at least 85%, only the two known
dependency deprecation warnings if still present. Run
`node --test evaluation/viewer/tests/model.test.mjs`,
`node --check evaluation/viewer/js/app.js`, and `git diff --check`.
Check `git diff --name-only` contains only the task files, and no evidence,
viewer data, frozen references, mapping or algorithm implementation changed.
Commit explicitly named task files with
`fix: clarify semantic contracts and remove ineffective overlap option`.
Record exact RED/GREEN commands, results, final tests and concerns in the task
report; return its path and the commit. Controller obtains task review and final
integration review before any deployment or PR action.

## Final whole-branch review follow-up

Task1 passed independent specification and quality review. Whole-branch review
then found two separate integration defects: candidate keyframes can outnumber
the projected evidence tracks, and an external BPE asset can vary under an
unchanged cache identity. The
[final hardening design](../specs/2026-09-06-final-integration-hardening-design.md)
defines one consolidated fix wave, including historical cleanup/test minors.
Its implementation brief and report are kept in this plan's SDD workspace.
The global preservation constraints still apply; no cache-key redesign or
new runtime experiment is part of the wave. One scoped re-review follows.

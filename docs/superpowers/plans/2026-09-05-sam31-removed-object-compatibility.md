# SAM3.1 removed-object compatibility implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Consume the pinned SAM removed-object sentinel without discarding the
entire request or inventing a confidence/occlusion observation.

**Architecture:** Recognize the exact sentinel inside the existing frame parser.
Keep structural, geometry and mask safety validation, omit only removed rows
before writing masks/detections, and leave actual probability validation intact.

**Tech Stack:** Existing Python adapter, pytest, pinned local SAM runtime.

## Global Constraints

- Accept only the numeric floating-array score exactly equal to `-10000.0` as the pinned removed-object sentinel. It is not a public confidence value.
- Preserve frame/timeline, array dtype/shape/size, ID uniqueness and range checks.
- Non-sentinel scores remain finite and within `[0,1]`; arbitrary negatives, values above one, booleans, NaN and infinities still fail closed.
- Removed rows produce no visible detection, confidence, mask reference or overlay. Do not treat removal as proof of absence or occlusion.
- No model/prompt/threshold changes, upstream patch, reference/mapping changes, probability rescaling, or publishing of diagnostics as acceptance evidence.
- Preserve valid rows, source frame identity and ordering, empty-frame handling, cleanup, cache integrity, OOM policy and all public schemas.

### Task 1: Normalize the pinned removal sentinel at the adapter boundary

**Files:**
- Modify: `src/las_repro/cv/sam31.py`
- Test: `tests/test_sam31_adapter.py`

**Interfaces and evidence:**
`_parse_frame_response(response, sampled, *, allowed_indices, expected_dimensions,
mask_writer)` validates arrays then emits `_Detection` rows consumed by existing
track/artifact construction. Use existing `frame_output`, `PredictorDouble`,
`MaterializerDouble`, `make_request`, `make_provider`, `read_npy` and
`read_int64_array` test helpers. No new runtime dependencies.
Confirmed exact runtime scalar is -10000.0 at frame27 in first request, not
roundoff/OOM. Pinned upstream sam3_multiplex_base.py1264-1271 explains removal.
No changes outside the two scoped files; root owns runtime operations/docs.

Task binding constraints are the Global Constraints above, verbatim, plus:
validate existing geometry and actual boolean mask content even for removed
rows; skip their archive write and detection creation after safety validation.
An empty per-prompt archive consistent with existing empty-frame behavior is
allowed, but it contains no removed masks and no observation references it.

- [ ] **Step 1: Add failing regressions using actual provider/analyzer path.**

  Minimal removed-only example (add additional cases below separately):

  ```python
  def test_removed_sentinel_does_not_publish_detection(tmp_path, fake_torch):
      request = make_request(tmp_path)
      def stream(session_number, prompt, payload):
          return [frame_output(i, probabilities=[-10000.0])
                  for i in range(payload["start_frame_index"],
                                 payload["start_frame_index"] + payload["max_frame_num_to_track"])]
      provider = make_provider(PredictorDouble(stream), fake_torch, MaterializerDouble())
      artifact = provider.analyze(request, tmp_path / "staging")
      assert artifact.tracks == ()
      assert artifact.overlay_records == ()
      assert artifact.processed_timeline == request.timeline
  ```

  Add mixed rows with IDs `[2,1]` and scores `[-10000.0,0.875]`: only ID1
  survives with unchanged geometry/confidence/frame order. Inspect archived mask
  contents/counts so removed rows cannot leak into files. Add visible/removed/
  visible frames0/1/2: observations only frames0/2, no generated occlusion.
  Parameterize non-sentinel invalid scores `-9999.0`, `-10000.001`, `-0.1`,
  `1.001`, NaN, positive/negative infinity and booleans. Existing malformed shape,
  duplicate ID, invalid frame, geometry and mask tests stay; exercise those with
  sentinel too to catch validation bypass. Verify close/cleanup on failures.

- [ ] **Step 2: Run RED before editing product code.**

  `.venv/bin/python -m pytest tests/test_sam31_adapter.py -q -k 'removed_sentinel'`
  The valid sentinel examples must fail in production probability validation;
  record the expected failures and already-passing negative cases in report.

- [ ] **Step 3: Implement the minimal normalization.**

  Keep `_bounded_float` unchanged. Add a private named constant with a concise
  pinned upstream source comment. In the existing row loop:

  ```python
  raw_probability = _array_item(probabilities, object_offset)
  removed = (not isinstance(raw_probability, bool)
             and isinstance(raw_probability, Real)
             and float(raw_probability) == _REMOVED_OBJECT_SCORE)
  probability = None if removed else _bounded_float(raw_probability, lower=0.0, upper=1.0)
  # Existing strict box validation remains here unchanged.
  if removed:
      _consume_mask_rows(masks, object_offset, masks_shape[1], masks_shape[2])
      continue
  # Existing mask writer/scan and _Detection construction remain unchanged.
  ```

  Do not weaken array metadata, ID, frame or mask checks, nor add broad score
  clamping/filtering. Preserve `_run_prompt` missing-frame/detection behavior.

- [ ] **Step 4: Verify and self-review.**

  Run full adapter, artifact-store, CV worker and summary test files (resolve
  existing exact paths with `rg --files tests`), then the full Python suite once
  `.venv/bin/python -m pytest -q --cov=las_repro --cov-branch --cov-report=term-missing`.
  Run scoped Ruff and `git diff --check`. Do not claim hardware verification.
  Ensure tests would catch sentinel being clamped/retained, arbitrary negative
  scores being accepted, mask safety being bypassed, or valid rows being dropped.

- [ ] **Step 5: Commit scoped files, report and independent review.**

  Commit only adapter/tests. Report RED/GREEN commands/output, full test evidence,
  self-review and concerns to the task report. Root reviews with the full base-to-
  head package before making any new immutable deployment. Existing failed runs
  remain frozen; exact-request hardware validation and later full acceptance
  remain root's original Task15 workflow, not claims of this unit-test task.

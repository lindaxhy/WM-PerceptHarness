# Viewer fine-source compatibility implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Root executes inline as Site owner, with independent scoped review before publication.

**Goal:** Make the already-approved closed viewer projection consume the actual
production fine-segment records without changing display schema or source data.

**Architecture:** Separate accepted source fields from published display fields.
Accept the existing producer's `start_boundary_id` and `end_boundary_id` as
source-only metadata, validate their text with the existing safety boundary,
and omit them from the unchanged closed Fine display schema. Original and
canonical digests continue to bind all original source fields.

Actual export then exposed a second existing-contract mismatch: scene semantic
events explicitly allow `target_object_id: "unknown"` without a declared object.
The browser must accept this sentinel only for scene event targets, display it
as `unknown`, and continue rejecting foreign IDs, null targets and spatial
references to undeclared objects. This is not a nullable-target contract.

**Tech Stack:** Existing Python projector/exporter, pytest and Node model.

## Global Constraints

- Preserve original canonical results, frozen references/mapping, API, models,
  prompts, thresholds and the existing `comparison_viewer_hybrid_v1` schema.
- Keep arbitrary unknown fields, unsafe strings and private payloads rejected.
- Root retains Site ownership; no subagent edits the Site checkout.
- No inference reruns, browser QA, new visual design or new hosting surface.
- This repairs existing Task 13 compatibility, not a new feature/design.

### Task 1: Accept production boundary metadata without publishing it

**Files:**
- Modify: `src/las_repro/evaluation/viewer_projection.py`
- Test: `tests/test_viewer_projection.py`
- Modify: `evaluation/viewer/js/model.js`
- Test: `evaluation/viewer/tests/model.test.mjs`

**Interfaces:** Production `FineSegmentTableRow.public_record()` includes both
boundary IDs. The projector's display allowlist and JS fine-event schema do not.
The five actual Doubao-only results validate and score, but all fail projector
validation with `unknown viewer fine field` on these existing source fields.

- [ ] Add a real producer-record regression, both Fine enabled and disabled,
  asserting successful canonical projection, original input immutability and
  unchanged source digests. For Fine enabled, assert boundary IDs are omitted
  while source segment identity remains and the real Node normalizer accepts
  the projection. Parameterize unsafe/non-string boundary metadata and retain
  unknown-field rejection tests.
- [ ] Run the focused tests before product editing and retain the expected RED.
- [ ] Keep `_FINE_FIELDS` unchanged. Define a source-only union containing the
  two boundary fields; use it only for the input-key check. Existing `_safe_text`
  validation applies to those values. Construct published Fine fields using:

  ```python
  {key: copy.deepcopy(value) for key, value in segment.items() if key in _FINE_FIELDS}
  ```

  Do not remove or rewrite anything in the input result or its digest inputs.
- [ ] Add a failing Node regression for the canonical scene target `unknown`
  (including absent object declarations), plus known/foreign/null targets and
  strict spatial-reference negatives. Accept the sentinel only in scene mode;
  preserve closed fields and render an explicit `unknown` target.
- [ ] Run projector/export/static Python tests, Node model tests, scoped Ruff
  and `git diff --check`. Run the actual five-control exporter and feed every
  generated projection to the real Node normalizer; verify frozen Qwen and
  demo-manifest bytes remain unchanged.
- [ ] Commit only scoped product/tests and request independent task review.
  Record RED/GREEN, actual export verification and any remaining findings.
  Keep the generated variant unbound until final Task 15 publication.

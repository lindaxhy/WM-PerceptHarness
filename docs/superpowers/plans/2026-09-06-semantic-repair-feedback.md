# Semantic Repair Feedback Implementation Plan

> **For agentic workers:** This is one tightly coupled validator-to-prompt change;
> implement inline with TDD and an independent final code review.

**Goal:** Make bounded semantic repair feedback actionable without weakening
evidence acceptance or rewriting model output.

**Architecture:** Shared provenance validation emits typed, closed categories.
Scene validation aggregates these with its temporal and text checks and retains
the existing umbrella code. Existing worker and pipeline envelopes carry only
allowlisted codes. Prompts explain the actual constraints.

**Tech Stack:** Python 3.12+, Pydantic, pytest; existing text prompt renderer.

## Global Constraints

- Preserve all previous rejection conditions and valid output identity.
- No sorting, snapping timestamps, inferred provenance, record deletion,
  extra retries, or forced nonempty spatial lists.
- Never serialize raw exceptions, IDs, paths, rejected text, or arbitrary codes
  in the repair envelope.
- No new paid calls, SAM runs, goal creation, deployment, or modification of
  frozen experiment reports/private responses.

## Task 1: Align the complete scene/occlusion repair contract

Files: modify `src/las_repro/pipelines/hybrid_result.py`,
`src/las_repro/pipelines/scene_semantics.py`,
`src/las_repro/pipelines/output_validation.py`, both semantic prompt templates,
and `tests/test_hybrid_result.py`, `tests/test_occlusion_semantics.py`.

Interfaces: `validate_event_provenance` retains its existing signature and
successful return. Add a ValueError subclass carrying closed `issue_codes`.
`validate_scene_semantics` continues raising TemporalValidationError.
The output registry continues returning only its existing invalid envelope.

- [ ] Add failing regression cases to the existing available_result fixture.
  Combine an unsorted location, wrong source segment list, unobserved end time,
  empty tracks, and slash evidence. The expected envelope contains the umbrella
  plus ORDER_INVALID, TIME_NOT_OBSERVED, SOURCE_SEGMENTS_INVALID,
  TRACKS_INVALID, PROVENANCE_INVALID, and PROHIBITED_CONTENT categories under
  the SCENE_SPATIAL prefix. Assert source immutability and pipeline code survival.
  Add isolated category tests for time bounds, references, tracks, keyframes,
  evidence availability, provenance metadata, and non-spatial prohibited text.
  Add controls for legal unchanged spatial output, valid empty lists, and
  rejected slash/path/mask evidence.

  Example independently specified assertion:
  ```python
  assert 'SCENE_SPATIAL_TIME_NOT_OBSERVED' in codes
  assert 'SCENE_SPATIAL_SOURCE_SEGMENTS_INVALID' in codes
  assert scene == before
  ```
- [ ] Run `.venv/bin/python -m pytest tests/test_hybrid_result.py tests/test_occlusion_semantics.py -q`;
  confirm the new detailed-category expectations fail on current production.
- [ ] Implement typed closed provenance errors from the shared predicates;
  retain ID validation and all existing checks. Aggregate deterministic categories
  without indexing unknown keyframes. Split the scene's combined temporal
  condition into independent closed checks, retaining umbrella compatibility.
  Register the categories and aggregate scene temporal, artifact-text, and
  per-row provenance errors after successful schema/context parsing. Fall back
  safely on unexpected errors; never match exception message strings.

  Error transport shape:
  ```python
  class ProvenanceValidationError(ValueError):
      def __init__(self, issue_codes):
          self.issue_codes = tuple(dict.fromkeys(issue_codes))
          super().__init__('event provenance is invalid')
  ```
- [ ] Update prompts with the positive-occlusion converse, strict plain-text
  rules, exact observed-clock boundaries, full temporal segment overlap rule,
  and closed-code explanations. Preserve unknown examples and spatial abstention.
- [ ] Run focused tests and manually compare prompt contract against validators.
  Validate existing rendered prompt examples with synthetic trusted contexts.
- [ ] Run full Python suite with coverage, Node tests, JS syntax and whitespace
  checks. Request independent code review; resolve substantive findings and
  record verification in a new report, not historical experiment artifacts.
- [ ] Commit only scoped files and hand off locally without merging or deployment.

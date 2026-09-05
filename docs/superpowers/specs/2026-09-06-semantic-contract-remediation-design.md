# Bounded semantic-contract remediation

Date: 2026-09-06

Status: written spec approved by the user; implementation authorized.
The user additionally requested default execution for routine work within the
confirmed goal, without repeated plan/implementation confirmation prompts.
Baseline: `d46dfd86e735d676cc03cc7f0554e0a754c680cb`.

## Objective and evidence

Correct three identified contract defects without changing evidence algorithms
or reinterpreting failed acceptance. The user confirmed this exact scope:
preserve the spatial error code, clarify occlusion interval fields, and remove
the ineffective overlap configuration. The second cold cohort, failed cache
resubmission, diagnostic responses and recovery evidence remain immutable.

The scene validator already emits `SCENE_SPATIAL_INVALID`, but the declared
scene registry omits it and substitutes the generic schema error. A separate
occlusion diagnostic used `timestamp` in two events instead of required `start`
and `end`; the production prompt currently shows only an empty events list.
`Settings.cv_overlap_threshold` and its example environment variable have no
product consumer: candidate occluders use the fixed positive-overlap rule.

These observations support the changes below, not a guarantee of improved
occlusion accuracy. The warm Pass B failure remains a separate diagnosis;
its original rejected description text was not retained.

## Chosen approach and alternatives

Use the existing validators and output shapes, improving the repair code and
prompt's explicit interval contract, and remove the unused setting. This is the
approved minimal change; no geometry or semantic acceptance rule is relaxed.

Wiring a new configurable overlap threshold would change candidate generation
and require a versioned evidence/cache contract and a separate experiment. It
is excluded. Leaving all three defects unchanged would preserve the misleading
configuration and nonspecific repair feedback; it is not the selected approach.

## 1. Preserve the closed spatial error code

Add `SCENE_SPATIAL_INVALID` to the existing allowed scene issue-code registry
in `output_validation.py`. Both spatial ordering failures and spatial
provenance failures must retain this code through normal worker sanitization
and pipeline repair-data construction.

Do not expose raw validation exceptions, rejected model output, filesystem paths
or uncontrolled error codes. Keep existing generic fallback for errors outside
the allowed registry. Invalid spatial output must remain invalid; do not sort,
drop, rewrite or auto-repair model spatial claims in this change. Do not change
the scene prompt or its geometry/provenance constraints.

## 2. Make occlusion interval fields explicit

Update only the occlusion prompt's contract guidance and example:

- Each event has exactly `event_type`, `start`, and `end`; `timestamp` is not a
  permitted substitute. Bounds are JSON numbers, not strings.
- Start and end must come from the corresponding candidate's allowed boundary
  lists, with positive duration; example values are illustrative, never defaults
  to copy when absent from the current candidate.
- Include a nonempty interval example without suggesting that every candidate
  is occluded or must contain all three occlusion event types.
- Preserve the unknown/empty-events example and the rule that non-occlusion
  decisions have no events. Preserve all visible-evidence requirements.
- On missing/extra-field repair codes, regenerate the complete decision set
  under the same contract. Never infer missing bounds from a timestamp.

No output schema, candidate cardinality, confidence rule, allowed times,
occluder constraints or semantic classification logic changes. Record the new
source/prompt identity for any subsequent inference; old results retain their
original identity. Prompt clarification alone does not prove model compliance.

## 3. Remove the no-op configuration

Remove `Settings.cv_overlap_threshold`, `LAS_CV_OVERLAP_THRESHOLD` from
`.env.example`, and tests that advertise or validate that nonexistent runtime
control. Update runtime documentation to state the existing candidate rule:
among otherwise eligible relations, at least one of bounding-box IoU, subject
covered fraction or object covered fraction must be strictly greater than zero.
This is candidate proposal only, never proof of semantic occlusion.

Do not modify `_possible_occluders`, `EvidenceThresholds`, entity normalization,
sampling or cache-key computation. Preserve validation of all actual thresholds.
Document removal for callers constructing Settings directly: obsolete keyword
arguments are no longer supported under the existing extra-field policy.
Do not add aliases, a compatibility shim or a new environment rejection policy.
Do not edit historical environment files or experiment metadata to erase the
previously recorded option; omit it in new runtime configuration only.

## Verification and release boundary

Use red/green regression tests against real sanitizer/validator and rendered
prompt boundaries, not assertions that merely mirror constants:

1. Spatial ordering and provenance violations remain rejected and retain the
   closed spatial code; valid spatial output is unchanged. Unknown errors stay
   generic and do not leak raw text.
2. A legal occlusion interval passes the production schema and candidate
   validator; timestamp-only, missing-bound and extra-field events still fail.
   Rendered guidance retains uncertain/non-occlusion cases and candidate-owned
   time constraints. Prompt examples are validated as actual structured output
   with matching synthetic candidate context, not treated as accuracy evidence.
3. Settings/example consumers no longer advertise the no-op control; actual
   threshold validation and existing candidate/cache-identity behavior remain
   covered. No historical frozen data changes.
4. Run focused tests, the full Python suite with at least 85% configured
   coverage, Node viewer tests, JavaScript syntax and whitespace checks. Obtain
   independent code review and resolve confirmed findings before deployment.

This spec authorizes no immediate new five-demo cohort, cache architecture
change, Pass A persistence, automatic normalization, timeout/watchdog change,
extra semantic replay, model/weight change or performance tuning. After local
implementation/review, propose the bounded verification experiment separately
within the continuing goal. Never replace a failed report with a later success.

Existing acceptance remains failed: zero positive occlusion events, two cold
samples above 720 seconds, and a failed first cache-resubmission task with one
new SAM analysis. Recovery passed independently. Human precision and browser
interaction acceptance are not fabricated. The previous evidence snapshot's
review was pending when this spec was written. The replacement review completed
on 2026-09-06; its Qwen viewer identity finding was fixed and independently
re-reviewed. See the
[remediation checkpoint](../../reports/2026-09-06-semantic-contract-remediation.md).

## Workflow checklist

- [x] Inspect current code, runtime evidence and clean tracked baseline.
- [x] Confirm intent and bounded scope with the user.
- [x] Compare minimal contract repair with threshold wiring and no change.
- [x] Present scope and obtain conversational approval.
- [x] Write this spec and check scope, consistency and explicit exclusions.
- [x] No visual companion needed: this is a textual contract decision.
- [x] User reviews this written spec.
- [x] Write the implementation plan, then execute TDD and independent review.

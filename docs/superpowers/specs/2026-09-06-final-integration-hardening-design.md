# Final integration review hardening

Date: 2026-09-06

Status: bounded review remediation under the user's default implementation
instruction. No new inference or deployment is included.

The whole-branch review of `b9ac98b..84e9e7c` found two Important integration
defects. These are separate from the completed three-part semantic-contract
repair and are not established causes of the failed real cohort.

## 1. Coherent occlusion evidence provenance

A candidate can contain overlays for its target and multiple possible
occluders. Positive-event projection currently copies all those keyframes but
only the target and selected occluder's tracks. A legal `unknown` decision or
one selected from several occluders can therefore fail final provenance
validation and lose otherwise valid action results.

Keep full candidate-context evidence: source tracks are the target followed
by all candidate occluder tracks in deterministic order, deduplicated. These
are considered evidence, not semantic claims that every source track occludes
the target. The actual `occluder_entity_id` stays exactly the validated choice,
including `unknown`. Candidate keyframes remain authoritative.

The alternative is filtering keyframes using an additional authenticated
ownership mapping. Retaining contextual tracks avoids changing the projection
interface or inferring ownership from filenames. Weakening validation is not
an option. Test both legal cases through final result validation and viewer
projection; truly foreign keyframes must remain rejected.

## 2. Invariant BPE vocabulary identity

The default BPE asset lies within the verified pinned source, but an explicit
external path can supply different bytes without changing the evidence cache
identity. Require the copied asset to match the vocabulary blob in the pinned
Git revision. Identical relocated bytes remain supported; differences fail
before builder invocation with the existing closed error and cleanup behavior.

Use existing bounded, replacement-ref-safe blob and snapshot mechanisms. Do not
trust only the mutable worktree copy. Preserve ownership, link, size and
TOCTOU defenses. This makes vocabulary identity invariant; adding a configurable
digest or changing request/cache schemas is unnecessary and excluded.

Test relocated identical bytes, altered external bytes, default asset behavior,
cleanup and existing pinned-source defenses. No actual altered-vocabulary SAM
inference is needed to validate this boundary.

## 3. Historical minor findings

In the same single final-review fix wave, close descriptors if prefix fsync or
child traversal setup fails; close already-constructed Fake semantic workers
if CV startup fails; and repair timestamp regressions whose invalid entity
fixtures could hide a missing timestamp check. Inject the specific failures and
assert resource cleanup/error preservation. Timestamp tests must use otherwise
valid fixtures and assert actual start/end error locations.

Lazy export reflection remains a deferred discoverability improvement, not a
runtime requirement. Correct the historical documentation: current lazy
attribute resolution neither populates module globals nor makes the names
appear in `dir()` after access.

## Verification and boundaries

Execute via the existing semantic-contract plan's single final-review fix
wave: test-first covering regressions, focused suites, one full Python suite
with configured coverage at least85%,20Node tests, syntax/whitespace checks,
then one independent scoped re-review. Adjudicate any residual findings;
do not start an unbounded fix loop.

Preserve actual thresholds, candidate generation, checkpoint, sampling, PTS,
cache keys, frozen references/results/mapping and Qwen projections. No model
replacement, paid inference, deployment, performance tuning, automatic semantic
repair, fabricated human precision or relabeled historical acceptance.

Self-review: the chosen fixes preserve existing public formats and cache
identity; the new BPE restriction makes an existing implicit invariant explicit.
Provenance tracks represent evidence context, not participant labels. No visual
design decision or additional user choice is needed for these bounded repairs.

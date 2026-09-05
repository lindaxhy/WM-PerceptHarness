# Semantic-contract remediation checkpoint

Date: 2026-09-06

The user approved the bounded repair and requested default routine execution
within the confirmed goal. This checkpoint does not authorize or claim a new
inference cohort, deployment, accepted result, human review, or PR.

## Changes and review

`df292e955a3bcf8fcacdf00318085ed63577ac40` implements the
[approved plan](../superpowers/plans/2026-09-06-semantic-contract-remediation.md):

- Preserve the existing closed `SCENE_SPATIAL_INVALID` code for ordering and
  provenance failures. Invalid output remains rejected and raw errors stay
  redacted; no spatial sorting or automatic repair was introduced.
- Version the occlusion prompt as `0906-occlusion-contract-v2`, require exact
  interval fields, and include executable positive and unknown examples.
  The validator and candidate constraints are unchanged.
- Remove the ineffective overlap setting and document the existing fixed
  positive-overlap candidate rule. Actual thresholds, algorithms and cache
  identity are unchanged.

An independent task review approved both specification compliance and code
quality with no findings. The implementer recorded failing regressions for the
generic spatial-code substitution, missing positive example, and accepted
obsolete Settings keyword before their respective fixes. Those historical RED
runs are implementation evidence, not independently reconstructed from the
final diff by the reviewer.

The separate frozen-evidence review of `04c5d70..d46dfd8` found one omission:
Qwen viewer descriptors lacked model identity. `84e9e7c` adds the recorded
`qwen3-vl-8b-instruct` identity to all five descriptors and tests its binding
to immutable metadata and source-result digests. The new test first failed
with a missing-identity `KeyError`; 35 exporter and 20 Node tests then passed.
Scoped independent re-review marked the finding addressed with no new issues.
The Sites workflow kept this correction local and metadata-only, without
initializing hosting or changing frozen projections.

That evidence review also verified all 15 display/source digests, five LAS
digests, model identities for the Doubao variants, and exactly eight referenced
PNG assets. It found no prohibited private material or fabricated acceptance
claims. Remote execution, database and GPU-idle records were checked for
internal consistency, not independently re-executed.

## Fresh combined-state verification

Verified at `84e9e7c`:

```text
.venv/bin/python -m pytest -q --cov=las_repro --cov-report=term-missing
1697 passed, 2 warnings in 75.76 seconds
Configured coverage: 85.78% (required: 85%)

node --test evaluation/viewer/tests/model.test.mjs
20 passed, 0 failed

node --check evaluation/viewer/js/app.js
exit 0

git diff --check
exit 0
```

Warnings remain the existing Starlette/httpx TestClient deprecation and
`anyio.abc.BlockingPortal` alias deprecation. No dependency changes were made.
Comparison against `3b58faa` confirms no changes under `evaluation/results`,
`evaluation/references`, `evaluation/config`, or frozen viewer `data/local`.
The user-owned untracked performance follow-up note remains outside commits.

## Acceptance and remaining work

The immutable [real acceptance report](2026-09-04-sam31-gpu-acceptance.md)
remains failed: zero positive occlusion events, two cold samples above 720
seconds, and a failed first full-pipeline cache resubmission with one new SAM
analysis and four unsubmitted samples. Isolated real-worker restart recovery
passed separately; it is not a cache-hit test.

Local prompt-example compliance does not establish real-model compliance or
improved quality. There are still no positive claims eligible for measured
human precision, and no live browser-interaction acceptance was performed.
The repaired source has not been installed into the isolated runtime.

Final whole-branch review is in progress against `b9ac98b..84e9e7c`, including
historical deferred findings. Any confirmed issues must be resolved before
release. A separately bounded verification experiment is the next planning
step; old failed reports must never be replaced or relabeled.

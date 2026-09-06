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

Final whole-branch review completed against `b9ac98b..84e9e7c`. It inspected
bounded load-bearing sections across orchestration, durable publication,
timeline/SAM lifecycle, candidate generation, provenance, ARK, evaluator,
exporter and viewer, not every line of all 119 files. It reproduced two valid
occlusion decisions failing provenance validation and identified an external
BPE asset identity gap. Code readiness is **with fixes**, separately from
failed real acceptance.

The [final hardening design](../superpowers/specs/2026-09-06-final-integration-hardening-design.md)
defines one consolidated fix wave for those two Important findings and the
historical descriptor/Fake-startup cleanup and timestamp-test minors. It keeps
all candidate evidence tracks without changing semantic labels, and requires
BPE bytes to match the pinned blob without a new cache key. Lazy import
reflection remains deferred; its historical description has been corrected.
Implementation completed at `cc6a1dbfa1b94526a8d3ba51158b95510f96eaab`.
Independent scoped re-review marked all six findings addressed, with no new
breakage or out-of-scope observations. The remaining lazy reflection item is
deferred as an optional discoverability change, not a correctness dependency.

### Final-wave regression evidence

The implementer recorded these RED/GREEN checks:

- `test_positive_occlusion_is_a_checked_projection_of_real_candidates`:
  two failures at final provenance validation before repair, then two passes
  including complete-result and viewer projection checks.
- `test_explicit_bpe_must_match_the_pinned_repository_blob`: altered external
  bytes were accepted before repair; identical and same-size altered cases
  now both pass their respective acceptance/rejection assertions, including
  pre-builder rejection and runtime-directory cleanup.
- `test_prefix_open_closes_descriptor_when_root_fsync_fails` and
  `test_tree_walk_closes_child_descriptor_when_scandir_setup_fails`: both
  exposed open descriptors before repair, then passed with original errors
  preserved and descriptors closed.
- `test_run_fake_closes_constructed_workers_when_cv_initialization_fails`:
  both constructed workers had zero close calls before repair; failure and
  normal-shutdown regressions then passed, checking exactly-once closure.
- Both negative-timestamp regression families now validate legal baseline
  fixtures first and assert exact erroneous timestamp locations; four cases
  passed, without unrelated entity-field failures satisfying the tests.

The implementer's focused integration run passed 488 tests, and the complete
SAM adapter run passed 158 tests. Its final full suite passed 1,703 tests in 77.26s.
These historical RED results are recorded implementation evidence; the scoped
review checked the fix and tests rather than rerunning that history.

The controller independently reran the final code at `cc6a1db`:

```text
.venv/bin/python -m pytest -q --cov=las_repro --cov-report=term-missing
1703 passed, 2 known warnings in 77.53 seconds
Configured coverage: 85.78% (required: 85%)

node --test evaluation/viewer/tests/model.test.mjs
20 passed, 0 failed

node --check evaluation/viewer/js/app.js
exit 0

git diff --check
exit 0
```

Frozen results, references, mapping and Qwen projection trees remain unchanged
against `3b58faa`. No runtime installation, remote inference, push or PR was
performed in this remediation. Code review is closed; real acceptance is not.

A separately bounded verification experiment is the next planning step after
review remediation; old failed reports must never be replaced or relabeled.
Keep this worktree for that continuation and the eventual new PR. The original
requirement to complete real acceptance before PR delivery remains unsatisfied.

## Subsequent bounded re-verification

The [2026-09-06 bounded real experiment](2026-09-06-semantic-reverification.md)
has now run on an isolated repaired wheel. Scene passed after one repair but
returned no spatial records; occlusion remained invalid after its one repair.
The four-call evidence is separate from the unchanged historical acceptance
results. A new five-video experiment is not recommended yet.

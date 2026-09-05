# SAM3.1 / Doubao integration acceptance

Status: **in progress, not accepted**. Recorded on 2026-09-05.

The final tool/runtime smoke below passed. Five-demo control/treatment and
cache-hit runs, recovery, human review, quantitative gates, actual viewer
publication and final branch review remain pending. No new PR is claimed.

## Immutable runtime

| Input | Identity |
| --- | --- |
| Harness source | `f4e797024df57792ac0c49cb615d88b01bf6d74f` |
| Wheel SHA-256 | `9099a0f092fcb59856e20670229c10f81b9aa893b93d266299146a6801c7e6a6` |
| Source archive SHA-256 | `65766663ed10e3847bd6a553a3f269e1e84e89dd905e4c5cafcf904a07dc4372` |
| Smoke script SHA-256 | `e6eda5bf5b1eada51289d3d59a6f3545b2f84729b0075f3c7379af8142709f8c` |
| SAM source | `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` |
| Checkpoint SHA-256 | `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` |
| Semantic backend | remote ARK `doubao-seed-2-1-pro-260628` |

The same wheel was installed without dependency resolution in separate ARK
and SAM environments. The original Qwen environment and deployment were not
modified. ARK's environment passes its dependency check. SAM reuses the accepted
Torch 2.10/CUDA 12.8 stack with its own NumPy and dependencies; inherited
unrelated package conflicts remain disclosed, not described as a clean global
environment. Runtime uses the verified FFmpeg 7.0.2 executable.

## Real final-tool smoke

The reviewed `scripts/sam31_smoke.py` ran unmodified, with local pinned inputs,
offline flags, a fresh private artifact cache and physical GPU 3. It returned
one canonical success record and exit 0. No diagnostic model or sampler
override was used.

| Measurement | Observed |
| --- | --- |
| Sample | `full_0024` |
| Original video SHA-256 | `a7a696bcdd835c083b27ca3705d13a2f22e069ebec9038581354fed39e6fbbe8` |
| GPU | RTX 5090, physical 3 |
| Source frames | 137 |
| Tracks / observations | 1 / 137 |
| Elapsed seconds | 55.689 |
| Peak Torch allocation, bytes | 16,423,134,208 |
| Artifact key | `ed7643e0e7799ce195d94fe2c8a0ff4cda27d1c7fad81510a7ade3a9d5288e0e` |
| Manifest SHA-256 | `db3c54d7a2897d735a5c25cdcdda02103fa96f550fdb545cbda16b2f06c53a2c` |

The smoke published and reloaded its artifact; a second read-only load through
the installed production artifact store verified the same manifest and 137
observations. GPUs 0–2 remained at 1 MiB during the observed GPU3 process; all
four devices returned to 1 MiB after exit, with no compute process reported.
This proves the bounded runtime/tool path, not occlusion accuracy.

## Local gates before the run

- Full Python suite: 1,652 passed, two existing dependency deprecations,
  73.65 seconds; branch-enabled combined coverage 85.69%.
- Viewer Node model: 19 passed; JavaScript syntax and branch whitespace checks
  passed.
- Frozen five-sample Qwen inputs passed the strict evaluator's original-byte,
  media, configuration and pinned-reference checks. With the unchanged mapping,
  action Event F1@0.3 is 6/44 (0.13636), and occlusion Event F1@0.3 is 0/13.
  Historical missing stage timings and checkpoint revision remain unknown.

## First cold control and transport failure

The first Doubao-only cold sample, `full_0001`, completed in 155.554 seconds.
All five inference jobs completed; the result contains nine action events, an
available scene branch, one repair and no degradation. Read-only canonical
validation of the stored result passed. Its canonical SHA-256 is
`6fc9ecb6eeaa42fadf359be486c6e082e6c9fecd1306ec3c0de43eea9924fc58`.

The actual authenticated Poll response failed the same validator because the
generic redactor replaced `source_keyframe_ids` and numeric token usage. The
original response is preserved privately (SHA-256
`cbb2c8814c10ac562c2136622f2da4e670fbf9b25b3574f8249e918461e3041c`).
The resumable driver stopped before submitting sample two. A bounded API-only
contract repair was implemented at `5f7a5d0` (1,657 tests passed, 85.75%
branch-enabled combined coverage). Independent review found a valid null-metrics
case, fixed and re-reviewed at `1046cf3` (163 covering tests passed), plus a
malformed-container `AttributeError` not covered by the original prescribed
exception tuple. The user approved the bounded amendment; the second fix and
independent re-review completed at `323393f`.

Final pre-deployment verification: 1,659 Python tests passed in 74.86 seconds,
85.75% branch-enabled combined coverage, two existing dependency warnings;
19 Node tests, JavaScript syntax and branch whitespace checks passed.

The repaired release identities are:

| Input | Identity |
| --- | --- |
| Source commit | `323393f5f5feacd87755e8bc5c7242fa05edace1` |
| Wheel SHA-256 | `de7b62489bd77a60a578daa10a15561952c6197a6cba100554eff9615395047c` |
| Source archive SHA-256 | `22b4fa7b2713eb203dffc003dd8b8510df75b15b8ac2a67acd5eba1e86926bfb` |

Only `api.py` differs from the actual GPU-smoke wheel; all 45 other product
files are byte-identical, including the pipeline, providers and prompts. Both
isolated environments were updated and all 46 installed product files compared
against the new wheel. Five owned idle control processes were restarted after
confirming terminal database state; their original logs were preserved. ARK's
dependency check remains clean. The original Qwen deployment was untouched.

Authenticated Poll of the same completed task now validates and has exactly the
stored canonical SHA-256 above. The database still held one task and five jobs;
unauthenticated Poll returned 401. No inference was repeated. The first sample
export contains 14 fine training rows. Its inference used `f4e7970`, while its
repaired transport used `323393f`; this API-only provenance distinction is
intentional and recorded. The resumed driver has submitted sample two. The
five-sample control/treatment and full acceptance gates remain incomplete.

## Completed Doubao-only cold control

All five tasks completed, passed canonical validation and caption export, and
matched their persisted results exactly. No sample degraded; two repairs were
used across the five tasks. All remained below the 12-minute bound.

| Sample | Wall seconds | Actions | Fine rows | Repairs |
| --- | ---: | ---: | ---: | ---: |
| `full_0001` | 155.554 | 9 | 14 | 1 |
| `full_0002` | 111.027 | 6 | 18 | 0 |
| `full_0024` | 64.175 | 4 | 6 | 0 |
| `full_0021` | 112.798 | 6 | 10 | 1 |
| `full_0004` | 100.263 | 7 | 11 | 0 |

Median wall time is 111.027 seconds. Timing uses actual database task creation
and terminal timestamps; the API-repair pause is not inference time.

The strict evaluator verified the five original media hashes, result hashes,
canonical structures, fixed configuration and unchanged reference/mapping:

- Action Event F1@0.3: 24/50 = **0.48** (12 matches, 32 predictions,
  18 reference action events); frozen Qwen was 6/44 = 0.13636.
- Action Event F1@0.5: 16/50 = **0.32** (8 matches); frozen Qwen was
  4/44 = 0.09091.
- Occlusion Event F1@0.3: 0/13 = **0**, with no positive control predictions.
- Configuration SHA-256:
  `98e6faccc826bc3fc0bdb957461dc4c2dddc125ada0cec0e83cac8bfd541a439`.
- Run metadata SHA-256:
  `1244cfd79df36e4eb55240af86def889b4505d4622b8daa63d0661fd68e98afa`.
- Verified control report SHA-256:
  `10fef3e1a8bd67a1658a3f390546eec361ec6b77ab5d10c434cf570cf5beab15`.

This supports a diagnostic semantic-backend comparison, not a dataset-wide
quality claim or a claim about SAM's contribution. The separate Doubao+SAM
cold run has started with the same semantic settings and a fresh CV cache.
Its CV worker runs the unmodified installed CLI under standard-library cProfile
to record actual provider `analyze` calls for cold/cache comparison; measured
hybrid timings will include that profiling overhead. GPUs 0–2 remained unused
at startup, while SAM loaded on physical GPU 3.

## Remaining acceptance

The isolated authenticated loopback service is running the Doubao-only control
with three remote semantic workers and no CV worker. Original media, query and
sampling settings are preserved. Treatment, cache, recovery, metrics and human
review evidence will be added from actual terminal jobs; no passing gate or
human verdict is inferred from this smoke or from unit tests.

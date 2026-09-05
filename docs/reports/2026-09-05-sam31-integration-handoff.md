# SAM3.1/LAS alignment integration handoff

Date: 2026-09-05

This document is the recovery point for the LAS comparison, English reference,
five-demo viewer, and SAM3.1-assisted alignment work discussed and implemented
through 2026-09-05. It deliberately contains no credentials, private host
address, temporary media URL, LAS task ID, or local absolute path.

## Continuation status — 2026-09-05

The original numbered sections below preserve the Tasks 1–9 checkpoint. The
continuation has completed Tasks 10 and 11 with independent review, plus the
Doubao backend and SAM runtime compatibility remediation. Task 12 completed its
review/fix loop at `0a66c06`; Task 13 completed independent review and its fix
round at `220efa2`. Task 14 completed review and two fix rounds at `761c242`;
its smoke tool validates local inputs, uses actual media duration and exact PTS,
suppresses upstream stdout, and reloads published artifacts. Its deployment
guide binds the compatible FFmpeg command for both smoke and workers. Task 15
is in progress; full five-demo acceptance remains unproven.
The subsequent unmodified final-wheel GPU smoke passed on physical GPU 3:
137 frames and observations, 55.689 seconds, digest-checked reload, and all
GPUs idle after exit. The
[in-progress acceptance report](2026-09-04-sam31-gpu-acceptance.md) records
immutable identities. The isolated Doubao-only five-demo run completed its
first sample in 155.554 seconds, with nine action events, an available scene
branch, one repair and no degradation. Its stored canonical result validates,
but Poll's generic sensitive-name redaction corrupts keyframe provenance and
numeric usage. The original transport response is preserved; the bounded
[Poll contract repair](../superpowers/plans/2026-09-05-poll-hybrid-provenance.md)
is in progress before re-polling that same task. No inference resubmission is
needed for this repair; full Task 15 acceptance remains incomplete.
Task 13 adds verified projection,
variant selection and bounded overlay access; its actual five-demo data and
acceptance publication remain Task 15 work. Focused verification passed 152
Python viewer/artifact tests, 19 Node model tests and 121 API/export/hybrid tests.
No integration PR or production-service deployment is claimed at this point.

The user explicitly authorized Doubao remote semantic inference. The
[Doubao amendment](../superpowers/specs/2026-09-05-doubao-semantic-backend-design.md)
therefore supersedes the historical Qwen-only/no-remote restrictions below.
Preserve frozen Qwen output and English LAS references, and evaluate both
Doubao-only and Doubao+SAM to distinguish model replacement from CV effects.

The user's ModelScope alternative succeeded: the pinned SAM3.1 multiplex
checkpoint was downloaded and hash-verified, and the corrected production
adapter processed all 137 frames of `full_0024` on physical GPU 3 with exact PTS
and digest-checked artifact reload. See the
[runtime preflight report](2026-09-05-sam31-runtime-preflight.md) for pins,
hashes, timings, and the distinction between diagnostics and acceptance.

The real Doubao-only pipeline completed and exported six fine rows, but the
scene branch degraded after its single repair. The bounded
[scene prompt contract repair](../superpowers/plans/2026-09-05-scene-prompt-contract.md)
completed at `1963dac` with independent review. A real no-override replay of the
production prompt passed structural, temporal and required-object validation
on its first response. It clarifies existing flat provenance and no-CV
spatial-list rules without relaxing validation. The
[feasibility report](2026-09-05-doubao-feasibility.md) records the actual result.

Continue using the per-plan SDD ledgers and committed history, not the historical
"next Task 10" instruction below. Remaining acceptance still requires real
five-demo control/treatment runs, cache verification, complete human occlusion
review, truthful quantitative gates, viewer projection, and a new PR.

## 1. Resume here

- GitHub repository: `lindaxhy/WM-PerceptHarness`
- Branch: `feat/sam31-evidence-integration`
- Base: `origin/main` at `b9ac98b`
  (`feat: add synchronized five-demo LAS comparison viewer (#3)`)
- Code checkpoint before this handoff: `1da02fd`
  (`fix: finish CV summary remediation`)
- Approved design:
  [`2026-09-04-sam31-las-alignment-design.md`](../superpowers/specs/2026-09-04-sam31-las-alignment-design.md)
- Implementation plan:
  [`2026-09-04-sam31-las-alignment.md`](../superpowers/plans/2026-09-04-sam31-las-alignment.md)
- Plan progress at this checkpoint: Tasks 1-9 complete; Tasks 10-15 pending.
- External state: no push and no PR were created for this branch.

The next implementation unit is Task 10, the dedicated evidence-constrained
occlusion adjudication stage. Continue on this branch only after confirming the
worktree is clean and that the current commit includes this handoff.

## 2. User intent and decisions

The conversation converged on the following requirements:

1. Preserve the existing LAS-compatible local service and compare it against
   official LAS output on the same five diagnostic demos.
2. Generate and freeze a new official LAS reference set whose free-text fields
   are English. Do not overwrite the earlier Chinese reference set.
3. Keep a maintainable, synchronized viewer with LAS on the left and the local
   result on the right while the shared video plays.
4. Align the local output to LAS by separating action events, occlusion events,
   and longer scene facts instead of forcing all granularities into one list.
5. Use the approved “Plan B” architecture: three Qwen semantic workers and one
   isolated SAM3.1 CV worker on a four-GPU host.
6. Keep Qwen3-VL-8B-Instruct as the semantic reasoner. SAM3.1 supplies object
   location, visibility, cross-frame identity, and bounded visual evidence.
7. SAM disappearance alone must never be classified as occlusion. Qwen must
   adjudicate whether a candidate is occlusion, out-of-frame motion, detector
   loss, or unknown using visible evidence.
8. Do not add Grounding DINO, DINO/DINOv3, optical flow, model ensembles,
   fine-tuning, LoRA, remote inference, or audio processing in this version.
9. Use quantitative acceptance gates and manual review for every positive
   occlusion claim.
10. The implementation must be a new PR based on the already merged PR #3,
    not an extension of or extra commits added to PR #3.

## 3. Earlier comparison questions resolved

### `full_0021`

The statement that `full_0021` remained a Pass B failure is now historical.
The current five-sample rerun completed all five samples. `full_0021` produced
14 continuous fine segments covering `[0.0, 7.3]` and five deterministic
grouped events. It passed the strict final validator after recording bounded
repair warnings. The old failure text is retained only in the report's frozen
pre-fix baseline section.

### `full_0001`

The repaired local `full_0001` result remains the same 13-segment artifact when
compared across the Chinese and English LAS evaluations. The LAS reference is
not the same: the new English LAS call regenerated boundaries and semantics,
increasing its event count from 7 to 9. Score differences therefore do not
represent a new local-model run.

### LAS language and granularity

LAS can produce English by placing an explicit English-only instruction in the
top-level query. The available calling document also describes a language
override in task context for an embodied template. There is no need to assume
that LAS output is inherently Chinese.

LAS also mixes different temporal granularities: short action events,
occlusion enter/hold/exit events, and longer scene-level facts. The local design
therefore keeps those products in separate validated branches and retains fine
segments as internal timing/provenance evidence.

### Occlusion semantics

LAS has explicit occlusion structures and event types; they are not merely a
display convention. The frozen English outputs nevertheless contain internal
inconsistencies, so they remain machine-only references rather than human
ground truth. In particular, `full_0021` has three `occlusions` records but no
corresponding occlusion semantic event, and `full_0004` lacks matching
enter/exit semantic events for its two records.

The complete comparison and the distinction between current and historical
results are documented in
[`2026-09-03-las-vs-local-implementation-report.md`](2026-09-03-las-vs-local-implementation-report.md).

## 4. Frozen English LAS reference

The new reference set is stored under
[`evaluation/references/las_official_english_2026-09-04`](../../evaluation/references/las_official_english_2026-09-04/README.md).

- Operator: `las_video_understanding`
- Version: `v1`
- Model recorded by the generation run: `doubao-seed-2-1-pro-260628`
- Samples: `full_0001`, `full_0002`, `full_0004`, `full_0021`, `full_0024`
- Total: 31 semantic events, including 13 occlusion-type events, plus 9
  `occlusions` records
- All five result JSON files use English free text and passed the recorded
  structural validation.
- The exact prompt, manifest, reference JSON, and comparison JSON are tracked.
- The output was preserved as returned; prompt-compliance gaps were not
  manually repaired.

Media caveat: `full_0002` could not be downloaded by LAS from the original
temporary endpoints. Its English reference used a complete 14.7-second,
441-frame, 30 fps, 1280x720 H.264 video-only transcode rather than the
byte-identical original. The manifest contains both hashes. The other four
samples used byte-identical frozen video bytes.

The previous 1,869 Chinese reference artifacts remain a separate historical
baseline. Every future evaluation must name which reference set it uses.

## 5. Existing five-demo viewer

The synchronized viewer was merged in PR #3 and is already on this branch's
base. Its entry point and maintenance instructions are in
[`evaluation/viewer/README.md`](../../evaluation/viewer/README.md).

Current behavior:

- one shared video clock;
- official English LAS annotations on the left;
- repaired Qwen-only local annotations on the right;
- selectable grouped events, fine segments, and scene events;
- clicking an event seeks the video;
- deterministic, validated, allowlisted static JSON;
- no backend and no upload when the session-only file picker is used.

The MP4 files are intentionally ignored and must be supplied locally. Viewer
data can be regenerated with
[`scripts/build_comparison_viewer_data.py`](../../scripts/build_comparison_viewer_data.py).

Task 13 will extend, not replace, this viewer. It will add a Qwen-only versus
Qwen+SAM3.1 selector, separate action/occlusion/scene layers, evidence and
provenance metadata, review state, and bounded relative overlay previews.

## 6. Approved hybrid architecture

The production allocation is:

| GPU | Process | Responsibility |
| --- | --- | --- |
| 0 | Qwen worker 0 | Qwen semantic inference |
| 1 | Qwen worker 1 | Qwen semantic inference |
| 2 | Qwen worker 2 | Qwen semantic inference |
| 3 | isolated CV worker | SAM3.1 segmentation and tracking |

The API and coordinator do not import GPU libraries. Qwen and SAM use separate
processes and may use separate Python environments. SAM source and checkpoint
identity are server-owned and pinned. Runtime code must not download arbitrary
code or weights.

The data flow is:

1. Qwen Pass A produces coarse actions and a bounded normalized entity list
   without an additional VLM call.
2. The CV worker samples the exact video timeline, runs one multiplex SAM3.1
   tracking stream, and publishes a content-addressed artifact.
3. CPU-only code converts tracks into bounded summaries, relations, keyframe
   references, and conservative occlusion candidates. Masks are not inserted
   into Qwen prompts.
4. Action enrichment, occlusion adjudication, and scene-fact generation run as
   independently validated semantic branches.
5. A deterministic merge preserves branch, model stage, source segment/track/
   keyframe IDs, evidence mode, confidence, repair history, warnings, and
   review status.
6. SAM OOM receives one same-resolution retry with a smaller execution chunk.
   Exhaustion degrades to Qwen-only. A failing semantic branch does not erase
   valid sibling branches.

## 7. Quantitative evaluation contract

The approved evaluator compares frozen Qwen-only and Qwen+SAM3.1 outputs to the
frozen English LAS references using immutable mappings.

Primary measurements:

- one-to-one Event F1 at temporal IoU 0.3;
- strict one-to-one Event F1 at temporal IoU 0.5;
- occlusion interval F1 and IoU;
- occlusion enter/exit boundary absolute error;
- actor, action, target, state, and result macro-F1;
- completion, degradation, repair, and review rates;
- manually reviewed factual precision and false-positive category counts;
- per-stage latency and cold/cache-hit behavior.

Acceptance gates for the five-demo integration:

1. All five demos complete and pass strict structural validation.
2. Action Event F1@0.3 falls by no more than 0.05 absolute from Qwen-only.
3. Occlusion Event F1@0.3 improves from zero to a positive value.
4. Every emitted positive occlusion is manually reviewed and at least 80% are
   supported. Emitting no positives fails the occlusion-improvement gate.
5. Every score is traceable to immutable video, LAS reference, local result,
   CV artifact, configuration, and model identity.

These five demos are diagnostic only and cannot support dataset-wide quality or
service-level claims.

## 8. Latency evidence

The last pre-SAM, Qwen-only five-demo run was highly variable:

| Sample | Video duration | End-to-end runtime |
| --- | ---: | ---: |
| `full_0001` | 10.93 s | 135.118 s |
| `full_0002` | 14.7 s | 570.381 s |
| `full_0024` | 4.57 s | 325.271 s |
| `full_0021` | 7.3 s | 255.208 s |
| `full_0004` | see frozen manifest | 465.334 s |

The closest measured example to a 10-second video took about 135 seconds, but
video duration alone did not predict runtime: the five-sample median was
325.271 seconds. No verified hybrid/SAM latency exists yet. Tasks 11, 14, and
15 must expose per-stage timing and record both cold-cache and cache-hit runs.

## 9. Implementation completed on this branch

The branch contains the approved spec/plan plus nine completed implementation
tasks. Every task received implementation review, bounded fix rounds, and
controller verification. The ignored local SDD ledger and reports are under
`.superpowers/sdd/2026-09-04-sam31-las-alignment/`.

| Task | Result | Commit span / final commit |
| --- | --- | --- |
| 1 | Per-job model routing and durable execution metrics | `3f9ac02..d741deb` |
| 2 | CV evidence contracts and deterministic entity normalization | `17bb7e7..4a239fd` |
| 3 | Exact frame PTS timeline and two-level sampling | `4664073..5b14f5c` |
| 4 | Secure content-addressed CV artifact store | `bf7be86..013847b` |
| 5 | Deterministic Fake provider and leased CV worker | `a770cf5..0365b5e` |
| 6 | 3+1 configuration and separate CV process role | `6ced581..5549054` |
| 7 | Entity-bearing Pass A without an extra VLM call | `44ee440..8b5b975` |
| 8 | Pinned SAM3.1 Object Multiplex adapter | `38143e0..5e1a5e5` |
| 9 | Bounded CV summaries and conservative occlusion candidates | `5f1b5a7..1da02fd` |

Task 9 required a user-authorized remediation cycle after its normal five-round
review cap. The final three Important findings were closed:

- the summary budget now guarantees a consumable bundle for every legal
  threshold and candidate limit;
- alias truncation is carried as bounded durable audit data through embodied
  output and export;
- normalized slug collision ownership is stable across input permutations.

The fresh formal review of `d4779df..1da02fd` reported no Critical, Important,
or Minor findings and assessed the task as Ready.

## 10. Current verification evidence

Verification was rerun from the clean `1da02fd` checkpoint on 2026-09-05:

| Check | Result |
| --- | --- |
| Task 9 focused Python group | 336 passed in 8.88 s |
| Full Python suite | 1432 passed, 2 warnings in 24.08 s |
| Python byte compilation | passed |
| `git diff --check` | passed |
| Base ancestry (`b9ac98b`) | passed |
| Viewer Node model tests | 8 passed |
| Viewer JavaScript syntax | passed |
| Worktree before adding this handoff | clean |

The two Python warnings are unchanged third-party deprecations:

- FastAPI's TestClient imports a deprecated Starlette/httpx integration;
- Starlette references the deprecated `anyio.abc.BlockingPortal` alias.

They are not product-code failures and existed before this branch.

## 11. Deferred findings for final triage

These reviewed Minor findings were intentionally deferred. They should be
triaged before the final branch-wide review, but they do not reopen Tasks 1-9.

### Task 4: artifact-store descriptor edge cases

- Failure while constructing `os.scandir(child_descriptor)` can leak that
  child descriptor before it enters the cleanup stack.
- An observer prefix descriptor is not immediately closed if its newly
  required root `fsync` fails.

### Task 6: Fake runner startup cleanup

- `run-fake` constructs Qwen workers before CV runtime startup enters the
  cleanup `finally`; CV initialization failure can leave those Qwen workers
  unclosed.

### Task 7: test intent and reflection compatibility

- Two legacy negative-timestamp tests have stale exact fixtures, so unrelated
  entity-field validation can fail before the intended timestamp assertion.
- Lazy CV worker exports work through attribute/star import but do not appear
  in module globals or `dir(las_repro.cv)` until first resolution.

Task 8 has no deferred code-review finding; its real CUDA/checkpoint smoke is
explicitly scheduled for Task 14. Task 9 has no deferred finding after the
authorized remediation review.

## 12. Work still pending

### Task 10: dedicated occlusion adjudication

Add the closed decision schema, evidence-constrained prompt, Qwen stage with one
repair, strict candidate/boundary validation, and positive event projection.
Zero candidates must create no Qwen job. Non-occlusion decisions remain audit
data and do not become positive events.

### Task 11: hybrid orchestration and fallback

Connect Pass A, CV job/artifact, summaries, action enrichment, occlusion, scene
facts, deterministic branch merge, metrics, and independent degradation. This
is where hybrid end-to-end behavior becomes available.

### Task 12: evaluator and acceptance gates

Implement deterministic one-to-one temporal matching, semantic macro-F1,
reliability metrics, review validation, canonical report JSON, and gate logic.

### Task 13: viewer projection and binding

Project allowlisted hybrid data and overlays, validate optional review records,
and extend the merged five-demo viewer with Qwen-only/Qwen+SAM3.1 selection.

### Task 14: runtime documentation and GPU smoke

Document disabled/Fake/SAM modes, independent Qwen/SAM environments, pinned
source/checkpoint identity, cache ownership, and 3+1 launch commands. Run real
SAM3.1 CUDA/checkpoint smoke on GPU 3 and Qwen isolation smoke on GPUs 0-2.

### Task 15: four-GPU acceptance and new PR

Run five-demo cold-cache and cache-hit A/B, review every positive occlusion,
generate immutable evaluation/viewer artifacts, audit SQLite/jobs/digests/GPU
memory, run final tests and branch-wide review, then push and open a new PR.

## 13. Security and external-state checklist

Credentials for GitHub, ARK, and LAS were pasted into the conversation. Their
literal values were not copied into this document or intentionally committed to
the repository. Treat all of them as exposed and revoke/rotate them before any
future API call, push, PR operation, or temporary media-host cleanup.

Before Tasks 14-15:

- confirm the user-supplied remote GPU host and the actual Qwen deployment;
- confirm the pinned local SAM3.1 source revision and checkpoint hash;
- inject credentials through protected files/environment only, never command
  history, logs, reports, JSON artifacts, prompts, or Git remotes;
- verify that any temporary repository used to serve LAS media has been
  deleted or remains private; deletion was previously pending an independently
  authenticated GitHub session with sufficient scope;
- verify that no temporary public media URL remains active;
- do not commit ignored MP4s, masks, local paths, service envelopes, task IDs,
  or raw provider errors.

No remote host was restarted and no external repository or LAS job was mutated
during this checkpoint/hand-off operation.

## 14. Recommended continuation sequence

1. Confirm this branch and a clean worktree with `git status --short` and
   `git log -1 --oneline`.
2. Read the approved design, global plan constraints, Task 10 section, and this
   handoff before editing code.
3. Create the Task 10 brief from the existing plan and continue the established
   test-first, implementation-review, and controller-verification loop.
4. Complete Tasks 10-13 locally before scheduling GPU-dependent Tasks 14-15.
5. Rotate credentials and confirm the private GPU environment before any
   external action.
6. Run the entire Python and Node suites, compile checks, repository policy,
   final branch diff review, and acceptance gates.
7. Only then push `feat/sam31-evidence-integration` and create a new PR against
   the current `main` of `lindaxhy/WM-PerceptHarness`.

Until Tasks 10-15 and their acceptance evidence are complete, this branch is a
clean, reviewed implementation checkpoint, not a PR-ready final delivery.

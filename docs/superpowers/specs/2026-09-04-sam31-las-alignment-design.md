# SAM3.1-assisted LAS alignment design

**Date:** 2026-09-04

**Status:** user-approved design, pending written-spec review

**Scope:** add SAM3.1-derived visual evidence to the existing Qwen3-VL
annotation pipeline and evaluate whether it improves alignment with the frozen
English LAS references.

## 1. Context

The current implementation runs an embodied-video pipeline around
Qwen3-VL-8B-Instruct. It produces strict continuous fine segments, enriched
action semantics, deterministic grouped events, and a full-video scene stage.
The five-sample rerun completes successfully, but explicit occlusion recall is
still zero against the frozen LAS references. The current scene schema can
represent occlusion, so the remaining gap is evidence and recall rather than
field availability.

The existing comparison also shows that LAS mixes action events, occlusion
events, and longer scene-level facts. Asking one output list to represent all
three makes event boundaries and scoring ambiguous. This design separates the
three semantic products while retaining fine segments as internal evidence.

The measured pre-SAM five-sample latency is highly variable: the nearest
ten-second example (`full_0001`, 10.93 seconds) took 135.118 seconds, while the
five-sample median was 325.271 seconds. Video duration alone is therefore not a
reliable latency predictor. The new implementation must expose per-stage timing
instead of reporting only total task time.

Relevant existing documents:

- [LAS video understanding architecture](../../architecture/las-video-understanding-design.md)
- [LAS versus local implementation report](../../reports/2026-09-03-las-vs-local-implementation-report.md)

The implementation branch starts from `main`; it does not contain the commits
from the comparison-viewer feature branch. Viewer integration is added to this
new PR only after the comparison viewer has itself entered `main`. Its commits
will not be copied or stacked into this branch.

## 2. Goals

1. Supply object location, visibility, and cross-frame identity evidence to the
   semantic pipeline through SAM3.1.
2. Improve explicit occlusion-event recall without materially regressing action
   event quality.
3. Keep Qwen3-VL-8B-Instruct as the semantic model. This phase does not add
   fine-tuning or replace the VLM.
4. Keep SAM-specific dependencies and failures isolated from the API,
   coordinator, and Qwen workers.
5. Produce versioned, cacheable, inspectable evidence with complete provenance.
6. Preserve deterministic Fake-mode development and strict output validation.
7. Publish a viewer-compatible projection so a reviewer can distinguish LAS,
   Qwen-only, Qwen-plus-SAM, and the evidence supporting each claim. Bind that
   projection to the viewer only when the viewer is available on `main`.

## 3. Non-goals

- Grounding DINO, DINO, DINOv3, optical-flow models, and model ensembles are not
  part of this version.
- The service will not train or fine-tune Qwen or SAM.
- SAM masks will not be treated as semantic truth. SAM supplies evidence;
  semantic stages remain responsible for action, occlusion, and scene claims.
- The public API will not accept arbitrary model paths, checkpoints, or Python
  modules.
- The five diagnostic videos will not be presented as a statistically stable
  dataset-level benchmark or service-level agreement.

## 4. Approaches considered

### 4.1 Chosen: separate semantic branches over shared evidence

The pipeline produces Action Events, Occlusion, and Scene Facts independently,
then validates and merges them into the public result. Fine segments remain an
internal timing and provenance layer. All branches consume the same versioned
Qwen and CV evidence.

This approach matches the distinct semantics observed in LAS, prevents
occlusion facts from being forced into action labels, and allows each branch to
be evaluated independently. It also lets the system degrade one branch without
discarding otherwise valid annotations.

### 4.2 Rejected: append SAM fields to the current scene-stage prompt

This is the smallest code change, but it retains the overloaded event list and
does not create a stable occlusion boundary contract. Failures would remain
difficult to attribute to segmentation, evidence summarization, prompting, or
event merging.

### 4.3 Deferred: deterministic CV-only event generation

Pure geometric rules are attractive for reproducibility, but a missing track
cannot by itself distinguish occlusion, leaving the frame, object deformation,
or detector failure. Deterministic logic will propose and validate candidates,
but it will not make unsupported semantic assertions.

## 5. Architecture

### 5.1 GPU allocation

The accepted production layout on the four RTX 5090 GPUs is:

| Device | Process | Responsibility |
| --- | --- | --- |
| GPU 0 | Qwen worker 0 | Qwen semantic inference |
| GPU 1 | Qwen worker 1 | Qwen semantic inference |
| GPU 2 | Qwen worker 2 | Qwen semantic inference |
| GPU 3 | SAM3.1 CV worker | segmentation and video tracking |

Each Qwen process loads one local Qwen3-VL-8B-Instruct snapshot. The SAM worker
runs in its own process, virtual environment, and model cache because its
official runtime requirements may differ from the existing Qwen environment.
The API and coordinator import neither GPU stack.

All work continues through the durable SQLite job queue. Jobs have immutable
backend/model routing, and large CV artifacts travel through validated local
artifact references rather than SQLite blobs. A Qwen worker is not reserved
while its task waits for SAM, so multiple videos can form a pipeline across the
three Qwen devices and the one SAM device.

### 5.2 Component boundaries

`CvEvidenceProvider` is the only interface exposed to the coordinator-facing
pipeline. It accepts a validated video, frame-time mapping, entity inventory,
and sampling configuration, and returns a versioned `CvEvidenceArtifact`.

Two implementations are required:

- `FakeCvEvidenceProvider` returns deterministic evidence without GPU packages.
- `Sam31EvidenceProvider` adapts the pinned local SAM3.1 code and checkpoint.

No action or occlusion vocabulary belongs in the SAM adapter. It detects and
tracks prompted entities. Evidence summarization and semantic interpretation
are separate components so that SAM can later be replaced without changing the
public result schema.

## 6. Data flow

1. **Media validation and probing.** The existing media layer resolves the
   allowlisted input and builds an authoritative original-frame PTS mapping.
2. **Pass A and entity inventory.** The existing first Qwen pass continues to
   propose the coarse action structure and also emits canonical entity
   candidates. This does not add another VLM call.
3. **Entity normalization.** A deterministic normalizer deduplicates aliases,
   assigns stable entity IDs, and ranks actors/hands, manipulated objects,
   containers or likely occluders, then other action-mentioned entities. The
   default cap is 16 prompts and is configurable. Omitted candidates are
   recorded in warnings.
4. **SAM3.1 evidence.** The CV worker detects and tracks the normalized prompts,
   preserving original frame indices and PTS-derived timestamps.
5. **Evidence summarization.** Masks stay in the artifact store. The semantic
   pipeline receives compact track summaries, deterministic spatial relations,
   and a bounded set of overlay keyframes around meaningful changes.
6. **Action branch.** Qwen produces LAS-aligned semantic action events from the
   existing fine timing evidence plus the CV summary. Fine segments remain
   available for audit but are not added to the primary scored event set.
7. **Occlusion branch.** Deterministic logic identifies visibility-change
   candidates. A dedicated Qwen stage adjudicates whether each candidate is
   occlusion, out-of-frame motion, detector loss, or unknown, and identifies an
   occluder only when supported.
8. **Scene Facts branch.** Qwen produces non-action state, location, relation,
   and outcome facts from the same shared evidence.
9. **Deterministic merge and validation.** Branch outputs are normalized,
   provenance-linked, sorted, and strictly validated. No merge step invents a
   semantic event or silently rewrites an unsupported boundary.
10. **Export and viewer projection.** The durable task result, training export,
    evaluator, and viewer use explicit projections of the same canonical result.

## 7. Evidence contract

### 7.1 Request

A CV request contains:

- canonical video path inside the allowed media root;
- video SHA-256, duration, frame count, and original PTS mapping;
- ordered entities with stable ID, canonical English label, aliases, and role;
- sampling policy and numerical thresholds;
- requested provider and immutable model snapshot identity.

API-supplied values cannot override the configured provider code or checkpoint
path.

### 7.2 Artifact manifest

Every `CvEvidenceArtifact` contains:

- schema version, provider name, checkpoint identity and checkpoint SHA-256;
- cache key, creation timestamp, normalized configuration and source-video hash;
- terminal evidence status and structured warnings;
- per-entity tracks;
- references to mask chunks and optional overlay frames;
- aggregate timings, processed-frame counts, and peak device memory when
  available.

Each track uses a provider-independent shape:

- `track_id` and `entity_id`;
- observations with original `frame_index` and `timestamp_seconds`;
- normalized `bbox_xyxy`, mask reference, visibility state, and confidence;
- lifecycle status and gaps;
- deterministic summaries such as area, center, containment, overlap, and edge
  proximity.

Masks are never embedded in prompts or top-level task JSON. Artifact references
must resolve beneath the configured cache root, pass symlink containment checks,
and match their recorded digest.

### 7.3 Provenance in semantic outputs

Every public semantic event records:

- producing branch and model stage;
- source fine-segment IDs, track IDs, and keyframe IDs;
- evidence mode: `hybrid`, `vlm_only`, or `unavailable`;
- confidence, repair history, and review flag.

Absence of a SAM track is never positive evidence of absence or occlusion.

## 8. Sampling and temporal boundaries

For videos no longer than 30 seconds, SAM processes the source frame rate capped
at 30 fps. For longer videos, it performs an initial pass capped at 8 fps and
refines one-second windows around visibility-state changes at up to 30 fps.

All processed frames retain their original indices and PTS-derived timestamps.
Reported boundaries snap only to observed frame times, and the artifact records
the effective temporal resolution. Repeated, non-monotonic, or missing timestamp
mappings fail validation rather than being replaced with a synthetic uniform
grid.

Occlusion candidate generation uses visible-to-uncertain-or-missing-to-visible
transitions together with spatial evidence such as an overlapping or containing
potential occluder. Edge departure, lack of a plausible occluder, and low track
confidence are explicit counter-signals. The Qwen adjudicator may return
`unknown`; it may not turn a weak candidate into a confident occlusion solely to
increase recall.

## 9. Cache design

The cache key is a SHA-256 over a canonical serialization of:

- evidence schema version;
- source video SHA-256;
- provider and checkpoint SHA-256;
- ordered normalized entity prompts;
- sampling configuration and evidence thresholds.

Artifacts are written to a task-local temporary directory, validated, fsynced,
and atomically published under the content-addressed cache root. A cache hit
requires a valid manifest and matching digests for every referenced chunk.
Corrupt, partial, or mismatched entries are quarantined from reads and recomputed;
they are never repaired in place.

Changing prompts, checkpoint, sampling, thresholds, or schema produces a new
key. Changing only a downstream Qwen prompt or evaluation mapping reuses the CV
artifact.

## 10. Failure handling and degradation

- Qwen structured-output failures retain the existing single repair attempt.
  Subsequent behavior follows the strict per-branch validation contract.
- SAM3.1 out-of-memory failure releases task resources and retries once with a
  smaller frame batch or chunk. It does not reduce temporal resolution or alter
  thresholds during the retry.
- SAM timeout, worker crash, invalid artifact, or failed retry degrades the task
  to Qwen-only evidence and adds `CV_EVIDENCE_UNAVAILABLE` with the sanitized
  cause.
- A failed Occlusion or Scene Facts branch does not discard valid Action Events.
  The unavailable branch is explicit in result status and warnings.
- Low-confidence or broken tracks produce `unknown` or no claim, never a
  fabricated occlusion.
- Existing leases, heartbeats, and expired-job recovery apply to both backends.
  Worker shutdown must release model and task-scoped GPU state.

The top-level task may remain `COMPLETED` when a non-required evidence branch
degrades, but its warnings and branch status must make that degradation visible
to API clients, exports, evaluation, and the viewer.

## 11. Performance observability and initial budget

Each task records queue wait, media decode, Pass A, SAM3.1, Action, Occlusion,
Scene Facts, merge, and end-to-end time separately. It also records:

- cold start versus cache hit;
- number of SAM frames, prompts, tracks, chunks, and retries;
- Qwen repair count and generated-token count when available;
- peak memory by stage and device;
- branch degradation and fallback counts.

For the initial five-demo GPU acceptance, every 8-to-15-second video must finish
within 12 minutes on a cold CV cache. A valid cache hit must skip SAM inference
entirely. Device memory must return to a stable post-task baseline, and routing
must show Qwen only on GPUs 0-2 and SAM only on GPU 3.

This limit is an engineering guardrail, not a production SLA. After at least 30
representative videos are available, the project will report end-to-end and
per-stage P50/P95 for both cold and cached execution. No linear extrapolation
from video duration will be used.

## 12. Evaluation

The evaluator runs Qwen-only and Qwen-plus-SAM3.1 against byte-identical frozen
inputs, the frozen English LAS references, and fixed generation settings.

Primary quantitative outputs are:

- one-to-one Event F1 at temporal IoU 0.3;
- strict one-to-one Event F1 at temporal IoU 0.5;
- occlusion interval F1 and IoU;
- occlusion enter and exit boundary absolute error;
- actor, action, target, state, and result macro-F1;
- completion, degradation, repair, and review rates;
- human factual precision with categorized false positives.

The initial integration acceptance gates are:

1. All five demos complete and pass strict structural validation.
2. Action Event F1 at tIoU 0.3 decreases by no more than 0.05 absolute versus
   Qwen-only on the same frozen inputs.
3. Occlusion Event F1 at tIoU 0.3 improves from the current zero baseline to a
   positive value.
4. At least 80% of positive occlusion statements are supported by manual video
   review. Every positive statement in the five-demo run is reviewed; a run
   that emits no positive statements fails gate 3.
5. Every score is traceable to an immutable input, LAS reference, local result,
   CV artifact, configuration, and model identity.

These gates establish that the integration works and does not trade away action
quality for unsupported occlusion recall. Broader model-quality claims require a
larger blinded and human-reviewed set.

## 13. Viewer projection and integration boundary

The SAM implementation owns a validated, static viewer projection but does not
own or duplicate the comparison-viewer feature. Once the viewer is present on
`main`, the new PR adds a selectable Qwen-plus-SAM result alongside the frozen
LAS and Qwen-only data. Its default comparison remains LAS on the left and the
selected local result on the right.

For each local event, the viewer displays branch, evidence mode, source track
IDs, confidence, warnings, and review status. When available, a reviewer can
toggle bounded overlay keyframes showing masks or boxes. The viewer does not
load the full mask store, require a backend, or expose local filesystem paths.
Generated viewer data is a validated projection containing only allowlisted
fields and relative static asset references.

## 14. Configuration and deployment

SAM configuration is server-owned and includes:

- enabled provider (`fake`, `sam31`, or disabled);
- allowlisted local repository and checkpoint paths;
- checkpoint SHA-256 and model identity;
- GPU device, timeout, lease, and chunk size;
- entity cap, sampling policy, thresholds, cache root, and cache limits.

The production SAM process uses a pinned source revision and local checkpoint.
It does not download code or weights during task execution. Secrets, access
tokens, absolute media paths, and raw model errors are excluded from logs and
public task results.

Qwen and SAM launch commands remain separate so either environment can be
upgraded or rolled back independently. Deployment documentation must include a
compatibility matrix and a GPU smoke command for each environment.

## 15. Test and acceptance strategy

### 15.1 Unit and contract tests

- deterministic Fake provider behavior;
- entity normalization, priority ordering, cap, and warning behavior;
- request and artifact schema validation;
- original-frame PTS mapping and boundary snapping;
- cache-key stability and invalidation;
- artifact containment, digest, atomic-publication, and corruption rejection;
- visibility transitions, edge exits, low-confidence gaps, and candidate
  generation;
- deterministic merge, provenance, and public projections.

### 15.2 Failure and worker tests

- SAM OOM followed by a smaller-chunk retry;
- timeout, worker crash, expired lease, and retry exhaustion;
- Qwen-only degradation with `CV_EVIDENCE_UNAVAILABLE`;
- independent Action, Occlusion, and Scene Facts branch degradation;
- no GPU imports in API, coordinator, Fake mode, or evaluator;
- immutable backend routing and cleanup after terminal tasks.

### 15.3 GPU and end-to-end tests

- Qwen isolation smoke on GPUs 0-2 and SAM isolation smoke on GPU 3;
- one real SAM artifact with valid masks, tracks, timestamps, and provenance;
- five-demo cold-cache completion;
- five-demo cache-hit rerun proving no SAM inference occurred;
- frozen Qwen-only versus Qwen-plus-SAM evaluation and manual occlusion review;
- viewer projection validation and synchronized playback checks;
- idle-memory, SQLite integrity, terminal job state, and artifact-digest audit.

The pre-change baseline for this branch is 744 passing local tests under Python
3.12. Any unrelated baseline failure must be resolved or explicitly separated
before SAM integration is evaluated.

## 16. Delivery sequence

1. Add provider-independent evidence contracts, cache primitives, and Fake
   provider without changing production output.
2. Add durable CV job routing and failure behavior with integration tests.
3. Package and smoke-test the pinned SAM3.1 adapter in its isolated GPU runtime.
4. Extend Pass A entity inventory and integrate evidence summarization.
5. Implement the Action, Occlusion, and Scene Facts branches and deterministic
   merge.
6. Add evaluation metrics, five-demo A/B artifacts, and manual-review records.
7. Add the viewer projection and deployment documentation. If the comparison
   viewer has entered `main`, bind the new projection to its UI; otherwise keep
   the projection independently testable and defer only the UI binding.
8. Run the complete local suite and four-GPU acceptance before creating the
   implementation PR.

No Grounding DINO or model fine-tuning is added during this sequence. Evidence
from the A/B evaluation will determine whether either becomes a justified later
project.

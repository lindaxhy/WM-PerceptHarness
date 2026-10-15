# Installation and evaluation

## Core grounded annotation and event scoring

From the repository root, use Python 3.12+, FFmpeg and FFprobe:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/verify_install.py
```

The core package includes the scorer and OpenAI-compatible clients. Grounded annotation requires a separate local SAM3.1 installation, checkpoint, tokenizer and compatible GPU runtime; installation does not download these assets. Scoring requires a judge endpoint even though the numeric matching runs locally. The metric extras below are for separate video-quality evaluators.

## Optional metric environments

Use separate virtual environments for CLIP-IQA+, VBench Motion Smoothness, and incompatible upstream evaluators. Install the core package in each environment to obtain the same `percept` command:

```bash
python3.12 -m venv .venv-clipiqa
source .venv-clipiqa/bin/activate
python -m pip install -e .
```

All dependency declarations live in `pyproject.toml`; there are no separate requirements files. Install a hardware-compatible Torch/TorchVision build first, then select the extra for this environment:

```bash
python -m pip install -e '.[clipiqa]'
# Or, in a separate CUDA environment:
python -m pip install -e '.[motion]'
python -m pip install --only-binary=:all: --no-deps vbench==0.1.5
```

Follow [metric runtime installation](metrics.md) for checkpoint setup and the reason VBench uses `--no-deps`. These extras retain the recorded direct dependency versions; they are not proof of a validated GPU environment. External evaluators such as DOVER still use their upstream-compatible runtime.

A unified CLI does not activate another environment automatically: `percept score motion` uses the currently active Python environment. Use that environment's full `bin/percept` path when orchestrating several environments.

## Reproducible core environment (optional uv)

```bash
uv sync --locked
uv run --locked percept --help
```

`pyproject.toml` defines dependencies; `uv.lock` fixes their resolved versions for uv users. Pip installation remains supported but does not consume `uv.lock`.

The lockfile also resolves optional extras, but does not select a hardware-specific CUDA wheel or lock separately installed VBench/upstream source trees and checkpoints. Record those independently. See [reproducibility](#reproducibility).

## Annotation configuration

The annotation client speaks the OpenAI-compatible chat completions protocol. Set `PERCEPT_OPENAI_BASE_URL`, `PERCEPT_OPENAI_MODEL`, and `PERCEPT_OPENAI_API_KEY` in the shell where you run `percept`. Provider examples and optional request settings are in [`.env.example`](../.env.example).

If you prefer a local environment file:

```bash
cp .env.example .env
# Edit .env with your endpoint, model and key, then load it:
set -a
. ./.env
set +a
```

The application does not automatically load `.env`. Keep keys outside committed configuration files. `--model` currently selects an alias from the configured registry; it is not an arbitrary provider model ID. For a single model, set `PERCEPT_OPENAI_MODEL` and omit `--model`. Multi-model runs can use `PERCEPT_OPENAI_MODEL_REGISTRY` instead; those two settings are mutually exclusive.

### Timeouts and transient failures

The OpenAI-compatible backend retries timeouts, transport errors, and HTTP 429,
500, 502, 503 and 504. Other HTTP errors fail immediately. Client class constants
set three extra attempts and exponential backoff with jitter from a five-second
base. Configure the fixed per-request timeout with
`PERCEPT_OPENAI_TIMEOUT_SECONDS` (default 180 seconds). Dynamic video-length
scaling and its upper timeout cap have been removed. This is not a batch deadline.

Each `ModelRequest` carries `max_frames` (default 128, including SAM overlays)
and optional `max_output_tokens`. When the token limit is omitted, stage defaults
are 16,384 for drafts/revision, 8,192 for evidence review and 4,096 for SAM prompt
review. Stage payloads accept these same fields. Effective budgets enter the
request-cache identity. The unused internal hints `media_resolution`,
`reasoning_effort` and `clip_context` have been removed; they never changed the
API payload. Provider-specific generation options remain in
`PERCEPT_OPENAI_EXTRA_BODY`. Removing these hints from cache identity makes old
request-cache entries miss; completed-result resume is unchanged. The CLI resolves model aliases before creating a client. Connection types, formats and numeric ranges are validated by `Settings`; the CLI checks required credentials. Direct Python callers of the internal client are responsible for supplying valid connection values. Reserved request-body overrides and external responses are still checked by the adapter.

`OpenAICompatVideoModel.MAX_REQUEST_BYTES` fixes the encoded request limit at
32 MiB; `MAX_OUTPUT_CHARS` limits model text to 1,000,000 characters. Oversized
requests thin original video frames while retaining mandatory SAM overlays;
requests that still exceed the limit fail. These limits and retry behavior are
internal class constants, not environment settings. The former
`PERCEPT_OPENAI_MAX_FRAMES`, request/output-size, retry and dynamic-timeout
settings no longer apply. For repeatability, see
[result identity and resume](usage.md#result-identity-and-resume).

### Targeted SAM3.1 grounding

Use the pinned SAM source revision `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`, its checkpoint and `sam3/assets/bpe_simple_vocab_16e6.txt.gz` tokenizer, a compatible GPU runtime, and FFmpeg 5.1+. The provider verifies the source revision. Annotation defaults to `--cv sam31`. SAM assets and the GPU are loaded only when the visual reviewer requests a targeted check. Configure them for production coverage; a failed check is recorded as unresolved and the final revision must remain conservative. Point these settings at actual local assets:

| Environment variable | Value |
|---|---|
| `PERCEPT_CV_REPOSITORY_PATH` | Local SAM source checkout |
| `PERCEPT_CV_CHECKPOINT_PATH` | Local checkpoint file |
| `PERCEPT_CV_BPE_PATH` | `<repository>/sam3/assets/bpe_simple_vocab_16e6.txt.gz` (default `models/sam3/sam3/assets/...`) |
| `PERCEPT_CV_DEVICE` | GPU device index |
| `PERCEPT_CV_CACHE_ROOT` | Writable evidence cache |

SAM loads the clean pinned checkout, checkpoint and tokenizer directly from
these local paths. Keep them unchanged for the lifetime of a run. The loader
checks the source revision, tokenizer identity and imported module locations;
it does not create private copies of the source tree or checkpoint. NumPy, Pillow and
OpenCV belong to the SAM runtime and are imported only when that runtime is used.

The core install neither installs this runtime nor validates a live GPU run. Run a small production annotation and inspect the `.audit.json` sidecar (`status`, `error`, `metadata.evidence`), overlay counts and archived CV manifests before scaling up. A successful zero-detection SAM result may have `no_tracks: true` and no overlays; the reviewer still sees the original video, and missing tracks do not prove absence.

## Annotation

`percept annotate` runs the annotation workflow. The backend is `openai`, for OpenAI-compatible model APIs.

```bash
percept annotate --videos video.mp4 \
  --output outputs/actions
```

Each successful `<video_stem>.json` is exactly `{ "duration": number, "events": [{ "start": number, "end": number, "description": string }] }`. The sibling `<video_stem>.audit.json` contains execution status, errors and review metadata. A failure writes its audit without a public annotation; an empty successful event list remains a valid annotation. `results.jsonl` summarizes the batch.

The validator canonicalizes model duration rounding within one microsecond to authoritative media metadata (including an endpoint overshoot within that tolerance); larger duration discrepancies still fail. Times are finite seconds relative to the original container playback start and lie within the video endpoint. Initial video-stream delays are preserved; trailing audio does not extend an available video-stream duration. SAM uses decoded frame PTS on this clock and preserves original decoder indices. Each event is an independently describable visible process or change, bounded by meaningful completion, interruption or a different activity. Long events, gaps, overlaps and unfinished actions are valid. Segmentation follows the original event-boundary rules: independently describable phases may remain separate, with no requirement to merge a continuous manipulation into one event or to exhaustively cover the timeline. There are no event categories or fixed event-length limits.

Experimental confidence-gated annotation is available on the experiment branch:

```bash
percept annotate --videos video.mp4 --cv sam31 \
  --event-workflow confidence-gated --output outputs/confidence
```

The default remains `--event-workflow reviewed` for comparison. The selector also
works with `--manifest` or `PERCEPT_EVENT_WORKFLOW=confidence-gated`; use a separate output directory for each workflow or pass `--force` to rerun.

In the experiment, `confidence_draft` returns event confidence and optional
uncertainties. Confidence <= 0.6 requires a question; every reported question is
processed regardless of confidence. With no questions, the draft-only route
still takes one VLM call. Confident errors can remain unreported.

The v4 workflow treats question kinds as descriptions, not SAM eligibility gates:

1. `grounded_draft` and `evidence_review` produce events and questions (or
   `confidence_draft` supplies both in confidence-gated mode).
2. Every concrete question enters `sam_target_plan`, which sees only its source
   video window plus up to 0.5 seconds of context on each side. It selects one
   relevant visible object/reference and a normalized XYXY box. A null target
   must include a reason; no target is invented when segmentation cannot help.
3. SAM probes that reference frame. `sam_prompt_review` checks the actual mask;
   only an accepted mask proceeds to tracking within the question window. One
   corrected reference retry is allowed. Empty masks do not prove disappearance.
4. `grounded_revision` responds to every concern ID. `revision_verification`
   independently checks candidate events without reading the revision rationale.
   Newly discovered structured questions also enter local SAM planning before
   the second revision. After two revision rounds, unresolved results remain
   `needs_review` rather than completed.

The shared per-video planning budget is 32 windows and 300 contextual seconds
across initial review and verification. Excess questions and unavailable targets
remain recorded for conservative revision/review; the budget never resets on a
retry. A final-round new question remains unresolved rather than starting an
unbounded loop. Each planned question adds a VLM call, and each nonempty SAM probe
adds a mask-review call. SAM still cannot decide semantic boundaries or reveal
fully hidden paths by itself. Public JSON remains `{duration, events}`.

The review records a short `motion_evidence` explanation naming a support/background reference or stating that the evidence is insufficient. `metadata.phases` lists completed VLM stages; `stage_timings`, `sam_checks`, `evidence`, `review`, `recheck_budget` and `unresolved_concerns` are audit information. Each SAM request records its question, object, reference frame, spatial prompt and time window. Detailed masks/tracks remain in content-addressed CV artifacts, not the public annotation. Internal SAM sampling remains up to 30 fps for short/local clips; no per-frame positions are added to event JSON. Cache identities bind selected original frames/times, source duration/count, probe/track mode and spatial prompts, so one window cannot reuse another window's evidence.

The camera check is a short instruction inside `evidence_review`: compare apparent object displacement with its support/background. It adds no optical flow, pose estimation, model or extra camera call. Uncertain motion is described conservatively. SAM image-coordinate geometry alone cannot establish world motion, cause, disappearance, completion or semantic boundaries. Successful execution and a revised description do not certify semantic correctness; unresolved concerns remain auditable.

### Result identity and resume

Each video writes `<video_stem>.json` and `<video_stem>.audit.json`.
Same-name videos use numeric suffixes such as `clip--1.json`; the audit records
which resolved source path owns a result. Use `result_file` in `results.jsonl`
to locate it. Existing unreadable files are preserved and a free suffix is used.

A result is skipped by default when its
source path matches, its status is completed, its annotation passes schema
validation, and its v4 audit records successful verification (or the explicit
confidence-gated draft-only route). Failed, incomplete or malformed results are retried. Legacy result
envelopes can still be read. There are no content checksums, code/environment
fingerprints, automatic history archives or per-run manifests.

```bash
percept annotate --videos video.mp4 --output outputs/actions
```

Resume does not detect changed video bytes, models, prompts or settings. Use a
new output directory for a different experiment. Manually edited valid event
annotations can be read without updating an audit digest. Each JSON is replaced
atomically; the audit's temporary `writing` status prevents an interrupted pair
from being treated as complete. A failed run removes its stale public annotation.
Use separate output directories for concurrent invocations; there is no writer
lock. `results.jsonl` describes the latest completed batch.

A separate VLM request cache is enabled by `PERCEPT_SEMANTIC_RESULT_CACHE=true`
under `<PERCEPT_WORK_ROOT>/semantic-cache`. It uses the video path, file size,
modification time and request settings to identify requests, and validates cached
responses against the current schema. `--force` disables this cache for an
independent annotation repeat and also bypasses completed-result skipping. Merely
changing the output directory does not disable request caching. SAM evidence has
its own cache. Internal hashes name cache entries and assign jobs to shards;
they are not file integrity checks.

The CV cache is a private, user-owned local directory. Cooperating workers use
per-key process locks and publish complete artifact directories atomically.
Each read checks the request identity, frame mapping, registered files, sizes
and overlay signatures. Do not modify cache entries while a run is using them.
The simplified internal CV schema accepts one target and one spatial prompt.
Each request uses the local ID `target` and retains the original target description.
Request scopes distinguish tracks; the description remains part of the cache key.
Old CV entries containing retired entity/sampling fields become cache misses and
are recomputed. Public annotation JSON and completed-result resume rules are
unchanged; use `--force` to regenerate an already completed annotation.

Targeted tracking samples the selected window at `PERCEPT_CV_MAX_FPS` and always
includes its reference frame. OOM retries use the same original frame indices.
Unused full-video scan/refinement settings and area/occlusion thresholds were
removed. Shared limits cover overlays (24), processed frames (10,000) and total
observations (64,000). SAM uses its pinned upstream defaults for confidence
(0.5), compilation (off), and object capacity (16). These defaults are not
separate environment settings. Artifact byte limits are internal storage
bounds, not model parameters or limits on the entire cache directory.

The retired `PERCEPT_CV_MIN_CONFIDENCE`, `PERCEPT_CV_MODEL_ALIAS`,
`PERCEPT_CV_COMPILE_MODEL`, `PERCEPT_CV_EXECUTION_CHUNK_FRAMES`,
`PERCEPT_CV_CACHE_MAX_FILES`, and `PERCEPT_CV_CACHE_MAX_BYTES` variables are
ignored; remove them from existing environments. CV requests no longer carry
a configurable confidence threshold, so previous CV cache keys are not reused.
Completed annotations still follow the existing resume rules.

Environment numbers and booleans use Pydantic Settings parsing, including
standard boolean spellings such as `true`/`false` and `1`/`0`. FPS and timeouts
must remain finite and positive. Model aliases are registry keys without a
restricted naming grammar; connection and provider-model errors are reported
when requests run. Setting both a single model and a model registry is rejected.

## Video frame sampling

VLM inputs use one FFmpeg decoding pass per extraction. Sampling intervals start
at the requested span's beginning; each interval contributes its first source
frame, and empty intervals contribute none. Timestamps come from the selected
frames' PTS on the container playback clock. Sampling faster than the source
does not duplicate frames. The model adapter still applies its frame budget.

Videos that change resolution use the first frame's output canvas; a change of
aspect ratio may stretch later frames. The source-PTS sampling contract has a
new request-cache version. Completed annotations still follow the resume rules
above: use `--force` to regenerate them with the new sampler.

## Grounded event scoring

The official [WGO-Bench grading description](https://macrodata.co/blog/annotating-robot-video-subtasks) motivates the temporal and semantic tests. This implementation is an adaptation: it uses natural observable events, its own strict judge prompt and independent one-to-one matching, with no timestamp snapping or duration normalization. It is not an exact reproduction of published WGO numbers.

Annotate matching sample IDs, then configure an OpenAI-compatible semantic judge:

```bash
export PERCEPT_JUDGE_BASE_URL="https://your-judge.example/v1"
export PERCEPT_JUDGE_MODEL="your-judge-model-id"
export PERCEPT_JUDGE_API_KEY="your-judge-key"
percept score events --reference outputs/reference \
  --system model=outputs/generated --out outputs/event_scores
# Inspect outputs/event_scores/events.json
```

The default temporal IoU threshold is `0.75` (`--tiou-threshold` changes it). Temporal assignment maximizes one-to-one match count, then IoU. The judge tests whether eligible descriptions express the same observed action, object, destination, direction and outcome, ignoring wording and timing. Segment precision is temporal matches / predicted events; recall is temporal matches / reference events; `segment_f1 = 2 * temporal_matches / (reference_events + prediction_events)`. `semantic_match_rate = semantic passes on the predicted temporal assignment / temporal matches`, or null if there are no temporal matches. This is **conditioned on predicted temporal matches**, unlike original WGO ground-truth-window label accuracy. Joint assignment independently maximizes one-to-one matches among edges passing both temporal and semantic tests; `joint_f1 = 2 * joint_match_count / (reference_events + prediction_events)`. Joint matching can select different event pairs from the temporal assignment, so `joint_f1` is not generally `segment_f1 * semantic_match_rate`. All counts and candidate judgments are retained for audit. A zero-event pair has null precision/recall/F1; a one-sided empty pair has zero F1 and undefined precision or recall where its denominator is zero.

The scorer accepts flat result directories (`video_path` stem or explicit `sample_id`/`job_id`) and nested sample directories. It excludes `.percept` histories, CV archives and root shard status files from primary annotations. Every sample ID in either side is reported. Missing, failed or invalid records, or a judge failure, yield an error sample, null system aggregate and nonzero exit. Valid empty `events` lists remain in the aggregate. `--out` is a directory containing `events.json`; the report includes source paths, status, counts, per-sample matching and a sanitized judge identity. Inspect invalid counts and source provenance before quoting scores. Old categorical JSON is rejected, not guessed or converted. In tests, `events_cli.main(argv, judge=callable)` permits fully offline scoring with a synthetic judge; that path does not validate hosted judge behavior.

## Direct video metrics

Run each command in its documented runtime environment:

```bash
percept score clipiqa --video video.mp4 --output outputs/clipiqa.json
percept score motion --video video.mp4 --cache-dir /path/to/vbench-cache \
  --output outputs/motion.json
```

Both outputs share:

| Field | Meaning |
|---|---|
| `metric` | `clipiqa+` or `vbench_motion_smoothness` |
| `score` | Metric-specific scalar |
| `higher_is_better` | Score direction |
| `video` | Input path |
| `versions` | Runtime package versions |

CLIP-IQA+ additionally records frame sampling, resizing, device, and MemoBench revision. Motion Smoothness records the checkpoint path, the official command, original result, and log. Both refuse to overwrite existing result files. Event reports are dataset-level reports with a different schema; no combined score is computed.

Use the short command selector `clipiqa` even though its output metric identifier is `clipiqa+`.

Other upstream evaluators use `percept score` commands listed in the [catalog](metrics.md). Their result schemas and validation stages are documented individually.

## Revision verification and unresolved annotations

The v4 annotation workflow assigns stable concern IDs after review. Revision
returns public events plus internal per-concern responses (`changed`, `retained`,
or `unresolved`), rationale, and source-video timestamps. A separate VLM call
checks the candidate against every concern and the original video. A changed
wording alone does not resolve an unsupported event boundary.

There are at most two revision/verification rounds. A rejected first revision is
retried with verifier feedback, including any newly identified issues. Remaining
issues yield `needs_review`, preserve the candidate JSON and audit history, and
cause a nonzero CLI exit. Manifest status counts distinguish these candidates
from completed annotations and execution failures. The event scorer rejects
`needs_review` records; keep their `.audit.json` sidecars with the JSON files.

Audit metadata contains `concern_ledger`, `revision_attempts`,
`initial_unresolved_concerns`, and `revision_validation`. A passed verification
clears the active unresolved list but preserves the original review and history.
Confidence-gated drafts without uncertainty retain their existing draft-only
route and record `revision_validation.status=not_required`.

Existing v3 annotations are not automatically reused by annotation commands;
they must pass the new workflow. Public event JSON remains `duration`/`events`.
Excluding target planning and mask checks, reviewed mode uses four VLM calls on
first-round success or six if a second round is necessary. Verification uses a separate call to the
configured VLM and the existing video sampling, not a different model or a
higher-density temporal probe. Passing is a model judgment, not human quality
certification. Every question is now considered for local SAM evidence regardless
of its category.

## Multi-GPU manifest annotation

Manifest mode requires SAM3.1 (`--cv sam31`).

A JSONL manifest supplies one absolute `video_path` and a unique relative
`job_id` per line. Input checksum fields are not used.
Each job writes normal annotation results under `<output>/<job_id>/`.

```bash
percept annotate --manifest corpus.jsonl \
  --cv sam31 --cv-devices 0,1,2,3 --workers-per-gpu 2 \
  --request-concurrency 9 --output outputs/annotations
```

One spawned SAM process stays loaded per GPU. Independent CPU threads own VLM
clients and serialize CV requests to their GPU. The per-process VLM limit ramps
from two healthy calls, up to `--request-concurrency`, and halves on observed
HTTP 429; ordinary backend transport retries remain enabled. Multiple hosts can
use the same manifest and output root with distinct `--shard-index` values and
one common `--shard-count`. SHA-256 of `job_id` assigns each job to exactly one
shard. All hosts must see the input and output paths.

Shard status files record worker PIDs, active jobs, accepted results, failures,
and observed request concurrency. Missing SAM evidence or a missing full review
is not accepted as a successful manifest job. Completed annotations are reused by default; request and CV-artifact
caches are independent of this option. To retry a subset,
create a manifest of failed/pending jobs after checking the prior worker is no
longer running. Do not restart a live shard just because an observation timed
out. Size CV-cache byte/file budgets and temporary storage for the full corpus.


## Verify the installation

Run the single check script in the environment you plan to use:

```bash
python scripts/verify_install.py
python scripts/verify_install.py --metric clipiqa --cache-dir ./weights
python scripts/verify_install.py --metric motion --cache-dir ./weights
```

The default checks Python, core imports, FFmpeg/FFprobe execution, the active environment’s `percept` launcher on `PATH`, and small deterministic JSON/event-scoring examples. Optional profiles also check runtime imports, the pinned PyIQA/VBench version, device availability and weight-file presence. CLIP-IQA+ can use `--device cpu`; Motion Smoothness requires CUDA. Only run the profile for the environment you installed. A failure exits with status 1 and lists the failed check.

The script does not download weights, contact a VLM API or perform model inference. It is an installation/preflight check, not a replacement for full regression or scientific validation. Other official evaluators retain their documented `--check-only` and upstream setup procedures.

## Reproducibility

For a reported run, record input paths and pair IDs; model/provider configuration without secrets; prompt/code revision; runtime versions and device; checkpoint sources and versions; sampling settings; sample counts and exclusions. Preserve raw results and separate output directories for independent repeats. Use `--force` for independent annotation repeats.

Save `python --version`, `python -m pip freeze` and `python -m pip check` for each evaluator environment. The motion-only VBench environment deliberately differs from the full upstream dependency metadata. A successful local check does not establish successful real-model inference or a complete benchmark run.

Historical LAS tools remain on [archive/las-comparison-viewer](https://github.com/lindaxhy/WM-PerceptHarness/tree/archive/las-comparison-viewer) and [legacy-las-service](https://github.com/lindaxhy/WM-PerceptHarness/tree/legacy-las-service).

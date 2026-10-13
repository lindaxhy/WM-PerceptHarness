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

The OpenAI-compatible backend retries timeouts, transport errors, and HTTP 429, 500, 502, 503 and 504. Other HTTP error statuses fail immediately. The default is three extra attempts, with exponential backoff and jitter starting from a five-second base. Configure these through:

```bash
export PERCEPT_OPENAI_TRANSPORT_RETRIES=3
export PERCEPT_OPENAI_RETRY_BACKOFF_SECONDS=5
export PERCEPT_OPENAI_TIMEOUT_SECONDS=180
export PERCEPT_OPENAI_TIMEOUT_SECONDS_PER_VIDEO_SECOND=0
export PERCEPT_OPENAI_MAX_TIMEOUT_SECONDS=1800
```

Set retries to `0` to disable transport retries. A positive per-video-second value scales each request's timeout as `min(max_timeout, base_timeout + per_video_second * requested_span_seconds)`; zero keeps the fixed base timeout. These are request settings, not an overall batch deadline. Model-output repair is a separate pipeline step. Check failed result records and retry logs before changing sampling or timeout settings.

For repeatability and the separate request-level cache, see [result identity and resume](usage.md#result-identity-and-resume).

The client limits each encoded JSON request to 32 MiB by default. Full-resolution SAM PNG overlays (including baseline and local evidence together) can exceed that limit. If a result reports `request exceeds configured size limit`, inspect the evidence sizes and set `PERCEPT_OPENAI_MAX_REQUEST_BYTES` to a budget supported by your endpoint; for example, `134217728` permits 128 MiB locally. This does not change the provider's limits. The client preserves all authenticated overlays and fails if they cannot fit.

### Mandatory SAM3.1 grounding

Use the pinned SAM source revision `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`, its checkpoint and `sam3/assets/bpe_simple_vocab_16e6.txt.gz` tokenizer, a compatible GPU runtime, and FFmpeg 5.1+. The provider verifies the source revision. Annotation defaults to `--cv sam31` and fails if SAM cannot run. Point these settings at actual local assets:

| Environment variable | Value |
|---|---|
| `PERCEPT_CV_REPOSITORY_PATH` | Local SAM source checkout |
| `PERCEPT_CV_CHECKPOINT_PATH` | Local checkpoint file |
| `PERCEPT_CV_CHECKPOINT_SHA256` | SHA-256 digest of that checkpoint |
| `PERCEPT_CV_BPE_PATH` | `<repository>/sam3/assets/bpe_simple_vocab_16e6.txt.gz` (default `models/sam3/sam3/assets/...`) |
| `PERCEPT_CV_DEVICE` | GPU device index |
| `PERCEPT_CV_CACHE_ROOT` | Writable evidence cache |

The core install neither installs this runtime nor validates a live GPU run. Run a small production annotation and inspect `status`, `error`, `metadata.evidence`, overlay counts and archived CV manifests before scaling up. `--backend fake --model fake --cv fake` is a synthetic offline test path; fake CV cannot be paired with a production model. A successful zero-detection SAM result may have `no_tracks: true` and no overlays; the reviewer still sees the original video, and missing tracks do not prove absence.

## Annotation

`percept annotate` runs the annotation workflow. The default backend is `openai`; `fake` remains an explicit test option.

```bash
percept annotate --videos video.mp4 \
  --template embodied_action_captioning --output outputs/actions
```

One successful record contains `video_path`, `template`, `status`, `data`, `error`, `elapsed_seconds`, `result_file`, `provenance`, and `data_sha256`. `data` holds the template-specific output; action times are seconds on the original video timeline. `results.jsonl` aggregates the batch but does not include every per-video metadata field.

A successful `data` object is exactly `{ "duration": number, "events": [{ "start": number, "end": number, "description": string }] }`. Times are finite seconds relative to the original container playback start and lie within the video endpoint. Initial video-stream delays are preserved; trailing audio does not extend an available video-stream duration. SAM uses decoded frame PTS on this clock and preserves original decoder indices. Each event is an independently describable visible process or change, bounded by meaningful completion, interruption or a different activity. Long events, gaps, overlaps, empty lists and visibly failed or unfinished actions are valid. The older categorical `segments`, `semantic_events`, actor and skill fields are incompatible; regenerate old annotations for this scorer.

The pipeline nominates entities, runs baseline SAM, drafts events, independently reviews original video frames and real SAM overlays, makes up to three contextual local SAM rechecks within a 30-second budget, then revises conservatively. A recheck can add entities; track IDs are scoped to each evidence call. `metadata.phases`, `metadata.evidence`, `metadata.review`, `metadata.recheck_budget` and `metadata.unresolved_concerns` expose the audit. SAM geometry never proves semantic cause, disappearance, completion or event boundaries. If SAM baseline fails, annotation fails; a local recheck failure is recorded as unresolved. The review and budget do not certify that every concern was resolved.

### Result identity and resume

A free `<video_stem>.json` is used for the first source claiming that name. Another source with the same stem gets `<stem-prefix>--<sha256-of-resolved-path>.json`. Ownership is checked against the full resolved source path, even when videos have identical bytes. Existing disambiguated paths remain stable on later runs. Use `result_file` in `results.jsonl` to find each result; do not guess a suffix or assume directory traversal order assigns the plain filename. An unreadable existing record is preserved and the new result uses the disambiguated path. If both names are occupied by unverifiable records, use a fresh output directory.

A result is skipped only when all of these match:

- Resolved input path and SHA-256 of the video bytes.
- Template, model alias, effective built-in backend settings and prompt context.
- Run settings, packaged Python/prompt asset digest, dependency versions, and FFmpeg/FFprobe version fingerprints.
- Completed status and intact annotation payload digest.

Legacy records without provenance are rerun. Changed settings or input bytes trigger evaluation. Credentials are included only as a digest: changing API keys conservatively reruns because an account may route a model ID differently. Raw API keys, endpoint URLs, proxies and extra request dictionaries/headers are not persisted; digests distinguish their values without writing credentials into manifests. Record the non-secret provider/model configuration separately for publication: a hash is not a recoverable configuration file.

Custom models, custom pipeline registries, custom transports/frame extractors, and any CV execution currently disable resume because the loaded external implementation/weights cannot yet be verified completely. They still receive provenance and run manifests, but run again each time. Local fingerprints cannot detect a provider silently changing the remote model behind an unchanged ID.

```bash
percept annotate --videos video.mp4 \
  --template embodied_action_captioning --output outputs/actions --force
```

`--force` reruns even a matching result, for independent repeats or changed remote model behavior. Before replacing a current result, its exact bytes are archived in `.percept/history/<record-sha256>.json`. `.percept/runs/<run-id>.json` records the run settings, timestamps, input/result mapping, status, and result-file digest; that digest also identifies a later archived record. Per-video and aggregate files are replaced atomically. Inputs or configuration that change during evaluation produce a failed record, not a reusable success.

There is also a separate request-level cache for the OpenAI-compatible backend, enabled by `PERCEPT_SEMANTIC_RESULT_CACHE=true`. It lives under `<PERCEPT_WORK_ROOT>/semantic-cache` (default `work/semantic-cache`) and can reuse validated model responses during reruns. Replayed responses pass through the current sanitizer and validator; rejected or malformed responses are not published as successful cache entries. `--force` disables this cache as well as whole-result skipping. Merely choosing a different output directory does not guarantee fresh model calls; use `--force` for independent annotation repeats, including self-agreement runs.

A writer lock prevents overlapping batches from using the same output directory. A normal exit or exception releases it. After an abrupt process/machine termination, inspect `.percept/writer.lock` (host/PID) and remove it only after confirming the original writer has stopped. The last manifest may remain `running` after an abrupt termination. Use separate output directories for parallel jobs. Each `results.jsonl` describes the most recent completed batch, while the per-run manifests retain earlier mappings.

Annotation supports only `embodied_action_captioning`, which is selected by default when `--template` is omitted. Pass confirmed object names using `--prompt-context` when helpful. This is naming guidance, not a prescribed action sequence; the action pipeline identifies relevant entities directly from the video.

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
| `video`, `video_sha256` | Input identity |
| `versions` | Runtime package versions |

CLIP-IQA+ additionally records frame sampling, resizing, device, and MemoBench revision. Motion Smoothness records checkpoint identity, the official command, original result, and log. Both refuse to overwrite existing result files. Event reports are dataset-level reports with a different schema; no combined score is computed.

Use the short command selector `clipiqa` even though its output metric identifier is `clipiqa+`.

Other upstream evaluators use `percept score` commands listed in the [catalog](metrics.md). Their result schemas and validation stages are documented individually.

## Multi-GPU manifest annotation

A JSONL manifest supplies one absolute `video_path` and a unique relative
`job_id` per line. Optional `expected_sha256` binds an input to a known version.
Each job writes normal annotation results under `<output>/<job_id>/`.

```bash
percept annotate --manifest corpus.jsonl --template embodied_action_captioning \
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
is not accepted as a successful manifest job. Existing conservative CV resume
rules remain: whole-result skipping is disabled for unverifiable CV execution;
validated request and CV-artifact caches can still be reused. To retry a subset,
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

The default checks Python, core imports, FFmpeg/FFprobe execution, the active environment’s `percept` launcher on `PATH`, and small deterministic JSON/event-scoring examples. Optional profiles also check runtime imports, the pinned PyIQA/VBench version, device availability and weight SHA-256. CLIP-IQA+ can use `--device cpu`; Motion Smoothness requires CUDA. Only run the profile for the environment you installed. A failure exits with status 1 and lists the failed check.

The script does not download weights, contact a VLM API or perform model inference. It is an installation/preflight check, not a replacement for full regression or scientific validation. Other official evaluators retain their documented `--check-only` and upstream setup procedures.

## Reproducibility

For a reported run, record input checksums and pair IDs; model/provider configuration without secrets; prompt/code revision; runtime versions and device; checkpoint sources and hashes; sampling settings; sample counts and exclusions. Preserve raw results and separate output directories for independent repeats. Use `--force` for independent annotation repeats.

Save `python --version`, `python -m pip freeze` and `python -m pip check` for each evaluator environment. The motion-only VBench environment deliberately differs from the full upstream dependency metadata. A successful local check does not establish successful real-model inference or a complete benchmark run.

Run the current regression suite with `python -m unittest discover -s tests -v`. Earlier development tests are preserved in [Git history](https://github.com/lindaxhy/WM-PerceptHarness/tree/ace1be715038296d445d904934d297a10ef6fbde/tests). Historical LAS tools remain on [archive/las-comparison-viewer](https://github.com/lindaxhy/WM-PerceptHarness/tree/archive/las-comparison-viewer) and [legacy-las-service](https://github.com/lindaxhy/WM-PerceptHarness/tree/legacy-las-service).

# Installation and evaluation

## Core annotation and temporal fidelity

From the repository root, with Python 3.12+:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
percept --help
```

Install FFmpeg and FFprobe through your system package manager and verify both are on `PATH`. SAM3.1 extraction needs FFmpeg 5.1+; plain annotation does not use that newer extraction option.

```bash
ffmpeg -version
ffprobe -version
```

The base package does not install Torch, SAM, PyIQA, or VBench. A remote VLM endpoint performs annotation inference. Temporal fidelity consumes existing annotations without API calls.

For semantic fidelity and paired statistical tests:

```bash
python -m pip install -e '.[fidelity]'
```

Semantic scoring loads a sentence-transformers model; provision/cache it before an offline run. The embedding model is controlled by `--embedding-model`.

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

Use `--extra fidelity` as well when reproducing semantic evaluation. `pyproject.toml` defines dependencies; `uv.lock` fixes their resolved versions for uv users. Pip installation remains supported but does not consume `uv.lock`.

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

### Optional SAM3.1 evidence

Add `--cv sam31` after configuring a compatible local SAM checkout, checkpoint, tokenizer assets and GPU runtime. This adds CV evidence for supported occlusion and scene semantics. Configure these settings for your environment:

| Environment variable | Value |
|---|---|
| `PERCEPT_CV_REPOSITORY_PATH` | Local SAM source checkout |
| `PERCEPT_CV_CHECKPOINT_PATH` | Local checkpoint file |
| `PERCEPT_CV_CHECKPOINT_SHA256` | SHA-256 digest of that checkpoint |
| `PERCEPT_CV_BPE_PATH` | Local tokenizer vocabulary file |
| `PERCEPT_CV_DEVICE` | GPU device index for your runtime |
| `PERCEPT_CV_CACHE_ROOT` | Writable local artifact cache directory |

The core installation does not install the GPU runtime or download weights. The old server-specific deployment procedure and its fixed-GPU smoke script are preserved on the [viewer archive branch](https://github.com/lindaxhy/WM-PerceptHarness/tree/archive/las-comparison-viewer/docs/deployment/sam31-runtime.md); they are not a required setup step for ordinary annotation. Run a small annotation with your own video and `--cv sam31` to validate an optional CV installation, and inspect the result's CV status and warnings rather than assuming that a completed annotation proves CV succeeded.

## Annotation

`percept annotate` runs the annotation workflow. The default backend is `openai`; `fake` remains an explicit test option.

```bash
percept annotate --videos video.mp4 \
  --template embodied_action_captioning --output outputs/actions
```

One successful record contains `video_path`, `template`, `status`, `data`, `error`, `elapsed_seconds`, `result_file`, `provenance`, and `data_sha256`. `data` holds the template-specific output; action times are seconds on the original video timeline. `results.jsonl` aggregates the batch but does not include every per-video metadata field.

An embodied-action result may legitimately contain `data.segments: []` when no action is observed. An empty action list is not itself a failed annotation; inspect `status`, `error`, and the template-specific output. Do not invent an action simply to make the timeline nonempty.

### Result identity and resume

A free `<video_stem>.json` is used for the first source claiming that name. Another source with the same stem gets `<stem-prefix>--<sha256-of-resolved-path>.json`. Ownership is checked against the full resolved source path, even when videos have identical bytes. Existing disambiguated paths remain stable on later runs. Use `result_file` in `results.jsonl` to find each result; do not guess a suffix or assume directory traversal order assigns the plain filename. An unreadable existing record is preserved and the new result uses the disambiguated path. If both names are occupied by unverifiable records, use a fresh output directory.

A result is skipped only when all of these match:

- Resolved input path and SHA-256 of the video bytes.
- Template, model alias, effective built-in backend settings, query and prompt context.
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

For wrist/main-camera annotation, first run `embodied_active_object_detection` on the wrist video, then pass confirmed object names to the main-video action annotation using `--prompt-context`. This is naming guidance, not a prescribed action sequence.

## Scene visibility labels

Scene events (`data.semantic_events` and `annotation_branches.scene_facts.events`)
support `out_of_frame` for an absence interval following visible departure
through a frame edge, and `enter_frame` for a visible entry or re-entry
transition. These labels do not imply motion relative to the world; camera
movement can also produce frame exit and entry. Physical covering/uncovering
uses `occlusion_enter`, `occluded`, and `occlusion_exit`. Missing detection alone
does not establish either cause; use `unknown` without supporting evidence.

The visibility labels extend the scene vocabulary only, not fine action skills.
When SAM evidence is available, the review VLM receives the complete scene draft,
video frames, spatial choices and tracking/candidate evidence. A single call
returns a complete revised scene and visibility decisions. The revised scene
replaces the draft in all final scene-event fields; draft responses remain in
the request cache when enabled. Review also runs with an empty candidate list.
SAM itself provides tracking, not semantic classifications. Contradictory
visibility labels, dropped draft objects and invalid scene/candidate references
trigger bounded repair; exhausted review fails annotation rather than publishing
an unreviewed draft as success. Without usable SAM evidence the original scene
path remains, with explicit CV status. Review scope is scene facts and events;
coarse task description and fine action segments remain separate outputs.
Current action fidelity excludes frame visibility labels, as it excludes
occlusion and other types outside its action families.

## Event-timeline fidelity

The scorer expects matching IDs in a nested layout:

```text
outputs/reference/sample_001/sample_001.json
outputs/generated/sample_001/sample_001.json
```

Prepare that layout directly by naming matching videos consistently and annotating each into its own sample directory:

```bash
percept annotate --videos data/real/sample_001.mp4 \
  --template embodied_action_captioning --output outputs/reference/sample_001
percept annotate --videos data/generated/sample_001.mp4 \
  --template embodied_action_captioning --output outputs/generated/sample_001
percept score fidelity --reference outputs/reference \
  --system model=outputs/generated --out outputs/fidelity.json
```

A flat annotation directory is not accepted by the current fidelity loader. Records with missing or null `data.segments` are excluded; an empty list is accepted. The loader extracts recognized action events from `data.semantic_events` and pairs shared sample IDs. Occlusion events and unknown event types are excluded from this fidelity calculation. Inspect `paired_samples` against your intended evaluation set and report exclusions; a missing result is not a zero score. Ensure the configured annotation pipeline produces the semantic events your protocol needs, and inspect an example before scaling up.

At the per-sample level, two empty event timelines have temporal IoU 1.0. The current aggregate reporter does not handle a comparison in which every paired timeline on both sides has zero recognized events: it raises `StatisticsError` while averaging undefined event F1 values. This is a reporting limitation, not a reason to discard valid actionless annotations. Comparisons of two systems also require at least one shared paired sample.

The report contains per-system frame event-family mIoU, event F1 at tIoU thresholds, outcome agreement, and per-family statistics. With two systems it adds paired comparisons and duration/event-count strata. The default temporal path needs no model downloads.

For semantic similarity:

```bash
python -m pip install -e '.[fidelity]'
percept score fidelity --reference outputs/reference \
  --system model=outputs/generated --semantic \
  --self-agreement a=outputs/real_repeat_a b=outputs/real_repeat_b \
  --out outputs/fidelity_semantic.json
```

Each repeat uses the same nested ID layout and should be annotated separately with `--force`. Self-agreement measures variation between two real-video annotation runs. The existing CLI prints ratios normalized by that baseline; the JSON stores raw system aggregates and `self_agreement`, not the printed normalized ratios. Preserve stdout alongside the report if using those ratios. No normalization is meaningful when the corresponding baseline is zero. The parent directory of `--out` must already exist; the scorer does not create it.

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

CLIP-IQA+ additionally records frame sampling, resizing, device, and MemoBench revision. Motion Smoothness records checkpoint identity, the official command, original result, and log. Both refuse to overwrite existing result files. Fidelity reports are dataset-level reports with a different schema; no combined score is computed.

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

The default checks Python, core imports, FFmpeg/FFprobe execution, the active environment’s `percept` launcher on `PATH`, and small deterministic JSON/temporal-matching examples. Optional profiles also check runtime imports, the pinned PyIQA/VBench version, device availability and weight SHA-256. CLIP-IQA+ can use `--device cpu`; Motion Smoothness requires CUDA. Only run the profile for the environment you installed. A failure exits with status 1 and lists the failed check.

The script does not download weights, contact a VLM API or perform model inference. It is an installation/preflight check, not a replacement for full regression or scientific validation. Other official evaluators retain their documented `--check-only` and upstream setup procedures.

## Reproducibility

For a reported run, record input checksums and pair IDs; model/provider configuration without secrets; prompt/code revision; runtime versions and device; checkpoint sources and hashes; sampling settings; sample counts and exclusions. Preserve raw results and separate output directories for independent repeats. Use `--force` for independent annotation repeats.

Save `python --version`, `python -m pip freeze` and `python -m pip check` for each evaluator environment. The motion-only VBench environment deliberately differs from the full upstream dependency metadata. A successful local check does not establish successful real-model inference or a complete benchmark run.

The previous development regression suite is preserved in [Git history](https://github.com/lindaxhy/WM-PerceptHarness/tree/ace1be715038296d445d904934d297a10ef6fbde/tests). The current repository ships the installation check above. Historical LAS tools remain on [archive/las-comparison-viewer](https://github.com/lindaxhy/WM-PerceptHarness/tree/archive/las-comparison-viewer) and [legacy-las-service](https://github.com/lindaxhy/WM-PerceptHarness/tree/legacy-las-service).

# WM-PerceptHarness

Annotate videos, compare real and generated action timelines, and evaluate video quality with individual metrics.

[Quick start](#quick-start) · [Compare videos](#compare-real-and-generated-videos) · [Video metrics](#evaluate-video-quality) · [Documentation](#documentation)

## What can I run?

| Task | Input | Output | Requirements |
|---|---|---|---|
| Grounded event annotation | One video or a directory of videos | `{duration, events: [{start, end, description}]}` in each successful result | FFmpeg, a vision-capable OpenAI-compatible API, local SAM3.1 assets and GPU |
| Event scoring | Matching reference/system event annotations | Segment F1, semantic match rate and joint F1 | Core package and a semantic judge endpoint |
| Video quality | One video | A score for the selected metric | The metric's runtime and model weights |

Annotation uses sampled video frames; it does not use audio. You supply the videos and, for event scoring, their reference/system pairing. Metrics produce separate results; the repository does not compute a unified benchmark score. See the [metric guide](docs/metrics.md) for implementation and validation status.

## Quick start

### 1. Install

Use **Python 3.12+**. Install **FFmpeg and FFprobe** with your system package manager, then:

```bash
git clone https://github.com/lindaxhy/WM-PerceptHarness.git
cd WM-PerceptHarness
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python scripts/verify_install.py
```

This installs the annotation client and event scorer. Grounded annotation also requires local SAM3.1 assets and a compatible GPU runtime; see [setup](docs/usage.md#targeted-sam31-grounding). Other GPU metrics have separate [runtime requirements](docs/usage.md#optional-metric-environments). `pip install` installs code and dependencies; it does not download model weights.

### 2. Configure the annotation model

Set your endpoint URL, provider model ID and API key in the same terminal:

```bash
export PERCEPT_OPENAI_BASE_URL="https://your-provider.example/v1"
export PERCEPT_OPENAI_MODEL="your-vision-model-id"
export PERCEPT_OPENAI_API_KEY="your-api-key"
```

Choose a model that accepts image inputs through the chat completions API. Configure local SAM3.1 for targeted checks before annotation; see [configuration](docs/usage.md#annotation-configuration). `.env` files are not loaded automatically.

### 3. Annotate your video

```bash
percept annotate \
  --videos ./videos/sample_001.mp4 \
  --output ./outputs/actions
```

Pass a directory to `--videos` to process its videos recursively. Annotation
produces event descriptions and bounds on the original video clock through a
single event annotation pipeline.

By default (`--event-workflow reviewed`), the VLM drafts events and independently reviews the original video. Only concrete object identity, location, occlusion or trajectory questions trigger SAM: a text + candidate box prompt is checked on one reference frame before tracking that object within the relevant window. There is no unconditional baseline SAM pass. Camera-motion doubts and editorial changes stay in the existing visual review. Tracking supports review but does not prove an action, outcome or boundary.

To try the experimental workflow that handles self-reported uncertainties:

```bash
percept annotate --videos ./videos/sample_001.mp4 \
  --event-workflow confidence-gated --output ./outputs/experiment
```

The first VLM call produces events, confidence scores and optional questions.
Despite the selector name, **every explicit question is processed regardless of
confidence**; a score of 0.6 or lower requires a question but is not the only SAM
trigger. Events retain the original segmentation rules, without fixed-length
windows or forced merging of continuous actions.

- No questions: return the draft after one VLM call.
- Every question enters local SAM target planning, regardless of its category.
  SAM probes and tracks a relevant visible target within that question's window;
  unavailable targets are explicitly recorded rather than guessed.
- Revision must respond to every concern, followed by a separate verification
  call. New verifier questions receive the same local SAM planning before one
  bounded retry. Remaining problems are saved as `needs_review` and return a
  nonzero exit code. See [workflow and budgets](docs/usage.md#revision-verification-and-unresolved-annotations).

Invalid optional SAM targets, such as a reference frame outside the question
window, are dropped and recorded in the audit. The valid question still reaches
revision, and other valid targets remain eligible. Invalid event data and invalid
question text/type/windows still fail validation. Confidence, questions and dropped
targets stay in the audit; public JSON remains `{duration, events}`.

This experiment remains opt-in: confident errors can go unreported, and SAM cannot
recover fully hidden paths or off-screen outcomes. A completed revision is not a
quality certification. See [workflow details](docs/usage.md#annotation)
for configuration, budgets and output behavior.

### 4. Read the results

For the example above, a fresh output directory contains:

```text
outputs/actions/
├── sample_001.json        # Only duration and events
├── sample_001.audit.json  # Execution state, review and evidence references
└── results.jsonl      # Batch summary, status and result filenames
```

Read `sample_001.json` directly for the annotation; execution `status`, `error` and intermediate review live in `sample_001.audit.json`. Failed runs have an audit file and no successful annotation file. Timestamps are seconds on the original video. If video filenames collide, use the `result_file` recorded in `results.jsonl` to locate the correct JSON.

Completed, structurally valid results for the same source path are skipped by default. Add `--force` to rerun them without the VLM request cache. Use a new output directory when changing videos, models or workflow settings; resume does not compare their contents or configuration. See [outputs and resume](docs/usage.md#result-identity-and-resume) for the exact rules.

## Annotate across multiple GPUs

Use `--manifest` to annotate multiple videos in parallel across GPUs. Ordinary
`--videos` runs process videos sequentially. Configure the VLM API and local
SAM3.1 runtime as above, then create `corpus.jsonl` with one unique relative
`job_id` and an absolute `video_path` per line:

```jsonl
{"job_id":"video_001","video_path":"/data/videos/video_001.mp4"}
{"job_id":"video_002","video_path":"/data/videos/video_002.mp4"}
```

```bash
percept annotate --manifest corpus.jsonl \
  --cv-devices 0,1,2,3 \
  --workers-per-gpu 2 --request-concurrency 9 \
  --output outputs/annotations
```

- `--cv-devices` selects physical GPUs. Each GPU has one SAM process, which
  loads the model on its first targeted request and reuses it for later requests.
  Different GPUs can process different videos concurrently; one video's SAM
  inference uses one GPU.
- `--workers-per-gpu` controls CPU threads running video workflows, each with its
  own VLM client. Two threads can overlap VLM calls and SAM work, but share one
  SAM instance and serialize SAM requests on their GPU.
- `--request-concurrency` caps concurrent VLM API calls across this invocation.
  The limit starts at two, increases after successful calls, and decreases when
  HTTP 429 responses are observed.

The example uses four GPUs and eight video workflow threads. SAM remains
conditional on the annotation workflow's questions. Add
`--event-workflow confidence-gated` to use the experimental workflow.
Results are written under `outputs/annotations/<job_id>/`, with progress in
`shard-00.status.json` at the output root. See
[multi-GPU manifest annotation](docs/usage.md#multi-gpu-manifest-annotation)
for multi-host sharding and retry behavior.

## Compare real and generated videos

Annotate each side with the SAM setup above. Name paired videos with the same sample ID and use matching sample directories:

```bash
percept annotate --videos ./videos/real/sample_001.mp4 --output outputs/reference/sample_001
percept annotate --videos ./videos/generated/sample_001.mp4 --output outputs/generated/sample_001
export PERCEPT_JUDGE_BASE_URL="https://your-judge.example/v1"
export PERCEPT_JUDGE_MODEL="your-judge-model-id"
export PERCEPT_JUDGE_API_KEY="your-judge-key"
percept score events --reference outputs/reference \
  --system model=outputs/generated --out outputs/event_scores
```

`--out` names a directory containing `events.json`. The scorer also accepts flat result directories; it identifies samples from audit metadata or the annotation filename. Plain human-authored `{duration, events}` ground truth is also accepted. A missing, failed or invalid annotation makes the system aggregate null and the command exits nonzero. A successful empty `events` list is valid. Old categorical `segments`/`semantic_events` JSON is incompatible and must be regenerated. The scoring formula and adaptation limits are in [event evaluation](docs/usage.md#grounded-event-scoring).

## Evaluate video quality

Video metrics run independently of annotation and do not need VLM API credentials. Use the environment for the metric you select; one environment is not expected to contain every upstream evaluator.

### CLIP-IQA+

Create a dedicated environment and install a hardware-compatible PyTorch/TorchVision build as described in the [CLIP-IQA+ setup](docs/metrics.md#clip-iqa). Then install the metric, prepare its weights, and score:

```bash
python3.12 -m venv .venv-clipiqa
source .venv-clipiqa/bin/activate
# Install the appropriate PyTorch/TorchVision build before the next step.
python -m pip install -e '.[clipiqa]'
mkdir -p ./weights/pyiqa ./weights/clip
curl --fail --location --output './weights/pyiqa/CLIP-IQA%2B_learned_prompts-603f3273.pth' \
  'https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/CLIP-IQA%2B_learned_prompts-603f3273.pth'
curl --fail --location --output ./weights/clip/RN50.pt \
  'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt'
python scripts/verify_install.py --metric clipiqa --cache-dir ./weights
percept score clipiqa --video ./videos/sample_001.mp4 \
  --cache-dir ./weights --output outputs/sample_001.clipiqa.json
```

The JSON includes the score, sampled-frame count, sampling settings and input path. `--device auto` selects CUDA when available, otherwise CPU. CLIP-IQA+ measures sampled-frame image quality.

### Motion Smoothness

Use a separate CUDA environment with a compatible PyTorch/TorchVision build; follow the [motion runtime notes](docs/metrics.md#motion-smoothness), including the reason for `--no-deps`:

```bash
python3.12 -m venv .venv-motion
source .venv-motion/bin/activate
# Install the appropriate PyTorch/TorchVision build before the next step.
python -m pip install -e '.[motion]'
python -m pip install --only-binary=:all: --no-deps vbench==0.1.5
mkdir -p ./weights/amt_model
curl --fail --location --output ./weights/amt_model/amt-s.pth \
  'https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth'
python scripts/verify_install.py --metric motion --cache-dir ./weights
percept score motion --video ./videos/sample_001.mp4 \
  --cache-dir ./weights --gpu 0 --output outputs/sample_001.motion.json
```

The JSON includes the score and paths to the official evaluator result and log. Motion Smoothness measures local interpolation smoothness; a smooth video can still depict an incorrect action.

**Weights:** Install Python dependencies with `pip`, then download model weights
with the `curl` commands above. `video_metrics/__init__.py` only records cache-relative paths;
installation checks and scoring check that the files exist. No SHA-256 verification is performed. Use the same `--cache-dir`
for checks and scoring; `./weights` is relative to the current directory.
Other metrics require their own [checkpoint preparation](docs/metrics.md#checkpoint-preparation).

### Other individual metrics

These commands select a metric directly:

| Command | Measures |
|---|---|
| `percept imaging-quality` | Image quality with MUSIQ |
| `percept aesthetic-quality` | Aesthetic quality |
| `percept temporal-flickering` | Temporal flickering on static/near-static inputs |
| `percept dynamic-degree` | Dynamic degree |
| `percept subject-consistency` | Subject consistency |
| `percept background-consistency` | Background consistency |
| `percept overall-consistency` | Video-text consistency with ViCLIP |

They share the official VBench implementation internally. Prepare the [pinned source checkout, runtime and per-metric weight manifest](docs/metrics.md#individual-metric-commands) first. For example:

```bash
percept imaging-quality \
  --vbench-root /path/to/VBench \
  --videos-path ./videos/sample_001.mp4 \
  --cache-dir /path/to/vbench-cache \
  --weights-manifest /path/to/imaging_quality.weights.json \
  --gpu 0 --output-dir outputs/sample_001.imaging-quality
```

Add `--check-only` to verify inputs, source and weights without starting inference. Use `percept <metric> --help` for options. `overall-consistency` also needs `--prompt` or `--prompt-file`; `temporal-flickering` needs `--static-subset-ack` after checking the input subset. Results retain the official metric identifiers.

CLIPScore, DOVER and other external evaluators have separate entries in the [metric guide](docs/metrics.md#vbench-and-external-evaluators). Their availability as commands does not imply a completed full-benchmark validation.

## Documentation

| Guide | Contents |
|---|---|
| [Usage](docs/usage.md) | Installation checks, configuration, annotation/event scoring workflows and outputs |
| [Metrics](docs/metrics.md) | Metric protocols, weights, input examples and reporting scope |
| [Third-party notices](docs/third-party-notices.md) | Upstream sources and component licenses |

`python scripts/verify_install.py` checks the local installation and a few basic behaviors. It does not call APIs, run model inference or establish benchmark validity. See [verification](docs/usage.md#verify-the-installation) for optional metric profiles.

No project-wide open-source license has been granted yet. Third-party components retain their own license terms.

## Code structure

Annotation flows through `cli → runner → pipelines → execution → models/cv`.
`batch.py` adds manifest scheduling and shared GPU workers. Event scoring and
video quality metrics have separate entry points.

Pipelines call `StageExecutor.execute()` directly for VLM and SAM requests;
there is no internal job queue or wait/poll layer.

| Module | Responsibility |
|---|---|
| `config.py` | `PERCEPT_` settings and model/CV validation |
| `backends.py` | Settings, VLM client and SAM executor setup shared by both annotation modes |
| `media.py` | FFprobe metadata and timestamped FFmpeg frame extraction |
| `pipelines/` | Event prompts, schemas, visual review and targeted SAM evidence |
| `models/`, `cv/` | VLM backends and computer-vision adapters |
| `runner.py` | Video discovery, per-video orchestration, resume and batch results |
| `execution.py` | Direct VLM/SAM stage execution and the VLM request cache |
| `batch.py` | Manifest scheduling, shared GPU workers and concurrency control |
| `results.py` | Annotation JSON, audit sidecars, filename collisions and resume |
| `evaluation/`, `video_metrics/` | Event comparison, video quality metrics and checkpoint checks |

`media.py` handles video input and extracts frames in one FFmpeg pass with actual
source timestamps. See [sampling behavior](docs/usage.md#video-frame-sampling).
`results.py` manages
files without a database. Its resume behavior is described in
[usage](docs/usage.md).

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

This installs the annotation client and event scorer. Grounded annotation also requires local SAM3.1 assets and a compatible GPU runtime; see [setup](docs/usage.md#mandatory-sam31-grounding). Other GPU metrics have separate [runtime requirements](docs/usage.md#optional-metric-environments). `pip install` installs code and dependencies; it does not download model weights.

### 2. Configure the annotation model

Set your endpoint URL, provider model ID and API key in the same terminal:

```bash
export PERCEPT_OPENAI_BASE_URL="https://your-provider.example/v1"
export PERCEPT_OPENAI_MODEL="your-vision-model-id"
export PERCEPT_OPENAI_API_KEY="your-api-key"
```

Choose a model that accepts image inputs through the chat completions API. Configure mandatory local SAM3.1 grounding before annotation; see [configuration](docs/usage.md#annotation-configuration). `.env` files are not loaded automatically.

### 3. Annotate your video

```bash
percept annotate \
  --videos ./videos/sample_001.mp4 \
  --template embodied_action_captioning \
  --output ./outputs/actions
```

Pass a directory to `--videos` to process its videos recursively. Annotation uses
`embodied_action_captioning`, the only supported template, and produces
event descriptions and bounds on the original video clock. You can omit `--template`; existing
commands that specify it explicitly remain valid.

The pipeline nominates visible entities, gathers baseline SAM evidence, drafts events, independently reviews the original video and overlays, performs bounded local SAM rechecks, and revises the result. Tracking supports visual review but does not prove an action, outcome or boundary.

### 4. Read the results

For the example above, a fresh output directory contains:

```text
outputs/actions/
├── sample_001.json    # Annotation and per-video metadata
└── results.jsonl      # Batch summary, status and result filenames
```

Read `status` and `error` to check success, then `data` for the annotation. Timestamps are seconds on the original video. The directory also contains `.percept/` run records used for resume and history. If video filenames collide, use the `result_file` recorded in `results.jsonl` to locate the correct JSON.

Grounded runs currently rerun because CV execution identity is not yet reusable. Add `--force` for an independent repeat that also disables the request cache. See [outputs and resume](docs/usage.md#result-identity-and-resume) for the exact rules.

## Compare real and generated videos

Annotate each side with the mandatory SAM setup above. Name paired videos with the same sample ID and use matching sample directories:

```bash
percept annotate --videos ./videos/real/sample_001.mp4 --output outputs/reference/sample_001
percept annotate --videos ./videos/generated/sample_001.mp4 --output outputs/generated/sample_001
export PERCEPT_JUDGE_BASE_URL="https://your-judge.example/v1"
export PERCEPT_JUDGE_MODEL="your-judge-model-id"
export PERCEPT_JUDGE_API_KEY="your-judge-key"
percept score events --reference outputs/reference \
  --system model=outputs/generated --out outputs/event_scores
```

`--out` names a directory containing `events.json`. The scorer also accepts flat result directories; it identifies samples from an explicit ID or `video_path`. A missing, failed or invalid annotation makes the system aggregate null and the command exits nonzero. A successful empty `events` list is valid. Old categorical `segments`/`semantic_events` JSON is incompatible and must be regenerated. The scoring formula and adaptation limits are in [event evaluation](docs/usage.md#grounded-event-scoring).

## Evaluate video quality

Video metrics run independently of annotation and do not need VLM API credentials. Use the environment for the metric you select; one environment is not expected to contain every upstream evaluator.

### CLIP-IQA+

Create a dedicated environment and install a hardware-compatible PyTorch/TorchVision build as described in the [CLIP-IQA+ setup](docs/metrics.md#clip-iqa). Then install the metric, prepare its weights, and score:

```bash
python3.12 -m venv .venv-clipiqa
source .venv-clipiqa/bin/activate
# Install the appropriate PyTorch/TorchVision build before the next step.
python -m pip install -e '.[clipiqa]'
percept download clipiqa --cache-dir ./weights
python scripts/verify_install.py --metric clipiqa --cache-dir ./weights
percept score clipiqa --video ./videos/sample_001.mp4 \
  --cache-dir ./weights --output outputs/sample_001.clipiqa.json
```

The JSON includes the score, sampled-frame count, sampling settings and input checksum. `--device auto` selects CUDA when available, otherwise CPU. CLIP-IQA+ measures sampled-frame image quality.

### Motion Smoothness

Use a separate CUDA environment with a compatible PyTorch/TorchVision build; follow the [motion runtime notes](docs/metrics.md#motion-smoothness), including the reason for `--no-deps`:

```bash
python3.12 -m venv .venv-motion
source .venv-motion/bin/activate
# Install the appropriate PyTorch/TorchVision build before the next step.
python -m pip install -e '.[motion]'
python -m pip install --only-binary=:all: --no-deps vbench==0.1.5
percept download motion --cache-dir ./weights
python scripts/verify_install.py --metric motion --cache-dir ./weights
percept score motion --video ./videos/sample_001.mp4 \
  --cache-dir ./weights --gpu 0 --output outputs/sample_001.motion.json
```

The JSON includes the score and paths to the official evaluator result and log. Motion Smoothness measures local interpolation smoothness; a smooth video can still depict an incorrect action.

**Weights:** `percept download` currently supports `clipiqa` and `motion`. It verifies SHA-256 and reuses valid files. Use the same `--cache-dir` for downloading and scoring; `./weights` is relative to the current directory. Other metrics require their own [checkpoint preparation](docs/metrics.md#checkpoint-preparation).

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

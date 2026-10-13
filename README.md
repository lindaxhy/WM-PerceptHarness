# WM-PerceptHarness

Annotate videos, compare real and generated action timelines, and evaluate video quality with individual metrics.

[Quick start](#quick-start) · [Compare videos](#compare-real-and-generated-videos) · [Video metrics](#evaluate-video-quality) · [Documentation](#documentation)

## What can I run?

| Task | Input | Output | Requirements |
|---|---|---|---|
| Video annotation | One video or a directory of videos | Captions, active objects, or action timelines in JSON | FFmpeg + a vision-capable OpenAI-compatible API |
| Event fidelity | Matching real/generated action annotations | Temporal event agreement; optional semantic similarity | Core package; `fidelity` extra for semantic scoring |
| Video quality | One video | A score for the selected metric | The metric's runtime and model weights |

Annotation uses sampled video frames; it does not use audio. You supply the videos and, for fidelity, their real/generated pairing. Metrics produce separate results; the repository does not compute a unified benchmark score. See the [metric guide](docs/metrics.md) for implementation and validation status.

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

This installs the annotation client and temporal fidelity scorer. GPU metrics have separate [runtime requirements](docs/usage.md#optional-metric-environments). `pip install` installs code and dependencies; it does not download model weights.

### 2. Configure the annotation model

Set your endpoint URL, provider model ID and API key in the same terminal:

```bash
export PERCEPT_OPENAI_BASE_URL="https://your-provider.example/v1"
export PERCEPT_OPENAI_MODEL="your-vision-model-id"
export PERCEPT_OPENAI_API_KEY="your-api-key"
```

Choose a model that accepts image inputs through the chat completions API. More options, including multiple model aliases and optional SAM3.1 evidence, are documented in [configuration](docs/usage.md#annotation-configuration). `.env` files are not loaded automatically.

### 3. Annotate your video

```bash
percept annotate \
  --videos ./videos/sample_001.mp4 \
  --template embodied_action_captioning \
  --output ./outputs/actions
```

Pass a directory to `--videos` to process its videos recursively. Select a template for your task:

| Template | Produces |
|---|---|
| `general_video_captioning` | Video summary and timestamped events |
| `embodied_active_object_detection` | Visibly interacted object inventory |
| `embodied_action_captioning` | Task description and time-bounded action segments |

### 4. Read the results

For the example above, a fresh output directory contains:

```text
outputs/actions/
├── sample_001.json    # Annotation and per-video metadata
└── results.jsonl      # Batch summary, status and result filenames
```

Read `status` and `error` to check success, then `data` for the annotation. Timestamps are seconds on the original video. The directory also contains `.percept/` run records used for resume and history. If video filenames collide, use the `result_file` recorded in `results.jsonl` to locate the correct JSON.

Matching completed built-in runs can resume. Add `--force` for an independent repeat. See [outputs and resume](docs/usage.md#result-identity-and-resume) for the exact reuse rules.

## Compare real and generated videos

Name matching videos with the same sample ID and annotate each into its own sample directory. With the API settings above:

```bash
percept annotate --videos ./videos/real/sample_001.mp4 \
  --template embodied_action_captioning --output outputs/reference/sample_001
percept annotate --videos ./videos/generated/sample_001.mp4 \
  --template embodied_action_captioning --output outputs/generated/sample_001

percept score fidelity \
  --reference outputs/reference \
  --system model=outputs/generated \
  --out outputs/fidelity.json
```

The scorer pairs this layout by sample ID:

```text
outputs/
├── reference/sample_001/sample_001.json
├── generated/sample_001/sample_001.json
└── fidelity.json
```

Repeat the annotation steps for each pair. A flat folder of annotation JSON files is not accepted. The output reports temporal event-family overlap, event F1 and outcome agreement. Check `paired_samples` against the number of pairs you intended to evaluate.

Temporal scoring needs no API calls or local model weights. For semantic similarity, install `python -m pip install -e '.[fidelity]'` and add `--semantic`; its embedding model is downloaded/cached separately. See [fidelity evaluation](docs/usage.md#event-timeline-fidelity) for input exclusions, actionless-case limitations and repeated-reference evaluation.

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
| [Usage](docs/usage.md) | Installation checks, configuration, annotation/fidelity workflows and outputs |
| [Metrics](docs/metrics.md) | Metric protocols, weights, input examples and reporting scope |
| [Third-party notices](docs/third-party-notices.md) | Upstream sources and component licenses |

`python scripts/verify_install.py` checks the local installation and a few basic behaviors. It does not call APIs, run model inference or establish benchmark validity. See [verification](docs/usage.md#verify-the-installation) for optional metric profiles.

No project-wide open-source license has been granted yet. Third-party components retain their own license terms.

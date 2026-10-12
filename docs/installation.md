# Installation and configuration

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
uv sync --locked --extra dev
uv run --locked percept --help
```

Use `--extra fidelity` as well when reproducing semantic evaluation. `pyproject.toml` defines dependencies; `uv.lock` fixes their resolved versions for uv users. Pip installation remains supported but does not consume `uv.lock`.

The lockfile also resolves optional extras, but does not select a hardware-specific CUDA wheel or lock separately installed VBench/upstream source trees and checkpoints. Record those independently. See [reproduction](reproduction.md).

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

For repeatability and the separate request-level cache, see [result identity and resume](evaluation.md#result-identity-and-resume).

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

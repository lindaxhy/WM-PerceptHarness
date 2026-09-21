# Installation

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

Then follow [metric runtime installation](metrics/usage.md). Install a hardware-compatible Torch/TorchVision build before the metric requirements. `requirements/metrics-*.txt` are runtime dependency lists, not complete environment locks. Do not install all of them into the core environment.

A unified CLI does not activate another environment automatically: `percept score motion` uses the currently active Python environment. Use that environment's full `bin/percept` path when orchestrating several environments.

## Reproducible core environment (optional uv)

```bash
uv sync --locked --extra dev
uv run --locked percept --help
```

Use `--extra fidelity` as well when reproducing semantic evaluation. `pyproject.toml` defines dependencies; `uv.lock` fixes their resolved versions for uv users. Pip installation remains supported but does not consume `uv.lock`.

External evaluators and checkpoints require their own version records; the core lockfile does not reproduce their environments. See [reproduction](reproduction.md).

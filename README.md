# WM-PerceptHarness

Video annotation and evaluation tools for world models and embodied agents.

The repository provides three complementary workflows:

| Workflow | Input → output | Entry point |
|---|---|---|
| Video annotation | Video → captions, active objects, or action timelines | `percept annotate` |
| Event-timeline fidelity | Real/generated video annotations → temporal and semantic comparisons | `percept score fidelity` |
| Video quality | Video → CLIP-IQA+ or VBench Motion Smoothness score | `percept score clipiqa` / `motion` |

Annotation uses sampled visual frames through an OpenAI-compatible VLM endpoint; it does not use audio. Direct quality metrics use their own model checkpoints and do not require VLM API credentials. The [metric catalog](docs/metrics/README.md) distinguishes implemented entry points, validation evidence, and planned integrations.

```text
Real video ────── annotate ── reference timeline ─┐
                                                ├── fidelity report
Generated video ─ annotate ── predicted timeline ─┘
       └──────────────────── video quality metrics
```

## Install

Use Python 3.12+ and install FFmpeg/FFprobe for annotation.

```bash
git clone https://github.com/lindaxhy/WM-PerceptHarness.git
cd WM-PerceptHarness
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional video metrics have separate environments: see [installation](docs/installation.md). `uv.lock` is available for the core project's reproducibility workflow; ordinary pip installation does not require uv.

## Try annotation

Set the URL, model ID, and key for your chosen vision-capable endpoint:

```bash
export PERCEPT_OPENAI_BASE_URL="https://your-provider.example/v1"
export PERCEPT_OPENAI_MODEL="your-vision-model-id"
export PERCEPT_OPENAI_API_KEY="your-api-key"

percept annotate --videos ./my_videos \
  --template embodied_action_captioning --output outputs/actions
```

The default backend is OpenAI-compatible. `percept eval` and `--backend openai` remain supported. Alternative providers and optional SAM3.1 evidence are described in [advanced configuration](docs/advanced.md).

| Template | Output |
|---|---|
| `general_video_captioning` | Video summary and timestamped events |
| `embodied_active_object_detection` | Visibly interacted object inventory |
| `embodied_action_captioning` | Task summary and time-bounded action segments |

Results use `<video_stem>.json`, with a source-path suffix when that name belongs to another video. `results.jsonl` records the exact result filenames. Verified built-in runs resume only when input bytes, model, prompts, configuration and code/runtime fingerprints match. Use `--force` for an independent repeat; displaced results and run manifests are retained under `.percept/`. See [input/output conventions](docs/evaluation.md) for conservative custom/CV behavior.

For a local check without API credentials or weights, run the [synthetic smoke example](examples/README.md). Its fake annotations test the software, not model quality.

## Compare real and generated videos

Prepare matching annotations under `reference/<id>/<id>.json` and `generated/<id>/<id>.json`:

```bash
percept score fidelity \
  --reference outputs/reference \
  --system model=outputs/generated \
  --out outputs/fidelity.json
```

Temporal scoring works with the core installation. For semantic similarity and statistical tests, install `python -m pip install -e '.[fidelity]'` and add `--semantic`. Use two independent real-video annotation runs with `--self-agreement` to estimate annotator self-agreement. See the [evaluation guide](docs/evaluation.md) for the full workflow and interpretation.

## Score video quality

After installing the corresponding [metric environment](docs/metrics/usage.md):

```bash
percept score clipiqa --video example.mp4 --output outputs/example.clipiqa.json

percept score motion --video example.mp4 \
  --cache-dir /path/to/vbench-cache --gpu 0 \
  --output outputs/example.motion.json
```

CLIP-IQA+ measures sampled-frame image quality; Motion Smoothness measures local interpolation smoothness. Neither establishes action correctness. Both preserve their existing scoring protocols and output fields. The original `scripts/score_*.py` entry points remain available.

## Research use

- [Metric catalog and readiness](docs/metrics/README.md)
- [Frozen benchmark selection](docs/metrics/FINAL_METRICS.md) — selection is separate from implementation status; CLIP-IQA+ is a backup metric.
- [Reproduction guide](docs/reproduction.md) — versions, inputs, environments, and evidence to retain.
- [Third-party sources and notices](docs/metrics/THIRD_PARTY_NOTICES.md)

The current release does not provide an end-to-end reproduction of every metric in the frozen selection. Per-metric limitations are listed in the catalog.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Core code lives in `src/percept_harness/`; optional metric implementations are in `video_metrics/`, fidelity scoring in `evaluation/`, compatibility and preparation commands in `scripts/`, and regression tests in `tests/`.

## Project history and license

The former LAS-compatible service is archived at `legacy-las-service` / `legacy/las-service`. Historical deployment notes are not prerequisites for current annotation.

No project-wide open-source license has been granted yet. Add an appropriate license before public release. Third-party components retain their own license terms. Paper citation information will be added when available.

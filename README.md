# WM-PerceptHarness

Video annotation and evaluation tools for world models and embodied agents.

The repository provides three complementary workflows:

| Workflow | Input → output | Entry point |
|---|---|---|
| Video annotation | Video → captions, active objects, or action timelines | `percept annotate` |
| Event-timeline fidelity | Real/generated video annotations → temporal and semantic comparisons | `percept score fidelity` |
| Video quality | Video → CLIP-IQA+ or VBench Motion Smoothness score | `percept score clipiqa` / `motion` |

Annotation uses sampled visual frames through an OpenAI-compatible VLM endpoint; it does not use audio. Direct quality metrics use their own model checkpoints and do not require VLM API credentials. The [metric catalog](docs/metrics.md) distinguishes implemented entry points, validation evidence, and planned integrations.

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

The default backend is OpenAI-compatible; `--backend openai` remains an explicit option. Alternative providers and optional SAM3.1 evidence are described in [advanced configuration](docs/installation.md#annotation-configuration).

| Template | Output |
|---|---|
| `general_video_captioning` | Video summary and timestamped events |
| `embodied_active_object_detection` | Visibly interacted object inventory |
| `embodied_action_captioning` | Task summary and time-bounded action segments |

Results use `<video_stem>.json`, with a source-path suffix when that name belongs to another video. `results.jsonl` records the exact result filenames. Verified built-in runs resume only when input bytes, model, prompts, configuration and code/runtime fingerprints match. Use `--force` for an independent repeat; displaced results and run manifests are retained under `.percept/`. See [input/output conventions](docs/evaluation.md) for conservative custom/CV behavior.

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

After installing the corresponding [metric environment](docs/metrics.md):

```bash
percept score clipiqa --video example.mp4 --output outputs/example.clipiqa.json

percept score motion --video example.mp4 \
  --cache-dir /path/to/vbench-cache --gpu 0 \
  --output outputs/example.motion.json
```

CLIP-IQA+ measures sampled-frame image quality; Motion Smoothness measures local interpolation smoothness. Neither establishes action correctness. Both preserve their existing scoring protocols and output fields.

## Documentation

- [Installation and configuration](docs/installation.md)
- [Evaluation workflows and outputs](docs/evaluation.md)
- [Metric commands and protocols](docs/metrics.md)
- [Reproduction and validation](docs/reproduction.md)
- [Frozen benchmark selection](docs/benchmark.md)
- [Third-party notices and license](docs/third-party-notices.md)

The current release does not provide an end-to-end reproduction of every metric in the frozen selection. Per-metric limitations are listed in the metric guide.

## Development

Use Python 3.12+ and FFmpeg/FFprobe. Run the regression suite from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The suite uses synthetic media, fake models and mocked external evaluators to check software behavior; it does not establish real-model quality. Optional runtime tests skip when their dependencies are unavailable.

### Repository layout

| Path | Responsibility |
|---|---|
| `src/percept_harness/` | Installable Python package and the public `percept` CLI |
| `src/percept_harness/pipelines/` | Annotation stages and output validation |
| `src/percept_harness/evaluation/` | Event-fidelity scoring and shared temporal event matching |
| `src/percept_harness/video_metrics/` | Video scorers, official evaluator wrappers and VideoPhy aggregation |
| `scripts/` | Input preparation and manifest validation |
| `tests/` | Python regression tests and small fixtures; retained in source control, excluded from the wheel |
| `pyproject.toml` | Core dependencies and optional `dev`, `fidelity`, `clipiqa`, `motion` extras |
| `docs/` | Current user guides, metric protocols and reproduction instructions |

Reusable logic lives in the package; `scripts/` contains only input preparation tools. Model interfaces and endpoint adapters live in `src/percept_harness/models/`; configure the provider/model through environment variables, not source constants.

All scorers use `percept score`: `fidelity`, `clipiqa`, `motion`, `vbench`, `vbench-clip`, `dover`, `external`, and `videophy-joint`. The former `scripts/score_*.py`, `run_official_external.py`, and `aggregate_videophy_joint.py` entry points have been replaced by these commands; their evaluator arguments are unchanged. See the [metric catalog](docs/metrics.md) for environment requirements and validation status.

Keep regression tests for scoring, timestamp boundaries, model-output validation, resume identity and command behavior. Add real-model validation separately with its environment and evidence, rather than making ordinary tests require API credentials or GPU checkpoints.

Local agent state (`.superpowers/`, `.claude/`, `docs/superpowers/`), caches, generated outputs and model weights are ignored. Keep current usage and design guidance in `docs/`; historical implementation notes remain available in Git history. Do not commit agent task transcripts. Existing files must also be removed from Git tracking before ignore rules take effect.

## Project history and license

The fixed two-sample Doubao semantic re-verification operator and its experiment-only tests were removed from active development. Their [last pre-cleanup version](https://github.com/lindaxhy/WM-PerceptHarness/tree/a201286f2b5b4deb55b2e99aaff0303e588f2295) remains available for historical reproduction. Current annotation uses the configurable OpenAI-compatible backend.

The old five-video LAS comparison viewer, machine-generated references, dated SAM3.1 evaluation results, LAS-specific evaluator and server-specific smoke script are preserved together on [`archive/las-comparison-viewer`](https://github.com/lindaxhy/WM-PerceptHarness/tree/archive/las-comparison-viewer). They are historical diagnostics, not inputs required by current annotation or fidelity scoring. To inspect them without changing your checkout:

```bash
git fetch origin
git worktree add ../WM-PerceptHarness-viewer origin/archive/las-comparison-viewer
```

Follow `evaluation/viewer/README.md` in that worktree. Its ignored video files must still be supplied locally. The old `scripts/evaluate_las_alignment.py`, `scripts/build_comparison_viewer_data.py` and `scripts/sam31_smoke.py` commands are available only on the archive branch; current scoring uses `percept score fidelity` with caller-supplied annotations.

The former LAS-compatible service is preserved by the [`legacy-las-service` tag](https://github.com/lindaxhy/WM-PerceptHarness/tree/legacy-las-service). The tag remains usable if the old development branch is deleted. Historical deployment notes are not prerequisites for current annotation.

No project-wide open-source license has been granted yet. Add an appropriate license before public release. Third-party components retain their own license terms. Paper citation information will be added when available.

# Reproduction and validation

The repository currently provides annotation, event-fidelity scoring, packaged video scorers, official evaluator wrappers, and input preparation tools. It does not yet ship a complete reproduction command for every metric in the frozen benchmark selection.

## Record the experiment

For each reported run, retain:

- Code commit and any local changes.
- Video IDs, input SHA-256 hashes, frozen prompts, auxiliary metadata, and reference-pairing rules.
- VLM provider/model ID, generation settings, sampling configuration, prompt contents, and optional CV settings. Do not record API keys.
- Checkpoint source and hashes, upstream evaluator revision, Python/package versions, FFmpeg version, device and CUDA configuration where applicable.
- Exact command, logs, original evaluator outputs, parsed reports, failures/exclusions, and evaluated sample count.

Annotation now saves per-video provenance and `.percept/runs/` manifests containing input/configuration/code/runtime fingerprints and result digests. External CV state and full remote model provenance still need a separate record. External API model behavior may change even when the client environment is locked; record the run date and repeatability evidence.

Use separate output directories for independently reported experiments. Verified built-in annotation runs reuse only matching input/configuration/code/runtime identities; `--force` requests an independent repeat. Displaced result bytes are retained in `.percept/history/`. Custom/CV executions conservatively rerun until their external identities can be verified. See [resume details](evaluation.md#result-identity-and-resume). Keep raw results, and document any later aggregation separately.

## Environment capture

For a uv-managed core environment, use `uv sync --locked` with the required extras. For every independent evaluator environment, save `python --version`, `python -m pip freeze`, and `python -m pip check` output. The dimension-specific VBench Motion environment intentionally differs from the full VBench dependency metadata; record that discrepancy rather than treating it as a fully validated general VBench installation.

Pin external weights and source trees as specified in [third-party notices](third-party-notices.md) and [checkpoint preparation](metrics.md#checkpoint-preparation). Optional dependency groups do not by themselves reproduce external evaluator environments.

## Validation stages

| Stage | What it establishes |
|---|---|
| Offline tests / preflight | Argument, manifest, source, cache, or result-contract behavior |
| Real single-video inference | The evaluator produced a score in a recorded environment |
| Batch validation | A frozen dataset completed with documented coverage and exclusions |

A passing preflight is not a model score. A single-video result does not establish full benchmark reproduction.

The [metric catalog](metrics.md) is the public navigation point. [benchmark.json](benchmark.json) records the frozen selection and its original status fields. Current entry points and validation limits are documented in the catalog; implementation status is not detected automatically.

## Reporting

Report event fidelity, image quality, and motion smoothness separately. Include sample counts and missing-data policy. Report PSNR/SSIM/LPIPS only with aligned real future frames, and FVD/FID only with an appropriate matching real distribution. Keep CLIP-IQA+ marked as a backup under the current frozen selection. Mechanics, Thermotics, and Material are selected specialist reports excluded from a unified main score; the five reference/distribution metrics are separate appendix reports.

Paper-specific configs, citation metadata, and published result tables should be added when those artifacts are finalized; no paper configuration or scientific result is invented by this release.


## Development

Tests are for contributors; running annotation or evaluation does not require them. With Python 3.12+ and FFmpeg/FFprobe installed:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The suite uses synthetic media, fake model responses and local evaluator fixtures. It covers annotation schemas, timestamp boundaries, result identity and resume, CV evidence, metric inputs, CLI dispatch and checkpoint integrity. Optional runtime tests skip when their dependencies are unavailable; passing these tests does not establish real-model quality.

Tests are grouped by behavior rather than implementation stage: model/backend contracts, runner/batch behavior, scene review, action contracts, CV evidence, fidelity and external metrics. To run a focused check, for example:

```bash
python -m pytest tests/test_cli.py tests/test_weights.py -q
```

Reusable runtime logic belongs in `src/percept_harness/`; `scripts/` holds input preparation tools. Generated outputs, local agent state and model weights are ignored and must stay outside commits. `uv.lock` supports the reproducible core workflow described in [installation](installation.md#reproducible-core-environment-optional-uv).

## Historical versions

The fixed two-sample Doubao re-verification experiment and its dedicated fixtures are available in the [last pre-cleanup commit](https://github.com/lindaxhy/WM-PerceptHarness/tree/a201286f2b5b4deb55b2e99aaff0303e588f2295). Current annotation uses the configurable OpenAI-compatible backend.

The five-video LAS viewer, generated references, dated SAM3.1 results and server-specific smoke scripts are preserved on [archive/las-comparison-viewer](https://github.com/lindaxhy/WM-PerceptHarness/tree/archive/las-comparison-viewer). To inspect that archive independently:

```bash
git fetch origin
git worktree add ../WM-PerceptHarness-viewer origin/archive/las-comparison-viewer
```

Follow `evaluation/viewer/README.md` there; the ignored video files must be supplied locally. The earlier LAS-compatible service is preserved by the [legacy-las-service tag](https://github.com/lindaxhy/WM-PerceptHarness/tree/legacy-las-service). These archives are not prerequisites for the current workflows.

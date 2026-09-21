# Reproduction and validation

The repository currently provides annotation, event-fidelity scoring, two standalone video metric entry points, and preparation tools for additional official evaluators. It does not yet ship a complete reproduction command for every metric in the frozen benchmark selection.

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

Pin external weights and source trees as specified in [third-party notices](metrics/THIRD_PARTY_NOTICES.md) and [weight provenance](metrics/weights.json). A runtime requirements file is not a complete environment lock.

## Validation stages

| Stage | What it establishes |
|---|---|
| Offline tests / preflight | Argument, manifest, source, cache, or result-contract behavior |
| Real single-video inference | The evaluator produced a score in a recorded environment |
| Batch validation | A frozen dataset completed with documented coverage and exclusions |

A passing preflight is not a model score. A single-video result does not establish full benchmark reproduction. The [2026-09-20 validation record](metrics/VALIDATION_20260920.md) records historical offline checks and explicitly limits its inference claims.

The [metric catalog](metrics/README.md) is the public navigation point. [FINAL_METRICS.json](metrics/FINAL_METRICS.json) records benchmark selection; its status fields and [STATUS_REMAINING.json](metrics/STATUS_REMAINING.json) are dated snapshots, not automatic detection of installed capabilities.

## Reporting

Report event fidelity, image quality, and motion smoothness separately. Include sample counts and missing-data policy. Report PSNR/SSIM/LPIPS only with aligned real future frames, and FVD/FID only with an appropriate matching real distribution. Keep CLIP-IQA+ marked as a backup under the current frozen selection. Mechanics, Thermotics, and Material are selected specialist reports excluded from a unified main score; the five reference/distribution metrics are separate appendix reports.

Paper-specific configs, citation metadata, and published result tables should be added when those artifacts are finalized; no paper configuration or scientific result is invented by this release.

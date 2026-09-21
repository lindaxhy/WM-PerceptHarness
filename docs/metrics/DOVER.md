# DOVER integration status

`scripts/score_dover_official.py` validates the pinned upstream source, runtime and checkpoint paths, then invokes the official evaluator. Offline wrapper tests exist; this release does not establish successful DOVER model inference.

Pinned source: `VQAssessment/DOVER` at `f1ddc96215bc7fbcf8f315c65d47905f339c3419`. The evaluator requires both `pretrained_weights/DOVER.pth` under the source root and `hub/checkpoints/convnext_tiny_1k_224_ema.pth` under `TORCH_HOME`. Record source URLs and full hashes; an observed hash alone is not verification against a trusted expected value.

Use a separate runtime compatible with upstream requirements. The original Torch 1.13 family is not compatible with every current GPU. A modern runtime is an explicitly unverified compatibility experiment until compared against a validated baseline; `--allow-unverified-runtime` does not certify correctness.

Inspect the wrapper's options without loading models:

```bash
python scripts/score_dover_official.py --help
```

With a provisioned official source tree, weights and environment:

```bash
python scripts/score_dover_official.py \
  --dover-root /path/to/DOVER \
  --video /path/to/video.mp4 \
  --torch-home /path/to/torch-cache \
  --device cpu \
  --output-dir /path/to/new-preflight-output \
  --check-only
```

A successful check is `preflight_passed_not_scored`. Omit `--check-only` and use a new output directory only when ready for actual model inference. Preserve the command, environment, hashes, `dover.log` and `result.json`. An unparsed official result is not a score.

The original server-specific dependency investigation and installation starting point are preserved in the [2026-09-20 DOVER archive](../archive/metrics-20260920/DOVER.md). See the [catalog](README.md) for overall readiness and the [reproduction guide](../reproduction.md) for validation stages.

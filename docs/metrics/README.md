# Metric catalog

Use this page to find an evaluator and understand its validation level. Benchmark selection, implemented code, and successful inference are different facts.

## Public scoring entry points

| Evaluator | Command | Environment | Evidence and scope |
|---|---|---|---|
| Event-timeline fidelity | `percept score fidelity` | Core; optional `fidelity` extra for semantic scoring | Existing hand-calculated regression examples; compares paired annotations |
| CLIP-IQA+ | `percept score clipiqa` | PyIQA 0.1.16; CPU or CUDA | Implemented sampled-frame scorer and protocol tests; a fresh full dependency installation is not validated by the historical record |
| VBench Motion Smoothness / AMT-S | `percept score motion` | VBench 0.1.5, CUDA, AMT-S weights | Existing record marks single-video smoke completed; this is not a full-dataset validation |

See [installation and exact protocols](usage.md), [input/output conventions](../evaluation.md), and [weight provenance](weights.json). The packaged commands use the active environment; they do not merge the different dependency stacks. Legacy script entry points remain supported.

CLIP-IQA+ is a **backup**, excluded from the current frozen main selection. Motion Smoothness is selected. Event fidelity is the repository's annotation-based workflow and is not silently inserted into the separate frozen 21-item list.

## Additional integration tools

The following code is preparation/integration support. Offline tests do not establish successful real-model inference.

| Scope | Existing tool | Remaining validation/resources |
|---|---|---|
| MUSIQ, temporal flickering, dynamic degree, subject/background consistency, ViCLIP; aesthetic diagnostic | `scripts/score_vbench_official.py` | Per-dimension weights and real smoke runs; frozen prompt mapping for ViCLIP; static subset for flickering |
| CLIPScore | `scripts/score_vbench_clip_score.py` | CLIP weights, original prompts, real inference validation |
| DOVER | `scripts/score_dover_official.py` | Independent compatible runtime, weights, real inference validation; see [DOVER](DOVER.md) |
| WorldModelBench, T2V-CompBench, VBench-2.0, PhyGenBench, VideoPhy-2 inputs | `scripts/official_input_adapters.py` | Official evaluators, environments, weights and task metadata; adapters do not run the models |
| Reference metrics | `scripts/build_reference_manifest.py` | Actual aligned future reference frames; pairing is not PSNR/SSIM/LPIPS scoring |
| VideoPhy-2 Joint analysis | `scripts/aggregate_videophy_joint.py` | Official PC/SA outputs matched by video ID; Joint is not an independent metric |
| Manifest preflight | `scripts/validate_metric_manifest.py` | Caller-provided frozen input metadata; no missing prompts or references are generated |

Detailed contracts: [remaining integrations](REMAINING_METRICS.md), [official input adapters](OFFICIAL_INPUT_ADAPTERS.md), and [2026-09-20 validation](VALIDATION_20260920.md). FID/FVD still require matching real distributions and the corresponding evaluator environments.

## Benchmark selection and historical records

- [Frozen selection](FINAL_METRICS.md) / [machine-readable selection](FINAL_METRICS.json): 21 selected entries, including three specialist physics reports excluded from a unified score, plus five separate appendix metrics. This is not a claim of 26 completed implementations.
- [Dated implementation snapshot](STATUS_REMAINING.json): preparation status and blockers recorded on 2026-09-20; labels such as `submitted` do not imply batch validation.
- [Third-party notices](THIRD_PARTY_NOTICES.md): upstream revisions and attribution.
- [Reproduction](../reproduction.md): run manifests and validation stages.
- [Archived server operations](../archive/metrics-20260920/DETAILED_OPERATION_GUIDE_20260920.md): historical machine-specific instructions.

Update this catalog and the linked validation evidence when a new evaluator completes real inference. Keep the selected metric list independent of readiness, and preserve the original upstream scoring protocols.

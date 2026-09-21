# Evaluation workflows and outputs

## Annotation

`percept annotate` is the public name for the existing `percept eval` workflow. Both accept the same annotation arguments. The default backend is `openai`; `fake` remains an explicit test option.

```bash
percept annotate --videos video.mp4 \
  --template embodied_action_captioning --output outputs/actions
```

One successful record contains `video_path`, `template`, `status`, `data`, `error`, and `elapsed_seconds`. `data` holds the template-specific output; action times are seconds on the original video timeline. `results.jsonl` aggregates the batch but does not include every per-video metadata field.

Each output filename uses the input video stem. Do not combine same-stem videos from different directories in one output directory. Resume currently checks completion and template, not input contents, model, or prompt; use a new output directory whenever those change. These are current implementation limits, not guarantees of a reproducible cache.

For wrist/main-camera annotation, first run `embodied_active_object_detection` on the wrist video, then pass confirmed object names to the main-video action annotation using `--prompt-context`. This is naming guidance, not a prescribed action sequence.

## Event-timeline fidelity

The scorer expects matching IDs in a nested layout:

```text
outputs/reference/sample_001/sample_001.json
outputs/generated/sample_001/sample_001.json
```

Prepare that layout directly by naming matching videos consistently and annotating each into its own sample directory:

```bash
percept annotate --videos data/real/sample_001.mp4 \
  --template embodied_action_captioning --output outputs/reference/sample_001
percept annotate --videos data/generated/sample_001.mp4 \
  --template embodied_action_captioning --output outputs/generated/sample_001
percept score fidelity --reference outputs/reference \
  --system model=outputs/generated --out outputs/fidelity.json
```

A flat annotation directory is not accepted by the current fidelity loader. It loads records with nonempty `data.segments`, extracts recognized `data.semantic_events`, and pairs shared sample IDs. Inspect `paired_samples` against your intended evaluation set and report exclusions; a missing result is not a zero score. Ensure the configured annotation pipeline produces the semantic events your protocol needs, and inspect an example before scaling up.

The report contains per-system frame event-family mIoU, event F1 at tIoU thresholds, outcome agreement, and per-family statistics. With two systems it adds paired comparisons and duration/event-count strata. The default temporal path needs no model downloads.

For semantic similarity:

```bash
python -m pip install -e '.[fidelity]'
percept score fidelity --reference outputs/reference \
  --system model=outputs/generated --semantic \
  --self-agreement a=outputs/real_repeat_a b=outputs/real_repeat_b \
  --out outputs/fidelity_semantic.json
```

Each repeat uses the same nested ID layout. Self-agreement measures variation between two real-video annotation runs. The existing CLI prints ratios normalized by that baseline; the JSON stores raw system aggregates and `self_agreement`, not the printed normalized ratios. Preserve stdout alongside the report if using those ratios. No normalization is meaningful when the corresponding baseline is zero.

## Direct video metrics

Run each command in its documented runtime environment:

```bash
percept score clipiqa --video video.mp4 --output outputs/clipiqa.json
percept score motion --video video.mp4 --cache-dir /path/to/vbench-cache \
  --output outputs/motion.json
```

Both outputs share:

| Field | Meaning |
|---|---|
| `metric` | `clipiqa+` or `vbench_motion_smoothness` |
| `score` | Metric-specific scalar |
| `higher_is_better` | Score direction |
| `video`, `video_sha256` | Input identity |
| `versions` | Runtime package versions |

CLIP-IQA+ additionally records frame sampling, resizing, device, and MemoBench revision. Motion Smoothness records checkpoint identity, the official command, original result, and log. Both refuse to overwrite existing result files. Fidelity reports are dataset-level reports with a different schema; no combined score is computed.

Use the short command selector `clipiqa` even though its output metric identifier is `clipiqa+`. Existing scripts remain compatible:

| Packaged command | Existing script |
|---|---|
| `percept score fidelity` | `scripts/evaluate_wm_fidelity.py` |
| `percept score clipiqa` | `scripts/score_clipiqa_plus.py` |
| `percept score motion` | `scripts/score_vbench_motion_smoothness.py` |

Other upstream wrappers remain separate preparation tools listed in the [catalog](metrics/README.md). Their result schemas and validation stages are documented individually.

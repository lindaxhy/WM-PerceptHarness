# Evaluation workflows and outputs

## Annotation

`percept annotate` runs the annotation workflow. The default backend is `openai`; `fake` remains an explicit test option.

```bash
percept annotate --videos video.mp4 \
  --template embodied_action_captioning --output outputs/actions
```

One successful record contains `video_path`, `template`, `status`, `data`, `error`, `elapsed_seconds`, `result_file`, `provenance`, and `data_sha256`. `data` holds the template-specific output; action times are seconds on the original video timeline. `results.jsonl` aggregates the batch but does not include every per-video metadata field.

An embodied-action result may legitimately contain `data.segments: []` when no action is observed. An empty action list is not itself a failed annotation; inspect `status`, `error`, and the template-specific output. Do not invent an action simply to make the timeline nonempty.

### Result identity and resume

A free `<video_stem>.json` is used for the first source claiming that name. Another source with the same stem gets `<stem-prefix>--<sha256-of-resolved-path>.json`. Ownership is checked against the full resolved source path, even when videos have identical bytes. Existing disambiguated paths remain stable on later runs. Use `result_file` in `results.jsonl` to find each result; do not guess a suffix or assume directory traversal order assigns the plain filename. An unreadable existing record is preserved and the new result uses the disambiguated path. If both names are occupied by unverifiable records, use a fresh output directory.

A result is skipped only when all of these match:

- Resolved input path and SHA-256 of the video bytes.
- Template, model alias, effective built-in backend settings, query and prompt context.
- Run settings, packaged Python/prompt asset digest, dependency versions, and FFmpeg/FFprobe version fingerprints.
- Completed status and intact annotation payload digest.

Legacy records without provenance are rerun. Changed settings or input bytes trigger evaluation. Credentials are included only as a digest: changing API keys conservatively reruns because an account may route a model ID differently. Raw API keys, endpoint URLs, proxies and extra request dictionaries/headers are not persisted; digests distinguish their values without writing credentials into manifests. Record the non-secret provider/model configuration separately for publication: a hash is not a recoverable configuration file.

Custom models, custom pipeline registries, custom transports/frame extractors, and any CV execution currently disable resume because the loaded external implementation/weights cannot yet be verified completely. They still receive provenance and run manifests, but run again each time. Local fingerprints cannot detect a provider silently changing the remote model behind an unchanged ID.

```bash
percept annotate --videos video.mp4 \
  --template embodied_action_captioning --output outputs/actions --force
```

`--force` reruns even a matching result, for independent repeats or changed remote model behavior. Before replacing a current result, its exact bytes are archived in `.percept/history/<record-sha256>.json`. `.percept/runs/<run-id>.json` records the run settings, timestamps, input/result mapping, status, and result-file digest; that digest also identifies a later archived record. Per-video and aggregate files are replaced atomically. Inputs or configuration that change during evaluation produce a failed record, not a reusable success.

There is also a separate request-level cache for the OpenAI-compatible backend, enabled by `PERCEPT_SEMANTIC_RESULT_CACHE=true`. It lives under `<PERCEPT_WORK_ROOT>/semantic-cache` (default `work/semantic-cache`) and can reuse validated model responses during reruns. Replayed responses pass through the current sanitizer and validator; rejected or malformed responses are not published as successful cache entries. `--force` disables this cache as well as whole-result skipping. Merely choosing a different output directory does not guarantee fresh model calls; use `--force` for independent annotation repeats, including self-agreement runs.

A writer lock prevents overlapping batches from using the same output directory. A normal exit or exception releases it. After an abrupt process/machine termination, inspect `.percept/writer.lock` (host/PID) and remove it only after confirming the original writer has stopped. The last manifest may remain `running` after an abrupt termination. Use separate output directories for parallel jobs. Each `results.jsonl` describes the most recent completed batch, while the per-run manifests retain earlier mappings.

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

A flat annotation directory is not accepted by the current fidelity loader. Records with missing or null `data.segments` are excluded; an empty list is accepted. The loader extracts recognized action events from `data.semantic_events` and pairs shared sample IDs. Occlusion events and unknown event types are excluded from this fidelity calculation. Inspect `paired_samples` against your intended evaluation set and report exclusions; a missing result is not a zero score. Ensure the configured annotation pipeline produces the semantic events your protocol needs, and inspect an example before scaling up.

At the per-sample level, two empty event timelines have temporal IoU 1.0. The current aggregate reporter does not handle a comparison in which every paired timeline on both sides has zero recognized events: it raises `StatisticsError` while averaging undefined event F1 values. This is a reporting limitation, not a reason to discard valid actionless annotations. Comparisons of two systems also require at least one shared paired sample.

The report contains per-system frame event-family mIoU, event F1 at tIoU thresholds, outcome agreement, and per-family statistics. With two systems it adds paired comparisons and duration/event-count strata. The default temporal path needs no model downloads.

For semantic similarity:

```bash
python -m pip install -e '.[fidelity]'
percept score fidelity --reference outputs/reference \
  --system model=outputs/generated --semantic \
  --self-agreement a=outputs/real_repeat_a b=outputs/real_repeat_b \
  --out outputs/fidelity_semantic.json
```

Each repeat uses the same nested ID layout and should be annotated separately with `--force`. Self-agreement measures variation between two real-video annotation runs. The existing CLI prints ratios normalized by that baseline; the JSON stores raw system aggregates and `self_agreement`, not the printed normalized ratios. Preserve stdout alongside the report if using those ratios. No normalization is meaningful when the corresponding baseline is zero. The parent directory of `--out` must already exist; the scorer does not create it.

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

Use the short command selector `clipiqa` even though its output metric identifier is `clipiqa+`.

Other upstream wrappers remain separate preparation tools listed in the [catalog](metrics/README.md). Their result schemas and validation stages are documented individually.

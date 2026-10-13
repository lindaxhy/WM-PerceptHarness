# Video metrics and input preparation

Use this page to find an evaluator and understand its validation level. Benchmark selection, implemented code, and successful inference are different facts.

Use the sections below for [CLIP-IQA+](#clip-iqa), [Motion Smoothness](#motion-smoothness), [VBench and external evaluators](#vbench-and-external-evaluators), [DOVER](#dover), and [input preparation](#input-preparation). The [reporting scope](#benchmark-reporting) records the benchmark selection; [third-party notices](third-party-notices.md) cover component licenses.

## Public scoring entry points

| Evaluator | Command | Environment | Evidence and scope |
|---|---|---|---|
| Grounded event scoring | `percept score events` | Core package and an OpenAI-compatible semantic judge endpoint | Offline tests cover real FFmpeg video with synthetic annotation/CV evidence and an injected judge; hosted judge and live SAM GPU inference remain unvalidated |
| CLIP-IQA+ | `percept score clipiqa` | PyIQA 0.1.16; CPU or CUDA | Implemented sampled-frame scorer; a fresh full dependency installation is not validated by the historical record |
| VBench Motion Smoothness / AMT-S | `percept score motion` | VBench 0.1.5, CUDA, AMT-S weights | Existing record marks single-video smoke completed; this is not a full-dataset validation |

See [installation](usage.md) and the metric protocols below, [input/output conventions](usage.md), and [checkpoint preparation](#checkpoint-preparation). The packaged commands use the active environment; they do not merge the different dependency stacks. Use the `percept score` commands above for these public workflows. The evaluator integrations below use the same CLI; input preparation remains in `scripts/`. Removed script paths are not compatibility entry points. JSON provenance retains the existing wrapper identifiers for compatibility.

CLIP-IQA+ is a **backup**, excluded from the current frozen main selection. Motion Smoothness is selected. Grounded event scoring is the repository's annotation-based workflow and is not silently inserted into the separate frozen 21-item list.

## Additional integration tools

The following code is preparation/integration support. Local checks do not establish successful real-model inference.

| Scope | Existing tool | Remaining validation/resources |
|---|---|---|
| MUSIQ, temporal flickering, dynamic degree, subject/background consistency, ViCLIP; aesthetic diagnostic | Independent commands below | Per-dimension weights and real smoke runs; frozen prompt mapping for ViCLIP; static subset for flickering |
| CLIPScore | `percept score vbench-clip` | CLIP weights, original prompts, real inference validation |
| DOVER | `percept score dover` | Independent compatible runtime, weights, real inference validation; see [DOVER](#dover) |
| WorldModelBench, T2V-CompBench, VBench-2.0, PhyGenBench, VideoPhy-2 inputs | `scripts/official_input_adapters.py` | Official evaluators, environments, weights and task metadata; adapters do not run the models |
| External evaluator launch | `percept score external` | Checks the required source revision and resource paths, then forwards a caller-supplied command; upstream runtime and scoring validation remain the caller's responsibility |
| Reference metrics | `scripts/build_reference_manifest.py` | Actual aligned future reference frames; pairing is not PSNR/SSIM/LPIPS scoring |
| VideoPhy-2 Joint analysis | `percept score videophy-joint` | Official PC/SA outputs matched by video ID; Joint is not an independent metric |
| Manifest preflight | `scripts/validate_metric_manifest.py` | Caller-provided frozen input metadata; no missing prompts or references are generated |

## Checkpoint preparation

`pip install` installs Python packages, not model weights. Use the explicit
`curl` commands in the README for [CLIP-IQA+](../README.md#clip-iqa) and
[Motion Smoothness](../README.md#motion-smoothness). The former
`percept download` command has been removed; downloading is handled outside the
package.

| Metric | Files relative to the cache directory | Source |
|---|---|---|
| CLIP-IQA+ | `pyiqa/CLIP-IQA%2B_learned_prompts-603f3273.pth`, `clip/RN50.pt` | PyIQA / OpenAI CLIP |
| Motion Smoothness | `amt_model/amt-s.pth` | AMT-S |

The examples save files under `./weights`, relative to the current directory.
For another location, change the download destinations and pass that same path
as `--cache-dir` to scoring and `scripts/verify_install.py`. Existing caches can
be reused when the required files are present. The shared
[weight registry](../src/percept_harness/weights.py) contains cache-relative paths;
installation checks and custom-cache scoring check file existence before model
loading. Checkpoints are ignored by Git and are not shipped in the wheel.

These instructions cover three checkpoints only. Other VBench dimensions,
DOVER, external evaluators and semantic event judges retain their respective
setup instructions. SAM3.1 has separate [grounding setup](usage.md#targeted-sam31-grounding).

## CLIP-IQA+

`percept score clipiqa` reuses the video decoding and frame-scoring protocol from MemoBench revision `f4edb0c4f9f1820bac837ee30d4957811cf275ff`. The model is supplied by PyIQA (`pyiqa==0.1.16`). The command does not implement CLIP-IQA+ itself and does not fall back to MUSIQ, Laplacian sharpness, or another metric.

The fixed defaults are:

- sample frames `0, 4, 8, ...` (`--sample-step 4`);
- resize only when the long side exceeds 640 pixels (`--max-side 640`), preserving the aspect ratio with rounded dimensions and OpenCV `INTER_AREA`;
- score each sampled frame with `pyiqa.create_metric("clipiqa+")`;
- clip each frame score to `[0, 1]`, average the frame scores, and round the video score to four decimal places.

The model score is a perceptual image-quality score applied to sampled video frames. It is not a temporal-consistency or action-correctness score.

### CLIP-IQA installation and command

Create a dedicated environment from the repository root:

```bash
python3.12 -m venv "$HOME/venvs/percept-metrics"
source "$HOME/venvs/percept-metrics/bin/activate"
python -m pip install -e .
```

Install a hardware-compatible PyTorch and TorchVision build first. Then install the `clipiqa` extra:

```bash
python -m pip install -e '.[clipiqa]'
```

The extra does not pin a PyTorch wheel; use the index and CUDA version appropriate for the server. The recorded validation environment uses `torch==2.10.0` / `torchvision==0.25.0` from the cu128 wheel index, for example:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

Run that command before installing the extra when this build matches your hardware. A fresh installation of the full CLIP-IQA+ dependency tree has not been validated by this package. Keep it outside the core harness environment.

Score one video and print JSON to stdout:

```bash
percept score clipiqa \
  --video /data/videos/example.mp4 \
  --cache-dir ./weights \
  --device auto \
  --sample-step 4 \
  --max-side 640
```

Save the same JSON to a new file (the script refuses to overwrite an existing file):

```bash
percept score clipiqa \
  --video /data/videos/example.mp4 \
  --cache-dir ./weights \
  --output /data/results/example.clipiqa_plus.json
```

`--device auto` selects `cuda:0` when CUDA is available and otherwise CPU. You can pass `cpu`, `cuda:0`, or another valid PyTorch device explicitly. The output contains `score` in `[0, 1]`, the resolved video path, frame counts, sampling settings, dependency versions, and the MemoBench revision. The model weights are not part of the repository; see [checkpoint preparation](#checkpoint-preparation) and [third-party notices](third-party-notices.md). Use `--cache-dir` to load the prepared cache; omitting it preserves PyIQA’s default cache and automatic downloading.

## Motion Smoothness

The official VBench CLI accepts single videos or directories; `percept score motion` scores one MP4 through that CLI. Prepare a separate environment:

```bash
python3.12 -m venv "$HOME/venvs/percept-motion"
source "$HOME/venvs/percept-motion/bin/activate"
python -m pip install -e .
```

After entering the session, install a hardware-compatible PyTorch/TorchVision build, install the runtime dependencies, and install the wheel without allowing pip to select a source distribution:

```bash
python -m pip install -e '.[motion]'
python -m pip install --only-binary=:all: --no-deps vbench==0.1.5
```

The `motion` extra in `pyproject.toml` retains the direct dependency versions from the recorded validation environment. Hardware-specific Torch/TorchVision and VBench are installed separately. The official VBench wheel declares a larger, older dependency set for other dimensions (including `numpy<2` and `transformers==4.33.2`); the command above deliberately installs only the Motion Smoothness runtime. Consequently, a global `pip check` can report unmet VBench metadata requirements. This environment is for this dimension only, not all VBench tasks.

Download the AMT-S checkpoint, then verify it in the configured motion environment:

```bash
mkdir -p /data/model-cache/vbench/amt_model
curl --fail --location --output /data/model-cache/vbench/amt_model/amt-s.pth \
  'https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth'
python scripts/verify_install.py --metric motion --cache-dir /data/model-cache/vbench
```

The scorer passes this directory through `VBENCH_CACHE_DIR` to the official evaluator and verifies the checkpoint before inference.

After preparing the environment and AMT-S checkpoint above, run in the active metric environment:

```bash
percept score motion \
  --video /data/videos/example.mp4 \
  --cache-dir /data/model-cache/vbench \
  --gpu 0 \
  --output /data/results/example.motion_smoothness.json
```

`--gpu` sets `CUDA_VISIBLE_DEVICES` for the child only; omit it to inherit the current setting. The official CLI requires CUDA. `--output` is a new JSON filename, whereas the official CLI's `--output_path` is a directory.

The wrapper checks VBench 0.1.5 and the AMT-S checkpoint path before launching the official CLI. It preserves the active Python environment for VBench's distributed child, scopes checkpoint compatibility settings to that process, and creates a fresh sibling directory named `<output-stem>.vbench-*` for official JSON files and `vbench.log`. These files are retained even on failure; output paths should be on a local filesystem supported by VBench. The wrapper validates a finite single-video result, then saves the unrounded official score plus input/checkpoint paths, VBench/Torch versions, command and original-result path. Existing output files are never overwritten. No inference is performed by `--help`, and the wrapper never resamples, resizes or interpolates frames itself.

This is only VBench Motion Smoothness (AMT-S), not all VBench dimensions and not MemoBench's RAFT-based MotionSmoothness. CLIP-IQA+ here is the MemoBench sampled-frame image-quality component, not its combined VisualQuality score. The two commands share the `metric`, `score`, `higher_is_better`, `video` and `versions` fields; other metadata is metric-specific.

Motion Smoothness estimates local interpolation smoothness. A smooth but incorrect action, a static video, or a video with incorrect object interactions can still score well; this metric does not establish physical correctness, causality, or long-term consistency.

Use a lowercase `.mp4` filename and at least three decodable frames. Keep the source resolution and frame rate fixed for comparisons. Large videos can exhaust GPU memory. Official results use `{"motion_smoothness": [mean, rows]}`, with `video_path` and `video_results` in each row.

## VBench and external evaluators

These independent commands use the pinned official VBench implementation internally:

| Command | Metric |
|---|---|
| `percept imaging-quality` | MUSIQ image quality |
| `percept aesthetic-quality` | Aesthetic quality |
| `percept temporal-flickering` | Temporal flickering |
| `percept dynamic-degree` | Dynamic degree |
| `percept subject-consistency` | Subject consistency |
| `percept background-consistency` | Background consistency |
| `percept overall-consistency` | Video-text consistency (ViCLIP) |

Each command selects one metric; there is no public `--dimension` selector or VBench total score. Shared source/cache checks and official invocation live in `video_metrics/vbench_backend.py`. Motion Smoothness uses `percept score motion`. The backend records the exact command, environment choices, source root, cache, and log. It does not alter VBench sampling, preprocessing, models, or formulas. `overall-consistency` requires a frozen prompt or video-to-prompt JSON. `temporal-flickering` requires an explicit static/near-static-subset acknowledgement. Result JSON keeps the official underscore-separated metric identifiers.

`percept score vbench-clip` calls the official competition
`competitions/clip_score.py` implementation.  It creates only the official
one-video metadata JSON and checks the official OpenAI CLIP ViT-B/32 cache;
frame sampling and similarity calculation remain in the upstream file.

`percept score external` provides preflight and command forwarding for
the external evaluator families. It checks required source revisions and local
resource paths, then runs the caller-supplied command after `--`. It does not
download models, infer the correct evaluator arguments, or compute scores itself.
Use `percept score external --help` for metric identifiers and
required resource options; `--check-only` performs preflight without inference.

The direct VBench wrapper is ready for environment-level smoke tests once the
corresponding cache and prompt manifest are available.  The wrapper does not
make these dimensions complete: MUSIQ/RAFT/CLIP/ViCLIP/DINO paths still need
cache mapping and a real single-video run; Temporal Flickering needs a frozen
static subset; and long-video VBench uses a separate official entry point.

Model-backed dimensions check the required checkpoint paths in `--cache-dir`.
An optional `--weights-manifest` supplies a JSON list of additional cache-relative
files to check. Legacy object manifests are accepted using their keys as paths;
values are ignored. The wrapper never downloads a missing checkpoint.
`--check-only` checks this inventory without starting inference. Provision trusted
checkpoints before opting into `--allow-trusted-pickle` for an official legacy loader.

The following remain external, conditional, or blocked and are intentionally
not faked by this package:

- DOVER needs its own Torch 1.13-compatible environment and `DOVER.pth`.
- Instruction Following needs WorldModelBench's VILA judge, data layout,
  and original question fields. Physical Adherence is not part of the
  frozen final benchmark list; it is retained only as an optional
  WorldModelBench diagnostic/compatibility reference.
- Action Binding / Motion Binding / Object Interactions need the official
  T2V-CompBench repositories, numeric IDs, metadata, and large judge/vision
  environments.
- Motion Order, Motion Rationality, Mechanics, Thermotics, and Material need
  VBench-2.0's full metadata and `auxiliary_info`; `custom_input` is rejected
  by the official code.
- PhyGenEval and VideoPhy-2 need their own checkpoints and environments.
- PSNR/SSIM/LPIPS need aligned future reference frames; FVD/FID need matched
  real collections.  The generated videos themselves are not valid GT.

### Individual metric commands

After installing the harness, verify the cache and choose a fresh output directory. Pass the separate pinned VBench source checkout through `--vbench-root`:

```bash
percept imaging-quality \
  --vbench-root /path/to/official-metrics/sources/VBench \
  --videos-path /path/to/video.mp4 \
  --cache-dir /path/to/vbench-cache \
  --weights-manifest /path/to/official-metrics/manifests/imaging_quality.weights.json \
  --gpu 0 \
  --output-dir /path/to/official-metrics/results/imaging_quality_one
```

For `percept overall-consistency`, add `--prompt` for a single video or
`--prompt-file /path/to/official-metrics/manifests/video_to_prompt.json`.
For `percept temporal-flickering`, add `--static-subset-ack` only after the subset
rule and excluded count have been recorded.

For CLIP Score, the official competition code hard-codes OpenAI CLIP
`ViT-B/32`.  Provision its official file at
`<clip-home>/.cache/clip/ViT-B-32.pt`, then run:

```bash
percept score vbench-clip \
  --vbench-root /path/to/official-metrics/sources/VBench \
  --videos-path /path/to/video.mp4 \
  --prompt 'the original frozen generation instruction' \
  --clip-home /path/to/official-metrics \
  --output-dir /path/to/official-metrics/results/clip_score_one
```

The wrapper does not copy or download the checkpoint.  It calls the
unchanged `competitions/clip_score.py` function and validates its one-video
result.

The first run should be one short, decodable video.  Inspect the official
`*_eval_results.json`, the wrapper metadata, and the log before scaling out.

## DOVER

`percept score dover` checks required upstream source files, runtime and checkpoint paths, then invokes the official evaluator. Source/cache preflight is available; this release does not establish successful DOVER model inference.

Pinned source: `VQAssessment/DOVER` at `f1ddc96215bc7fbcf8f315c65d47905f339c3419`. The evaluator requires both `pretrained_weights/DOVER.pth` under the source root and `hub/checkpoints/convnext_tiny_1k_224_ema.pth` under `TORCH_HOME`. Record the source URLs used to acquire these files.

Checkpoint sources are the [official DOVER release](https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth) and the [ConvNeXt initialization checkpoint](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth).

Use a separate runtime compatible with upstream requirements. The original Torch 1.13 family is not compatible with every current GPU. A modern runtime is an explicitly unverified compatibility experiment until compared against a validated baseline; `--allow-unverified-runtime` does not certify correctness.

Inspect the wrapper's options without loading models:

```bash
percept score dover --help
```

With a provisioned official source tree, weights and environment:

```bash
percept score dover \
  --dover-root /path/to/DOVER \
  --video /path/to/video.mp4 \
  --torch-home /path/to/torch-cache \
  --device cpu \
  --output-dir /path/to/new-preflight-output \
  --check-only
```

A successful check is `preflight_passed_not_scored`. Omit `--check-only` and use a new output directory only when ready for actual model inference. Preserve the command, environment, checkpoint paths, `dover.log` and `result.json`. An unparsed official result is not a score.

See the [catalog](#public-scoring-entry-points) for overall readiness and the [reproducibility notes](usage.md#reproducibility) for validation stages.

If the original evaluator requires Python older than the harness's Python 3.12 minimum, run the dependency-free wrapper file directly with that environment's Python:

```bash
python /path/to/WM-PerceptHarness/src/percept_harness/video_metrics/dover.py --help
```

The wrapper still lives in the package source; this does not require installing the harness into an incompatible upstream environment.

## Input preparation

`scripts/validate_metric_manifest.py` validates caller-supplied prompts, auxiliary questions and reference paths before evaluation. `scripts/build_reference_manifest.py` pairs generated and real future frames by identical relative filenames and reports unmatched files. Neither generates missing task inputs or reference frames.

`percept score videophy-joint --pc pc.csv --sa sa.csv --output joint.json` joins official VideoPhy-2 outputs by video ID and computes `SA >= 4 and PC >= 4`. This is an analysis field, not another model or an independent metric.

Keep source trees, checkpoints, manifests and outputs outside Git. Substitute paths appropriate to your environment in these examples.

### Manifest validation

```json
[
  {
    "video_id": "sample-0001",
    "video": "/path/to/video.mp4",
    "prompt": "the original frozen generation instruction",
    "dimensions": ["overall_consistency", "clip_score"],
    "auxiliary_info": "/path/to/official-metrics/manifests/sample-0001.json"
  }
]
```

Validate it before running:

```bash
python scripts/validate_metric_manifest.py \
  --manifest /path/to/official-metrics/manifests/videos.json \
  --dimension overall_consistency --check-files
```

Do not fill a prompt, action order, physical question, or reference path from
an automatic caption of the generated result.  Those fields are task inputs
and must be frozen from the original task design.

### Official input adapters

`scripts/official_input_adapters.py` is a format checker and field-preserving
adapter. It does not score videos, download checkpoints, generate prompts, or
replace an upstream evaluator. The caller remains responsible for supplying
the original task metadata and for running the official command in its
documented environment.

The emitted `official` files deliberately contain only the fields consumed by
the upstream repositories:

| adapter | upstream file shape |
| --- | --- |
| `worldmodelbench` | WorldModelBench `worldmodelbench.json`: `domain`, `subdomain`, `text_first_frame`, `text_instruction`, `first_frame` (domain/subdomain are optional in the upstream examples) |
| `action_binding` | T2V-CompBench V2 `meta_data/action_binding.json`: `prompt`, `phrase_0`, `phrase_1`; each phrase is exactly two strings |
| `motion_binding` | T2V-CompBench V2 `meta_data/motion_binding.json`: `prompt`, `object_1`, `d_1`, `object_2`, `d_2`; directions are `left/right/up/down` and the second pair is empty for a one-object task |
| `object_interactions` | T2V-CompBench V2 `meta_data/object_interactions.json`: `prompt` |
| `vbench2` | VBench-2.0 `prompts/meta_info/*.json` / `VBench2_full_info.json`: `prompt_en`, one retained `dimension`, and a non-empty string-list `auxiliary_info`; Motion Order has exactly two ordered actions |
| `phygen_*` | PhyGenBench `prompts.json` and the three question assets; records are validated and passed through without renaming fields |
| `videophy_pc` / `videophy_sa` | VideoPhy-2 AutoEval CSV: `videopath` for PC; `caption,videopath` for SA |

The official metadata files intentionally do not contain arbitrary local video
paths. For WorldModelBench, T2V-CompBench, and VBench-2.0 the script therefore
returns a separate `video_map` sidecar (`video_id` + `video`). The sidecar is a
local mapping aid and must not be mixed into an upstream metadata file.

Example local input for T2V-CompBench:

```json
[
  {
    "video_id": "sample-0001",
    "video": "/data/videos/sample-0001.mp4",
    "prompt": "A dog runs while a cat climbs a tree.",
    "phrase_0": ["a dog?", "a dog runs?"],
    "phrase_1": ["a cat?", "a cat climbs a tree?"]
  }
]
```

```bash
python scripts/official_input_adapters.py \
  --adapter action_binding \
  --input /data/manifests/action_binding.local.json \
  --output /data/manifests/action_binding.official.json \
  --video-map-output /data/manifests/action_binding.video_map.json
```

The command refuses to overwrite an existing output. It also refuses duplicate
video IDs/paths, unknown directions, missing prompts, malformed phrase lists,
and incomplete VBench-2.0 auxiliary questions. For PhyGenEval question files,
the adapter deliberately does not invent a schema: use the files shipped by
PhyGenBench or the exact schema accepted by the pinned upstream revision.

### Adapter input examples

Select the array for your adapter below, save it as your input JSON, and replace the example paths and task fields.

```json
{
  "worldmodelbench": [
    {
      "video_id": "sample-0001",
      "video": "/data/videos/sample-0001.mp4",
      "domain": "human",
      "subdomain": "hands",
      "text_first_frame": "A person sits at a table with both hands visible.",
      "text_instruction": "The person waves with one hand.",
      "first_frame": "images/sample-0001.jpg"
    }
  ],
  "action_binding": [
    {
      "video_id": "sample-0001",
      "video": "/data/videos/sample-0001.mp4",
      "prompt": "A dog runs through a field while a cat climbs a tree.",
      "phrase_0": ["a dog?", "a dog runs through a field?"],
      "phrase_1": ["a cat?", "a cat climbs a tree?"]
    }
  ],
  "motion_binding": [
    {
      "video_id": "sample-0001",
      "video": "/data/videos/sample-0001.mp4",
      "prompt": "A cat walks to the left.",
      "object_1": "cat",
      "d_1": "left",
      "object_2": "",
      "d_2": ""
    }
  ],
  "object_interactions": [
    {
      "video_id": "sample-0001",
      "video": "/data/videos/sample-0001.mp4",
      "prompt": "A child pushes a toy car."
    }
  ],
  "vbench2": [
    {
      "video_id": "sample-0001",
      "video": "/data/videos/sample-0001.mp4",
      "prompt_en": "A person sits, then gets up and sweeps the floor.",
      "dimension": "Motion_Order_Understanding",
      "auxiliary_info": ["sitting", "get up and sweeping"]
    }
  ],
  "videophy_pc": [
    {"videopath": "/data/videos/sample-0001.mp4"}
  ],
  "videophy_sa": [
    {"caption": "A person sits, then gets up and sweeps the floor.", "videopath": "/data/videos/sample-0001.mp4"}
  ]
}
```


## Benchmark reporting

The frozen selection from 2026-09-20 contains 21 main/specialist metrics and five reference/distribution appendix metrics. Selection is not evidence that every evaluator is provisioned or validated; use the current entry-point tables above for implementation scope.

| Group | Selected metrics |
|---|---|
| Image/video quality | MUSIQ, DOVER, Motion Smoothness, Temporal Flickering, Dynamic Degree |
| Text alignment | CLIPScore, ViCLIP / Overall Consistency |
| Instruction/action | Instruction Following, Action Binding, Motion Binding, Motion Order Understanding, Motion Rationality, Object Interactions |
| Physics | PhyGenEval PCA, VideoPhy-2 PC, VideoPhy-2 SA |
| Specialist physics | Mechanics, Thermotics, Material |
| Consistency | Subject Consistency, Background Consistency |
| Appendix | PSNR, SSIM, LPIPS, FVD, FID |

Mechanics, Thermotics and Material are separate specialist reports, excluded from a unified main score. PSNR/SSIM/LPIPS need aligned real future frames; FVD/FID need a matching real collection or statistics. Without those references, report `N/A`.

Aesthetic Quality and long-video consistency variants are diagnostics; CLIP-IQA+ is a backup. Physical Adherence is outside the frozen selection. VideoPhy-2 Joint is an analysis of paired PC/SA scores (`PC >= 4` and `SA >= 4`), not an additional model or metric. Grounded event scoring is the separate annotation-based workflow. Report scores, coverage and exclusions separately rather than averaging incompatible metrics.

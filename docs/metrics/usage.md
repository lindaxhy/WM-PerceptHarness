# Video metric installation and protocols

See the [metric catalog](README.md) for scope and validation status. Long remote runs can optionally use tmux; it is not a dependency.

This directory documents standalone use of two video metrics, separate from the core `percept` package. CLIP-IQA+ supports CPU or CUDA; the VBench 0.1.5 CLI requires CUDA. Both use external model checkpoints.

Both metric commands accept one video and produce a JSON result. The Motion Smoothness wrapper invokes the unchanged official `vbench==0.1.5` CLI; it does not reimplement its scoring algorithm or alter its video preprocessing.

## CLIP-IQA+

`percept score clipiqa` reuses the video decoding and frame-scoring protocol from MemoBench revision `f4edb0c4f9f1820bac837ee30d4957811cf275ff`. The model is supplied by PyIQA (`pyiqa==0.1.16`). The command does not implement CLIP-IQA+ itself and does not fall back to MUSIQ, Laplacian sharpness, or another metric.

The fixed defaults are:

- sample frames `0, 4, 8, ...` (`--sample-step 4`);
- resize only when the long side exceeds 640 pixels (`--max-side 640`), preserving the aspect ratio with rounded dimensions and OpenCV `INTER_AREA`;
- score each sampled frame with `pyiqa.create_metric("clipiqa+")`;
- clip each frame score to `[0, 1]`, average the frame scores, and round the video score to four decimal places.

The model score is a perceptual image-quality score applied to sampled video frames. It is not a temporal-consistency or action-correctness score.

### Run it

Create a dedicated environment from the repository root:

```bash
python3.12 -m venv "$HOME/venvs/percept-metrics"
source "$HOME/venvs/percept-metrics/bin/activate"
python -m pip install -e .
```

Install a hardware-compatible PyTorch and TorchVision build first. Then install the CLIP-IQA+ runtime requirements:

```bash
python -m pip install -r requirements/metrics-clipiqa.txt
```

The requirements file deliberately does not choose a PyTorch wheel; use the index and CUDA version appropriate for the server. The recorded validation environment uses `torch==2.10.0` / `torchvision==0.25.0` from the cu128 wheel index, for example:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

Run that command before installing the requirements when this build matches your hardware. A fresh installation of the full CLIP-IQA+ dependency tree has not been validated by this package. Keep it outside the core harness environment.

Score one video and print JSON to stdout:

```bash
percept score clipiqa \
  --video /data/videos/example.mp4 \
  --device auto \
  --sample-step 4 \
  --max-side 640
```

Save the same JSON to a new file (the script refuses to overwrite an existing file):

```bash
percept score clipiqa \
  --video /data/videos/example.mp4 \
  --output /data/results/example.clipiqa_plus.json
```

`--device auto` selects `cuda:0` when CUDA is available and otherwise CPU. You can pass `cpu`, `cuda:0`, or another valid PyTorch device explicitly. The output contains `score` in `[0, 1]`, the resolved video path and SHA-256, frame counts, sampling settings, dependency versions, and the MemoBench revision. The model weights are not part of the repository; see `THIRD_PARTY_NOTICES.md` and `weights.json` for provenance and expected cache locations.

## VBench Motion Smoothness

VBench already accepts a single MP4 or a directory of videos for `motion_smoothness`. Prepare its environment as follows; the standalone wrapper below uses this same official CLI:

```bash
python3.12 -m venv "$HOME/venvs/percept-motion"
source "$HOME/venvs/percept-motion/bin/activate"
python -m pip install -e .
```

After entering the session, install a hardware-compatible PyTorch/TorchVision build, install the runtime dependencies, and install the wheel without allowing pip to select a source distribution:

```bash
python -m pip install -r requirements/metrics-motion-runtime.txt
python -m pip install --only-binary=:all: --no-deps vbench==0.1.5
```

`metrics-motion-runtime.txt` records the versions selected from the recorded validation environment. It is an installation starting point, not a complete lock file. The official VBench wheel declares a larger, older dependency set for other dimensions (including `numpy<2` and `transformers==4.33.2`); the command above deliberately installs only the Motion Smoothness runtime. Consequently, a global `pip check` can report unmet VBench metadata requirements. This environment is for this dimension only, not all VBench tasks.

The AMT-S checkpoint is stored outside the repository. Set a cache directory and verify the downloaded file against the SHA-256 listed in `weights.json` before evaluation:

```bash
export VBENCH_CACHE_DIR=/data/model-cache/vbench
mkdir -p "$VBENCH_CACHE_DIR/amt_model"
# Place amt-s.pth at "$VBENCH_CACHE_DIR/amt_model/amt-s.pth".
sha256sum "$VBENCH_CACHE_DIR/amt_model/amt-s.pth"
```

The official checkpoint URL and expected digest are recorded in `weights.json`. VBench's default cache is `~/.cache/vbench`; `VBENCH_CACHE_DIR` is used so that the cache is explicit and reproducible.

Select the GPU with `CUDA_VISIBLE_DEVICES`. The official 0.1.5 CLI has no `--device` or `--local` flag. Its local-checkpoint option is named `--load_ckpt_from_local`, but that option does not change the Motion Smoothness checkpoint path; use the explicit `VBENCH_CACHE_DIR` above instead. For `--mode=custom_input`, `--full_json_dir` is normally unnecessary because VBench builds the input metadata from the supplied video path.

Keep the modern PyTorch checkpoint compatibility setting scoped to this command, and put the selected virtual environment first in `PATH`: VBench launches its distributed child with the literal command `python`.

```bash
VENV="$VIRTUAL_ENV"
OUT=/data/results/vbench_motion_example
PATH="$VENV/bin:$PATH" \
VBENCH_CACHE_DIR=/data/model-cache/vbench \
CUDA_VISIBLE_DEVICES=0 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"$VENV/bin/vbench" evaluate \
  --videos_path /data/videos/example.mp4 \
  --dimension motion_smoothness \
  --mode=custom_input \
  --output_path "$OUT"
```

The same command accepts a directory instead of one MP4. `--output_path` is a directory, not a filename. Use a lowercase `.mp4` filename and at least three decodable frames; this dimension's implementation does not support every video extension. Keep the source resolution and frame rate for comparisons. Choose a new output directory per run. Long or large videos can exhaust GPU memory; the allocator setting is not a guarantee that any video will fit.

A successful run creates files like:

```text
$OUT/results_<timestamp>_full_info.json
$OUT/results_<timestamp>_eval_results.json
```

The evaluation JSON has the form `{"motion_smoothness": [mean, rows]}`. Each row contains `video_path` and `video_results`; the first value is the mean over the evaluated videos. Read the newest result without depending on a hard-coded timestamp:

```bash
python - <<'PY'
import glob
import json

paths = sorted(glob.glob("/data/results/vbench_motion_example/results_*_eval_results.json"))
if not paths:
    raise SystemExit("No VBench evaluation result was found")
with open(paths[-1], encoding="utf-8") as stream:
    result = json.load(stream)["motion_smoothness"]
mean, rows = result
print("mean:", mean)
for row in rows:
    print(row["video_path"], row["video_results"])
PY
```

### Standalone single-video wrapper

After preparing the environment and AMT-S checkpoint above, run in the active metric environment:

```bash
percept score motion \
  --video /data/videos/example.mp4 \
  --cache-dir /data/model-cache/vbench \
  --gpu 0 \
  --output /data/results/example.motion_smoothness.json
```

`--gpu` sets `CUDA_VISIBLE_DEVICES` for the child only; omit it to inherit the current setting. The official CLI requires CUDA. `--output` is a new JSON filename, whereas the official CLI's `--output_path` above is a directory.

The wrapper verifies VBench 0.1.5 and the documented AMT-S SHA-256 before launching the official CLI. It preserves the active Python environment for VBench's distributed child, scopes checkpoint compatibility settings to that process, and creates a fresh sibling directory named `<output-stem>.vbench-*` for official JSON files and `vbench.log`. These files are retained even on failure; output paths should be on a local filesystem supported by VBench. The wrapper validates a finite single-video result and an unchanged input hash, then saves the unrounded official score plus input/checkpoint hashes, VBench/Torch versions, command and original-result path. Existing output files are never overwritten. No inference is performed by `--help`, and the wrapper never resamples, resizes or interpolates frames itself.

This is only VBench Motion Smoothness (AMT-S), not all VBench dimensions and not MemoBench's RAFT-based MotionSmoothness. CLIP-IQA+ here is the MemoBench sampled-frame image-quality component, not its combined VisualQuality score. The two scripts share the `metric`, `score`, `higher_is_better`, `video`, `video_sha256` and `versions` fields; other metadata is metric-specific.

Motion Smoothness estimates local interpolation smoothness. A smooth but incorrect action, a static video, or a video with incorrect object interactions can still score well; this metric does not establish physical correctness, causality, or long-term consistency.

## Sources and files

- `percept score clipiqa` (`src/percept_harness/video_metrics/clipiqa.py`) — one-video CLIP-IQA+ entry point.
- `percept score motion` (`src/percept_harness/video_metrics/motion.py`) — one-video wrapper around the official VBench Motion Smoothness CLI.
- `requirements/metrics-clipiqa.txt` — CLIP-IQA+ user-space dependencies; install PyTorch separately for the target hardware.
- `requirements/metrics-motion-runtime.txt` — dependencies imported by the VBench Motion Smoothness path.
- `THIRD_PARTY_NOTICES.md` — source revisions, licenses, and modifications.
- `weights.json` — checkpoint URLs, cache locations, and SHA-256 digests.

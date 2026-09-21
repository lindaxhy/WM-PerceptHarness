# Local smoke example

This synthetic clip and the fake backend exercise decoding, annotation output, and the CLI without external models or API keys. They are not a scientific example or benchmark result.

From the repository root after installing the core package and FFmpeg:

```bash
mkdir -p outputs/smoke
ffmpeg -n -f lavfi -i color=c=blue:s=64x64:r=8 \
  -t 2 -c:v libx264 -pix_fmt yuv420p outputs/smoke/sample.mp4
percept annotate --videos outputs/smoke/sample.mp4 \
  --template general_video_captioning --backend fake \
  --output outputs/smoke/annotations
```

Expected: the CLI reports `1 completed`, and writes `sample.json` with `status: "completed"` plus `results.jsonl`. Repeating annotation reports a skipped result. FFmpeg's `-n` refuses to replace an existing clip; use a new directory for a fresh run.

For a real evaluation, replace the clip with your data, configure a VLM as shown in the [README](../README.md), and omit `--backend fake`.

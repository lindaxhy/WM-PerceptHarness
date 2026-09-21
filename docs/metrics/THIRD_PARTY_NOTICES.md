# Metric sources and third-party notices

`percept score clipiqa` (`src/percept_harness/video_metrics/clipiqa.py`) is a
standalone invocation tool, not a new
CLIP-IQA+ model or an official MemoBench release. The default video protocol is
the one used by MemoBench's evaluation runner: frame stride 4 and maximum image
side 640. MemoBench's lower-level function has a different default stride of 5;
this tool passes 4 explicitly.

## Code adapted into this repository

MemoBench, revision `f4edb0c4f9f1820bac837ee30d4957811cf275ff`:

- [Video reader and resizing](https://github.com/MemoBench-Team/MemoBench/blob/f4edb0c4f9f1820bac837ee30d4957811cf275ff/evaluation/automated/io/frames.py)
- [CLIP-IQA+ frame scoring and averaging](https://github.com/MemoBench-Team/MemoBench/blob/f4edb0c4f9f1820bac837ee30d4957811cf275ff/evaluation/automated/metrics/visual_quality.py)
- [Evaluation runner settings](https://github.com/MemoBench-Team/MemoBench/blob/f4edb0c4f9f1820bac837ee30d4957811cf275ff/evaluation/run_eval.py)

The original code is MIT licensed, copyright (c) 2026 Haoyu Chen. The complete
notice is retained in [licenses/MemoBench-MIT.txt](licenses/MemoBench-MIT.txt).

The script extracts the video reader, BGR-to-RGB/PIL/ToTensor conversion, frame
score clipping, and arithmetic mean. It preserves the interpolation method,
rounded resize dimensions, and four-decimal final rounding. It removes unrelated
aesthetic/ImageReward functions, frame-directory handling, global model caches,
and the MUSIQ/Laplacian fallback. It adds CLI arguments, optional imports, input
validation, explicit video release, non-finite score rejection, JSON output,
version metadata, and video SHA-256. Model loading errors now fail the command.

## External packages and models (not vendored)

| Component | Selected implementation | Upstream information |
| --- | --- | --- |
| CLIP-IQA+ | PyIQA 0.1.16, model name `clipiqa+`, RN50 with learned prompts | [PyIQA](https://github.com/chaofengc/IQA-PyTorch), [original CLIP-IQA](https://github.com/IceClear/CLIP-IQA) |
| Motion Smoothness | VBench 0.1.5 official CLI | [VBench 0.1.5](https://github.com/Vchitect/VBench/tree/v0.1.5) |
| Motion interpolation | AMT-S as selected by VBench | [AMT](https://github.com/MCG-NKU/AMT) |
| CLIP image backbone | RN50 as loaded by PyIQA | [OpenAI CLIP](https://github.com/openai/CLIP) |

PyIQA 0.1.16's wheel declares PolyForm Noncommercial 1.0.0 and includes an
additional NTU S-Lab license for applicable components. The original CLIP-IQA
repository uses NTU S-Lab License 1.0. VBench 0.1.5 is Apache-2.0 licensed.
The copied MemoBench MIT notice does not relicense these external dependencies
or their model weights. Refer to the licenses shipped with the exact installed
versions and to the upstream checkpoint terms.

No model checkpoints or third-party model implementation are distributed here.
[weights.json](weights.json) records download URLs, cache-relative filenames,
and reference SHA-256 values from the existing evaluation environment. It is
documentation for CLIP-IQA+; that script does not enforce these hashes at runtime.
The Motion Smoothness wrapper verifies the recorded AMT-S hash before enabling
legacy checkpoint loading in its child process.

`percept score motion` (`src/percept_harness/video_metrics/motion.py`) is a
local invocation wrapper. It
executes the installed official CLI without copying or modifying VBench/AMT
scoring code. It adds input/checkpoint checks, isolated run directories, logging,
result validation and single-video JSON metadata. Official results are retained.

## Metric interpretation

CLIP-IQA+ here is mean sampled-frame image quality. It does not learn temporal
relationships. This is not MemoBench's combined VisualQuality score, which also
uses an aesthetic component. VBench Motion Smoothness uses AMT-S interpolation
error; it is distinct from MemoBench's RAFT-based metric of a similar name.
Neither score alone establishes physical plausibility or action correctness.

References: [CLIP-IQA paper](https://arxiv.org/abs/2207.12396),
[MemoBench](https://arxiv.org/abs/2606.27537),
[VBench](https://arxiv.org/abs/2311.17982).

# SAM3.1 runtime provisioning evidence

Date: 2026-09-05

The four-GPU host was recovered through an existing authenticated SSH session.
Read-only device inspection reported four RTX 5090 GPUs, each with 32,607 MiB
total memory and 1 MiB used before any model was loaded. Its Python is 3.12.13
and PyTorch is 2.10.0+cu128 (CUDA 12.8).

Following the user's ModelScope suggestion, the SAM3.1 Object Multiplex
checkpoint was downloaded successfully and its complete file hash verified:

| Property | Verified value |
| --- | --- |
| Publisher repository | `facebook/sam3.1` on ModelScope |
| Repository revision | `616acbee0b9ed4177f1f389e3c13594a0a1f6398` |
| File | `sam3.1_multiplex.pt` |
| Bytes | 3,502,755,717 |
| SHA-256 | `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` |
| Source repository | `facebookresearch/sam3` |
| Checked-out source revision | `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` |

The download used the pinned revision, first wrote a staging file, and promoted
it only after `sha256sum -c` succeeded against ModelScope's published digest.
It did not require Hugging Face credentials. Hugging Face's public file listing
shows the same filename and size but redacts its LFS hash for unauthenticated
requests; this report does not claim an independent Hugging Face hash check.

A dedicated SAM virtual environment was created. It reuses the host's existing
PyTorch installation through system site packages and contains its own
NumPy 1.26.4, ftfy 6.1.1, iopath 0.1.10, pycocotools 2.0.11, and editable pinned
SAM source. The existing Qwen virtual environment was not modified. Importing
`sam3.model_builder` succeeds after adding `pycocotools`, which the upstream base
install did not include. Upstream pkg_resources and timm deprecation warnings
remain visible. The host's unrelated globally installed packages have reported
dependency conflicts; full inference must still verify the actual SAM path.

## Real CUDA preflight

The pinned provider loaded on physical GPU 3 in 26.307 seconds and closed
successfully. The other three GPUs remained idle. The initial real-video probe
then exposed three adapter/runtime compatibility issues:

- Host FFmpeg 4.4.2 does not recognize `-fps_mode:v`. The dedicated SAM
  environment now supplies imageio-ffmpeg 0.6.0's FFmpeg 7.0.2-static. Exact
  extraction of the source's 137 frames succeeds; system FFprobe still supplies
  the original PTS.
- The upstream builder defaults to FlashAttention 3, but `flash_attn_interface`
  is not installed. Its supported `use_fa3=False` path works with existing Torch.
- The adapter passed the last frame index as `max_frame_num_to_track`. The
  pinned tracker's inclusive bound and detector's exclusive chunk bound then
  produced an empty feature tensor for the last frame. Passing the frame count
  resolves that mismatch without modifying upstream code.

Separately, rounded FFprobe timestamps make nominal 30 fps appear as
30.000002205882517 fps. The previous exact sampler selected only 92 of 137 source
frames. The adapter now uses a one-microsecond tolerance for timestamp comparisons,
bounded by the difference between two six-decimal FFprobe roundings. It preserves
all 137 source indices and their observed PTS unchanged, while genuinely higher-rate
sources, long-video scans, and refinement windows remain capped by the approved
sampling rates.

A diagnostic run on frozen `full_0024` using the baseline `ba1f46f` wheel and
the two explicit attention/count overrides succeeded in 46.010 seconds:
two entity prompts, one track, 92 observations, 15,700,532,224 bytes peak Torch
allocation, artifact publication and digest-checked cache reload successful.
Source SHA-256 is
`a7a696bcdd835c083b27ca3705d13a2f22e069ebec9038581354fed39e6fbbe8`.
This demonstrates actual checkpoint inference and artifact generation; it is
not production-adapter acceptance because of the diagnostic overrides and
incomplete sampling. The production adapter now passes `use_fa3=False` and the
sampled-frame count to the pinned builder/tracker boundary. A fresh immutable-wheel
real-GPU run with all 137 original index/PTS mappings remains controller-owned and
has not yet been accepted. The formal corrected run, memory
stability checks, and five-demo acceptance gates remain required.

Sources: [ModelScope model](https://www.modelscope.cn/models/facebook/sam3.1),
[ModelScope file metadata](https://www.modelscope.cn/api/v1/models/facebook/sam3.1/repo/files?Revision=616acbee0b9ed4177f1f389e3c13594a0a1f6398&Recursive=true),
[Meta release at the pinned source revision](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/RELEASE_SAM3p1.md).

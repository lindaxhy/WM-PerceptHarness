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

This is provisioning evidence only. It does not establish successful CUDA
checkpoint loading, valid track/artifact generation, memory stability, or the
five-demo acceptance gates. Those remain required before final delivery.

Sources: [ModelScope model](https://www.modelscope.cn/models/facebook/sam3.1),
[ModelScope file metadata](https://www.modelscope.cn/api/v1/models/facebook/sam3.1/repo/files?Revision=616acbee0b9ed4177f1f389e3c13594a0a1f6398&Recursive=true),
[Meta release at the pinned source revision](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/RELEASE_SAM3p1.md).

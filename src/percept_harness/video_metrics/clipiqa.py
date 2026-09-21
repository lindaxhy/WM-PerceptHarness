#!/usr/bin/env python3
"""Score one video with CLIP-IQA+ using MemoBench's frame sampling protocol.

Video decoding and frame scoring are adapted from MemoBench, revision
f4edb0c4f9f1820bac837ee30d4957811cf275ff (MIT, Copyright 2026 Haoyu Chen).
See docs/metrics/THIRD_PARTY_NOTICES.md for sources and modifications.
The CLIP-IQA+ model itself is supplied by PyIQA, not implemented here.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

MEMOBENCH_REVISION = "f4edb0c4f9f1820bac837ee30d4957811cf275ff"
PYIQA_VERSION = "0.1.16"


class VideoReader:
    """MemoBench's on-demand OpenCV reader, with explicit resource cleanup."""

    def __init__(self, video_path: str, max_side: int = 640):
        import cv2

        self.max_side = max_side
        self.video_path = video_path
        self._cap = cv2.VideoCapture(video_path)
        self._pos = -1
        count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(count) or count < 1:
            self.close()
            raise ValueError(f"Cannot decode video or frame count is zero: {video_path}")
        self.num_frames = int(count)

    def get(self, index: int):
        import cv2

        if index != self._pos + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {index}: {self.video_path}")
        self._pos = index
        height, width = frame.shape[:2]
        if max(height, width) > self.max_side:
            scale = self.max_side / float(max(height, width))
            frame = cv2.resize(
                frame,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def close(self):
        self._cap.release()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_video(
    video_path: str | Path,
    device: str = "auto",
    sample_step: int = 4,
    max_side: int = 640,
) -> dict:
    """Return one CLIP-IQA+ video score and its sampling/version metadata.

    Frame scores are clipped to [0, 1], averaged with NumPy, and rounded
    to four decimals, matching MemoBench. There is no alternate metric fallback.
    """
    path = Path(video_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a video file: {path}")
    if sample_step < 1 or max_side < 1:
        raise ValueError("sample_step and max_side must be positive")
    version = importlib.metadata.version("pyiqa")
    if version != PYIQA_VERSION:
        raise RuntimeError(f"Expected pyiqa=={PYIQA_VERSION}; found {version}")

    # Optional GPU dependencies must not be imported by --help or the harness.
    import cv2
    import numpy as np
    import pyiqa
    import torch
    import torchvision
    from PIL import Image
    from torchvision.transforms import ToTensor

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reader = VideoReader(str(path), max_side=max_side)
    try:
        # Load exactly this model; propagate loading errors rather than substituting.
        metric = pyiqa.create_metric("clipiqa+", device=device).eval()
        transform = ToTensor()
        indices = range(0, reader.num_frames, sample_step)
        scores = []
        for index in indices:
            frame = reader.get(index)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                raw = float(metric(tensor).item())
            if not math.isfinite(raw):
                raise RuntimeError(f"Non-finite CLIP-IQA+ score at frame {index}")
            scores.append(float(np.clip(raw, 0.0, 1.0)))
        score = round(float(np.mean(scores)), 4)
    finally:
        reader.close()

    return {
        "metric": "clipiqa+",
        "score": score,
        "higher_is_better": True,
        "video": str(path),
        "video_sha256": _sha256(path),
        "frame_count": reader.num_frames,
        "sampled_frames": len(scores),
        "sampling": {
            "start_frame": 0,
            "sample_step": sample_step,
            "max_side": max_side,
            "resize": "round dimensions; OpenCV INTER_AREA; no upscaling",
            "aggregation": "mean of clipped frame scores; round to 4 decimals",
        },
        "device": device,
        "versions": {
            "pyiqa": version,
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "memobench_revision": MEMOBENCH_REVISION,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="One video file")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--sample-step", type=_positive_int, default=4)
    parser.add_argument("--max-side", type=_positive_int, default=640)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; refuses overwrite")
    args = parser.parse_args(argv)
    try:
        output = args.output.expanduser().resolve() if args.output else None
        if output is not None and output.exists():
            raise FileExistsError(f"Output already exists; choose a new filename: {output}")
        # Keep stdout machine-readable; model loading messages go to stderr.
        with redirect_stdout(sys.stderr):
            result = score_video(args.video, args.device, args.sample_step, args.max_side)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        print(payload, end="")
        return 0
    except Exception as error:
        print(f"CLIP-IQA+ failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

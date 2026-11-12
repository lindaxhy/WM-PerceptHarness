"""Sampling plans for the SA/PC experimental entrypoints.

Official mode reproduces each author's fixed-frame indices. FPS mode uses
presentation timestamps, retains the endpoints, and never silently caps frames.
VideoPhy half mode uses at least 32 frames, following the author's 32-frame
indices below that floor. Otherwise it selects source indices 0, 2, 4, ...;
an explicit cap resamples the whole clip uniformly when necessary.
"""
from bisect import bisect_left
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass
class SamplingPlan:
    mode: str
    family: str
    source_frames: int
    source_fps: float
    duration_s: float
    requested_fps: float | None
    indices: list[int]
    timestamps_s: list[float]
    timestamp_basis: str
    frame_cap: int | None = None
    uncapped_sampled_frames: int | None = None
    sampling_cap_applied: bool = False
    frame_floor: int | None = None
    sampling_floor_applied: bool = False

    def to_dict(self):
        result = asdict(self)
        gaps = [b - a for a, b in zip(self.timestamps_s, self.timestamps_s[1:])]
        result.update(sampled_frames=len(self.indices), unique_frames=len(set(self.indices)),
                      max_gap_s=max(gaps, default=0.0),
                      effective_interval_fps=(len(gaps) / sum(gaps) if sum(gaps) else None),
                      source_frame_stride=2 if self.mode == 'half' and not (
                          self.sampling_cap_applied or self.sampling_floor_applied) else None,
                      requested_frame_ratio=0.5 if self.mode == 'half' else None,
                      selection_policy=('uniform source indices over full clip, including endpoints'
                                        if self.sampling_cap_applied else
                                        'official VideoPhy-2 32-frame segment centers; duplicates allowed'
                                        if self.sampling_floor_applied else
                                        'source frame stride 2' if self.mode == 'half' else self.mode),
                      actual_frame_ratio=len(set(self.indices)) / self.source_frames)
        return result


def make_plan(total_frames, source_fps, *, family, mode='official', sample_fps=None,
              timestamps=None, duration_s=None, frame_cap=None):
    if family != 'videophy' or mode not in ('official', 'fps', 'half'):
        raise ValueError('Unknown sampling family/mode')
    if frame_cap is not None and (mode != 'half' or not isinstance(frame_cap, int) or frame_cap < 2):
        raise ValueError('frame_cap must be an integer >=2 and applies only to half sampling')
    if total_frames < 1 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError('Video must have frames and a positive finite FPS')
    uncapped_count = None
    cap_applied = False
    frame_floor = 32 if mode == 'half' else None
    floor_applied = False
    if mode == 'official':
        # Exact pinned author algorithms, including repeated frames for short clips.
        import numpy as np
        segment = float(total_frames - 1) / 32
        start = int(segment / 2)
        indices = [start + int(np.round(segment * i)) for i in range(32)]
        duration = total_frames / source_fps
        times = [i / source_fps for i in indices]
        basis = 'official average FPS'
    else:
        if mode == 'fps' and (sample_fps is None or not math.isfinite(sample_fps) or sample_fps <= 0):
            raise ValueError('FPS mode requires a positive finite sample_fps')
        starts = list(timestamps) if timestamps is not None else [i / source_fps for i in range(total_frames)]
        if len(starts) != total_frames or not all(math.isfinite(t) for t in starts):
            raise ValueError('Invalid presentation timestamp array')
        origin = starts[0]
        starts = [t - origin for t in starts]
        if any(b <= a for a, b in zip(starts, starts[1:])):
            raise ValueError('Presentation timestamps must strictly increase')
        duration = duration_s if duration_s is not None else total_frames / source_fps
        if not math.isfinite(duration) or duration <= starts[-1]:
            raise ValueError('Duration must extend beyond the final frame start')
        if mode == 'half':
            uncapped_count = (total_frames + 1) // 2
            floor_applied = uncapped_count < frame_floor
            cap_applied = frame_cap is not None and max(uncapped_count, frame_floor) > frame_cap
            if cap_applied:
                # Equal source-index intervals over the entire clip; includes both
                # endpoints and cannot repeat indices because frame_cap < ceil(N/2).
                indices = [i * (total_frames - 1) // (frame_cap - 1) for i in range(frame_cap)]
            elif floor_applied:
                # Exact author algorithm, including repeated source indices when
                # the clip contains fewer than 32 distinct source frames.
                import numpy as np
                segment = float(total_frames - 1) / 32
                start = int(segment / 2)
                indices = [start + int(np.round(segment * i)) for i in range(32)]
            else:
                indices = list(range(0, total_frames, 2))
        # For high requested rates, select all available frames without duplication.
        elif sample_fps >= total_frames / duration:
            indices = list(range(total_frames))
        else:
            indices = []
            for k in range(math.floor(starts[-1] * sample_fps + 1e-9) + 1):
                target = k / sample_fps
                right = bisect_left(starts, target)
                left = max(0, right - 1)
                right = min(right, total_frames - 1)
                index = left if target - starts[left] <= starts[right] - target else right
                if not indices or index != indices[-1]:
                    indices.append(index)
            if indices[-1] != total_frames - 1:
                indices.append(total_frames - 1)
        times = [starts[i] for i in indices]
        basis = 'presentation timestamps' if timestamps is not None else 'CFR fallback'
    return SamplingPlan(mode, family, total_frames, source_fps, duration,
                        sample_fps if mode == 'fps' else None, indices, times, basis,
                        frame_cap, uncapped_count, cap_applied, frame_floor, floor_applied)


def open_video_plan(path, *, family, mode, sample_fps, resize=None, frame_cap=None):
    from decord import VideoReader, cpu
    kwargs = dict(ctx=cpu(0), num_threads=1)
    if resize is not None:
        kwargs.update(width=resize, height=resize)
    reader = VideoReader(str(path), **kwargs)
    times = None
    duration = None
    if mode in ('fps', 'half'):
        pts = reader.get_frame_timestamp(list(range(len(reader))))
        times = pts[:, 0].tolist()
        duration = float(pts[-1, 1] - pts[0, 0])
    plan = make_plan(len(reader), float(reader.get_avg_fps()), family=family,
                     mode=mode, sample_fps=sample_fps,
                     timestamps=times, duration_s=duration, frame_cap=frame_cap)
    return reader, plan


def official_rating(text):
    # Preserve the author's parser. Record raw output separately for later review.
    num_map = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
               '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    value = text.lower().strip()
    for key, score in num_map.items():
        if key in value:
            return score
    digits = ''.join(c for c in value if c.isdigit())
    return int(digits) if digits and int(digits) in num_map.values() else 0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def experiment_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

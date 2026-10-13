"""Provider-independent computer-vision evidence execution boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

from .contracts import (
    ArtifactFile,
    CvEvidenceArtifact,
    CvEvidenceRequest,
    CvTrack,
    EvidenceStatus,
    FrameTimeline,
    TrackObservation,
    OverlayRecord,
)


class CvProviderError(RuntimeError):
    """A sanitized CV provider boundary failure."""


class CvOutOfMemoryError(CvProviderError):
    """A provider-reported allocation failure eligible for one retry."""


@runtime_checkable
class CvEvidenceProvider(Protocol):
    def analyze(
        self, request: CvEvidenceRequest, staging_dir: Path
    ) -> CvEvidenceArtifact:
        """Write provider-owned files and return their strict manifest."""


class FakeCvEvidenceProvider:
    """Deterministic CPU-only evidence provider for tests and local demos."""

    def __init__(self, *, execution_chunk_frames: int = 1) -> None:
        self.execution_chunk_frames = _positive_integer(
            execution_chunk_frames, "execution_chunk_frames"
        )
        self._metrics = {
            "processed_frames": 0,
            "execution_chunk_frames": self.execution_chunk_frames,
            "entity_prompts": 0,
            "track_count": 0,
            "peak_allocated_bytes": 0,
        }

    def analyze(
        self, request: CvEvidenceRequest, staging_dir: Path
    ) -> CvEvidenceArtifact:
        """Write stable mask fixtures and return matching stable tracks."""
        try:
            staging = Path(staging_dir)
            staging.mkdir(mode=0o700, parents=True, exist_ok=True)
            prompts = {prompt.entity_id: prompt for prompt in request.spatial_prompts}
            references = {prompt.frame_index for prompt in request.spatial_prompts}
            processed_frames = tuple(
                frame for frame in request.timeline.frames
                if request.mode != "probe" or frame.frame_index in references
            )
            tracks: list[CvTrack] = []
            files: list[ArtifactFile] = []
            for entity_index, entity in enumerate(request.entities):
                spatial = prompts.get(entity.entity_id)
                entity_frames = tuple(
                    frame for frame in processed_frames
                    if request.mode != "probe" or frame.frame_index == spatial.frame_index
                )
                observations: list[TrackObservation] = []
                mask_ref = f"masks/{entity.entity_id}.mask"
                payload = (
                    f"fake-cv-mask-v1\n{entity.entity_id}\n"
                    + "\n".join(
                        f"{frame.frame_index}:{frame.timestamp_seconds:.9f}"
                        for frame in entity_frames
                    )
                    + "\n"
                ).encode("ascii")
                (staging / mask_ref).parent.mkdir(mode=0o700, exist_ok=True)
                (staging / mask_ref).write_bytes(payload)
                files.append(
                    ArtifactFile(
                        path=mask_ref,
                        size_bytes=len(payload),
                    )
                )
                for observation_index, frame in enumerate(entity_frames):
                    visible = (
                        len(entity_frames) == 1
                        or observation_index < len(entity_frames) - 1
                    )
                    left = 0.1 + 0.1 * (entity_index % 5)
                    top = 0.1 + 0.05 * (entity_index % 5)
                    box = (
                        spatial.bbox_xyxy if spatial is not None
                        else (left, top, left + 0.08, top + 0.08)
                    )
                    observations.append(
                        TrackObservation(
                            frame_index=frame.frame_index,
                            timestamp_seconds=frame.timestamp_seconds,
                            bbox_xyxy=box,
                            mask_ref=mask_ref if visible else None,
                            visible=visible,
                            confidence=max(request.thresholds.min_confidence, 0.9),
                            area_fraction=max(
                                request.thresholds.min_area_fraction, 0.01
                            ),
                            center_xy=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                        )
                    )
                tracks.append(
                    CvTrack(
                        track_id=f"{entity.entity_id}_1",
                        entity_id=entity.entity_id,
                        observations=tuple(observations),
                    )
                )
            # Synthetic overlays use decoded source pixels and the fixture box.
            # These are explicitly synthetic; no learned segmentation is implied.
            overlays = []
            for track in tracks[:24]:
                observation = track.observations[0]
                relative = f'overlays/{track.track_id}.png'
                destination = staging / relative
                destination.parent.mkdir(mode=0o700, exist_ok=True)
                x, y, right, bottom = observation.bbox_xyxy
                subprocess.run(['ffmpeg', '-v', 'error', '-i', str(request.video_path),
                    '-vf', f"select=eq(n\\,{observation.frame_index}),drawbox=x=iw*{x}:y=ih*{y}:w=iw*{right-x}:h=ih*{bottom-y}:color=lime:t=2",
                    '-frames:v', '1', '-update', '1', str(destination)], check=True, capture_output=True)
                payload = destination.read_bytes()
                files.append(ArtifactFile(path=relative, size_bytes=len(payload)))
                overlays.append(OverlayRecord(path=relative, track_id=track.track_id, frame_index=observation.frame_index))
            artifact = CvEvidenceArtifact(
                schema_version="cv_evidence_v1",
                status=EvidenceStatus.AVAILABLE,
                provider=request.provider,
                model_identity=request.model_identity,
                video_identity=request.video_identity,
                checkpoint_identity=request.checkpoint_identity,
                processed_timeline=FrameTimeline(frames=processed_frames),
                entities=request.entities,
                tracks=tuple(tracks),
                files=tuple(files),
                overlay_records=tuple(overlays),
            )
            self._metrics = {
                "processed_frames": len(processed_frames),
                "execution_chunk_frames": self.execution_chunk_frames,
                "entity_prompts": len(request.entities),
                "track_count": len(tracks),
                "peak_allocated_bytes": 0,
            }
            return artifact
        except CvProviderError:
            raise
        except Exception:
            raise CvProviderError("fake CV evidence inference failed") from None

    def request_metrics(self) -> dict[str, int]:
        """Return bounded deterministic metrics for the most recent request."""
        return dict(self._metrics)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value

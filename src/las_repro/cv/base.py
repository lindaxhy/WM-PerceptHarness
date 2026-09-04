"""Provider-independent computer-vision evidence execution boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

from .contracts import (
    ArtifactFile,
    CvEvidenceArtifact,
    CvEvidenceRequest,
    CvTrack,
    EvidenceStatus,
    TrackObservation,
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
            tracks: list[CvTrack] = []
            files: list[ArtifactFile] = []
            for entity_index, entity in enumerate(request.entities):
                observations: list[TrackObservation] = []
                for observation_index, frame in enumerate(request.timeline.frames):
                    visible = (
                        len(request.timeline.frames) == 1
                        or observation_index < len(request.timeline.frames) - 1
                    )
                    mask_ref: str | None = None
                    if visible:
                        mask_ref = (
                            f"masks/{entity.entity_id}-{frame.frame_index:08d}.mask"
                        )
                        payload = (
                            f"fake-cv-mask-v1\n{entity.entity_id}\n"
                            f"{frame.frame_index}\n{frame.timestamp_seconds:.9f}\n"
                        ).encode("ascii")
                        (staging / mask_ref).parent.mkdir(
                            mode=0o700, exist_ok=True
                        )
                        (staging / mask_ref).write_bytes(payload)
                        files.append(
                            ArtifactFile(
                                path=mask_ref,
                                sha256=hashlib.sha256(payload).hexdigest(),
                                size_bytes=len(payload),
                            )
                        )
                    left = 0.1 + 0.1 * (entity_index % 5)
                    top = 0.1 + 0.05 * (entity_index % 5)
                    observations.append(
                        TrackObservation(
                            frame_index=frame.frame_index,
                            timestamp_seconds=frame.timestamp_seconds,
                            bbox_xyxy=(left, top, left + 0.08, top + 0.08),
                            mask_ref=mask_ref,
                            visible=visible,
                            confidence=max(request.thresholds.min_confidence, 0.9),
                            area_fraction=max(
                                request.thresholds.min_area_fraction, 0.01
                            ),
                            center_xy=(left + 0.04, top + 0.04),
                        )
                    )
                tracks.append(
                    CvTrack(
                        track_id=f"{entity.entity_id}_1",
                        entity_id=entity.entity_id,
                        observations=tuple(observations),
                    )
                )
            artifact = CvEvidenceArtifact(
                schema_version="cv_evidence_v1",
                status=EvidenceStatus.AVAILABLE,
                provider=request.provider,
                model_identity=request.model_identity,
                video_sha256=request.video_sha256,
                checkpoint_sha256=request.checkpoint_sha256,
                entities=request.entities,
                tracks=tuple(tracks),
                files=tuple(files),
            )
            self._metrics = {
                "processed_frames": len(request.timeline.frames),
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

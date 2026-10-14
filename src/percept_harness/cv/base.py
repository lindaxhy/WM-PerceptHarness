"""Provider-independent computer-vision evidence execution boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .contracts import CvEvidenceArtifact, CvEvidenceRequest


class CvProviderError(RuntimeError):
    """A sanitized CV provider boundary failure."""


class CvOutOfMemoryError(CvProviderError):
    """A provider-reported allocation failure eligible for one retry."""


class CvEvidenceProvider(Protocol):
    execution_chunk_frames: int

    def analyze(
        self, request: CvEvidenceRequest, staging_dir: Path
    ) -> CvEvidenceArtifact:
        """Write provider-owned files and return their strict manifest."""

    def request_metrics(self) -> dict[str, int]:
        """Return execution metrics for the most recent completed request."""

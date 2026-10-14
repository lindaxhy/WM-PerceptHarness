"""Provider-independent computer-vision evidence execution boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .contracts import CvEvidenceArtifact, CvEvidenceRequest


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

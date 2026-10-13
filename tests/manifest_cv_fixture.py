"""Test-only CV injection while retaining real spawned worker communication."""
from contextlib import contextmanager
from unittest.mock import patch

from percept_harness.batch import GpuSlot


@contextmanager
def synthetic_executor(provider_name, arguments, settings):
    from percept_harness.cv.artifacts import CvArtifactStore
    from percept_harness.cv.base import FakeCvEvidenceProvider
    from percept_harness.cv.executor import SyncCvExecutor

    assert provider_name == "sam31"
    with CvArtifactStore(settings.cv_cache_root) as store:
        yield SyncCvExecutor(FakeCvEvidenceProvider(), store)


def synthetic_cv_worker(connection, cv_settings, gpu):
    from percept_harness.batch import _cv_worker

    with patch("percept_harness.cli._cv_executor", synthetic_executor):
        _cv_worker(connection, cv_settings, gpu)


class SyntheticGpuSlot(GpuSlot):
    def __init__(self, context, settings, gpu):
        with patch("percept_harness.batch._cv_worker", synthetic_cv_worker):
            super().__init__(context, settings, gpu)

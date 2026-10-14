"""Lazy annotation CV keeps clean VLM runs free of SAM assets and imports."""
import builtins
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from percept_harness.cli import _cv_executor
from percept_harness.config import Settings
from percept_harness.cv.artifacts import CvArtifactStore, CvArtifactError
from percept_harness.cv.base import FakeCvEvidenceProvider, CvOutOfMemoryError


def payload(root):
    return dict(schema_version='cv_request_v1', provider='sam31', model_identity='sam3.1',
        video_path=str(root / 'source.mp4'), video_identity='source-video',
        checkpoint_identity='fixture-checkpoint', duration_seconds=1.0, frame_count=1,
        timeline={'frames': [{'frame_index': 0, 'timestamp_seconds': 0.0}]}, entities=[{'entity_id': 'target', 'canonical_label': 'target'}],
        sampling={'max_fps': 4.0},
        spatial_prompts=[{'entity_id': 'target', 'frame_index': 0, 'bbox_xyxy': [0.1, 0.1, 0.8, 0.8]}],
        thresholds={'min_confidence': 0.1})


class CountingProvider(FakeCvEvidenceProvider):
    def __init__(self, *, oom=False):
        super().__init__()
        self.calls = 0
        self.closed = False
        self.oom = oom

    def analyze(self, request, staging):
        self.calls += 1
        if self.oom and self.calls == 1:
            raise CvOutOfMemoryError('fixture allocation failure')
        from percept_harness.cv.contracts import CvEvidenceArtifact
        self._metrics = {'processed_frames': 1, 'execution_chunk_frames': self.execution_chunk_frames,
                         'peak_allocated_bytes': 0}
        return CvEvidenceArtifact(schema_version='cv_evidence_v1', status='available',
            provider=request.provider, model_identity=request.model_identity,
            video_identity=request.video_identity, checkpoint_identity=request.checkpoint_identity,
            processed_timeline=request.timeline, entities=request.entities,
            tracks=(), files=(), overlay_records=())

    def close(self):
        self.closed = True


class LazyCvTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.args = Namespace(backend='openai', cv_device=None)
        self.settings = Settings(cv_cache_root=self.root / 'cache',
            cv_repository_path=self.root / 'missing-sam',
            cv_checkpoint_path=self.root / 'missing-checkpoint',
            cv_bpe_path=self.root / 'missing-bpe', cv_execution_chunk_frames=8)

    def assets(self):
        self.settings.cv_repository_path.mkdir()
        self.settings.cv_checkpoint_path.write_bytes(b'fixture')
        self.settings.cv_bpe_path.write_bytes(b'fixture')

    def test_unused_executor_needs_no_assets_import_or_cache(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.endswith('sam31') or name in {'torch', 'sam3'}:
                self.fail('unused annotation imported a SAM dependency')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded_import):
            with _cv_executor('sam31', self.args, self.settings) as execute:
                self.assertTrue(callable(execute))
                self.assertFalse(self.settings.cv_cache_root.exists())
        self.assertFalse(self.settings.cv_cache_root.exists())

    def test_missing_assets_fail_only_when_evidence_is_requested(self):
        with _cv_executor('sam31', self.args, self.settings) as execute:
            with self.assertRaisesRegex(ValueError, 'existing local directory'):
                execute(payload(self.root))
        self.assertFalse(self.settings.cv_cache_root.exists())

    def test_first_job_loads_once_and_preserves_oom_retry_cache_and_cleanup(self):
        self.assets()
        provider = CountingProvider(oom=True)
        stores = []
        class RecordingStore(CvArtifactStore):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                stores.append(self)
        with patch('percept_harness.cv.sam31.Sam31EvidenceProvider.load', return_value=provider) as load, \
             patch('percept_harness.cv.artifacts.CvArtifactStore', RecordingStore):
            with _cv_executor('sam31', self.args, self.settings) as execute:
                self.assertEqual(load.call_count, 0)
                self.assertFalse(stores)
                first, metrics = execute(payload(self.root))
                self.assertEqual(load.call_count, 1)
                self.assertEqual(metrics['processed_frames'], 1)
                self.assertEqual(metrics['execution_chunk_frames'], 4)
                self.assertTrue(metrics['oom_retry'])
                self.assertFalse(first['cache_hit'])
                second, cached_metrics = execute(payload(self.root))
                self.assertEqual(second['artifact_key'], first['artifact_key'])
                self.assertTrue(second['cache_hit'])
                self.assertEqual(cached_metrics['processed_frames'], 0)
                self.assertEqual(provider.calls, 2)
                self.assertEqual(load.call_count, 1)
                self.assertFalse(provider.closed)
            self.assertTrue(provider.closed)
            with self.assertRaises(CvArtifactError):
                stores[0].lookup(first['artifact_key'])

    def test_provider_and_store_close_when_annotation_body_fails(self):
        self.assets()
        provider = CountingProvider()
        with patch('percept_harness.cv.sam31.Sam31EvidenceProvider.load', return_value=provider):
            with self.assertRaisesRegex(RuntimeError, 'annotation failed'):
                with _cv_executor('sam31', self.args, self.settings) as execute:
                    execute(payload(self.root))
                    raise RuntimeError('annotation failed')
        self.assertTrue(provider.closed)

    def test_failed_load_closes_allocated_store(self):
        self.assets()
        store = CvArtifactStore(self.settings.cv_cache_root)
        with patch('percept_harness.cv.artifacts.CvArtifactStore', return_value=store), \
             patch('percept_harness.cv.sam31.Sam31EvidenceProvider.load', side_effect=RuntimeError('load failed')):
            with _cv_executor('sam31', self.args, self.settings) as execute:
                with self.assertRaisesRegex(RuntimeError, 'load failed'):
                    execute(payload(self.root))
                with self.assertRaises(CvArtifactError):
                    store.lookup('source-video')

    def test_runner_rejects_production_fake_cv_and_disabled_settings(self):
        from percept_harness.runner import SyncRunner
        class ProductionModel:
            def generate(self, request):
                raise AssertionError('invalid CV policy must reject before VLM use')
        for provider, message in (('fake', 'Fake CV'), ('disabled', 'disabled CV')):
            with self.subTest(provider=provider):
                settings = self.settings.model_copy(update={'cv_provider': provider})
                result = SyncRunner(ProductionModel(), settings, model_alias='production').evaluate(
                    self.root / 'source.mp4')
                self.assertEqual(result.status, 'failed')
                self.assertIn(message, result.error)

    def test_provider_policy_is_enforced_even_when_unused(self):
        for provider in ('disabled', 'fake', 'unknown'):
            with self.subTest(provider=provider), self.assertRaises(ValueError):
                with _cv_executor(provider, self.args, self.settings):
                    pass


if __name__ == '__main__':
    unittest.main()

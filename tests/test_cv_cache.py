"""Behavioral checks for atomic CV cache publication and bounded file access."""
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pytest

from percept_harness.cv.artifacts import CvArtifactError, CvArtifactStore, cv_cache_key
from percept_harness.cv.contracts import CvEvidenceArtifact, CvEvidenceRequest


def cache_request():
    return CvEvidenceRequest.model_validate(dict(
        schema_version='cv_request_v1', provider='sam31', model_identity='fixture',
        video_path=Path('/tmp/video.mp4'), video_identity='video', checkpoint_identity='weights',
        duration_seconds=1.0, frame_count=1,
        timeline={'frames': [{'frame_index': 0, 'timestamp_seconds': 0.0}]},
        entities=[{'entity_id': 'target', 'canonical_label': 'target'}], sampling={'max_fps': 4.0},
        spatial_prompts=[{'entity_id': 'target', 'frame_index': 0, 'bbox_xyxy': [0.1, 0.1, 0.8, 0.8]}],
        thresholds={'min_confidence': 0.1}))


def cache_artifact(request, stage, content=b'mask'):
    (stage / 'masks').mkdir()
    (stage / 'masks' / 'fixture.mask').write_bytes(content)
    return CvEvidenceArtifact.model_validate(dict(
        schema_version='cv_evidence_v1', status='available', provider=request.provider,
        model_identity=request.model_identity, video_identity=request.video_identity,
        checkpoint_identity=request.checkpoint_identity,
        processed_timeline=request.timeline, entities=request.entities, tracks=[], overlay_records=[],
        files=[{'path': 'masks/fixture.mask', 'size_bytes': len(content)}]))


def publish_worker(root, start, ready):
    request = cache_request()
    with CvArtifactStore(Path(root)) as store:
        with store.staging(cv_cache_key(request)) as stage:
            artifact = cache_artifact(request, stage)
            ready.put(True)
            if not start.wait(15):
                raise RuntimeError('publisher start timed out')
            handle = store.publish(request, stage, artifact)
        assert store.load(handle) == artifact


def test_concurrent_processes_publish_one_complete_entry(tmp_path):
    context = multiprocessing.get_context('spawn')
    start, ready = context.Event(), context.Queue()
    root = tmp_path / 'cache'
    processes = [context.Process(target=publish_worker, args=(str(root), start, ready)) for _ in range(3)]
    try:
        for process in processes:
            process.start()
        for _ in processes:
            assert ready.get(timeout=20)
        # Prepared work is not visible as a completed cache result.
        with CvArtifactStore(root) as store:
            assert store.lookup(cv_cache_key(cache_request())) is None
        start.set()
        for process in processes:
            process.join(20)
            assert process.exitcode == 0
        with CvArtifactStore(root) as store:
            assert store.lookup(cv_cache_key(cache_request())) is not None
        assert not list(root.glob('*/.staging-*'))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        ready.close()


def test_failed_publication_leaves_no_result_or_staging(tmp_path):
    request = cache_request()
    with CvArtifactStore(tmp_path / 'cache') as store:
        with pytest.raises(CvArtifactError), store.staging(cv_cache_key(request)) as stage:
            artifact = cache_artifact(request, stage)
            with patch('percept_harness.cv.artifacts.os.replace', side_effect=OSError('fixture interruption')):
                store.publish(request, stage, artifact)
        assert not stage.exists()
        assert store.lookup(cv_cache_key(request)) is None


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'extra', 'oversize', 'fifo'])
def test_invalid_artifact_files_cannot_be_published(tmp_path, kind):
    request = cache_request()
    with CvArtifactStore(tmp_path / 'cache', max_bytes=4) as store:
        with store.staging(cv_cache_key(request)) as stage:
            artifact = cache_artifact(request, stage)
            mask = stage / 'masks/fixture.mask'
            external = tmp_path / 'outside'
            external.write_bytes(b'mask')
            if kind in {'symlink', 'hardlink', 'fifo'}:
                mask.unlink()
                if kind == 'symlink':
                    mask.symlink_to(external)
                elif kind == 'hardlink':
                    mask.hardlink_to(external)
                else:
                    import os
                    os.mkfifo(mask)
            elif kind == 'extra':
                (stage / 'unexpected').write_text('extra')
            else:
                mask.write_bytes(b'longer-than-declared')
            with pytest.raises(CvArtifactError):
                store.publish(request, stage, artifact)
        assert store.lookup(cv_cache_key(request)) is None
        assert external.read_bytes() == b'mask'


def test_corrupt_entry_is_recomputed_without_reusing_it(tmp_path):
    request = cache_request()
    key = cv_cache_key(request)
    with CvArtifactStore(tmp_path / 'cache') as store:
        with store.staging(key) as stage:
            artifact = cache_artifact(request, stage)
            handle = store.publish(request, stage, artifact)
        (store._entry_path(key) / 'manifest.json').write_text('{broken')
        assert store.lookup(key) is None
        with store.staging(key) as stage:
            artifact = cache_artifact(request, stage)
            assert store.publish(request, stage, artifact) == handle
        assert store.load(handle) == artifact


def test_same_key_cache_is_readable_across_store_instances(tmp_path):
    request = cache_request()
    with CvArtifactStore(tmp_path / 'cache') as store:
        with store.staging(cv_cache_key(request)) as stage:
            artifact = cache_artifact(request, stage)
            handle = store.publish(request, stage, artifact)
    with CvArtifactStore(tmp_path / 'cache') as reader:
        assert reader.load(handle) == artifact
        with pytest.raises(CvArtifactError):
            reader.read_overlays(handle, ('masks/fixture.mask',))

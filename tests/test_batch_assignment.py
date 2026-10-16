"""CPU-only regression tests for the real manifest queue routing."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from percept_harness import batch
from percept_harness.config import Settings


class FakeSlot:
    def __init__(self, context, settings, gpu):
        self.gpu = gpu
        self.ready = False
        self.process = SimpleNamespace(pid=0, is_alive=lambda: True)

    def wait_ready(self):
        self.ready = True

    def close(self):
        pass


class AssignmentTests(unittest.TestCase):
    def test_host_shards_remain_backward_compatible(self):
        for count in (1, 2, 3, 8):
            for jid in ('sample', 'model/视频', *map(str, range(100))):
                old = int.from_bytes(hashlib.sha256(jid.encode()).digest()[:8], 'big') % count
                self.assertEqual(batch.assignment(jid, count), old)

    def test_manifest_routes_every_job_once_to_all_listed_gpus(self):
        # Exercise run_manifest itself: changing only the helper while leaving
        # the old GPU call site would still fail this regression.
        for devices_by_host in ((range(8), range(8)), (range(8), range(7)),
                                ((1, 3, 6), (0, 2, 4, 5)), (range(8),)):
            with self.subTest(devices=devices_by_host), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows = [{'job_id': f'model/{i}', 'video_path': '/unused.mp4'} for i in range(4096)]
                all_jobs = []
                for shard, devices in enumerate(devices_by_host):
                    captured = {}

                    class CaptureExecutor:
                        def __init__(self, **kwargs):
                            pass

                        def __enter__(self):
                            return self

                        def __exit__(self, *args):
                            pass

                        def submit(self, worker, gpu, slot, tasks):
                            captured[gpu] = [row['job_id'] for row in list(tasks.queue)]
                            return SimpleNamespace(result=lambda: None)

                    args = SimpleNamespace(workers_per_gpu=1, request_concurrency=2,
                        shard_index=shard, shard_count=len(devices_by_host), cv='sam31',
                        cv_devices=','.join(map(str, devices)), event_workflow=None,
                        manifest=root / 'manifest.jsonl', output=root / str(shard))
                    with patch.object(batch, 'Settings', return_value=Settings(_env_file=None)), \
                         patch.object(batch, 'read_manifest', return_value=rows), \
                         patch.object(batch, 'GpuSlot', FakeSlot), \
                         patch.object(batch, 'ThreadPoolExecutor', CaptureExecutor):
                        batch.run_manifest(args)
                    self.assertEqual(set(captured), set(devices))
                    self.assertTrue(all(captured.values()), captured)
                    for jobs in captured.values():
                        self.assertTrue(all(batch.assignment(jid, len(devices_by_host)) == shard for jid in jobs))
                        all_jobs.extend(jobs)
                self.assertCountEqual(all_jobs, [row['job_id'] for row in rows])


if __name__ == '__main__':
    unittest.main()

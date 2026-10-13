import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from percept_harness import result_store
from percept_harness.runner import EvalOutcome, _write_result, SyncRunner
from percept_harness.batch import ReviewRequiredRunner
from percept_harness.evaluation.events import evaluate, load_group


class AnnotationPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.path = self.root / 'clip.json'
        self.video = self.root / 'clip.mp4'
        self.identity = {'input': {'path': str(self.video)}, 'run': {
            'reusable': True, 'template': 'embodied_action_captioning'}}
        self.data = {'duration': 2, 'events': []}

    def outcome(self, **changes):
        values = dict(video_path=self.video, template='embodied_action_captioning',
                      status='completed', data=self.data, provenance=self.identity,
                      metadata={'phases': ['grounded_draft', 'evidence_review', 'grounded_revision'], 'evidence': []})
        values.update(changes)
        return EvalOutcome(**values)

    def test_public_is_minimal_audit_resumes_and_digest_tampering_does_not(self):
        _write_result(self.path, self.outcome())
        self.assertEqual(json.loads(self.path.read_text()), self.data)
        audit = json.loads((self.root / 'clip.audit.json').read_text())
        self.assertNotIn('data', audit)
        self.assertEqual(audit['data_sha256'], result_store.json_digest(self.data))
        self.assertEqual(result_store.completed_result(self.path, self.identity)['data'], self.data)
        self.assertEqual(result_store.result_path(self.root, self.video), self.path)
        self.path.write_text(json.dumps({'duration': 3, 'events': []}))
        self.assertIsNone(result_store.completed_result(self.path, self.identity))

    def test_failure_removes_stale_public_and_archives_both_records(self):
        _write_result(self.path, self.outcome())
        result_store.archive_result(self.path, self.root)
        _write_result(self.path, self.outcome(status='failed', data=None, error='review failed'))
        self.assertFalse(self.path.exists())
        self.assertEqual(json.loads((self.root / 'clip.audit.json').read_text())['status'], 'failed')
        self.assertEqual(len(list((self.root / '.percept/history').glob('*.json'))), 2)
        self.assertEqual(result_store.result_path(self.root, self.video), self.path)
        self.assertIsNone(result_store.completed_result(self.path, self.identity))

    def test_plain_ground_truth_and_audited_prediction_score_empty_as_valid(self):
        ref = self.root / 'ref'; pred = self.root / 'pred'
        ref.mkdir(); pred.mkdir()
        (ref / 'clip.json').write_text(json.dumps(self.data))
        _write_result(pred / 'clip.json', self.outcome())
        system = evaluate(ref, {'sys': pred}, lambda a,b: None)['systems']['sys']
        self.assertEqual((system['valid_count'], system['invalid_count']), (1, 0))
        self.assertEqual(set(load_group(pred)), {'clip'})
        (pred / 'clip.json').write_text(json.dumps({'duration': 3, 'events': []}))
        self.assertEqual(evaluate(ref, {'sys': pred}, lambda a,b: None)['systems']['sys']['invalid_count'], 1)

    def test_orphan_failed_audit_is_error_sample_not_empty_dataset(self):
        ref = self.root / 'ref'; pred = self.root / 'pred'
        ref.mkdir(); pred.mkdir()
        (ref / 'clip.json').write_text(json.dumps(self.data))
        _write_result(pred / 'clip.json', self.outcome(status='failed', data=None, error='review failed'))
        system = evaluate(ref, {'sys': pred}, lambda a,b: None)['systems']['sys']
        self.assertEqual(system['invalid_count'], 1)
        self.assertIn('review failed', system['samples']['clip']['error'])
        self.assertIsNone(system['aggregate'])

    def test_completed_orphan_and_malformed_audit_are_not_ignored(self):
        _write_result(self.path, self.outcome())
        self.path.unlink()
        self.assertIn('clip', load_group(self.root))
        (self.root / 'clip.audit.json').write_text('{')
        with self.assertRaises(ValueError):
            load_group(self.root)

    def test_required_review_accepts_zero_sam_but_rejects_missing_phase(self):
        runner = object.__new__(ReviewRequiredRunner)
        with patch.object(SyncRunner, 'evaluate', return_value=self.outcome()):
            self.assertEqual(runner.evaluate().status, 'completed')
        with patch.object(SyncRunner, 'evaluate', return_value=self.outcome(metadata={'phases': ['grounded_draft']})):
            self.assertEqual(runner.evaluate().status, 'failed')

    def test_interrupted_public_write_leaves_non_resumable_error_sample(self):
        _write_result(self.path, self.outcome())
        original = result_store.write_json
        def interrupted(path, value):
            if path == self.path:
                raise OSError('disk full')
            original(path, value)
        with patch.object(result_store, 'write_json', side_effect=interrupted):
            with self.assertRaises(OSError):
                _write_result(self.path, self.outcome())
        self.assertIsNone(result_store.completed_result(self.path, self.identity))
        self.assertEqual(load_group(self.root)['clip']['record']['status'], 'writing')

    def test_legacy_envelope_still_resumes_and_scores(self):
        legacy = {'video_path': str(self.video), 'template': 'embodied_action_captioning',
                  'status': 'completed', 'data': self.data, 'provenance': self.identity,
                  'data_sha256': result_store.json_digest(self.data)}
        self.path.write_text(json.dumps(legacy))
        self.assertEqual(result_store.completed_result(self.path, self.identity)['data'], self.data)
        report = evaluate(self.root, {'sys': self.root}, lambda a,b: None)
        self.assertEqual(report['systems']['sys']['valid_count'], 1)

    def test_audit_write_failure_cannot_leave_old_successful_public(self):
        _write_result(self.path, self.outcome())
        with patch.object(result_store, 'write_json', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                _write_result(self.path, self.outcome(status='failed', data=None))
        self.assertFalse(self.path.exists())
        self.assertIsNone(result_store.completed_result(self.path, self.identity))

    def test_idle_gpu_worker_does_not_initialize_sam(self):
        from percept_harness.batch import _cv_worker
        class Connection:
            def __init__(self):
                self.messages = []
                self.closed = False
            def send(self, message):
                self.messages.append(message)
            def recv(self):
                return None
            def close(self):
                self.closed = True
        connection = Connection()
        with patch.dict(os.environ), patch('percept_harness.cli._cv_executor', side_effect=AssertionError('SAM initialized without a target')):
            _cv_worker(connection, {'cv_provider': 'fake'}, 0)
        self.assertEqual(connection.messages[0][0], 'ready')
        self.assertTrue(connection.closed)

    def test_reserved_audit_video_name_roundtrips_beside_normal_video(self):
        paths = []
        for name in ('foo.audit', 'foo'):
            video = self.root / (name + '.mp4')
            path = result_store.result_path(self.root, video)
            self.assertFalse(path.name.endswith('.audit.json'))
            _write_result(path, self.outcome(video_path=video))
            self.assertEqual(result_store.result_path(self.root, video), path)
            paths.append(path)
        self.assertNotEqual(paths[0], paths[1])
        self.assertEqual(set(load_group(self.root)), {'foo', 'foo.audit'})
        system = evaluate(self.root, {'sys': self.root}, lambda a,b: None)['systems']['sys']
        self.assertEqual((system['valid_count'], system['invalid_count']), (2, 0))

    def test_literal_audit_named_plain_and_legacy_annotations_are_samples(self):
        for name, record in [('foo', self.data), ('foo.audit', self.data),
                             ('bar.audit', {'status': 'completed', 'data': self.data})]:
            (self.root / (name + '.json')).write_text(json.dumps(record))
        self.assertEqual(set(load_group(self.root)), {'foo', 'foo.audit', 'bar.audit'})
        system = evaluate(self.root, {'sys': self.root}, lambda a,b: None)['systems']['sys']
        self.assertEqual((system['valid_count'], system['invalid_count']), (3, 0))

    def test_old_literal_audit_named_public_with_nested_sidecar_scores(self):
        path = self.root / 'foo.audit.json'
        _write_result(path, self.outcome(video_path=self.root / 'foo.audit.mp4'))
        self.assertEqual(set(load_group(self.root)), {'foo.audit'})
        system = evaluate(self.root, {'sys': self.root}, lambda a,b: None)['systems']['sys']
        self.assertEqual(system['valid_count'], 1)

    def test_duplicate_status_cannot_turn_failed_legacy_record_into_success(self):
        self.path.write_text('{"status":"failed","status":"completed","data":{"duration":2,"events":[]}}')
        with self.assertRaises(ValueError):
            load_group(self.root)

    def test_unrelated_annotation_in_audit_slot_is_never_overwritten(self):
        other = self.root / 'clip.audit.json'
        other.write_text(json.dumps(self.data))
        payload = other.read_bytes()
        self.assertNotEqual(result_store.result_path(self.root, self.video), self.path)
        with self.assertRaises(ValueError):
            _write_result(self.path, self.outcome())
        self.assertEqual(other.read_bytes(), payload)

    def test_nonfinite_metadata_and_duplicate_audit_status_are_rejected(self):
        for text in ('{"status":"failed","status":"completed"}',
                     '{"status":"completed","elapsed_seconds":NaN}',
                     '{"status":"completed","elapsed_seconds":1e400}'):
            with self.subTest(text=text):
                (self.root / 'clip.audit.json').write_text(text)
                with self.assertRaises(ValueError):
                    load_group(self.root)

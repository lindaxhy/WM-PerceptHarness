import unittest

from percept_harness.evaluation import events
from percept_harness.pipelines.event_schemas import GroundedEvents


def annotation(*segments):
    return GroundedEvents.model_validate({"duration": 20, "events": [
        {"start": start, "end": end, "description": description}
        for start, end, description in segments
    ]})


class EventScoreTests(unittest.TestCase):
    def test_joint_matching_is_independent_of_temporal_assignment(self):
        reference = annotation((0, 10, "A"), (1, 11, "B"))
        prediction = annotation((0, 10, "B"), (1, 11, "A"))
        score = events.score_pair(reference, prediction, lambda a, b: {"match": a == b, "reason": "action"})
        self.assertEqual([(x["reference_index"], x["prediction_index"]) for x in score["temporal_matches"]], [(0, 0), (1, 1)])
        self.assertEqual([(x["reference_index"], x["prediction_index"]) for x in score["joint_matches"]], [(0, 1), (1, 0)])
        self.assertEqual((score["segment_f1"], score["semantic_match_rate"], score["joint_f1"]), (1, 0, 1))
        self.assertEqual(len(score["candidate_judgments"]), 4)
        self.assertEqual(score["temporal_match_count"], 2)
        self.assertEqual(score["joint_match_count"], 2)
        self.assertEqual(score["unmatched_temporal_reference_indices"], [])
        self.assertEqual(score["unmatched_joint_prediction_indices"], [])

    def test_boundary_and_empty_denominators(self):
        a, b, empty = annotation((0, 10, "walk")), annotation((2, 10, "walk")), annotation()
        judge = lambda x, y: {"match": True, "reason": "same"}
        self.assertEqual(events.score_pair(a, b, judge, .8)["segment_f1"], 1)
        self.assertEqual(events.score_pair(a, b, judge, .81)["segment_f1"], 0)
        self.assertIsNone(events.score_pair(empty, empty, judge)["segment_f1"])
        self.assertIsNone(events.score_pair(empty, empty, judge)["semantic_match_rate"])
        self.assertEqual(events.score_pair(a, empty, judge)["joint_f1"], 0)

    def test_invalid_threshold_and_judgment_fail(self):
        with self.assertRaises(ValueError):
            events.score_pair(annotation(), annotation(), lambda a, b: None, float("nan"))
        with self.assertRaises(events.JudgeError):
            events.score_pair(annotation((0, 1, "A")), annotation((0, 1, "A")), lambda a, b: {"match": "yes", "reason": "x"})


class DatasetAndJudgeTests(unittest.TestCase):
    def test_http_judge_rejects_bad_schema_and_status(self):
        import httpx
        ok = events.OpenAIJudge("http://local/v1", "model", "secret", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": '{"match":true,"reason":"same action"}'}}]})))
        self.assertEqual(ok("A", "A"), {"match": True, "reason": "same action"})
        malformed = events.OpenAIJudge("http://local/v1", "model", "secret", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": '{"match":"yes","reason":"x"}'}}]})))
        with self.assertRaises(events.JudgeError):
            malformed("A", "B")
        failed = events.OpenAIJudge("http://local/v1", "model", "secret", transport=httpx.MockTransport(
            lambda request: httpx.Response(503)))
        with self.assertRaises(events.JudgeError):
            failed("A", "B")

    def test_missing_sample_invalidates_aggregate(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            ref, pred = Path(temporary, "ref"), Path(temporary, "pred")
            ref.mkdir(); pred.mkdir()
            def write(where, name, video):
                Path(where, name).write_text(json.dumps({"status": "completed", "video_path": video,
                    "data": {"duration": 20, "events": []}}))
            write(ref, "a.json", "/ref/a.mp4")
            write(ref, "b.json", "/ref/b.mp4")
            write(pred, "a.json", "/pred/a.mp4")
            write(pred, "c.json", "/pred/c.mp4")
            report = events.evaluate(ref, {"sys": pred}, lambda a, b: None)
            system = report["systems"]["sys"]
            self.assertEqual((system["valid_count"], system["invalid_count"]), (1, 2))
            self.assertIsNone(system["aggregate"])
            self.assertEqual(set(system["samples"]), {"a", "b", "c"})

    def test_nested_identity_duplicate_and_malformed_json(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Path(root, "job-7").mkdir()
            Path(root, "job-7", "sample.json").write_text(json.dumps({"status": "completed", "data": {"duration": 20, "events": []}}))
            self.assertIn("job-7", events.load_group(root))
            Path(root, "a.json").write_text(json.dumps({"video_path": "/one/clip.mp4"}))
            Path(root, "b.json").write_text(json.dumps({"video_path": "/two/clip.mp4"}))
            with self.assertRaises(ValueError):
                events.load_group(root)
            Path(root, "b.json").unlink()
            Path(root, "broken.json").write_text("{")
            with self.assertRaises(ValueError):
                events.load_group(root)

    def test_macro_counts_defined_samples_only(self):
        scores = [events.score_pair(annotation(), annotation(), lambda a, b: None),
                  events.score_pair(annotation((0, 1, "A")), annotation(), lambda a, b: None)]
        result = events.aggregate(scores)
        self.assertEqual(result["segment"]["micro"]["f1"], 0)
        self.assertEqual(result["segment"]["macro"]["f1"], {"value": 0, "contributing_count": 1})
        self.assertIsNone(result["semantic_match_rate"]["micro"])

class CliTests(unittest.TestCase):
    def test_events_cli_writes_report_and_returns_nonzero_on_missing_sample(self):
        import json
        import tempfile
        from pathlib import Path
        from percept_harness import cli
        from percept_harness.evaluation import events_cli
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ref = root / "ref"; pred = root / "pred"; out = root / "out"
            ref.mkdir(); pred.mkdir()
            (ref / "a.json").write_text(json.dumps({"status": "completed", "video_path": "/r/a.mp4", "data": {"duration": 1, "events": []}}))
            (pred / "a.json").write_text(json.dumps({"status": "completed", "video_path": "/p/a.mp4", "data": {"duration": 1, "events": []}}))
            self.assertEqual(events_cli.main(["--reference", str(ref), "--system", f"sys={pred}", "--out", str(out)], judge=lambda a, b: None), 0)
            self.assertIsNone(json.loads((out / "events.json").read_text())["systems"]["sys"]["aggregate"]["segment"]["micro"]["f1"])
            self.assertIn("events", cli._SCORE_MODULES)
            (ref / "b.json").write_text(json.dumps({"status": "completed", "video_path": "/r/b.mp4", "data": {"duration": 1, "events": []}}))
            self.assertEqual(events_cli.main(["--reference", str(ref), "--system", f"sys={pred}", "--out", str(out)], judge=lambda a, b: None), 1)
            self.assertIsNone(json.loads((out / "events.json").read_text())["systems"]["sys"]["aggregate"])

class FailureAndTimeTests(unittest.TestCase):
    def test_judge_failure_invalidates_sample_without_partial_aggregate(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            ref, pred = Path(temporary, "ref"), Path(temporary, "pred")
            ref.mkdir(); pred.mkdir()
            data = {"duration": 20, "events": [{"start": 0, "end": 1, "description": "open door"}]}
            for directory in (ref, pred):
                (directory / "clip.json").write_text(json.dumps({"status": "completed", "video_path": f"/{directory.name}/clip.mp4", "data": data}))
            def broken(a, b):
                raise events.JudgeError("HTTP 503")
            row = events.evaluate(ref, {"sys": pred}, broken)["systems"]["sys"]
            self.assertEqual((row["valid_count"], row["invalid_count"]), (0, 1))
            self.assertIsNone(row["aggregate"])
            self.assertIn("HTTP 503", row["samples"]["clip"]["error"])

    def test_unequal_durations_use_original_seconds(self):
        ref = annotation((0, 10, "walk"))
        pred = GroundedEvents.model_validate({"duration": 40, "events": [{"start": 20, "end": 30, "description": "walk"}]})
        self.assertEqual(events.score_pair(ref, pred, lambda a, b: {"match": True, "reason": "same"})["segment_f1"], 0)

class ProvenanceTests(unittest.TestCase):
    def test_report_never_contains_judge_key_or_url_credentials(self):
        import json
        import tempfile
        from pathlib import Path
        from percept_harness.evaluation import events_cli
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ref = root / "ref"; pred = root / "pred"; out = root / "out"
            ref.mkdir(); pred.mkdir()
            payload = {"status": "completed", "video_path": "/a/clip.mp4", "data": {"duration": 1, "events": []}}
            (ref / "clip.json").write_text(json.dumps(payload))
            (pred / "clip.json").write_text(json.dumps(payload))
            events_cli.main(["--reference", str(ref), "--system", f"sys={pred}", "--out", str(out),
                "--judge-base-url", "http://user:password@local/token-in-path/v1?token=hidden", "--judge-model", "model", "--judge-api-key", "secret"])
            report = (out / "events.json").read_text()
            for secret in ("password", "hidden", "secret", "token-in-path"):
                self.assertNotIn(secret, report)

class StrictBoundaryTests(unittest.TestCase):
    def test_duplicate_judge_key_is_rejected(self):
        import httpx
        bad = events.OpenAIJudge("http://local/v1", "model", "secret", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": '{"match":true,"match":false,"reason":"x"}'}}]})))
        with self.assertRaises(events.JudgeError):
            bad("A", "B")

    def test_blank_explicit_identity_is_error(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "clip.json").write_text(json.dumps({"sample_id": "", "video_path": "/a/clip.mp4"}))
            with self.assertRaises(ValueError):
                events.load_group(root)

class BatchOutputTests(unittest.TestCase):
    def test_manifest_shard_status_is_not_annotation(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "job-1").mkdir()
            (root / "job-1" / "job-1.json").write_text(json.dumps({"status": "completed", "data": {"duration": 1, "events": []}}))
            (root / "shard-00.status.json").write_text(json.dumps({"phase": "done", "pid": 123, "shard_index": 0,
                "shard_count": 1, "started_unix": 0, "updated_unix": 1, "total": 1,
                "counts": {"completed": 1}, "jobs": {"job-1": {"status": "completed"}}}))
            self.assertEqual(set(events.load_group(root)), {"job-1"})

class MalformedReportTests(unittest.TestCase):
    def test_malformed_primary_json_has_invalid_count_and_null_aggregate(self):
        import json
        import tempfile
        from pathlib import Path
        from percept_harness.evaluation import events_cli
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ref = root / "ref"; pred = root / "pred"; out = root / "out"
            ref.mkdir(); pred.mkdir()
            (ref / "clip.json").write_text("{")
            (pred / "clip.json").write_text(json.dumps({"status": "completed", "video_path": "/b/clip.mp4", "data": {"duration": 1, "events": []}}))
            self.assertEqual(events_cli.main(["--reference", str(ref), "--system", f"sys={pred}", "--out", str(out)], judge=lambda a, b: None), 1)
            report = json.loads((out / "events.json").read_text())
            self.assertEqual(report["systems"]["sys"]["invalid_count"], 1)
            self.assertIsNone(report["systems"]["sys"]["aggregate"])
            self.assertIn("clip.json", report["systems"]["sys"]["error"])

class EndpointAndAuditTests(unittest.TestCase):
    def test_invalid_judge_port_produces_sanitized_error_report(self):
        import json
        import tempfile
        from pathlib import Path
        from percept_harness.evaluation import events_cli
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref, pred, out = root / "ref", root / "pred", root / "out"
            ref.mkdir(); pred.mkdir()
            payload = {"status": "completed", "video_path": "/x/clip.mp4", "data": {"duration": 1, "events": []}}
            for directory in (ref, pred):
                (directory / "clip.json").write_text(json.dumps(payload))
            code = events_cli.main(["--reference", str(ref), "--system", f"sys={pred}", "--out", str(out),
                "--judge-base-url", "http://user:secret@localhost:bad/private", "--judge-model", "m", "--judge-api-key", "key"])
            self.assertEqual(code, 1)
            report = (out / "events.json").read_text()
            self.assertIn('invalid judge', report.lower())
            self.assertNotIn('secret', report)
            self.assertNotIn('private', report)
            self.assertNotIn('key', report)

    def test_unmatched_indices_distinguish_temporal_and_joint_assignments(self):
        ref = annotation((0, 10, "A"))
        pred = annotation((0, 10, "B"))
        score = events.score_pair(ref, pred, lambda a, b: {"match": False, "reason": "different action"})
        self.assertEqual(score["temporal_match_count"], 1)
        self.assertEqual(score["joint_match_count"], 0)
        self.assertEqual(score["unmatched_temporal_reference_indices"], [])
        self.assertEqual(score["unmatched_joint_reference_indices"], [0])
        self.assertEqual(score["unmatched_joint_prediction_indices"], [0])

class InstallSmokeTests(unittest.TestCase):
    def test_core_smoke_covers_grounded_event_scoring(self):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "scripts" / "verify_install.py"
        spec = importlib.util.spec_from_file_location("verify_install", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIn("grounded event scoring", module.check_smoke())

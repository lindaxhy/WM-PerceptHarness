"""Manifest annotation with persistent GPU workers and bounded VLM requests.

Each logical job has an independent normal run_batch output directory. SAM is
loaded once per physical GPU, in spawned processes; CPU threads own independent
VLM clients and serialize access to their assigned GPU. No credentials enter
worker arguments, manifests or progress files.
"""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import tempfile
import uuid
import threading
from dataclasses import replace
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

from .config import Settings
from .results import write_json
from .runner import SyncRunner, run_batch, _write_result


class RequestGate:
    """Ramp concurrency after healthy calls and back off on observed HTTP 429."""
    def __init__(self, maximum):
        self.maximum = maximum
        self.limit = min(2, maximum)
        self.active = self.peak = self.completed = self.rate_limits = 0
        self.healthy = 0
        self.condition = threading.Condition()

    def __enter__(self):
        with self.condition:
            self.condition.wait_for(lambda: self.active < self.limit)
            self.active += 1
            self.peak = max(self.peak, self.active)

    def __exit__(self, kind, value, traceback):
        with self.condition:
            self.active -= 1
            self.completed += 1
            self.healthy = self.healthy + 1 if kind is None else 0
            if self.healthy >= 4:
                self.limit = min(self.maximum, self.limit + 1)
                self.healthy = 0
            self.condition.notify_all()

    def throttled(self):
        with self.condition:
            self.rate_limits += 1
            self.limit = max(1, self.limit // 2)
            self.healthy = 0

    def snapshot(self):
        with self.condition:
            return {k: getattr(self, k) for k in ("maximum", "limit", "active", "peak", "completed", "rate_limits")}


class _ThrottleHandler(logging.Handler):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def emit(self, record):
        if "service status 429" in record.getMessage():
            self.gate.throttled()


def read_manifest(path: Path) -> list[dict]:
    rows, seen = [], set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("job_id"), str):
            raise ValueError(f"manifest row {number} requires a job_id")
        job_id = Path(row["job_id"])
        if job_id.is_absolute() or not job_id.parts or job_id.as_posix() != row["job_id"] or any(p in {".", ".."} for p in row["job_id"].split("/")):
            raise ValueError(f"unsafe job_id at row {number}")
        if row["job_id"] in seen:
            raise ValueError(f"duplicate job_id at row {number}")
        video = Path(row["video_path"])
        if not video.is_absolute() or not video.is_file():
            raise ValueError(f"missing absolute video path at row {number}: {video}")
        seen.add(row["job_id"])
        rows.append(row)
    return rows


def assignment(job_id: str, count: int) -> int:
    return int.from_bytes(hashlib.sha256(job_id.encode()).digest()[:8], "big") % count


def _cv_worker(connection, cv_settings, gpu):
    # Set visibility before the first Torch/CUDA import in this interpreter.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.pop("PERCEPT_OPENAI_API_KEY", None)
    try:
        from .cli import _cv_executor
        settings = Settings(**cv_settings)
        from contextlib import ExitStack
        with ExitStack() as stack:
            execute = None
            connection.send(("ready", {"pid": os.getpid(), "provider": settings.cv_provider,
                                       "initialization": "deferred_until_target"}))
            while True:
                payload = connection.recv()
                if payload is None:
                    break
                try:
                    if execute is None:
                        execute = stack.enter_context(_cv_executor(settings.cv_provider, SimpleNamespace(
                            cv_device=gpu, backend="openai"), settings))
                    result, metrics = execute(payload)
                    connection.send(("ok", (result, metrics)))
                except Exception as error:
                    connection.send(("error", f"{type(error).__name__}: {error}"))
    except Exception as error:
        try:
            connection.send(("error", f"{type(error).__name__}: {error}"))
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class GpuSlot:
    def __init__(self, context, settings: Settings, gpu: int):
        self.gpu, self.lock = gpu, threading.Lock()
        parent, child = context.Pipe()
        self.connection = parent
        cv_settings = {k: v for k, v in settings.model_dump().items() if k.startswith("cv_")}
        cv_settings["cv_device"] = gpu
        self.process = context.Process(target=_cv_worker, args=(child, cv_settings, gpu))
        self.process.start()
        child.close()
        self.ready = False

    def receive(self):
        while not self.connection.poll(1):
            if not self.process.is_alive():
                raise RuntimeError(f"SAM worker GPU {self.gpu} exited: {self.process.exitcode}")
        kind, value = self.connection.recv()
        if kind == "error":
            raise RuntimeError(value)
        return kind, value

    def wait_ready(self):
        kind, identity = self.receive()
        if kind != "ready":
            raise RuntimeError("invalid SAM worker startup response")
        self.identity = identity
        self.ready = True

    def __call__(self, payload):
        with self.lock:
            self.connection.send(payload)
            kind, value = self.receive()
            if kind != "ok":
                raise RuntimeError("invalid SAM worker response")
            return value

    def close(self):
        if self.ready and self.process.is_alive():
            self.connection.send(None)
            self.process.join()
        elif self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.connection.close()


def archive_cv_evidence(evidence, cache_root: Path, archive_root: Path) -> Path:
    """Retain SAM files outside the bounded local runtime cache."""
    from .cv.artifacts import CvArtifactStore, CvArtifactHandle
    handle = CvArtifactHandle(evidence["artifact_key"])
    destination = archive_root / handle.key[:2] / handle.key
    if not destination.exists():
        with CvArtifactStore(cache_root) as store:
            store.load(handle)  # Check the entry structure before exporting.
        source = cache_root / handle.key[:2] / handle.key
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".cv-copy-", dir=destination.parent))
        try:
            shutil.copytree(source, temporary, dirs_exist_ok=True)
            _verify_cv_archive(temporary)
            try:
                os.replace(temporary, destination)
            except OSError:
                if not destination.exists():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    _verify_cv_archive(destination)
    return destination


def _verify_cv_archive(root: Path):
    manifest = root / "manifest.json"
    if manifest.is_symlink():
        raise ValueError("CV archive manifest must be a regular file")
    content = manifest.read_bytes()
    artifact = json.loads(content)["artifact"]
    for record in artifact["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("CV archive file path escapes archive")
        path = root / relative
        if path.is_symlink() or path.resolve().is_relative_to(root.resolve()) is False:
            raise ValueError("CV archive contains a foreign file")
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError("CV archive file size mismatch")


def required_event_phases(metadata):
    if metadata.get('event_workflow') == 'confidence-gated':
        required = {'confidence_draft'}
        if (metadata.get('recheck_budget', {}).get('selected_windows', 0)
                or metadata.get('confidence_gate', {}).get('requires_revision', False)):
            required.add('grounded_revision')
        return required
    return {'grounded_draft', 'evidence_review', 'grounded_revision'}


class ReviewRequiredRunner(SyncRunner):
    """Require complete review, while allowing no targeted SAM requests."""
    def evaluate(self, *args, **kwargs):
        outcome = super().evaluate(*args, **kwargs)
        if outcome.status != "completed":
            return outcome
        metadata = outcome.metadata or {}
        phases = metadata.get('phases', [])
        required = required_event_phases(metadata)
        if not required.issubset(phases):
            return replace(outcome, status='failed', data=None,
                           error='Complete grounded review unavailable')
        return outcome


def run_manifest(arguments) -> int:
    from .cli import _load_backend
    if arguments.workers_per_gpu <= 0 or arguments.request_concurrency <= 0:
        raise ValueError("worker and request concurrency must be positive")
    if not 0 <= arguments.shard_index < arguments.shard_count:
        raise ValueError("invalid shard index or count")
    settings = Settings()
    provider = arguments.cv or settings.cv_provider
    if provider != "sam31":
        raise ValueError("manifest annotation requires --cv sam31")
    devices = [int(s) for s in (arguments.cv_devices or str(settings.cv_device)).split(",")]
    if any(g < 0 for g in devices) or len(set(devices)) != len(devices):
        raise ValueError("CV devices must be distinct nonnegative integers")
    settings = Settings.model_validate({**settings.model_dump(), "cv_provider": provider,
        "event_workflow": getattr(arguments, "event_workflow", None) or settings.event_workflow})
    rows = [row for row in read_manifest(arguments.manifest)
            if assignment(row["job_id"], arguments.shard_count) == arguments.shard_index]
    root = arguments.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / f"shard-{arguments.shard_index:02d}.status.json"
    states = {row["job_id"]: {"status": "pending"} for row in rows}
    mutex = threading.Lock()
    stopped = threading.Event()
    request_gate = RequestGate(arguments.request_concurrency)
    throttle_handler = _ThrottleHandler(request_gate)
    provider_logger = logging.getLogger("percept_harness.models.openai_compat")
    provider_logger.addHandler(throttle_handler)
    slots = []
    live_keys = {}
    archived_keys = {}
    archive_lock = threading.Lock()
    start = time.time()

    def status(phase):
        with mutex:
            snapshot = {k: dict(v) for k, v in states.items()}
        counts = {s: sum(v["status"] == s for v in snapshot.values())
                  for s in ("pending", "running", "completed", "failed")}
        write_json(status_path, {"phase": phase, "pid": os.getpid(), "host": os.uname().nodename,
            "shard_index": arguments.shard_index, "shard_count": arguments.shard_count,
            "started_unix": start, "updated_unix": time.time(), "total": len(rows),
            "counts": counts, "cv_devices": devices,
            "request_concurrency": arguments.request_concurrency,
            "vlm_requests": request_gate.snapshot(),
            "sam_workers": [{"gpu": s.gpu, "pid": s.process.pid,
                             "alive": s.process.is_alive(), "ready": s.ready,
                             "identity": getattr(s, "identity", None)} for s in slots],
            "jobs": snapshot})

    def reporter():
        while not stopped.wait(15):
            status("running" if slots and all(s.ready for s in slots) else "initializing")

    def worker(gpu, cv_execute, tasks):
        lane_settings = settings.model_copy(update={"cv_device": gpu})
        model, alias, closer = _load_backend(arguments, lane_settings)
        original_generate = model.generate
        def generate(request):
            with request_gate:
                return original_generate(request)
        model.generate = generate
        pending_keys = set()
        def protected_cv(payload):
            from .cv.artifacts import cv_cache_key
            from .cv.contracts import CvEvidenceRequest
            key = cv_cache_key(CvEvidenceRequest.model_validate(payload, strict=True))
            with mutex:
                live_keys[key] = live_keys.get(key, 0) + 1
                pending_keys.add(key)
            return cv_execute(payload)
        runner = ReviewRequiredRunner(model, lane_settings, model_alias=alias, cv_executor=protected_cv)
        try:
            while True:
                try:
                    row = tasks.get_nowait()
                except queue.Empty:
                    break
                jid = row["job_id"]
                with mutex:
                    states[jid] = {"status": "running", "gpu": gpu, "started_unix": time.time()}
                outcome = None
                try:
                    video = Path(row["video_path"])
                    job_root = root / jid
                    outcomes = run_batch(runner, [video], job_root,
                                         force=arguments.force, log=lambda message: None)
                    outcome = outcomes[0]
                    if outcome.status not in {"completed", "skipped"}:
                        raise RuntimeError(outcome.error or "annotation failed")
                    # Completed, structurally valid annotations are reused by default.
                    metadata = outcome.metadata or {}
                    evidence = metadata.get('evidence', [])
                    if not required_event_phases(metadata).issubset(metadata.get('phases', [])):
                        raise RuntimeError('required event workflow phases are incomplete')
                    archives = []
                    with archive_lock:
                        for record in evidence:
                            if record.get('status') != 'available':
                                continue
                            archived = archive_cv_evidence(record, lane_settings.cv_cache_root, root / '.cv-archive')
                            archived_keys[record['artifact_key']] = archived
                            archives.append({'scope': record['scope'], 'archive_path': str(archived)})
                    write_json(job_root / '.percept' / 'cv-artifact.json', {
                        'artifacts': archives, 'host': os.uname().nodename, 'gpu': gpu})
                    with mutex:
                        states[jid].update(status='completed', finished_unix=time.time(),
                            result_file=str(job_root / outcome.result_file), cv_archives=archives, resumed=outcome.status == 'skipped')
                except Exception as error:
                    if outcome is not None and outcome.status in {"completed", "skipped"}:
                        _write_result(job_root / outcome.result_file, replace(
                            outcome, status="failed", data=None, error=f"Evidence export failed: {error}"))
                    with mutex:
                        states[jid].update(status="failed", finished_unix=time.time(), error=str(error))
                finally:
                    with mutex:
                        for key in pending_keys:
                            live_keys[key] -= 1
                        pending_keys.clear()
                    # Evict only verified, durably archived, unused cache copies.
                    # Atomic detachment under the same lock as CV registration
                    # keeps active source entries intact. The evidence remains
                    # recoverable in the archive; never evict unexported entries.
                    with archive_lock:
                        for key in list(archived_keys):
                            if shutil.disk_usage(lane_settings.cv_cache_root).free >= 12 * 1024**3:
                                break
                            source = lane_settings.cv_cache_root / key[:2] / key
                            retired = lane_settings.cv_cache_root / (".retired-" + uuid.uuid4().hex)
                            with mutex:
                                if live_keys.get(key, 0) or not source.exists():
                                    continue
                                os.replace(source, retired)
                            shutil.rmtree(retired)
                            del archived_keys[key]
                    tasks.task_done()
        finally:
            closer()

    status("initializing")
    monitor = threading.Thread(target=reporter, daemon=True)
    monitor.start()
    try:
        context = mp.get_context("spawn")
        slots = [GpuSlot(context, settings, gpu) for gpu in devices]
        for slot in slots:
            slot.wait_ready()
        tasks = [queue.Queue() for _ in slots]
        for row in rows:
            tasks[assignment(row["job_id"], len(slots))].put(row)
        status("running")
        with ThreadPoolExecutor(max_workers=len(slots) * arguments.workers_per_gpu) as executor:
            futures = [executor.submit(worker, slot.gpu, slot, tasks[i])
                       for i, slot in enumerate(slots) for _ in range(arguments.workers_per_gpu)]
            for future in futures:
                future.result()
        status("complete")
        return 1 if any(v["status"] == "failed" for v in states.values()) else 0
    except Exception:
        status("failed")
        raise
    finally:
        provider_logger.removeHandler(throttle_handler)
        stopped.set()
        monitor.join()
        for slot in slots:
            slot.close()

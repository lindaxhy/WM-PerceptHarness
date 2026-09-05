from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

from las_repro.cv.base import FakeCvEvidenceProvider
from las_repro.cv.contracts import FrameTimeline, FrameTimestamp

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "sam31_smoke.py"
PINNED_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
SECRET_PROMPT = "prompt-secret-person"
SECRET_TOKEN = "sk-" + "smoke-secret-token-value"


def _load_script():
    spec = importlib.util.spec_from_file_location("sam31_smoke", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_paths(tmp_path: Path) -> tuple[list[str], str, Path, Path]:
    repository = tmp_path / "sam-source"
    repository.mkdir()
    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"pinned checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    media_root = tmp_path / "allowed-media"
    media_root.mkdir()
    video = media_root / "one-video.mp4"
    video.write_bytes(b"one allowed video")
    cache = tmp_path / "cache"
    arguments = [
        "--repository",
        str(repository),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        digest,
        "--video",
        str(video),
        "--allowed-media-root",
        str(media_root),
        "--cache-root",
        str(cache),
        "--device",
        "3",
    ]
    return arguments, digest, video, cache


class RecordingProvider(FakeCvEvidenceProvider):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.request = None

    def analyze(self, request, staging_dir):
        self.request = request
        return super().analyze(request, staging_dir)

    def close(self) -> None:
        self.closed = True


def _dependencies(
    module,
    provider: RecordingProvider,
    *,
    duration: float = 1.25,
    noisy_load: bool = False,
    timeline: FrameTimeline | None = None,
):
    def load_provider(**kwargs):
        assert kwargs["repository_path"].name == "sam-source"
        assert kwargs["checkpoint_path"].name == "sam3.1_multiplex.pt"
        if noisy_load:
            print(f"loader {SECRET_PROMPT}")
            os.write(1, f"loader-native {SECRET_TOKEN}\n".encode())
        return provider

    return module.SmokeDependencies(
        load_provider=load_provider,
        probe_timeline=lambda _path: (
            timeline
            or FrameTimeline(
                frames=(
                    FrameTimestamp(frame_index=0, timestamp_seconds=0.0),
                    FrameTimestamp(frame_index=1, timestamp_seconds=1.0),
                )
            )
        ),
        probe_duration=lambda _path: duration,
        probe_gpu=lambda: ("Test GPU", 0),
        monotonic=iter((10.0, 12.5)).__next__,
    )


def test_smoke_requires_every_local_runtime_argument(capsys) -> None:
    module = _load_script()

    with pytest.raises(SystemExit) as error:
        module.main([])

    assert error.value.code == 2
    assert "required" in capsys.readouterr().err


def test_smoke_runs_real_request_store_reload_and_emits_one_canonical_record(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    arguments, digest, video, _cache = _fixture_paths(tmp_path)
    provider = RecordingProvider()
    monkeypatch.setenv("ARK_API_KEY", SECRET_TOKEN)

    assert (
        module.main(
            arguments,
            dependencies=_dependencies(module, provider, noisy_load=True),
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert (
        captured.out
        == json.dumps(json.loads(captured.out), sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    record = json.loads(captured.out)
    assert record == {
        "checkpoint_sha256": digest,
        "elapsed_seconds": 2.5,
        "frame_count": 2,
        "gpu_index": 3,
        "gpu_name": "Test GPU",
        "observation_count": 4,
        "pass": True,
        "peak_allocated_bytes": 0,
        "source_revision": PINNED_REVISION,
        "track_count": 2,
    }
    assert provider.closed is True
    assert provider.request.duration_seconds == 1.25
    assert str(video.resolve()) not in captured.out
    assert SECRET_PROMPT not in captured.out
    assert SECRET_TOKEN not in captured.out


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda args, _tmp: args.__setitem__(args.index("3"), "2"), "device"),
        (
            lambda args, tmp: args.__setitem__(
                args.index(
                    next(item for item in args if item.endswith("one-video.mp4"))
                ),
                str(tmp / "outside.mp4"),
            ),
            "video",
        ),
        (
            lambda args, _tmp: args.__setitem__(
                args.index(next(item for item in args if len(item) == 64)), "0" * 64
            ),
            "checkpoint",
        ),
    ],
    ids=("wrong-device", "outside-media-root", "checkpoint-hash-mismatch"),
)
def test_smoke_rejects_wrong_device_hash_and_outside_media_without_gpu_loading(
    tmp_path: Path, capsys, mutation, expected_error: str
) -> None:
    module = _load_script()
    arguments, _digest, _video, _cache = _fixture_paths(tmp_path)
    (tmp_path / "outside.mp4").write_bytes(b"outside")
    mutation(arguments, tmp_path)
    loaded = False

    def forbidden_loader(**_kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError

    dependencies = module.SmokeDependencies(
        load_provider=forbidden_loader,
        probe_timeline=lambda _path: (_ for _ in ()).throw(AssertionError()),
        probe_duration=lambda _path: (_ for _ in ()).throw(AssertionError()),
        probe_gpu=lambda: (_ for _ in ()).throw(AssertionError()),
        monotonic=lambda: 0.0,
    )

    assert module.main(arguments, dependencies=dependencies) == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {"error": expected_error, "pass": False}
    assert loaded is False
    assert str(tmp_path) not in output


def test_smoke_rejects_symlink_video_and_network_flags(tmp_path: Path, capsys) -> None:
    module = _load_script()
    arguments, _digest, video, _cache = _fixture_paths(tmp_path)
    link = video.parent / "linked.mp4"
    link.symlink_to(video)
    arguments[arguments.index(str(video))] = str(link)

    assert (
        module.main(arguments, dependencies=_dependencies(module, RecordingProvider()))
        == 1
    )
    assert json.loads(capsys.readouterr().out) == {"error": "video", "pass": False}

    with pytest.raises(SystemExit) as error:
        module.main(arguments + ["--download", "https://example.invalid/video"])
    assert error.value.code == 2


def test_smoke_suppresses_python_native_and_subprocess_output_on_success(
    tmp_path: Path, capfd
) -> None:
    module = _load_script()
    arguments, _digest, _video, _cache = _fixture_paths(tmp_path)

    class NoisyProvider(RecordingProvider):
        def analyze(self, request, staging_dir):
            print(f"analyze {SECRET_PROMPT}")
            os.write(1, f"native {SECRET_TOKEN}\n".encode())
            subprocess.run(
                ["sh", "-c", "printf 'subprocess /private/source/path\\n'"],
                check=True,
            )
            return super().analyze(request, staging_dir)

        def close(self):
            print(f"cleanup {SECRET_TOKEN}")
            super().close()

    provider = NoisyProvider()

    assert module.main(arguments, dependencies=_dependencies(module, provider)) == 0

    captured = capfd.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out)["pass"] is True
    assert SECRET_PROMPT not in captured.out
    assert SECRET_TOKEN not in captured.out
    assert "/private/source/path" not in captured.out


def test_smoke_flushes_buffered_native_stdout_before_restoring_fd(
    tmp_path: Path, capfd
) -> None:
    module = _load_script()
    arguments, _digest, _video, _cache = _fixture_paths(tmp_path)
    libc = ctypes.CDLL(None)
    libc.printf.argtypes = [ctypes.c_char_p]
    libc.printf.restype = ctypes.c_int
    libc.fflush.argtypes = [ctypes.c_void_p]
    libc.fflush.restype = ctypes.c_int

    class BufferedNativeProvider(RecordingProvider):
        def analyze(self, request, staging_dir):
            assert libc.printf(b"BUFFERED_NATIVE_SENTINEL") > 0
            return super().analyze(request, staging_dir)

    provider = BufferedNativeProvider()

    assert module.main(arguments, dependencies=_dependencies(module, provider)) == 0
    assert libc.fflush(None) == 0
    captured = capfd.readouterr()
    assert captured.out.count("\n") == 1
    assert "BUFFERED_NATIVE_SENTINEL" not in captured.out


def test_smoke_sanitizes_noisy_analyze_failure_and_closes_provider(
    tmp_path: Path, capfd
) -> None:
    module = _load_script()
    arguments, _digest, _video, _cache = _fixture_paths(tmp_path)

    class FailingProvider(RecordingProvider):
        def analyze(self, request, staging_dir):
            print(f"analyze {SECRET_PROMPT}")
            raise RuntimeError(f"{SECRET_TOKEN} /private/host/path")

        def close(self):
            os.write(1, f"cleanup {SECRET_TOKEN}\n".encode())
            super().close()

    provider = FailingProvider()

    assert module.main(arguments, dependencies=_dependencies(module, provider)) == 1

    output = capfd.readouterr().out
    assert json.loads(output) == {"error": "runtime", "pass": False}
    assert SECRET_PROMPT not in output
    assert SECRET_TOKEN not in output
    assert str(tmp_path) not in output
    assert "RuntimeError" not in output
    assert provider.closed is True


def test_smoke_sanitizes_noisy_cleanup_failure(tmp_path: Path, capfd) -> None:
    module = _load_script()
    arguments, _digest, _video, _cache = _fixture_paths(tmp_path)

    class CleanupFailure(RecordingProvider):
        def close(self):
            print(f"cleanup {SECRET_TOKEN} /private/cleanup/path")
            raise RuntimeError(SECRET_PROMPT)

    provider = CleanupFailure()

    assert module.main(arguments, dependencies=_dependencies(module, provider)) == 1
    output = capfd.readouterr().out
    assert json.loads(output) == {"error": "runtime", "pass": False}
    assert SECRET_PROMPT not in output
    assert SECRET_TOKEN not in output
    assert "/private/cleanup/path" not in output


@pytest.mark.parametrize(
    ("timeline", "duration"),
    [
        (
            FrameTimeline(
                frames=(FrameTimestamp(frame_index=0, timestamp_seconds=0.0),)
            ),
            0.04,
        ),
        (
            FrameTimeline(
                frames=(
                    FrameTimestamp(frame_index=0, timestamp_seconds=0.0),
                    FrameTimestamp(frame_index=1, timestamp_seconds=35.0),
                )
            ),
            35.5,
        ),
    ],
    ids=("single-frame-at-zero", "long-video"),
)
def test_smoke_uses_media_duration_instead_of_final_pts(
    tmp_path: Path, capsys, timeline: FrameTimeline, duration: float
) -> None:
    module = _load_script()
    arguments, _digest, _video, _cache = _fixture_paths(tmp_path)
    provider = RecordingProvider()

    assert (
        module.main(
            arguments,
            dependencies=_dependencies(
                module, provider, duration=duration, timeline=timeline
            ),
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["frame_count"] == len(timeline.frames)
    assert provider.request.duration_seconds == duration


def test_smoke_closes_provider_when_reload_raises(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    arguments, _digest, _video, _cache = _fixture_paths(tmp_path)
    provider = RecordingProvider()

    def fail_reload(_self, _handle):
        print(f"reload {SECRET_PROMPT}")
        os.write(1, f"reload-native {SECRET_TOKEN}\n".encode())
        raise RuntimeError(f"{SECRET_PROMPT} {SECRET_TOKEN}")

    monkeypatch.setattr(module.CvArtifactStore, "load", fail_reload)

    assert module.main(arguments, dependencies=_dependencies(module, provider)) == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {"error": "runtime", "pass": False}
    assert SECRET_PROMPT not in output
    assert SECRET_TOKEN not in output
    assert provider.closed is True


def test_smoke_rejects_invalid_reloaded_artifact_and_closes_lifecycle(
    tmp_path: Path, capsys
) -> None:
    module = _load_script()
    arguments, _digest, _video, _cache = _fixture_paths(tmp_path)

    class EmptyProvider(RecordingProvider):
        def analyze(self, request, staging_dir):
            artifact = super().analyze(request, staging_dir)
            return artifact.model_copy(update={"tracks": (), "files": ()})

    provider = EmptyProvider()

    assert module.main(arguments, dependencies=_dependencies(module, provider)) == 1
    assert json.loads(capsys.readouterr().out) == {"error": "artifact", "pass": False}
    assert provider.closed is True

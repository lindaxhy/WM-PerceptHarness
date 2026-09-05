from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from las_repro.cv.base import FakeCvEvidenceProvider
from las_repro.cv.contracts import FrameTimeline, FrameTimestamp

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "sam31_smoke.py"
PINNED_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
SECRET_PROMPT = "prompt-secret-person"
SECRET_TOKEN = "sk-smoke-secret-token-value"


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

    def close(self) -> None:
        self.closed = True


def _dependencies(module, provider: RecordingProvider, *, fail: bool = False):
    def load_provider(**kwargs):
        assert kwargs["repository_path"].name == "sam-source"
        assert kwargs["checkpoint_path"].name == "sam3.1_multiplex.pt"
        if fail:
            raise RuntimeError(f"{SECRET_PROMPT} {SECRET_TOKEN} /private/host/path")
        return provider

    return module.SmokeDependencies(
        load_provider=load_provider,
        probe_timeline=lambda _path: FrameTimeline(
            frames=(
                FrameTimestamp(frame_index=0, timestamp_seconds=0.0),
                FrameTimestamp(frame_index=1, timestamp_seconds=1.0),
            )
        ),
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

    assert module.main(arguments, dependencies=_dependencies(module, provider)) == 0

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


def test_smoke_sanitizes_runtime_failure_and_closes_provider(
    tmp_path: Path, capsys
) -> None:
    module = _load_script()
    arguments, _digest, _video, _cache = _fixture_paths(tmp_path)
    provider = RecordingProvider()

    assert (
        module.main(arguments, dependencies=_dependencies(module, provider, fail=True))
        == 1
    )

    output = capsys.readouterr().out
    assert json.loads(output) == {"error": "runtime", "pass": False}
    assert SECRET_PROMPT not in output
    assert SECRET_TOKEN not in output
    assert str(tmp_path) not in output
    assert "RuntimeError" not in output
    assert provider.closed is False


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

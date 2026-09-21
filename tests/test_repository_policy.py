from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from percept_harness.config import Settings


_ACCESS_TOKEN_PATTERN = re.compile(
    rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|"
    rb"sk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{20,})"
)
_CV_GPU_PACKAGES = frozenset({"cv2", "numpy", "sam3", "torch"})


def _access_token_match(payload: bytes) -> re.Match[bytes] | None:
    return _ACCESS_TOKEN_PATTERN.search(payload)


def _imported_cv_gpu_packages(source: str) -> set[str]:
    tree = ast.parse(source)
    imported: set[str] = set()
    importlib_aliases: set[str] = set()
    import_module_aliases: set[str] = set()
    builtins_aliases: set[str] = set()
    builtin_import_aliases = {"__import__"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                package = alias.name.split(".", 1)[0]
                if package in _CV_GPU_PACKAGES:
                    imported.add(package)
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
                elif alias.name == "builtins":
                    builtins_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            package = node.module.split(".", 1)[0]
            if package in _CV_GPU_PACKAGES:
                imported.add(package)
            if node.module == "importlib":
                import_module_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            elif node.module == "builtins":
                builtin_import_aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "__import__"
                )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        dynamic_import = (
            isinstance(function, ast.Name)
            and function.id in import_module_aliases | builtin_import_aliases
        ) or (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and (
                (
                    function.attr == "import_module"
                    and function.value.id in importlib_aliases
                )
                or (
                    function.attr == "__import__"
                    and function.value.id in builtins_aliases
                )
            )
        )
        name_keywords = [
            keyword.value for keyword in node.keywords if keyword.arg == "name"
        ]
        if node.args and name_keywords:
            continue
        if node.args:
            argument = node.args[0]
        elif len(name_keywords) == 1:
            argument = name_keywords[0]
        else:
            continue
        if (
            dynamic_import
            and isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
        ):
            package = argument.value.split(".", 1)[0]
            if package in _CV_GPU_PACKAGES:
                imported.add(package)

    return imported


def _cv_gpu_import_offenders(repository_root: Path) -> list[str]:
    source_root = repository_root / "src" / "percept_harness"
    adapters = {
        source_root / "cv" / "sam31.py": _CV_GPU_PACKAGES,
        source_root / "video_metrics" / "clipiqa.py": {"cv2", "numpy", "torch"},
    }
    offenders: list[str] = []

    for path in sorted(source_root.rglob("*.py")):
        imported = _imported_cv_gpu_packages(path.read_text(encoding="utf-8"))
        allowed = adapters.get(path, set())
        if imported - allowed:
            offenders.append(path.relative_to(repository_root).as_posix())

    return offenders


def test_repository_has_no_asr_module_dependency_configuration_or_legacy_test():
    """Reintroducing any removed ASR surface must fail the no-audio policy gate."""
    repository_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (repository_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    optional_dependencies = project["project"]["optional-dependencies"]
    dependency_groups = [
        project["project"]["dependencies"],
        *optional_dependencies.values(),
    ]
    dependency_names = {
        re.sub(r"[-_.]+", "-", match.group(1).lower())
        for group in dependency_groups
        for requirement in group
        if (match := re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)) is not None
    }

    assert importlib.util.find_spec("percept_harness.asr") is None
    assert all(extra.lower() != "asr" for extra in optional_dependencies)
    assert "faster-whisper" not in dependency_names
    assert {
        "asr_backend",
        "asr_model_path",
        "asr_device",
        "asr_compute_type",
        "asr_required",
    }.isdisjoint(Settings.model_fields)
    assert not (repository_root / "tests" / "test_asr.py").exists()


def _tracked_text_files(repository_root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        capture_output=True,
        check=True,
    )
    return [
        repository_root / Path(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw and (repository_root / Path(raw.decode("utf-8"))).is_file()
    ]


def test_repository_contains_no_committed_access_token_values() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    offenders = [
        path.relative_to(repository_root).as_posix()
        for path in _tracked_text_files(repository_root)
        if _access_token_match(path.read_bytes()) is not None
    ]

    assert offenders == []


@pytest.mark.parametrize(
    "candidate",
    [
        b"hf" + b"_" + b"A" * 24,
        b"gh" + b"p_" + b"B" * 32,
        b"gh" + b"o_" + b"C" * 32,
        b"gh" + b"u_" + b"D" * 32,
        b"gh" + b"s_" + b"E" * 32,
        b"gh" + b"r_" + b"F" * 32,
        b"github" + b"_pat_" + b"G" * 32,
        b"sk" + b"-" + b"H" * 32,
        b"sk" + b"-proj-" + b"I" * 32,
        b"sk" + b"-svcacct-" + b"J" * 32,
    ],
    ids=(
        "hugging-face",
        "github-personal",
        "github-oauth",
        "github-user",
        "github-server",
        "github-refresh",
        "github-fine-grained",
        "openai-legacy",
        "openai-project",
        "openai-service-account",
    ),
)
def test_access_token_detector_recognizes_common_formats_without_self_reporting(
    candidate: bytes,
) -> None:
    assert _access_token_match(b"prefix=" + candidate + b";suffix") is not None
    assert _access_token_match(Path(__file__).read_bytes()) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import torch", {"torch"}),
        ("import numpy as np", {"numpy"}),
        ("from sam3.runtime import Predictor", {"sam3"}),
        ("from cv2 import imread", {"cv2"}),
        ("import importlib as il\nil.import_module('torch')", {"torch"}),
        (
            "from importlib import import_module as load\nload('numpy.linalg')",
            {"numpy"},
        ),
        ("__import__('sam3')", {"sam3"}),
        ("import builtins as bi\nbi.__import__('cv2')", {"cv2"}),
        (
            "import importlib\nimportlib.import_module(name='torch')",
            {"torch"},
        ),
        (
            "from importlib import import_module as load\nload(name='numpy')",
            {"numpy"},
        ),
        ("__import__(name='sam3')", {"sam3"}),
        ("import builtins as bi\nbi.__import__(name='cv2')", {"cv2"}),
        ("import importlib\nimportlib.import_module(package_name)", set()),
        (
            "import importlib\nimportlib.import_module(name=package_name)",
            set(),
        ),
        (
            "import importlib\nimportlib.import_module('torch', name='sam3')",
            set(),
        ),
        ("import importlib\nimportlib.import_module('json')", set()),
    ],
)
def test_cv_gpu_import_detector_handles_static_and_constant_dynamic_imports(
    source: str,
    expected: set[str],
) -> None:
    assert _imported_cv_gpu_packages(source) == expected


def test_cv_gpu_import_policy_recurses_and_uses_exact_runtime_adapter_paths(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src" / "percept_harness"
    sources = {
        "cv/sam31.py": "import torch\nimport numpy\n__import__('sam3')",
        "models/qwen3_vl.py": "import importlib\nimportlib.import_module('torch')",
        "nested/deeper/bad.py": "import importlib as il\nil.import_module('torch')",
        "models/sam31.py": "__import__('sam3')",
        "cv/safe.py": "import importlib\nimportlib.import_module('json')",
    }
    for relative_path, source in sources.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    assert _cv_gpu_import_offenders(tmp_path) == [
        "src/percept_harness/models/qwen3_vl.py",
        "src/percept_harness/models/sam31.py",
        "src/percept_harness/nested/deeper/bad.py",
    ]


def test_cv_gpu_packages_are_imported_only_by_explicit_runtime_adapters() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    assert _cv_gpu_import_offenders(repository_root) == []


def test_env_example_documents_disabled_local_cv_defaults() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    values = {
        key: value.strip("'\"")
        for raw_line in (repository_root / ".env.example").read_text(
            encoding="utf-8"
        ).splitlines()
        if raw_line.strip() and not raw_line.lstrip().startswith("#")
        for key, value in [raw_line.split("=", 1)]
    }

    # CV evidence must stay disabled unless explicitly configured: the
    # example environment must not switch the provider on, and the coded
    # defaults must keep every CV field at its conservative value.
    assert "PERCEPT_CV_PROVIDER" not in values

    from percept_harness.config import Settings

    defaults = Settings()
    assert defaults.cv_provider == "disabled"
    assert defaults.cv_model_alias == "sam3.1"
    assert defaults.cv_entity_limit == 16
    assert defaults.cv_short_video_seconds == 30.0
    assert defaults.cv_scan_fps == 8.0
    assert defaults.cv_max_fps == 30.0
    assert defaults.cv_refinement_radius_seconds == 1.0
    assert defaults.cv_min_confidence == 0.5
    assert defaults.cv_min_area_fraction == 0.01
    assert defaults.cv_occlusion_visibility_drop == 0.5
    assert defaults.cv_execution_chunk_frames == 8
    assert defaults.cv_timeout_seconds == 300.0
    assert defaults.cv_compile_model is False
    assert defaults.cv_cache_max_bytes == 8 * 1024 * 1024 * 1024
    assert defaults.cv_cache_max_files == 10000
    assert re.fullmatch(r"[0-9a-f]{64}", defaults.cv_checkpoint_sha256)
    for field in (
        defaults.cv_repository_path,
        defaults.cv_checkpoint_path,
        defaults.cv_bpe_path,
        defaults.cv_cache_root,
    ):
        assert not Path(field).is_absolute()

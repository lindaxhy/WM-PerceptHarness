"""Runtime configuration loaded from the environment."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BeforeValidator, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _comma_separated_paths(value: Any) -> tuple[Path, ...]:
    if isinstance(value, str):
        return tuple(Path(item.strip()) for item in value.split(",") if item.strip())
    return tuple(Path(item) for item in value)


def _comma_separated_ints(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(
            _integer(item.strip(), "GPU device")
            for item in value.split(",")
            if item.strip()
        )
    return tuple(_integer(item, "GPU device") for item in value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _nonnegative_integer(value: Any) -> int:
    parsed = _integer(value, "value")
    if parsed < 0:
        raise ValueError("value must be non-negative")
    return parsed


def _positive_integer(value: Any) -> int:
    parsed = _integer(value, "value")
    if parsed <= 0:
        raise ValueError("value must be positive")
    return parsed


def _positive_finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("value must be finite and positive")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError("value must be finite and positive") from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("value must be finite and positive")
    return parsed


def _positive_fraction(value: Any) -> float:
    parsed = _positive_finite(value)
    if parsed > 1:
        raise ValueError("value must be no greater than one")
    return parsed


def _strict_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError("value must be a boolean")


CsvPaths = Annotated[tuple[Path, ...], NoDecode, BeforeValidator(_comma_separated_paths)]
CsvInts = Annotated[tuple[int, ...], NoDecode, BeforeValidator(_comma_separated_ints)]
NonnegativeInteger = Annotated[int, BeforeValidator(_nonnegative_integer)]
PositiveInteger = Annotated[int, BeforeValidator(_positive_integer)]
PositiveFinite = Annotated[float, BeforeValidator(_positive_finite)]
PositiveFraction = Annotated[float, BeforeValidator(_positive_fraction)]
StrictEnvironmentBool = Annotated[bool, BeforeValidator(_strict_boolean)]


class Settings(BaseSettings):
    """Configuration for one local LAS-compatible service instance."""

    model_config = SettingsConfigDict(env_prefix="LAS_", extra="forbid")

    database_path: Path = Path("data/las-repro.sqlite3")
    work_root: Path = Path("work")
    allowed_media_roots: CsvPaths = (Path("data/media"),)
    model_registry: dict[str, Path] = Field(
        default_factory=lambda: {"qwen3-vl-8b-instruct": Path("models/qwen3-vl-8b-instruct")}
    )
    backend: str = "qwen3_vl"
    api_key_sha256: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    max_download_bytes: int = 10 * 1024 * 1024 * 1024
    max_model_output_chars: int = Field(default=1_000_000, gt=0)
    segment_seconds: float = 30.0
    segment_overlap_seconds: float = 2.0
    max_fine_segment_seconds: float = 1.0
    lease_seconds: int = 300
    gpu_devices: CsvInts = (0, 1, 2)
    cv_device: NonnegativeInteger = 3
    cv_provider: Literal["disabled", "fake", "sam31"] = "disabled"
    cv_model_alias: Literal["sam3.1"] = "sam3.1"
    cv_repository_path: Path = Path("models/sam3")
    cv_checkpoint_path: Path = Path("models/sam3/checkpoints/sam3.pt")
    cv_bpe_path: Path = Path("models/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    cv_checkpoint_sha256: str = "0" * 64
    cv_cache_root: Path = Path("work/cv-cache")
    cv_cache_max_bytes: PositiveInteger = 8 * 1024 * 1024 * 1024
    cv_cache_max_files: PositiveInteger = 10_000
    cv_entity_limit: PositiveInteger = Field(default=16, le=16)
    cv_short_video_seconds: PositiveFinite = 30.0
    cv_scan_fps: PositiveFinite = 8.0
    cv_max_fps: PositiveFinite = 30.0
    cv_refinement_radius_seconds: PositiveFinite = 1.0
    cv_min_confidence: PositiveFraction = 0.5
    cv_min_area_fraction: PositiveFraction = 0.01
    cv_occlusion_visibility_drop: PositiveFraction = 0.5
    cv_overlap_threshold: PositiveFraction = 0.1
    cv_execution_chunk_frames: PositiveInteger = 8
    cv_timeout_seconds: PositiveFinite = 300.0
    cv_compile_model: StrictEnvironmentBool = False
    tos_endpoint: SecretStr | None = None
    tos_region: SecretStr | None = None
    tos_access_key: SecretStr | None = None
    tos_secret_key: SecretStr | None = None

    @property
    def qwen_gpu_devices(self) -> tuple[int, ...]:
        """Expose the legacy GPU setting under its explicit 3+1 role name."""
        return self.gpu_devices

    @field_validator("gpu_devices")
    @classmethod
    def validate_qwen_devices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("gpu_devices must contain at least one Qwen device")
        if any(device < 0 for device in value):
            raise ValueError("gpu_devices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("gpu_devices must be distinct")
        return value

    @field_validator("cv_checkpoint_sha256")
    @classmethod
    def validate_checkpoint_digest(cls, value: str) -> str:
        if re.fullmatch(r"[0-9A-Fa-f]{64}", value) is None:
            raise ValueError("cv_checkpoint_sha256 must be exactly 64 hexadecimal characters")
        return value.lower()

    @model_validator(mode="after")
    def validate_cv_configuration(self) -> Self:
        if self.cv_device in self.gpu_devices:
            raise ValueError("Qwen and CV devices must be distinct")
        if self.cv_scan_fps > self.cv_max_fps:
            raise ValueError("cv_scan_fps must not exceed cv_max_fps")
        if self.cv_provider == "sam31":
            path_kinds = (
                ("cv_repository_path", self.cv_repository_path, "directory"),
                ("cv_checkpoint_path", self.cv_checkpoint_path, "file"),
                ("cv_bpe_path", self.cv_bpe_path, "file"),
                ("cv_cache_root", self.cv_cache_root, "directory"),
            )
            for field_name, path, expected_kind in path_kinds:
                matches = path.is_dir() if expected_kind == "directory" else path.is_file()
                if not matches:
                    raise ValueError(
                        f"{field_name} must be an existing local {expected_kind}"
                    )
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        """Construct settings from ``LAS_``-prefixed environment variables."""
        return cls()

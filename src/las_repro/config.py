"""Runtime configuration loaded from the environment."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import BeforeValidator, Field, SecretStr
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _comma_separated_paths(value: Any) -> tuple[Path, ...]:
    if isinstance(value, str):
        return tuple(Path(item.strip()) for item in value.split(",") if item.strip())
    return tuple(Path(item) for item in value)


def _comma_separated_ints(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


CsvPaths = Annotated[tuple[Path, ...], NoDecode, BeforeValidator(_comma_separated_paths)]
CsvInts = Annotated[tuple[int, ...], NoDecode, BeforeValidator(_comma_separated_ints)]


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
    gpu_devices: CsvInts = (0, 1, 2, 3)
    tos_endpoint: SecretStr | None = None
    tos_region: SecretStr | None = None
    tos_access_key: SecretStr | None = None
    tos_secret_key: SecretStr | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        """Construct settings from ``LAS_``-prefixed environment variables."""
        return cls()

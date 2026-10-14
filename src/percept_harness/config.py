"""Runtime configuration loaded from the environment."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BeforeValidator, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("value must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value):
        return int(value)
    raise ValueError("value must be an integer")


def _finite_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("value must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError("value must be a finite number") from None
    if not math.isfinite(parsed):
        raise ValueError("value must be a finite number")
    return parsed


def _strict_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError("value must be a boolean")


StrictInteger = Annotated[int, BeforeValidator(_integer)]
NonnegativeInteger = Annotated[StrictInteger, Field(ge=0)]
PositiveInteger = Annotated[StrictInteger, Field(gt=0)]
FiniteNumber = Annotated[float, BeforeValidator(_finite_number)]
PositiveFinite = Annotated[FiniteNumber, Field(gt=0)]
NonnegativeFinite = Annotated[FiniteNumber, Field(ge=0)]
PositiveFraction = Annotated[FiniteNumber, Field(gt=0, le=1)]
StrictEnvironmentBool = Annotated[bool, BeforeValidator(_strict_boolean)]


class Settings(BaseSettings):
    """Configuration for the local evaluation harness."""

    model_config = SettingsConfigDict(env_prefix="PERCEPT_", extra="ignore")

    # Local workspace and model selection.
    work_root: Path = Path("work")
    # Alias registry for the fake backend and stored-result identity; the
    # openai backend derives its aliases from PERCEPT_OPENAI_MODEL[_REGISTRY].
    model_registry: dict[str, Path] = Field(
        default_factory=lambda: {"qwen3-vl-8b-instruct": Path("models/qwen3-vl-8b-instruct")}
    )
    # OpenAI-compatible model transport and request limits.
    openai_base_url: str | None = None
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    openai_model_registry: dict[str, str] = Field(default_factory=dict)
    openai_response_format: Literal["json_schema", "json_object", "none"] = "json_object"
    openai_extra_body: dict[str, Any] = Field(default_factory=dict)
    openai_extra_headers: dict[str, str] = Field(default_factory=dict)
    openai_timeout_seconds: PositiveFinite = 180.0
    # Per-request timeout grows with clip length: base + per_video_second * span,
    # capped at max. Zero keeps the fixed base timeout for every request.
    openai_timeout_seconds_per_video_second: NonnegativeFinite = 0.0
    openai_max_timeout_seconds: PositiveFinite = 1_800.0
    # Extra attempts for transient transport/service failures (timeouts, 429, 5xx).
    openai_transport_retries: NonnegativeInteger = 3
    openai_retry_backoff_seconds: PositiveFinite = 5.0
    openai_max_frames: PositiveInteger = 128
    # Replay validated per-request results for byte-identical inputs across
    # repair-loop retries and task reruns (openai backend only).
    semantic_result_cache: StrictEnvironmentBool = True
    openai_max_request_bytes: PositiveInteger = 32 * 1024 * 1024
    openai_max_output_chars: PositiveInteger = 1_000_000
    openai_proxy: SecretStr | None = None
    event_workflow: Literal["reviewed", "confidence-gated"] = "reviewed"
    # Optional computer-vision evidence.
    cv_device: NonnegativeInteger = 3
    cv_provider: Literal["disabled", "fake", "sam31"] = "sam31"
    cv_model_alias: Literal["sam3.1"] = "sam3.1"
    cv_repository_path: Path = Path("models/sam3")
    cv_checkpoint_path: Path = Path("models/sam3/checkpoints/sam3.pt")
    cv_bpe_path: Path = Path("models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    cv_cache_root: Path = Path("work/cv-cache")
    cv_cache_max_bytes: PositiveInteger = 8 * 1024 * 1024 * 1024
    cv_cache_max_files: PositiveInteger = 10_000
    cv_max_fps: PositiveFinite = 30.0
    cv_min_confidence: PositiveFraction = 0.5
    cv_execution_chunk_frames: PositiveInteger = 8
    cv_compile_model: StrictEnvironmentBool = False

    @property
    def openai_model_aliases(self) -> dict[str, str]:
        """Alias -> provider model id for the OpenAI-compatible backend."""
        if self.openai_model_registry:
            return dict(self.openai_model_registry)
        if self.openai_model:
            from .models.base import validate_model_alias

            alias = re.sub(r"[^A-Za-z0-9._-]+", "-", self.openai_model).lstrip("._-")[:128]
            return {validate_model_alias(alias): self.openai_model}
        return {}

    @model_validator(mode="after")
    def validate_model_configuration(self) -> Self:
        if self.openai_model and self.openai_model_registry:
            raise ValueError(
                "openai_model and openai_model_registry are mutually exclusive"
            )
        openai_aliases = set(self.openai_model_aliases)
        if openai_aliases & set(self.model_registry):
            raise ValueError("OpenAI-compatible aliases must not overlap other registries")
        if self.openai_base_url is not None and not self.openai_base_url.startswith(
            ("https://", "http://")
        ):
            raise ValueError("openai_base_url must be an http(s) URL")
        return self

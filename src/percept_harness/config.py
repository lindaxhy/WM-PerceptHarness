"""Runtime configuration loaded from the environment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the local evaluation harness."""

    model_config = SettingsConfigDict(env_prefix="PERCEPT_", extra="ignore", allow_inf_nan=False)

    # Local workspace and model selection.
    work_root: Path = Path("work")
    # OpenAI-compatible model transport and request limits.
    openai_base_url: str | None = None
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    openai_model_registry: dict[str, str] = Field(default_factory=dict)
    openai_response_format: Literal["json_schema", "json_object", "none"] = "json_object"
    openai_extra_body: dict[str, Any] = Field(default_factory=dict)
    openai_extra_headers: dict[str, str] = Field(default_factory=dict)
    openai_timeout_seconds: float = Field(default=180.0, gt=0)
    # Replay validated per-request results for byte-identical inputs across
    # repair-loop retries and task reruns (openai backend only).
    semantic_result_cache: bool = True
    openai_proxy: SecretStr | None = None
    event_workflow: Literal["reviewed", "confidence-gated"] = "reviewed"
    # Optional computer-vision evidence.
    cv_device: int = Field(default=3, ge=0)
    cv_provider: Literal["disabled", "sam31"] = "sam31"
    cv_repository_path: Path = Path("models/sam3")
    cv_checkpoint_path: Path = Path("models/sam3/checkpoints/sam3.pt")
    cv_bpe_path: Path = Path("models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    cv_cache_root: Path = Path("work/cv-cache")
    cv_max_fps: float = Field(default=30.0, gt=0)

    @property
    def openai_model_aliases(self) -> dict[str, str]:
        """Alias -> provider model id for the OpenAI-compatible backend."""
        if self.openai_model_registry:
            return dict(self.openai_model_registry)
        if self.openai_model:
            alias = re.sub(r"[^A-Za-z0-9._-]+", "-", self.openai_model).lstrip("._-")[:128]
            return {alias: self.openai_model}
        return {}

    @model_validator(mode="after")
    def validate_model_configuration(self) -> Self:
        if self.openai_model and self.openai_model_registry:
            raise ValueError(
                "openai_model and openai_model_registry are mutually exclusive"
            )
        return self

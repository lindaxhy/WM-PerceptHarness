"""Public LAS-compatible request and response contracts."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from .domain import TaskStatus
from .model_alias import DEFAULT_MODEL_ALIAS, validate_model_alias


OperatorId = Literal["las_long_video_understand", "las_video_understanding"]
TaskTemplate = Literal[
    "general_video_captioning",
    "embodied_active_object_detection",
    "embodied_action_captioning",
]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonnegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class ContractModel(BaseModel):
    """Reject accidental top-level contract fields."""

    model_config = ConfigDict(extra="forbid")


class TaskContext(ContractModel):
    prompt_context: str


class SubmitData(ContractModel):
    """Submit payload, retaining documented tuning fields but not cloud secrets."""

    model_config = ConfigDict(extra="allow")

    video_url: str
    task_template: TaskTemplate | None = None
    query: str | None = None
    model_name: str = DEFAULT_MODEL_ALIAS
    task_context: TaskContext | None = None
    fps: PositiveFiniteFloat | None = None
    media_resolution: Literal["low", "medium", "high"] | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    clip_context: Literal["low", "medium", "high"] | None = None
    start: NonnegativeFiniteFloat | None = None
    end: FiniteFloat | None = None
    ark_api_key: SecretStr | None = None
    ark_endpoint_id: str | None = None
    use_responses_api: bool | None = None
    previous_response_ids: list[str] | None = None
    expire_in: int | None = None

    @field_validator("model_name")
    @classmethod
    def require_local_model_alias(cls, value: str) -> str:
        return validate_model_alias(value)

    @model_validator(mode="after")
    def require_template_or_query(self) -> SubmitData:
        """Reject work that cannot be routed to a supported local pipeline."""
        if self.task_template is None and not (self.query and self.query.strip()):
            raise ValueError("task_template or non-blank query is required")
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must be supplied together")
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("start must be less than end")
        return self

    def effective_template(self) -> TaskTemplate:
        """Resolve quickstart query mode onto the local general pipeline."""
        return self.task_template or "general_video_captioning"


class SubmitRequest(ContractModel):
    operator_id: OperatorId
    operator_version: Literal["v1"]
    data: SubmitData

    def sanitized_data(self) -> dict[str, Any]:
        """Return a new persistence-safe payload without cloud-only fields."""
        sanitized = self.data.model_dump(
            mode="json",
            exclude={
                "ark_api_key",
                "ark_endpoint_id",
                "use_responses_api",
                "previous_response_ids",
                "expire_in",
            },
            exclude_none=True,
        )
        sanitized["task_template"] = self.data.effective_template()
        return sanitized

    def compatibility_warnings(self) -> list[str]:
        """List ignored cloud-only inputs in stable request-field order."""
        warnings: list[str] = []
        supplied_fields = self.data.model_fields_set
        if "ark_api_key" in supplied_fields:
            warnings.append("ark_api_key was ignored; inference is fully local")
        if "ark_endpoint_id" in supplied_fields:
            warnings.append("ark_endpoint_id was ignored; inference is fully local")
        if "use_responses_api" in supplied_fields:
            warnings.append(
                "use_responses_api was ignored; local VideoSession caching is used"
            )
        if "previous_response_ids" in supplied_fields:
            warnings.append(
                "previous_response_ids was ignored; local VideoSession caching is used"
            )
        if "expire_in" in supplied_fields:
            warnings.append(
                "expire_in was ignored; local task retention is configured by the service"
            )
        return warnings


class PollRequest(ContractModel):
    operator_id: OperatorId
    operator_version: Literal["v1"]
    task_id: str


class ResponseMetadata(ContractModel):
    task_id: str
    task_status: TaskStatus
    business_code: str = "0"
    error_msg: str = ""
    warnings: list[str] = Field(default_factory=list)
    progress: dict[str, Any] | None = None


class SubmitResponse(ContractModel):
    metadata: ResponseMetadata


class PollResponse(ContractModel):
    metadata: ResponseMetadata
    data: dict[str, Any] | None = None

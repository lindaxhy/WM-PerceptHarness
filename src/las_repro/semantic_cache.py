"""Bounded content identities and validation for ARK semantic result replay."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .models.base import ModelRequest
    from .pipelines.output_validation import OutputSchemaRegistry

CACHE_SCHEMA_VERSION = 1
# Bump when trusted schema validation or normalization semantics change.
VALIDATOR_CONTRACT_VERSION = "embodied-output-v9"
MAX_ENTRY_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ENTRIES = 256
STAGE_SCHEMAS = {
    "embodied_pass_a": "CoarsePlan",
    "embodied_pass_b": "BoundaryPlan",
    "embodied_enrichment": "EnrichmentResult",
    "scene_semantics": "SceneSemanticsChoices",
    "occlusion_semantics": "OcclusionDecisionSet",
}
ResultValidator = Callable[[Mapping[str, Any]], bool]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VideoBinding:
    digest: str
    identity: tuple[int, int, int, int, int]


def bind_video(path: Path) -> VideoBinding | None:
    """Hash actual bytes, rejecting files that change while being read."""
    try:
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            digest = hashlib.file_digest(source, "sha256").hexdigest()
            after = os.fstat(source.fileno())
        def identity(st: os.stat_result) -> tuple[int, int, int, int, int]:
            return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if identity(before) != identity(after) or identity(after) != identity(path.stat()):
            return None
        return VideoBinding(digest, identity(after))
    except OSError:
        return None


@dataclass(frozen=True)
class SemanticIdentity:
    key: str
    json: str
    video: VideoBinding


def make_identity(
    model: Any, request: ModelRequest, context: Mapping[str, Any] | None
) -> SemanticIdentity | None:
    if (getattr(model, "supports_semantic_result_cache", False) is not True
            or STAGE_SCHEMAS.get(request.stage) != request.schema_name):
        return None
    video = bind_video(request.video_path)
    if video is None:
        return None
    value = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "validator_contract_version": VALIDATOR_CONTRACT_VERSION,
        "video_sha256": video.digest,
        "span": {"start": request.span.start, "end": request.span.end},
        "fps": request.fps,
        "prompt_sha256": sha256(request.prompt),
        "stage": request.stage,
        "schema_name": request.schema_name,
        "schema_context": context,
        "request": model.semantic_cache_identity(request),
    }
    encoded = canonical_json(value)
    if len(encoded.encode()) > MAX_ENTRY_BYTES:
        return None
    return SemanticIdentity(sha256(encoded), encoded, video)


def safe_result(
    result: Any,
    schema: str,
    context: Mapping[str, Any] | None,
    registry: OutputSchemaRegistry,
) -> bool:
    """Revalidate exact safe shapes, never sanitize an envelope as provider data."""
    try:
        if not isinstance(result, dict):
            return False
        if "_schema_validation" in result:
            return (registry.failure_codes(schema, result) is not None
                    or registry.normalized_result(schema, result, context) is not None)
        validated = registry.sanitize(schema, result, context)
        return canonical_json(validated) == canonical_json(result)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return False


def decode_entry(
    row: Any, identity: SemanticIdentity, validate: ResultValidator
) -> dict[str, Any] | None:
    """Digest, schema, canonical encoding and current validator must all agree."""
    try:
        if (row is None or row["schema_version"] != CACHE_SCHEMA_VERSION
                or row["cache_key"] != identity.key
                or row["identity_json"] != identity.json
                or sha256(row["identity_json"]) != identity.key):
            return None
        encoded = row["result_json"]
        if (not isinstance(encoded, str)
                or len(identity.json.encode()) + len(encoded.encode()) > MAX_ENTRY_BYTES):
            return None
        if sha256(encoded) != row["result_sha256"]:
            return None
        result = json.loads(encoded)
        if canonical_json(result) != encoded or not validate(result):
            return None
        return result
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return None

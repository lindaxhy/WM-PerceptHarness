"""Deterministic, non-semantic normalization for CV entity prompts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
from typing import Annotated, Literal
import unicodedata

from pydantic import Field, StrictStr, field_validator, model_validator

from .contracts import EntityPrompt, EntityRole, StrictModel


_MAX_RAW_ENTITY_CANDIDATES = 64
_MAX_NORMALIZED_ENTITIES = 16
_MAX_ENTITY_NAME_CHARS = 256
_MAX_ENTITY_ALIASES = 256
_MAX_ENTITY_ALIAS_CHARS = 128
_MAX_ALIAS_OMISSIONS = _MAX_RAW_ENTITY_CANDIDATES * _MAX_ENTITY_ALIASES
_MAX_ENTITY_ID_CHARS = 128
_MAX_ENTITY_ID_COLLISION_SUFFIX_CHARS = 3
_HASH_SUFFIX_CHARS = 16
_MAX_NORMALIZATION_WARNINGS = 1
_MAX_WARNING_CHARS = 256
_MAX_CANONICAL_INT = 2**63 - 1


class EntityCandidate(StrictModel):
    """A candidate named upstream without adding any NLP inference here."""

    name: Annotated[StrictStr, Field(max_length=_MAX_ENTITY_NAME_CHARS)]
    aliases: Annotated[
        tuple[
            Annotated[StrictStr, Field(max_length=_MAX_ENTITY_ALIAS_CHARS)], ...
        ],
        Field(max_length=_MAX_ENTITY_ALIASES),
    ]
    role: EntityRole

    @field_validator("role", mode="before")
    @classmethod
    def parse_role(cls, value: EntityRole | str) -> EntityRole:
        if type(value) is EntityRole:
            return value
        if type(value) is not str:
            raise ValueError("entity role must be a plain string or EntityRole")
        return EntityRole(value)


class NormalizedEntities(StrictModel):
    entities: Annotated[
        tuple[EntityPrompt, ...], Field(max_length=_MAX_NORMALIZED_ENTITIES)
    ]
    omitted_count: int = Field(ge=0, le=_MAX_CANONICAL_INT, strict=True)
    alias_omitted_count: int = Field(
        default=0, ge=0, le=_MAX_ALIAS_OMISSIONS, strict=True
    )
    alias_warning: Literal["ENTITY_ALIASES_TRUNCATED"] | None = None
    warnings: Annotated[
        tuple[Annotated[StrictStr, Field(max_length=_MAX_WARNING_CHARS)], ...],
        Field(max_length=_MAX_NORMALIZATION_WARNINGS),
    ] = ()

    @model_validator(mode="after")
    def require_alias_audit_pair(self) -> NormalizedEntities:
        if (self.alias_omitted_count > 0) != (self.alias_warning is not None):
            raise ValueError("alias omission count and warning must be paired")
        return self


@dataclass(slots=True)
class _MergedEntity:
    canonical_label: str
    role: EntityRole
    position: int
    aliases: dict[str, str] = field(default_factory=dict)


_ROLE_PRIORITY = {
    EntityRole.ACTOR: 0,
    EntityRole.MANIPULATED_OBJECT: 1,
    EntityRole.CONTAINER: 2,
    EntityRole.OCCLUDER: 3,
    EntityRole.SURFACE: 4,
    EntityRole.OTHER: 5,
}


def _normalize_text(value: str) -> str:
    """Apply only deterministic whitespace and Unicode case normalization."""
    return " ".join(value.split()).casefold()


def _truncate_with_hash(value: str, maximum: int) -> str:
    """Bound normalized text without making equal prefixes collide silently."""
    if len(value) <= maximum:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:_HASH_SUFFIX_CHARS]
    prefix_length = maximum - _HASH_SUFFIX_CHARS - 1
    return f"{value[:prefix_length]}_{digest}"


def _ascii_slug(label: str) -> str:
    decomposed = unicodedata.normalize("NFKD", label)
    ascii_label = decomposed.encode("ascii", "ignore").decode("ascii")
    pieces = []
    previous_was_separator = True
    for character in ascii_label.casefold():
        if character.isascii() and character.isalnum():
            pieces.append(character)
            previous_was_separator = False
        elif not previous_was_separator:
            pieces.append("_")
            previous_was_separator = True
    slug = "".join(pieces).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"entity_{slug}" if slug else "entity"
    maximum_base = _MAX_ENTITY_ID_CHARS - _MAX_ENTITY_ID_COLLISION_SUFFIX_CHARS
    if len(slug) <= maximum_base:
        return slug
    digest = hashlib.sha256(slug.encode("ascii")).hexdigest()[:_HASH_SUFFIX_CHARS]
    prefix_length = maximum_base - _HASH_SUFFIX_CHARS - 1
    prefix = slug[:prefix_length].rstrip("_")
    return f"{prefix}_{digest}"


def normalize_entities(
    candidates: Sequence[EntityCandidate], *, limit: int = 16
) -> NormalizedEntities:
    """Deduplicate and order explicit candidates without deriving new entities."""
    if type(candidates) not in {list, tuple}:
        raise ValueError("entity candidate container must be a plain list or tuple")
    if len(candidates) > _MAX_RAW_ENTITY_CANDIDATES:
        raise ValueError("entity candidate container exceeds the raw Pass-A cap")
    if type(limit) is not int or not 0 < limit <= _MAX_NORMALIZED_ENTITIES:
        raise ValueError(
            f"limit must be a positive integer no greater than {_MAX_NORMALIZED_ENTITIES}"
        )

    for candidate in candidates:
        if type(candidate) is not EntityCandidate:
            raise ValueError("entity candidate must be a validated EntityCandidate")
        if (
            type(candidate.name) is not str
            or len(candidate.name) > _MAX_ENTITY_NAME_CHARS
        ):
            raise ValueError("entity candidate name exceeds its resource bound")
        if (
            type(candidate.aliases) is not tuple
            or len(candidate.aliases) > _MAX_ENTITY_ALIASES
        ):
            raise ValueError("entity candidate aliases exceed their resource bound")
        if any(
            type(alias) is not str or len(alias) > _MAX_ENTITY_ALIAS_CHARS
            for alias in candidate.aliases
        ):
            raise ValueError("entity candidate alias exceeds its resource bound")
        if type(candidate.role) is not EntityRole:
            raise ValueError("entity candidate role must be canonical")

    merged: dict[str, _MergedEntity] = {}
    for position, candidate in enumerate(candidates):
        canonical_key = _normalize_text(candidate.name)
        if canonical_key in {"", "unknown"}:
            continue

        record = merged.setdefault(
            canonical_key,
            _MergedEntity(
                canonical_label=_truncate_with_hash(
                    canonical_key, _MAX_ENTITY_NAME_CHARS
                ),
                role=candidate.role,
                position=position,
            ),
        )
        if _ROLE_PRIORITY[candidate.role] < _ROLE_PRIORITY[record.role]:
            record.role = candidate.role

        for alias in candidate.aliases:
            alias_key = _normalize_text(alias)
            if alias_key in {"", "unknown", canonical_key}:
                continue
            record.aliases.setdefault(
                alias_key,
                _truncate_with_hash(alias_key, _MAX_ENTITY_ALIAS_CHARS),
            )

    ordered = sorted(
        merged.values(),
        key=lambda record: (_ROLE_PRIORITY[record.role], record.position),
    )
    omitted_count = max(0, len(ordered) - limit)
    alias_omitted_count = 0
    used_ids: set[str] = set()
    next_occurrence: dict[str, int] = {}
    prompts: list[EntityPrompt] = []
    for record in ordered[:limit]:
        base_id = _ascii_slug(record.canonical_label)
        occurrence = next_occurrence.get(base_id, 1)
        entity_id = base_id if occurrence == 1 else f"{base_id}_{occurrence}"
        while entity_id in used_ids:
            occurrence += 1
            entity_id = f"{base_id}_{occurrence}"
        used_ids.add(entity_id)
        next_occurrence[base_id] = occurrence + 1
        aliases_over_limit = len(record.aliases) > _MAX_ENTITY_ALIASES
        alias_items = (
            sorted(record.aliases.items(), key=lambda item: (item[1], item[0]))
            if aliases_over_limit
            else record.aliases.items()
        )
        aliases = tuple(
            value for _, value in tuple(alias_items)[:_MAX_ENTITY_ALIASES]
        )
        alias_omitted_count += max(0, len(record.aliases) - len(aliases))
        prompts.append(
            EntityPrompt(
                entity_id=entity_id,
                canonical_label=record.canonical_label,
                aliases=aliases,
                role=record.role,
            )
        )

    warnings = (
        (f"{omitted_count} entity candidate omitted by limit {limit}",)
        if omitted_count == 1
        else (f"{omitted_count} entity candidates omitted by limit {limit}",)
        if omitted_count
        else ()
    )
    return NormalizedEntities(
        entities=tuple(prompts),
        omitted_count=omitted_count,
        alias_omitted_count=alias_omitted_count,
        alias_warning=(
            "ENTITY_ALIASES_TRUNCATED" if alias_omitted_count else None
        ),
        warnings=warnings,
    )

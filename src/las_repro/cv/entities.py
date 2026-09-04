"""Deterministic, non-semantic normalization for CV entity prompts."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Annotated

from pydantic import Field, StrictStr, field_validator

from .contracts import EntityPrompt, EntityRole, StrictModel


_MAX_RAW_ENTITY_CANDIDATES = 64
_MAX_NORMALIZED_ENTITIES = 16
_MAX_ENTITY_NAME_CHARS = 256
_MAX_ENTITY_ALIASES = 256
_MAX_ENTITY_ALIAS_CHARS = 128
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
        return EntityRole(value)


class NormalizedEntities(StrictModel):
    entities: Annotated[
        tuple[EntityPrompt, ...], Field(max_length=_MAX_NORMALIZED_ENTITIES)
    ]
    omitted_count: int = Field(ge=0, le=_MAX_CANONICAL_INT, strict=True)
    warnings: Annotated[
        tuple[Annotated[StrictStr, Field(max_length=_MAX_WARNING_CHARS)], ...],
        Field(max_length=_MAX_NORMALIZATION_WARNINGS),
    ] = ()


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
    return slug


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

    merged: dict[str, dict[str, object]] = {}
    for position, candidate in enumerate(candidates):
        canonical_label = _normalize_text(candidate.name)
        if canonical_label in {"", "unknown"}:
            continue

        record = merged.setdefault(
            canonical_label,
            {
                "canonical_label": canonical_label,
                "role": candidate.role,
                "position": position,
                "aliases": [],
                "alias_keys": set(),
            },
        )
        if _ROLE_PRIORITY[candidate.role] < _ROLE_PRIORITY[record["role"]]:  # type: ignore[index]
            record["role"] = candidate.role

        aliases = record["aliases"]  # type: ignore[assignment]
        alias_keys = record["alias_keys"]  # type: ignore[assignment]
        for alias in candidate.aliases:
            normalized_alias = _normalize_text(alias)
            if normalized_alias in {"", "unknown", canonical_label} or normalized_alias in alias_keys:
                continue
            aliases.append(normalized_alias)
            alias_keys.add(normalized_alias)

    ordered = sorted(
        merged.values(),
        key=lambda record: (_ROLE_PRIORITY[record["role"]], record["position"]),  # type: ignore[index]
    )
    omitted_count = max(0, len(ordered) - limit)
    used_ids: set[str] = set()
    next_occurrence: dict[str, int] = {}
    prompts: list[EntityPrompt] = []
    for record in ordered[:limit]:
        base_id = _ascii_slug(record["canonical_label"])  # type: ignore[arg-type]
        occurrence = next_occurrence.get(base_id, 1)
        entity_id = base_id if occurrence == 1 else f"{base_id}_{occurrence}"
        while entity_id in used_ids:
            occurrence += 1
            entity_id = f"{base_id}_{occurrence}"
        used_ids.add(entity_id)
        next_occurrence[base_id] = occurrence + 1
        prompts.append(
            EntityPrompt(
                entity_id=entity_id,
                canonical_label=record["canonical_label"],  # type: ignore[arg-type]
                aliases=tuple(record["aliases"]),  # type: ignore[arg-type]
                role=record["role"],  # type: ignore[arg-type]
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
        entities=tuple(prompts), omitted_count=omitted_count, warnings=warnings
    )

from __future__ import annotations

import pytest

from las_repro.cv.contracts import EntityRole
from las_repro.cv.entities import EntityCandidate, normalize_entities


def test_normalize_entities_deduplicates_and_applies_role_priority():
    """Dropping role ordering or canonical deduplication would alter stable prompts."""
    normalized = normalize_entities(
        [
            EntityCandidate(name="Cup", aliases=("mug",), role=EntityRole.MANIPULATED_OBJECT),
            EntityCandidate(name=" cup ", aliases=("vessel",), role=EntityRole.OTHER),
            EntityCandidate(name="right hand", aliases=(), role=EntityRole.ACTOR),
        ],
        limit=2,
    )

    assert [item.entity_id for item in normalized.entities] == ["right_hand", "cup"]
    assert normalized.entities[1].aliases == ("mug", "vessel")
    assert normalized.omitted_count == 0


def test_normalize_entities_casefolds_unicode_and_uses_ascii_slug_collisions():
    """Removing Unicode casefolding or collision suffixes would lose distinct prompts."""
    normalized = normalize_entities(
        [
            EntityCandidate(name="Caf\u00e9", aliases=(), role=EntityRole.OTHER),
            EntityCandidate(name="CAF\u00c9", aliases=("coffee shop",), role=EntityRole.OTHER),
            EntityCandidate(name="cafe", aliases=(), role=EntityRole.OTHER),
        ]
    )

    assert [(item.entity_id, item.canonical_label, item.aliases) for item in normalized.entities] == [
        ("cafe", "caf\u00e9", ("coffee shop",)),
        ("cafe_2", "cafe", ()),
    ]


def test_normalize_entities_discards_blank_or_unknown_candidates_without_mutating_input():
    """Treating unknown labels as prompts would fabricate unsupported entities."""
    aliases = ["vessel", "cup"]
    candidates = [
        EntityCandidate(name="  ", aliases=(), role=EntityRole.OTHER),
        EntityCandidate(name="UNKNOWN", aliases=(), role=EntityRole.OTHER),
        EntityCandidate(name="Cup", aliases=tuple(aliases), role=EntityRole.MANIPULATED_OBJECT),
    ]

    normalized = normalize_entities(candidates)

    assert normalized.entities[0].canonical_label == "cup"
    assert normalized.entities[0].aliases == ("vessel",)
    assert aliases == ["vessel", "cup"]
    assert candidates[2].name == "Cup"


def test_normalize_entities_uses_every_declared_role_priority():
    """Changing the declared role ordering must change deterministic prompt order."""
    normalized = normalize_entities(
        [
            EntityCandidate(name="other", aliases=(), role=EntityRole.OTHER),
            EntityCandidate(name="surface", aliases=(), role=EntityRole.SURFACE),
            EntityCandidate(name="occluder", aliases=(), role=EntityRole.OCCLUDER),
            EntityCandidate(name="container", aliases=(), role=EntityRole.CONTAINER),
            EntityCandidate(name="object", aliases=(), role=EntityRole.MANIPULATED_OBJECT),
            EntityCandidate(name="actor", aliases=(), role=EntityRole.ACTOR),
        ]
    )

    assert [item.canonical_label for item in normalized.entities] == [
        "actor",
        "object",
        "container",
        "occluder",
        "surface",
        "other",
    ]


def test_normalize_entities_keeps_sixteen_by_default_and_reports_omission_warning():
    """Removing the cap would allow an unbounded prompt list into CV inference."""
    normalized = normalize_entities(
        [EntityCandidate(name=f"object {index}", aliases=(), role=EntityRole.OTHER) for index in range(17)]
    )

    assert len(normalized.entities) == 16
    assert normalized.omitted_count == 1
    assert normalized.warnings == ("1 entity candidate omitted by limit 16",)

from __future__ import annotations

import pytest
from pydantic import ValidationError

from las_repro.cv.contracts import EntityPrompt, EntityRole
from las_repro.cv.entities import EntityCandidate, NormalizedEntities, normalize_entities


class BombTuple(tuple):
    """A hostile tuple subclass that must be rejected before traversal."""

    def __iter__(self):
        raise RuntimeError("hostile tuple was iterated")


class BombList(list):
    """A hostile oversized JSON array that must be rejected before traversal."""

    def __iter__(self):
        raise RuntimeError("hostile list was iterated")


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


def test_normalize_entities_allocates_ids_globally_across_slug_bases():
    """Per-base suffix counters would emit duplicate prompt IDs across labels."""
    normalized = normalize_entities(
        [
            EntityCandidate(name="foo", aliases=(), role=EntityRole.OTHER),
            EntityCandidate(name="foo 2", aliases=(), role=EntityRole.OTHER),
            EntityCandidate(name="f\u00f3o", aliases=(), role=EntityRole.OTHER),
        ]
    )

    assert [item.entity_id for item in normalized.entities] == ["foo", "foo_2", "foo_3"]
    assert len({item.entity_id for item in normalized.entities}) == 3


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


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "x" * 1_000_000, "aliases": [], "role": "other"},
        {"name": "item", "aliases": ["x" * 1_000_000], "role": "other"},
        {"name": "item", "aliases": ["alias"] * 257, "role": "other"},
    ],
)
def test_entity_candidate_enforces_text_and_alias_resource_caps(payload) -> None:
    """Pass-A entity text cannot create unbounded downstream prompt state."""
    with pytest.raises(ValidationError):
        EntityCandidate.model_validate(payload)


def test_normalized_entities_enforces_container_text_and_integer_caps() -> None:
    """Direct normalized payloads share the finite canonical JSON envelope."""
    entity = EntityPrompt(
        entity_id="item",
        canonical_label="item",
        aliases=(),
        role=EntityRole.OTHER,
    )
    payloads = (
        {
            "entities": BombList([entity] * 65),
            "omitted_count": 0,
            "warnings": [],
        },
        {
            "entities": [],
            "omitted_count": 0,
            "warnings": BombList(["warning"] * 65),
        },
        {"entities": [], "omitted_count": 0, "warnings": ["x" * 1_000_000]},
        {"entities": [], "omitted_count": 2**100, "warnings": []},
    )

    for payload in payloads:
        with pytest.raises(ValidationError):
            NormalizedEntities.model_validate(payload)


def test_entity_json_length_cap_precedes_hostile_item_iteration() -> None:
    """An oversized alias array is rejected using its size, not its elements."""
    payload = {
        "name": "item",
        "aliases": BombList(["alias"] * 257),
        "role": "other",
    }

    with pytest.raises(ValidationError):
        EntityCandidate.model_validate(payload)


def test_normalize_entities_rejects_custom_candidate_tuple_before_iteration() -> None:
    """The public normalizer accepts only plain bounded JSON-style sequences."""
    candidate = EntityCandidate(name="item", aliases=(), role=EntityRole.OTHER)

    with pytest.raises(ValueError, match="candidate container"):
        normalize_entities(BombTuple((candidate,)))


def test_normalize_entities_preserves_the_pass_a_raw_cap_of_sixty_four() -> None:
    """All 64 schema-valid raw candidates may reach deterministic normalization."""
    candidates = [
        EntityCandidate(
            name=f"object {index}",
            aliases=(),
            role=EntityRole.OTHER,
        )
        for index in range(64)
    ]

    normalized = normalize_entities(candidates)

    assert len(normalized.entities) == 16
    assert normalized.omitted_count == 48

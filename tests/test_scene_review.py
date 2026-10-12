"""Scene review behavior and regression tests."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from percept_harness.cv.summary import summarize_cv_evidence
from percept_harness.pipelines.embodied import PromptRenderer
from percept_harness.pipelines.output_validation import (
    DEFAULT_OUTPUT_SCHEMAS,
    DEFAULT_OUTPUT_SCHEMAS as registry,
)
from percept_harness.pipelines.scene_provenance import scene_spatial_prompt_data
from percept_harness.pipelines.scene_semantics import unavailable_scene_semantics

from test_cv_summary import _artifact, _entity, _observation, _track
from test_runner import silent_video


def source(count=2, frames=31):
    tracks = tuple(
        _track(
            f"item_{i}_track",
            f"item_{i}",
            tuple(_observation(f) for f in range(frames)),
        )
        for i in range(count)
    )
    summary = summarize_cv_evidence(
        _artifact(tracks),
        max_prompt_chars=200000,
        max_relations=128,
        max_observations_per_track=16,
    )
    segments = [
        {
            "segment_index": i,
            "start": float(i),
            "end": float(i + 1),
            "target": f"item {i % count}",
        }
        for i in range(max(1, (frames + 8) // 10))
    ]
    return summary, segments


def options(summary, segments, duration=None):
    prompt = PromptRenderer().scene_semantics(
        segments,
        video_duration=duration or segments[-1]["end"],
        evidence_summary=summary,
    )
    marker = "[SCENE_SPATIAL_PROVENANCE_OPTIONS_JSON]\n"
    assert marker in prompt, "trusted provenance choices must precede inference"
    from percept_harness.pipelines.scene_provenance import scene_spatial_prompt_data
    envelope, _ = scene_spatial_prompt_data(
        summary, segments, duration=duration or segments[-1]["end"])
    compact = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]
    assert compact is None if envelope is None else compact["options"] == [
        {k: o[k] for k in ("option_id", "kind", "object_ids", "object_names", "start", "end")}
        for o in envelope["options"]]
    return envelope, prompt


def scene_from(envelope):
    scene = unavailable_scene_semantics()
    objects = {}
    for option in envelope["options"]:
        objects.update(zip(option["object_ids"], option["object_names"]))
        row = {
            k: copy.deepcopy(v)
            for k, v in option.items()
            if k not in {"option_id", "kind", "object_ids", "object_names"}
        }
        row.update(envelope["flat_provenance"])
        row.update(
            visual_evidence="visible support independently inspected", confidence=0.5
        )
        if option["kind"] == "location":
            row.update(object_id=option["object_ids"][0], location="visible workspace")
            scene["locations"].append(row)
        else:
            row.update(
                subject_object_id=option["object_ids"][0],
                object_object_id=option["object_ids"][1],
                relation="unknown",
            )
            scene["relations"].append(row)
    scene["objects"] = [
        {"object_id": k, "name": v, "description": "visible entity"}
        for k, v in objects.items()
    ]
    return scene


def context(summary, segments):
    return {
        "duration": segments[-1]["end"],
        "require_observed_content": False,
        "required_object_ids": [],
        "evidence_summary": summary.model_dump(mode="json"),
        "segments": segments,
    }


def test_renderer_supplies_valid_exact_options_and_local_windows():
    summary, segments = source()
    envelope, _ = options(summary, segments)
    assert {o["kind"] for o in envelope["options"]} == {"location", "relation"}
    assert any(o["end"] - o["start"] >= 0.8 for o in envelope["options"])
    assert max(o["end"] for o in envelope["options"]) == 3.0
    assert all(
        o["object_ids"][0] != o["object_ids"][1]
        for o in envelope["options"]
        if o["kind"] == "relation"
    )
    scene = scene_from(envelope)
    assert (
        DEFAULT_OUTPUT_SCHEMAS.sanitize(
            "SceneSemantics", scene, context(summary, segments)
        )
        == scene
    )
    for option in envelope["options"]:
        fields = {k: v for k, v in option.items() if k != "option_id"}
        encoded = json.dumps(
            fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        assert option["option_id"] == "spv_" + hashlib.sha256(encoded).hexdigest()
        assert (
            not {"location", "relation", "confidence", "visual_evidence"}
            & option.keys()
        )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("start", 0.015, "TIME_NOT_OBSERVED"),
        ("source_keyframe_ids", ["item_0_track_frame_0"], "KEYFRAMES_INVALID"),
        ("source_track_ids", ["invented_track"], "TRACKS_INVALID"),
        ("source_segment_indices", [], "SOURCE_SEGMENTS_INVALID"),
    ],
)
def test_altered_generated_provenance_is_rejected_without_rewriting(field, value, code):
    summary, segments = source()
    envelope, _ = options(summary, segments)
    scene = scene_from(envelope)
    scene["locations"][0][field] = value
    original = copy.deepcopy(scene)
    result = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "SceneSemantics", scene, context(summary, segments)
    )
    assert "SCENE_SPATIAL_" + code in result["_schema_validation"]["issue_codes"]
    assert scene == original


def test_empty_and_last_observation_only_are_safe():
    prompt = PromptRenderer().scene_semantics([], video_duration=1.0)
    marker = "[SCENE_SPATIAL_PROVENANCE_OPTIONS_JSON]\n"
    assert marker in prompt
    envelope = json.JSONDecoder().raw_decode(prompt.split(marker)[1])[0]
    assert envelope["options"] == []
    assert "field_shape_example" not in envelope
    summary, segments = source(frames=1)
    envelope, _ = options(summary, segments)
    assert envelope["options"] == []


def test_budget_is_shared_and_selection_covers_objects_and_time():
    summary, segments = source(count=6, frames=101)
    envelope, prompt = options(summary, segments)
    assert 0 < len(envelope["options"]) <= 64
    assert envelope["options_complete"] is False
    size = len(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode())
    assert size <= 24000
    suffix = json.dumps(summary.prompt_record(), ensure_ascii=False, separators=(",", ":"))
    assert size + len(suffix) <= summary.prompt_char_limit
    assert {f"item_{i}" for i in range(6)} <= {
        x for o in envelope["options"] for x in o["object_ids"]
    }
    assert any(o["start"] >= 7 for o in envelope["options"])
    assert any(3 <= o["start"] <= 6 for o in envelope["options"])
    assert any(o["start"] < 2 for o in envelope["options"])
    assert options(summary, segments)[0] == envelope


def test_general_visibility_and_inventory_rules():
    from importlib import resources

    read = lambda name: (
        resources.files("percept_harness.prompts").joinpath(name + ".txt").read_text()
    )
    scene = read("scene_semantics")
    assert "provenance choices, not facts" in scene
    assert "Selection order is irrelevant" in scene
    occ = read("occlusion_semantics")
    assert "inspection cues, not proof" in occ
    assert "edge_departure" in occ
    assert "tracked portion" in occ
    assert "incorrect proposed" in occ
    inventory = read("embodied_pass_a")
    assert "later entrants" in inventory
    assert "distinct canonical entities" in inventory


def test_no_room_uses_null_without_changing_summary_or_availability():
    summary, segments = source()
    suffix = json.dumps(
        summary.prompt_record(), ensure_ascii=False, separators=(",", ":")
    )
    from percept_harness.cv.summary import _summary_identity

    tight = summary.model_copy(update={"prompt_char_limit": len(suffix)})
    tight = tight.model_copy(update={"summary_id": _summary_identity(tight)})
    envelope, prompt = options(tight, segments)
    assert envelope is None
    from percept_harness.pipelines.scene_choices import prepare_scene_choices
    assert prepare_scene_choices(tight, segments, duration=segments[-1]["end"]).context()["evidence_summary"] == tight.model_dump(mode="json")
    assert '[CV_EVIDENCE_AVAILABILITY_JSON]\n{"available":true}' in prompt
    small = summary.model_copy(update={"prompt_char_limit": len(suffix) + 45})
    small = small.model_copy(update={"summary_id": _summary_identity(small)})
    envelope, _ = options(small, segments)
    assert envelope == {"options": [], "options_complete": False}


def test_options_do_not_depend_on_track_or_entity_input_order():
    tracks = tuple(
        _track(
            f"item_{i}_track", f"item_{i}", tuple(_observation(f) for f in range(21))
        )
        for i in range(3)
    )
    entities = tuple(_entity(f"item_{i}") for i in range(3))
    first = summarize_cv_evidence(_artifact(tracks, entities=entities))
    second = summarize_cv_evidence(
        _artifact(tuple(reversed(tracks)), entities=tuple(reversed(entities)))
    )
    segments = [{"segment_index": 0, "start": 0.0, "end": 2.0, "target": "item 0"}]
    assert options(first, segments)[0] == options(second, segments)[0]


def test_ambiguous_names_and_same_entity_fragments_never_form_distinct_pair():
    tracks = tuple(
        _track(
            f"track_{i}",
            "item" if i < 2 else "other",
            (_observation(0), _observation(10)),
        )
        for i in range(3)
    )
    entities = (
        _entity("item", aliases=("shared",)),
        _entity("other", aliases=("shared",)),
    )
    summary = summarize_cv_evidence(_artifact(tracks, entities=entities))
    segments = [{"segment_index": 0, "start": 0.0, "end": 1.0, "target": "shared"}]
    envelope, _ = options(summary, segments)
    assert all("shared" not in o["object_ids"] for o in envelope["options"])
    assert all(
        len(set(o["object_ids"])) == len(o["object_ids"]) for o in envelope["options"]
    )
    # Each location option chooses just one fragment, never a synthetic entity.
    assert all(
        len(o["source_track_ids"]) == 1
        for o in envelope["options"]
        if o["kind"] == "location"
    )


def test_nonpositive_relation_geometry_gives_locations_only():
    tracks = (
        _track("a_track", "a", (_observation(0), _observation(10))),
        _track(
            "b_track",
            "b",
            (
                _observation(0, bbox_xyxy=(0.6, 0.6, 0.8, 0.8)),
                _observation(10, bbox_xyxy=(0.6, 0.6, 0.8, 0.8)),
            ),
        ),
    )
    summary = summarize_cv_evidence(_artifact(tracks))
    envelope, _ = options(
        summary, [{"segment_index": 0, "start": 0.0, "end": 1.0, "target": "a"}]
    )
    assert envelope["options"]
    assert {o["kind"] for o in envelope["options"]} == {"location"}


def test_segment_boundary_snaps_inward_and_last_witness_is_not_extended():
    summary, _ = source(frames=11)
    segments = [
        {"segment_index": 0, "start": 0.0, "end": 0.55, "target": "item 0"},
        {"segment_index": 1, "start": 0.55, "end": 1.03, "target": "item 1"},
    ]
    envelope, _ = options(summary, segments)
    assert any((o["start"], o["end"]) == (0.0, 0.5) for o in envelope["options"])
    assert any((o["start"], o["end"]) == (0.6, 1.0) for o in envelope["options"])
    assert all(o["end"] != 1.03 for o in envelope["options"])
    assert all(
        o["source_segment_indices"]
        == [
            s["segment_index"]
            for s in segments
            if s["start"] < o["end"] and s["end"] > o["start"]
        ]
        for o in envelope["options"]
    )


def test_retained_overlay_stems_are_copied_exactly():
    from percept_harness.cv.contracts import ArtifactFile, OverlayRecord

    path = "overlays/retained_visual_001.png"
    summary = summarize_cv_evidence(
        _artifact(
            (_track("a_track", "a", (_observation(0), _observation(10))),),
            files=(ArtifactFile(path=path, sha256="a" * 64, size_bytes=1),),
            overlay_records=(
                OverlayRecord(path=path, track_id="a_track", frame_index=0),
            ),
        )
    )
    envelope, _ = options(
        summary, [{"segment_index": 0, "start": 0.0, "end": 1.0, "target": "a"}]
    )
    assert envelope["options"][0]["source_keyframe_ids"] == ["retained_visual_001"]


def test_unicode_budget_and_oversized_segment_list_omit_whole_options():
    from percept_harness.cv.contracts import EntityPrompt

    entities = tuple(
        EntityPrompt(
            entity_id=f"item_{i}",
            canonical_label="物体" + str(i),
            aliases=(),
            role="other",
        )
        for i in range(4)
    )
    tracks = tuple(
        _track(
            f"item_{i}_track", f"item_{i}", tuple(_observation(f) for f in range(31))
        )
        for i in range(4)
    )
    summary = summarize_cv_evidence(_artifact(tracks, entities=entities))
    segments = [
        {
            "segment_index": i,
            "start": float(i),
            "end": float(i + 1),
            "target": "物体" + str(i),
        }
        for i in range(3)
    ]
    envelope, _ = options(summary, segments)
    assert (
        len(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode())
        <= 24000
    )
    assert DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "SceneSemantics", scene_from(envelope), context(summary, segments)
    ) == scene_from(envelope)
    overlapping = [
        {"segment_index": i, "start": 0.0, "end": 3.0, "target": "物体0"}
        for i in range(10000)
    ]
    envelope, _ = options(summary, overlapping)
    assert envelope["options"] == []
    assert envelope["options_complete"] is False


def test_hostile_summary_and_segment_bounds_fail_before_iteration():
    from test_cv_summary import BombList

    from percept_harness.pipelines.scene_provenance import scene_spatial_prompt_data

    summary, segments = source()
    with pytest.raises(ValueError):
        scene_spatial_prompt_data(summary, BombList([{}] * 10001), duration=3.0)
    hostile = summary.model_copy(update={"tracks": BombList([None] * 10000)})
    with pytest.raises(ValueError):
        options(hostile, segments)


def test_generated_options_validate_source_only_once(monkeypatch):
    from percept_harness.cv.summary import CvEvidenceSummary

    summary, segments = source()
    original = CvEvidenceSummary.prompt_record
    calls = []

    def counted(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(CvEvidenceSummary, "prompt_record", counted)
    from percept_harness.pipelines.scene_provenance import scene_spatial_prompt_data
    envelope, _ = scene_spatial_prompt_data(summary, segments, duration=segments[-1]["end"])
    assert len(envelope["options"]) > 1
    assert len(calls) == 1


def test_hash_collisions_are_rejected_even_for_otherwise_valid_options(monkeypatch):
    from types import SimpleNamespace

    import percept_harness.pipelines.scene_provenance as module

    summary, segments = source()
    monkeypatch.setattr(
        module,
        "hashlib",
        SimpleNamespace(sha256=lambda _: SimpleNamespace(hexdigest=lambda: "0" * 64)),
    )
    with pytest.raises(ValueError, match="identity collision"):
        options(summary, segments)


def test_empty_available_summary_and_count_limit():
    summary = summarize_cv_evidence(_artifact(()))
    envelope, _ = options(
        summary, [{"segment_index": 0, "start": 0.0, "end": 1.0, "target": "unknown"}]
    )
    assert envelope == {"options": [], "options_complete": True}
    summary, segments = source(count=16, frames=11)
    envelope, _ = options(summary, segments)
    assert len(envelope["options"]) <= 64
    assert envelope["options_complete"] is False


def test_unusable_optional_entity_names_do_not_break_scene_rendering():
    from percept_harness.cv.contracts import EntityPrompt

    summary = summarize_cv_evidence(
        _artifact(
            (_track("a_track", "a", (_observation(0), _observation(10))),),
            entities=(
                EntityPrompt(
                    entity_id="a",
                    canonical_label="unsafe/name",
                    aliases=(),
                    role="other",
                ),
            ),
        )
    )
    envelope, _ = options(
        summary, [{"segment_index": 0, "start": 0.0, "end": 1.0, "target": "unknown"}]
    )
    assert envelope["options"] == []


def test_returned_choices_do_not_mutate_future_provenance_constants():
    from percept_harness.pipelines.scene_provenance import scene_spatial_prompt_data

    summary, segments = source()
    original, _ = scene_spatial_prompt_data(summary, segments, duration=3.0)
    expected = copy.deepcopy(original)
    original["flat_provenance"]["repair_history"].append("repair")
    original["field_shape_example"]["repair_history"].append("repair")
    repeated, _ = scene_spatial_prompt_data(summary, segments, duration=3.0)
    assert repeated == expected


def test_output_prohibited_overlay_stem_is_not_offered_as_a_copyable_value():
    from percept_harness.cv.contracts import ArtifactFile, OverlayRecord

    path = "overlays/retained.npy.png"
    summary = summarize_cv_evidence(
        _artifact(
            (_track("a_track", "a", (_observation(0), _observation(10))),),
            files=(ArtifactFile(path=path, sha256="a" * 64, size_bytes=1),),
            overlay_records=(
                OverlayRecord(path=path, track_id="a_track", frame_index=0),
            ),
        )
    )
    envelope, _ = options(
        summary, [{"segment_index": 0, "start": 0.0, "end": 1.0, "target": "a"}]
    )
    assert envelope["options"]
    assert all(option["source_keyframe_ids"] == [] for option in envelope["options"])


@pytest.mark.parametrize("count", [48, 64])
def test_selection_covers_every_eligible_object_before_repeating_lexical_hubs(count):
    summary, segments = source(count=count, frames=2)
    envelope, prompt = options(summary, segments)
    expected = {entity.entity_id for entity in summary.entities}
    assert len(expected) == count
    represented = {
        object_id
        for option in envelope["options"]
        for object_id in option["object_ids"]
    }
    assert represented == expected
    assert len(envelope["options"]) <= 64
    assert {option["kind"] for option in envelope["options"]} == {
        "location",
        "relation",
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(encoded) <= 24000
    suffix = json.dumps(summary.prompt_record(), ensure_ascii=False, separators=(",", ":"))
    assert len(encoded) + len(suffix) <= summary.prompt_char_limit
    assert envelope["options_complete"] is False
    assert options(summary, segments)[0] == envelope
    scene = scene_from(envelope)
    assert (
        DEFAULT_OUTPUT_SCHEMAS.sanitize(
            "SceneSemantics", scene, context(summary, segments)
        )
        == scene
    )

    tracks = tuple(
        _track(f"item_{i}_track", f"item_{i}", (_observation(0), _observation(1)))
        for i in reversed(range(count))
    )
    reverse_summary = summarize_cv_evidence(
        _artifact(tracks),
        max_prompt_chars=200000,
        max_relations=128,
        max_observations_per_track=16,
    )
    assert options(reverse_summary, segments)[0] == envelope


def test_unrepresented_object_gets_a_later_fitting_window_before_repeats():
    tracks = (
        _track("a_track", "a", tuple(_observation(f) for f in (0, 6, 7))),
    ) + tuple(
        _track(
            f"item_{i}_track",
            f"item_{i}",
            tuple(_observation(f, bbox_xyxy=(0.6, 0.6, 0.8, 0.8)) for f in (6, 7)),
        )
        for i in range(63)
    )
    summary = summarize_cv_evidence(_artifact(tracks), max_relations=128)
    segments = [
        {"segment_index": i, "start": 0.0, "end": 0.5, "target": "unknown"}
        for i in range(5000)
    ]
    segments.append(
        {"segment_index": 5000, "start": 0.5, "end": 1.0, "target": "unknown"}
    )
    envelope, _ = options(summary, segments)
    assert len(envelope["options"]) <= 64
    assert {entity.entity_id for entity in summary.entities} == {
        obj for option in envelope["options"] for obj in option["object_ids"]
    }
    assert any(
        option["object_ids"] == ["a"]
        and option["start"] >= 0.5
        and option["source_segment_indices"] == [5000]
        for option in envelope["options"]
    )


def fixture():
    from percept_harness.pipelines.scene_choices import prepare_scene_choices
    summary, segments = source()
    package = prepare_scene_choices(summary, segments, duration=segments[-1]['end'])
    ctx = package.context()
    envelope = ctx['spatial_options']
    scene = scene_from(envelope)
    # The target skeleton requires one independently visible semantic event.
    scene['semantic_events'] = [dict(event_index=0, start=0.01, end=0.99,
        event_type='move', actor='unknown', target_object_id='item_0',
        description='item moves', confidence=0.7)]
    draft = copy.deepcopy(scene)
    draft['locations'] = [dict(option_id=o['option_id'], location='visible workspace',
        visual_evidence='visible support', confidence=0.6)
        for o in envelope['options'] if o['kind'] == 'location']
    draft['relations'] = [dict(option_id=o['option_id'], direction=d, relation='near',
        visual_evidence='visible pair', confidence=0.5)
        for o in envelope['options'] if o['kind'] == 'relation' for d in ('forward', 'reverse')]
    return draft, ctx


def test_legacy_contacting_has_closed_enum_code():
    summary, segments = source()
    envelope, _ = scene_spatial_prompt_data(summary, segments, duration=segments[-1]['end'])
    scene = scene_from(envelope)
    scene['relations'][0]['relation'] = 'contacting'
    failure = registry.sanitize('SceneSemantics', scene, context(summary, segments))
    assert failure['_schema_validation']['issue_codes'] == ['SCENE_SEMANTICS_ENUM_VALUE']


@pytest.mark.parametrize('event_type', ['out_of_frame', 'enter_frame'])
def test_frame_visibility_survives_model_validation_and_public_projection(event_type):
    from percept_harness.pipelines.scene_choices import project_scene_choices
    from percept_harness.pipelines.scene_semantics import SceneSemantics
    from percept_harness.pipelines.validators import Skill

    draft, ctx = fixture()
    draft['semantic_events'][0].update(
        event_type=event_type,
        description='item crosses the visible frame edge',
    )
    accepted = registry.sanitize('SceneSemanticsChoices', draft, ctx)
    assert accepted == draft
    public = project_scene_choices(accepted, ctx)
    assert public['semantic_events'] == draft['semantic_events']
    assert SceneSemantics.model_validate(public).model_dump(mode='json')['semantic_events'] == draft['semantic_events']
    # A scene visibility observation must not expand the fine-action taxonomy.
    with pytest.raises(ValueError):
        Skill(event_type)


def test_projection_preserves_nonspatial_and_exact_sources_and_direction():
    from percept_harness.pipelines.scene_choices import project_scene_choices
    draft, ctx = fixture()
    draft['locations'].reverse()
    draft['relations'].reverse()
    accepted = registry.sanitize('SceneSemanticsChoices', draft, ctx)
    assert accepted == draft
    assert registry.sanitize('SceneSemanticsChoices', accepted, ctx) == accepted
    public = project_scene_choices(accepted, ctx, repair_history=('initial', 'repair'))
    for key in ('objects', 'initial_state', 'final_state', 'outcome', 'semantic_events'):
        assert public[key] == draft[key]
    for collection in ('locations', 'relations'):
        for row in public[collection]:
            assert row['repair_history'] == ['initial', 'repair']
            assert 'option_id' not in row
            match = [o for o in ctx['spatial_options']['options'] if
                     o['start'] == row['start'] and o['end'] == row['end'] and
                     o['source_track_ids'] == row['source_track_ids']]
            assert match
            for key in ('source_keyframe_ids', 'source_segment_indices'):
                assert row[key] == match[0][key]
    pair = public['relations'][:2]
    assert pair[0]['subject_object_id'] == pair[1]['object_object_id']


@pytest.mark.parametrize('mutation,code', [
    ('unknown', 'OPTION_UNKNOWN'), ('kind', 'OPTION_KIND'),
    ('duplicate', 'DUPLICATE'), ('binding', 'OBJECT_BINDING'),
    ('undeclared', 'OBJECT_BINDING'), ('direction', 'RELATION_DIRECTION_ENUM_VALUE'),
    ('relation', 'RELATION_PREDICATE_ENUM_VALUE'), ('extra', 'EXTRA_FIELD'),
    ('history', 'EXTRA_FIELD'), ('artifact', 'PROHIBITED_CONTENT'),
])
def test_invalid_choices_fail_closed(mutation, code):
    draft, ctx = fixture()
    if mutation == 'unknown': draft['locations'][0]['option_id'] = 'spv_stale'
    if mutation == 'kind': draft['locations'][0]['option_id'] = draft['relations'][0]['option_id']
    if mutation == 'duplicate': draft['locations'].append(copy.deepcopy(draft['locations'][0]))
    if mutation == 'binding': draft['objects'][0]['name'] = 'wrong name'
    if mutation == 'undeclared': draft['objects'].pop(0)
    if mutation == 'direction': draft['relations'][0]['direction'] = 'backward'
    if mutation == 'relation': draft['relations'][0]['relation'] = 'contacting'
    if mutation == 'extra': draft['locations'][0]['start'] = 0.0
    if mutation == 'history': draft['repair_history'] = ['initial']
    if mutation == 'artifact': draft['locations'][0]['location'] = '/tmp/mask.npy'
    result = registry.sanitize('SceneSemanticsChoices', draft, ctx)
    assert 'SCENE_SEMANTICS_CHOICES_' + code in result['_schema_validation']['issue_codes']


@pytest.mark.parametrize('mutation', ['offer', 'digest', 'source', 'skeleton', 'version'])
def test_forged_context_is_internal_failure(mutation):
    draft, ctx = fixture()
    if mutation == 'offer': ctx['spatial_options']['options'][0]['end'] += 0.01
    if mutation == 'digest': ctx['spatial_options_sha256'] = '0' * 64
    if mutation == 'source': ctx['evidence_summary']['summary_id'] = 'forged'
    if mutation == 'skeleton': ctx['known_targets'][0]['name'] = 'forged'
    if mutation == 'version': ctx['choice_contract'] = 'old'
    with pytest.raises(ValueError): registry.sanitize('SceneSemanticsChoices', draft, ctx)


def test_no_cv_preserves_supported_nonspatial_and_rejects_fake_envelopes():
    from percept_harness.pipelines.scene_choices import prepare_scene_choices, project_scene_choices
    draft, _ = fixture()
    _, segments = source()
    ctx = prepare_scene_choices(None, segments, duration=segments[-1]['end']).context()
    draft['locations'] = []; draft['relations'] = []
    assert project_scene_choices(draft, ctx) == draft
    for value in ({'_schema_validation': {'status': 'normalized'}, 'data': draft},
                  {'projected': draft}):
        assert '_schema_validation' in registry.sanitize('SceneSemanticsChoices', value, ctx)


def test_compiled_schema_separates_kind_enums_and_empty_branches():
    from percept_harness.pipelines.scene_choices import prepare_scene_choices
    _, ctx = fixture()
    contract = registry.model_response_contract('SceneSemanticsChoices', ctx)
    schema = json.loads(contract.schema_json)
    assert schema['additionalProperties'] is False
    for field, kind in [('locations', 'location'), ('relations', 'relation')]:
        assert schema['properties'][field]['items']['properties']['option_id']['enum'] == [
            o['option_id'] for o in ctx['spatial_options']['options'] if o['kind'] == kind]
    empty = prepare_scene_choices(None, [], duration=1.0).context()
    schema = json.loads(registry.model_response_contract('SceneSemanticsChoices', empty).schema_json)
    assert schema['properties']['locations']['maxItems'] == 0
    assert schema['properties']['relations']['maxItems'] == 0
    assert registry.model_response_contract('SceneSemantics', {}) is None


def test_selection_order_is_irrelevant_even_for_distinct_predicates_on_one_pair():
    from percept_harness.pipelines.scene_choices import project_scene_choices
    draft, ctx = fixture()
    draft['relations'].append(dict(draft['relations'][0], relation='on'))
    reversed_draft = copy.deepcopy(draft)
    reversed_draft['locations'].reverse(); reversed_draft['relations'].reverse()
    assert project_scene_choices(draft, ctx) == project_scene_choices(reversed_draft, ctx)


@pytest.mark.parametrize('branch', ['empty_choices', 'empty_offers', 'null_offers'])
def test_available_cv_fallbacks_preserve_nonspatial(branch):
    from percept_harness.cv.summary import _summary_identity
    from percept_harness.pipelines.scene_choices import prepare_scene_choices, project_scene_choices
    draft, _ = fixture()
    summary, segments = source(frames=1 if branch == 'empty_offers' else 31)
    if branch == 'null_offers':
        size = len(json.dumps(summary.prompt_record(), ensure_ascii=False, separators=(',', ':')))
        summary = summary.model_copy(update={'prompt_char_limit': size})
        summary = summary.model_copy(update={'summary_id': _summary_identity(summary)})
    ctx = prepare_scene_choices(summary, segments, duration=segments[-1]['end']).context()
    draft['locations'] = []; draft['relations'] = []
    public = project_scene_choices(draft, ctx)
    assert public == draft
    assert ctx['evidence_summary']['status'] == 'available'
    if branch == 'null_offers': assert ctx['spatial_options'] is None
    if branch == 'empty_offers': assert ctx['spatial_options']['options'] == []


def test_combined_selection_cap_is_local_and_duplicates_are_not_collapsed():
    draft, ctx = fixture()
    # All individually legal choices are distinct, but the combined count is 65.
    predicates = ['left_of', 'right_of', 'above', 'below', 'inside', 'on',
                  'overlapping', 'near', 'occluding', 'unknown']
    draft['relations'] = [dict(r, relation=p) for r in draft['relations'] for p in predicates][:64]
    draft['locations'] = draft['locations'][:1]
    result = registry.sanitize('SceneSemanticsChoices', draft, ctx)
    assert result['_schema_validation']['issue_codes'] == ['SCENE_SEMANTICS_CHOICES_SELECTION_LIMIT']


@pytest.mark.parametrize('attack', ['bytes', 'nodes', 'depth', 'external_ref'])
def test_schema_compilation_rejects_unbounded_or_external_structures(monkeypatch, attack):
    from percept_harness.pipelines.scene_choices import SceneSemanticsChoices, prepare_scene_choices
    schema = {'type': 'object'}
    if attack == 'bytes': schema['title'] = 'a' * 65537
    if attack == 'nodes': schema['required'] = ['x'] * 5000
    if attack == 'depth':
        for _ in range(33): schema = {'items': schema}
    if attack == 'external_ref': schema = {'$ref': 'https://example.invalid/schema'}
    monkeypatch.setattr(SceneSemanticsChoices, 'model_json_schema', lambda: schema)
    with pytest.raises(ValueError):
        registry.model_response_contract('SceneSemanticsChoices', prepare_scene_choices(None, [], duration=1.0).context())


def test_source_preflight_rejects_nonplain_containers_before_traversal():
    from percept_harness.pipelines.scene_choices import authenticate_scene_context
    class Hostile(dict):
        def items(self): raise AssertionError('unbounded traversal')
    with pytest.raises(ValueError): authenticate_scene_context(Hostile())


ENUM_CASES = [
    ('semantic_events', 'event_type', 'hold', 'EVENT_TYPE'),
    ('semantic_events', 'actor', 'private actor value', 'ACTOR'),
    ('outcome', 'status', 'private outcome value', 'OUTCOME_STATUS'),
    ('relations', 'direction', 'unknown', 'RELATION_DIRECTION'),
    ('relations', 'relation', 'contacting', 'RELATION_PREDICATE'),
]


@pytest.mark.parametrize('collection,field,value,suffix', ENUM_CASES)
def test_scene_choice_enum_feedback_is_field_specific_and_model_free(collection, field, value, suffix):
    draft, ctx = fixture()
    # Reproduce an enum error at event index 2, without exposing its index.
    if collection == 'semantic_events':
        draft[collection] *= 3
        draft[collection] = [dict(row, event_index=i) for i, row in enumerate(draft[collection])]
    row = draft[collection] if collection == 'outcome' else draft[collection][-1]
    row[field] = value
    original = copy.deepcopy(draft)
    failure = registry.sanitize('SceneSemanticsChoices', draft, ctx)
    code = 'SCENE_SEMANTICS_CHOICES_' + suffix + '_ENUM_VALUE'
    assert failure == {'_schema_validation': {'schema_name': 'SceneSemanticsChoices',
                                             'status': 'invalid', 'issue_codes': [code]}}
    assert draft == original
    assert registry.failure_codes('SceneSemanticsChoices', failure) == (code,)


def test_scene_choice_enum_feedback_deduplicates_in_closed_domain_order():
    draft, ctx = fixture()
    for collection, field, value, _ in reversed(ENUM_CASES):
        rows = [draft[collection]] if collection == 'outcome' else draft[collection]
        for row in rows:
            row[field] = value
    failure = registry.sanitize('SceneSemanticsChoices', draft, ctx)
    assert failure['_schema_validation']['issue_codes'] == [
        'SCENE_SEMANTICS_CHOICES_EVENT_TYPE_ENUM_VALUE',
        'SCENE_SEMANTICS_CHOICES_ACTOR_ENUM_VALUE',
        'SCENE_SEMANTICS_CHOICES_OUTCOME_STATUS_ENUM_VALUE',
        'SCENE_SEMANTICS_CHOICES_RELATION_DIRECTION_ENUM_VALUE',
        'SCENE_SEMANTICS_CHOICES_RELATION_PREDICATE_ENUM_VALUE',
    ]


@pytest.mark.parametrize('kind,path', [
    ('enum', ('private field', 2, 'event_type')),
    ('enum', ('semantic_events', 'private index', 'event_type')),
    ('enum', ('semantic_events', -1, 'event_type')),
    ('enum', ('semantic_events', 2, 'event_type', 'extra')),
    ('enum', ('event_type',)),
    ('enum', ('outcome', 0, 'status')),
    ('literal_error', ('semantic_events', 2, 'event_type')),
    ('enum', ('relations', 0, 'direction')),
])
def test_scene_choice_enum_unexpected_paths_or_kinds_remain_generic(kind, path):
    from pydantic import ValidationError
    from percept_harness.pipelines.output_validation import _scene_choice_pydantic_issue_codes
    error = ValidationError.from_exception_data('synthetic', [
        dict(type=kind, loc=path, input='private raw value', ctx={'expected': 'private message'})])
    assert _scene_choice_pydantic_issue_codes(error) == ('SCENE_SEMANTICS_CHOICES_ENUM_VALUE',)


def test_scene_unknown_requires_explicit_model_value_and_retains_supported_records():
    from percept_harness.pipelines.scene_choices import project_scene_choices
    draft, ctx = fixture()
    draft['semantic_events'][0].update(event_type='hold', description='hand holds item visibly')
    original = copy.deepcopy(draft)
    assert '_schema_validation' in registry.sanitize('SceneSemanticsChoices', draft, ctx)
    assert draft == original
    draft['semantic_events'][0]['event_type'] = 'unknown'
    assert registry.sanitize('SceneSemanticsChoices', draft, ctx) == draft
    public = project_scene_choices(draft, ctx)
    assert public['semantic_events'] == draft['semantic_events']
    assert public['objects'] == draft['objects']
    draft['semantic_events'] = []
    assert registry.sanitize('SceneSemanticsChoices', draft, ctx)['_schema_validation']['issue_codes'] == ['EMPTY_SCENE_EVENTS']


def test_scene_choice_enum_mapping_preserves_other_issue_families():
    draft, ctx = fixture()
    del draft['semantic_events'][0]['actor']
    draft['semantic_events'][0]['event_type'] = 'hold'
    draft['semantic_events'][0]['private field'] = 'private input'
    draft['semantic_events'][0]['confidence'] = 2.0
    failure = registry.sanitize('SceneSemanticsChoices', draft, ctx)
    assert failure['_schema_validation']['issue_codes'] == [
        'SCENE_SEMANTICS_CHOICES_MISSING_FIELD',
        'SCENE_SEMANTICS_CHOICES_EXTRA_FIELD',
        'SCENE_SEMANTICS_CHOICES_NUMBER_RANGE',
        'SCENE_SEMANTICS_CHOICES_EVENT_TYPE_ENUM_VALUE',
    ]
    assert registry.failure_codes('SceneSemanticsChoices', failure) is not None


def test_legacy_event_enum_feedback_remains_generic():
    from percept_harness.pipelines.scene_choices import project_scene_choices
    draft, ctx = fixture()
    public = project_scene_choices(draft, ctx)
    public['semantic_events'][0]['event_type'] = 'hold'
    public_context = {key: ctx[key] for key in ('duration', 'require_observed_content',
                      'required_object_ids', 'evidence_summary', 'segments')}
    assert registry.sanitize('SceneSemantics', public, public_context) == {
        '_schema_validation': {'schema_name': 'SceneSemantics', 'status': 'invalid',
                               'issue_codes': ['SCENE_SEMANTICS_ENUM_VALUE']}}


def test_scene_normalization_repairs_mechanical_faults_without_a_model_call():
    """Sorting, renumbering, dangling targets, and bad predicates fix locally."""
    from percept_harness.pipelines.output_validation import (
        DEFAULT_OUTPUT_SCHEMAS, NormalizedSchemaOutput,
    )
    from percept_harness.pipelines.scene_choices import normalize_scene_choice_mechanics

    draft, ctx = fixture()
    draft['semantic_events'] = [
        dict(event_index=0, start=0.5, end=0.99, event_type='move',
             actor='unknown', target_object_id='item_0',
             description='item moves late', confidence=0.7),
        dict(event_index=1, start=0.01, end=0.4, event_type='move',
             actor='unknown', target_object_id='not_declared',
             description='item moves early', confidence=0.7),
    ]
    if draft['relations']:
        draft['relations'][0]['relation'] = 'gripping'

    fixed, codes, count = normalize_scene_choice_mechanics(draft)
    assert 'SCENE_EVENT_START_NOT_ORDERED' in codes
    assert 'SCENE_EVENT_UNKNOWN_OBJECT' in codes
    assert count >= 2
    assert [e['event_index'] for e in fixed['semantic_events']] == [0, 1]
    assert fixed['semantic_events'][0]['start'] == 0.01
    assert fixed['semantic_events'][0]['target_object_id'] == 'unknown'

    flagged = dict(ctx)
    flagged['allow_scene_normalization'] = True
    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize('SceneSemanticsChoices', draft, flagged)
    assert sanitized['_schema_validation']['status'] == 'normalized'
    normalized = DEFAULT_OUTPUT_SCHEMAS.normalized_result(
        'SceneSemanticsChoices', sanitized, flagged
    )
    assert isinstance(normalized, NormalizedSchemaOutput)
    assert normalized.issue_codes == tuple(codes)


def test_scene_normalization_never_hides_a_non_mechanical_fault():
    """A fault outside the mechanical set must still fail and trigger repair."""
    from percept_harness.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

    draft, ctx = fixture()
    draft['outcome']['status'] = 'victorious'
    flagged = dict(ctx)
    flagged['allow_scene_normalization'] = True
    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize('SceneSemanticsChoices', draft, flagged)
    assert sanitized['_schema_validation']['status'] == 'invalid'


def test_invented_option_ids_are_dropped_locally_without_a_model_call():
    """A selection referencing an unoffered option is removed, not retried."""
    from percept_harness.pipelines.output_validation import (
        DEFAULT_OUTPUT_SCHEMAS, NormalizedSchemaOutput,
    )

    draft, ctx = fixture()
    assert draft['locations'] or draft['relations']
    victim = 'locations' if draft['locations'] else 'relations'
    forged = json.loads(json.dumps(draft[victim][0]))
    forged['option_id'] = 'opt_invented_9999'
    draft[victim] = draft[victim] + [forged]

    flagged = dict(ctx)
    flagged['allow_scene_normalization'] = True
    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize('SceneSemanticsChoices', draft, flagged)
    envelope = sanitized['_schema_validation']
    assert envelope['status'] == 'normalized'
    assert 'SCENE_SEMANTICS_CHOICES_OPTION_UNKNOWN' in envelope['issue_codes']
    assert all(
        row['option_id'] != 'opt_invented_9999'
        for row in sanitized['data'][victim]
    )
    normalized = DEFAULT_OUTPUT_SCHEMAS.normalized_result(
        'SceneSemanticsChoices', sanitized, flagged
    )
    assert isinstance(normalized, NormalizedSchemaOutput)


def test_option_kind_mismatch_still_fails_closed():
    """Only wholly unknown option ids are mechanical; kind confusion is not."""
    from percept_harness.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

    draft, ctx = fixture()
    offered = ctx['spatial_options']['options']
    location_options = [o for o in offered if o['kind'] == 'location']
    relation_options = [o for o in offered if o['kind'] == 'relation']
    if not (draft['relations'] and location_options):
        return
    draft['relations'][0]['option_id'] = location_options[0]['option_id']
    flagged = dict(ctx)
    flagged['allow_scene_normalization'] = True
    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize('SceneSemanticsChoices', draft, flagged)
    assert sanitized['_schema_validation']['status'] == 'invalid'


def revision_fixture():
    draft, ctx = fixture()
    return {"scene": copy.deepcopy(draft), "decisions": []}, {
        "scene_context": ctx, "draft": draft,
        "occlusion_context": {"duration": ctx["duration"],
            "candidates": [], "evidence_summary": ctx["evidence_summary"]},
    }


def test_review_without_candidates_returns_revised_scene():
    result, ctx = revision_fixture()
    result["scene"]["semantic_events"][0]["event_type"] = "out_of_frame"
    assert registry.sanitize("SceneRevision", result, ctx) == result
    assert ctx["draft"]["semantic_events"][0]["event_type"] == "move"


def test_out_of_frame_review_rejects_remaining_occluded_claim():
    from test_cv_summary import _candidate_artifact
    from percept_harness.cv.contracts import EvidenceThresholds
    from percept_harness.cv.summary import summarize_cv_evidence, build_cv_prompt_bundle
    from percept_harness.pipelines.scene_choices import prepare_scene_choices
    artifact, timeline = _candidate_artifact()
    summary = summarize_cv_evidence(artifact, timeline=timeline)
    bundle = build_cv_prompt_bundle(summary, EvidenceThresholds(
        min_confidence=0.5, min_area_fraction=0.01, occlusion_visibility_drop=0.5))
    candidate = next(c for c in bundle.candidates if c.target_entity_id == "edge_target")
    gap = next(o for o in candidate.allowed_event_intervals if o.event_type == "occluded")
    segments = [{"segment_index": 0, "start": 0.0, "end": 0.5, "target": "edge target"}]
    context = prepare_scene_choices(summary, segments, duration=0.5).context()
    scene = {"objects": [{"object_id": "edge_target", "name": "edge target", "description": "visible target"}],
        "locations": [], "relations": [], "initial_state": [], "final_state": [],
        "outcome": {"status": "unknown", "description": "not established", "confidence": 0.2},
        "semantic_events": [{"event_index": 0, "start": gap.start, "end": gap.end,
            "event_type": "occluded", "actor": "unknown", "target_object_id": "edge_target",
            "description": "target disappears", "confidence": 0.8}]}
    decisions = [{"candidate_id": c.candidate_id, "target_entity_id": c.target_entity_id,
        "classification": "out_of_frame" if c.candidate_id == candidate.candidate_id else "unknown",
        "occluder_entity_id": "unknown", "events": [], "visual_evidence": "target exits at the edge",
        "confidence": 0.8} for c in bundle.candidates]
    ctx = {"scene_context": context, "draft": copy.deepcopy(scene), "occlusion_context": {
        "duration": 0.5, "evidence_summary": summary.model_dump(mode="json"),
        "candidates": [c.model_dump(mode="json") for c in bundle.candidates]}}
    result = {"scene": scene, "decisions": decisions}
    rejected = registry.sanitize("SceneRevision", result, ctx)
    assert rejected["_schema_validation"]["issue_codes"] == ["SCENE_REVISION_VISIBILITY_CONFLICT"]
    result["scene"]["semantic_events"][0]["event_type"] = "out_of_frame"
    assert registry.sanitize("SceneRevision", result, ctx) == result


@pytest.mark.parametrize("fault", ["shape", "objects", "bounds"])
def test_invalid_revision_is_not_published_as_success(fault):
    result, ctx = revision_fixture()
    if fault == "shape":
        del result["scene"]
    elif fault == "objects":
        result["scene"]["objects"] = []
    else:
        result["scene"]["semantic_events"][0]["end"] = ctx["scene_context"]["duration"] + 1
    assert registry.sanitize("SceneRevision", result, ctx)["_schema_validation"]["status"] == "invalid"


def test_fake_cv_runs_complete_review_without_candidates(tmp_path, silent_video, monkeypatch):
    import json
    from percept_harness.cli import main
    monkeypatch.setenv("PERCEPT_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("PERCEPT_CV_CACHE_ROOT", str(tmp_path / "cv-cache"))
    output = tmp_path / "out"
    assert main(["annotate", "--videos", str(silent_video), "--template",
                 "embodied_action_captioning", "--backend", "fake", "--cv", "fake",
                 "--output", str(output)]) == 0
    data = json.loads((output / "demo.json").read_text())["data"]
    assert any(s.get("model_stage") == "occlusion_semantics" for s in data["performance"]["stages"])
    scene = data["annotation_branches"]["scene_facts"]["events"]
    assert all(e["model_stage"] == "occlusion_semantics" for e in scene)
    assert [e["event_type"] for e in scene] == [e["event_type"] for e in data["semantic_events"]]


def test_exhausted_review_fails_annotation_instead_of_publishing_draft(tmp_path, silent_video, monkeypatch):
    import json
    from percept_harness import cli
    from percept_harness.models.fake import FakeVideoModel
    model = FakeVideoModel(failure_script={"occlusion_semantics": [{}, {}, {}]})
    monkeypatch.setattr(cli, "_load_backend", lambda args, settings: (model, "qwen3-vl-8b-instruct", lambda: None))
    monkeypatch.setenv("PERCEPT_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("PERCEPT_CV_CACHE_ROOT", str(tmp_path / "cache"))
    output = tmp_path / "out"
    assert cli.main(["annotate", "--videos", str(silent_video), "--template",
                     "embodied_action_captioning", "--backend", "fake", "--cv", "fake",
                     "--output", str(output)]) == 1
    result = json.loads((output / "demo.json").read_text())
    assert result["status"] == "failed"
    assert result["data"] is None

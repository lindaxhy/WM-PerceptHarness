"""Trusted generation choices must remain acceptable to the unchanged boundary."""

import copy
import hashlib
import json

import pytest
from test_cv_summary import _artifact, _entity, _observation, _track

from las_repro.cv.summary import summarize_cv_evidence
from las_repro.pipelines.embodied import PromptRenderer
from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
from las_repro.pipelines.scene_semantics import unavailable_scene_semantics


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
    from las_repro.pipelines.scene_provenance import scene_spatial_prompt_data
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
    suffix = prompt.split("[CV_EVIDENCE_SUMMARY_JSON]\n")[1]
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
        resources.files("las_repro.prompts").joinpath(name + ".txt").read_text()
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
    from las_repro.cv.summary import _summary_identity

    tight = summary.model_copy(update={"prompt_char_limit": len(suffix)})
    tight = tight.model_copy(update={"summary_id": _summary_identity(tight)})
    envelope, prompt = options(tight, segments)
    assert envelope is None
    assert prompt.endswith(
        json.dumps(tight.prompt_record(), ensure_ascii=False, separators=(",", ":"))
    )
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
    from las_repro.cv.contracts import ArtifactFile, OverlayRecord

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
    from las_repro.cv.contracts import EntityPrompt

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

    from las_repro.pipelines.scene_provenance import scene_spatial_prompt_data

    summary, segments = source()
    with pytest.raises(ValueError):
        scene_spatial_prompt_data(summary, BombList([{}] * 10001), duration=3.0)
    hostile = summary.model_copy(update={"tracks": BombList([None] * 10000)})
    with pytest.raises(ValueError):
        options(hostile, segments)


def test_generated_options_validate_source_only_once(monkeypatch):
    from las_repro.cv.summary import CvEvidenceSummary

    summary, segments = source()
    original = CvEvidenceSummary.prompt_record
    calls = []

    def counted(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(CvEvidenceSummary, "prompt_record", counted)
    from las_repro.pipelines.scene_provenance import scene_spatial_prompt_data
    envelope, _ = scene_spatial_prompt_data(summary, segments, duration=segments[-1]["end"])
    assert len(envelope["options"]) > 1
    assert len(calls) == 1


def test_hash_collisions_are_rejected_even_for_otherwise_valid_options(monkeypatch):
    from types import SimpleNamespace

    import las_repro.pipelines.scene_provenance as module

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
    from las_repro.cv.contracts import EntityPrompt

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
    from las_repro.pipelines.scene_provenance import scene_spatial_prompt_data

    summary, segments = source()
    original, _ = scene_spatial_prompt_data(summary, segments, duration=3.0)
    expected = copy.deepcopy(original)
    original["flat_provenance"]["repair_history"].append("repair")
    original["field_shape_example"]["repair_history"].append("repair")
    repeated, _ = scene_spatial_prompt_data(summary, segments, duration=3.0)
    assert repeated == expected


def test_output_prohibited_overlay_stem_is_not_offered_as_a_copyable_value():
    from las_repro.cv.contracts import ArtifactFile, OverlayRecord

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
    suffix = prompt.split("[CV_EVIDENCE_SUMMARY_JSON]\n")[1]
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

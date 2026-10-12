"""Cv candidates behavior and regression tests."""
from __future__ import annotations

import copy
import json

from pydantic import ValidationError
import pytest

from percept_harness.cv.identity import identity_bytes
import percept_harness.cv.summary as sm
from percept_harness.pipelines.occlusion import (
    OcclusionDecisionSet,
    validate_occlusion_decisions,
)
from percept_harness.pipelines.validators import TemporalValidationError

from test_cv_summary import _artifact, _observation, _thresholds, _timeline, _track


def gap_bundle(*, visible=(0, 4), frames=(0, 1, 2, 3, 4), extras=()):
    summary = sm.summarize_cv_evidence(
        _artifact((_track('item_1', 'item', tuple(_observation(i) for i in visible)), *extras), processed_timeline=_timeline(*frames)),
        timeline=_timeline(*frames),
    )
    return sm.build_cv_prompt_bundle(summary, _thresholds())


def decisions(candidate, events):
    described = [
        event if event.get('description') else dict(event, description=f"visible {event['event_type']} phase of the target")
        for event in events
    ]
    return OcclusionDecisionSet.model_validate({'decisions': [{
        'candidate_id': candidate.candidate_id, 'classification': 'occlusion',
        'target_entity_id': candidate.target_entity_id, 'occluder_entity_id': 'unknown',
        'events': described, 'visual_evidence': 'Visible partial covering and return', 'confidence': .8,
    }]})


def test_truthful_three_phase_lifecycle_accepted_and_whole_gap_rejected():
    candidate = gap_bundle().candidates[0]
    phases = [
        {'event_type': 'occlusion_enter', 'start': 0., 'end': .1},
        {'event_type': 'occluded', 'start': .1, 'end': .3},
        {'event_type': 'occlusion_exit', 'start': .3, 'end': .4},
    ]
    validate_occlusion_decisions(decisions(candidate, phases), (candidate,), duration=.4)
    assert candidate.prompt_record()['allowed_event_intervals'] == phases
    whole_gap = [dict(p, start=0., end=.4) for p in phases]
    with pytest.raises(TemporalValidationError) as exc:
        validate_occlusion_decisions(decisions(candidate, whole_gap), (candidate,), duration=.4)
    assert {i.code for i in exc.value.issues} >= {'OCCLUSION_INTERVAL_NOT_ALLOWED'}


@pytest.mark.parametrize(('visible', 'frames', 'expected'), [
    ((0, 2), (0, 1, 2), [('occlusion_enter', 0., .1), ('occlusion_exit', .1, .2)]),
    ((0,), (0, 1, 2), [('occlusion_enter', 0., .1), ('occluded', .1, .2)]),
])
def test_one_frame_and_permanent_gap_options(visible, frames, expected):
    candidate = gap_bundle(visible=visible, frames=frames).candidates[0]
    assert [(o.event_type, o.start, o.end) for o in candidate.allowed_event_intervals] == expected


def test_retained_continuation_is_hypothesis_with_actual_two_frame_pts():
    other = _track('item_2', 'item', tuple(_observation(i) for i in (1, 2, 3)))
    bundle = gap_bundle(extras=(other,))
    cue = bundle.candidates[0].identity_evidence
    assert cue.basis == 'retained_geometry'
    assert cue.target_scope == 'predictor_track'
    assert [(c.other_track_id, c.target_frame_index, c.other_frame_index,
             c.target_timestamp_seconds, c.other_timestamp_seconds) for c in cue.continuation_cues] == [
        ('item_2', 0, 1, 0., .1), ('item_2', 4, 3, .4, .3)]
    assert all(c.bbox_iou == 1. and c.area_similarity == 1. and c.center_distance_fraction == 0.
               for c in cue.continuation_cues)
    assert [t.track_id for t in bundle.summary.tracks] == ['item_1', 'item_2']


def test_identity_metric_tampering_rejected_after_resealing_candidate():
    from test_cv_summary import _reseal_candidate_payload
    other = _track('item_2', 'item', (_observation(1), _observation(2), _observation(3)))
    bundle = gap_bundle(extras=(other,))
    payload = bundle.model_dump(mode='json')
    payload['candidates'][0]['identity_evidence']['continuation_cues'][0]['bbox_iou'] = .5
    _reseal_candidate_payload(payload['candidates'][0])
    with pytest.raises(ValidationError, match='canonical candidate set'):
        sm.CvPromptBundle.model_validate(payload)


@pytest.mark.parametrize('options', [
    [],
    [{'event_type':'occluded','start':.1,'end':.3}]*2,
    [{'event_type':'occlusion_exit','start':.3,'end':.4}, {'event_type':'occlusion_enter','start':0.,'end':.1}],
    [{'event_type':'occluded','start':float('nan'),'end':.3}],
    [{'event_type':'other','start':0.,'end':.3}],
    [{'event_type':'occluded','start':True,'end':.3}],
    [{'event_type':'occluded','start':.3,'end':.3}],
    [{'event_type':'occluded','start':i/100.,'end':(i+1)/100.} for i in range(9)],
    [{'event_type':'occluded','start':i/100.,'end':(i+1)/100.} for i in range(25)],
])
def test_malformed_options_rejected(options):
    payload = gap_bundle().candidates[0].model_dump(mode='json')
    payload['allowed_event_intervals'] = options
    with pytest.raises(ValueError):
        sm.OcclusionCandidate.model_validate(payload)


def test_legacy_endpoint_pool_record_explicitly_rejected():
    payload = gap_bundle().candidates[0].model_dump(mode='json')
    del payload['allowed_event_intervals']
    payload.update(allowed_start_times=[0.], allowed_end_times=[.4])
    with pytest.raises(ValueError):
        sm.OcclusionCandidate.model_validate(payload)


@pytest.mark.parametrize('event', [
    {'event_type':'occlusion_enter','start':.1,'end':.3},
    {'event_type':'occluded','start':0.,'end':.3},
    {'event_type':'occlusion_exit','start':.1,'end':.4},
])
def test_wrong_type_and_cartesian_recombination_rejected(event):
    candidate = gap_bundle().candidates[0]
    with pytest.raises(TemporalValidationError) as exc:
        validate_occlusion_decisions(decisions(candidate,[event]), (candidate,), duration=.4)
    assert 'OCCLUSION_INTERVAL_NOT_ALLOWED' in {i.code for i in exc.value.issues}


def test_bypass_nested_option_rejected_before_hostile_numeric_operations():
    from test_cv_summary import BombFloat
    candidate = gap_bundle().candidates[0]
    forged = sm.AllowedEventInterval.model_construct(event_type='occluded', start=BombFloat(.1), end=.3)
    payload = {name:getattr(candidate,name) for name in sm.OcclusionCandidate.model_fields}
    payload['allowed_event_intervals'] = (forged,)
    with pytest.raises(ValueError):
        sm.OcclusionCandidate.model_validate(payload)


@pytest.mark.parametrize(('field','value'), [
    ('other_track_id','foreign_1'), ('other_frame_index',2),
    ('other_timestamp_seconds',.2), ('target_frame_index',1),
    ('area_similarity',.9), ('center_distance_fraction',.2),
])
def test_identity_resealed_source_forgery_rejected(field,value):
    from test_cv_summary import _reseal_candidate_payload
    other = _track('item_2','item',tuple(_observation(i) for i in (1,2,3)))
    bundle = gap_bundle(extras=(other,))
    payload = bundle.model_dump(mode='json')
    payload['candidates'][0]['identity_evidence']['continuation_cues'][0][field]=value
    _reseal_candidate_payload(payload['candidates'][0])
    with pytest.raises(ValueError):
        sm.CvPromptBundle.model_validate(payload)


def test_static_border_cross_label_conflict_retains_all_classifications():
    target = _track('item_1','item',tuple(_observation(i,bbox_xyxy=(0.,0.,.8,.8)) for i in (0,4)))
    alias = _track('label_1','label',tuple(_observation(i,bbox_xyxy=(0.,0.,.8,.8)) for i in range(5)))
    summary = sm.summarize_cv_evidence(_artifact((target,alias)),timeline=_timeline(0,1,2,3,4))
    bundle = sm.build_cv_prompt_bundle(summary,_thresholds())
    candidate = bundle.candidates[0]
    assert candidate.edge_departure is True
    assert [(c.left_track_id,c.right_track_id,c.frame_index,c.bbox_iou) for c in candidate.identity_evidence.cross_label_cues] == [('item_1','label_1',0,1.),('item_1','label_1',4,1.)]
    for classification in ('occlusion','unknown','detector_loss','out_of_frame'):
        event = candidate.allowed_event_intervals[0].model_dump()
        result = decisions(candidate,[event] if classification=='occlusion' else [])
        payload = result.model_dump(mode='json')
        payload['decisions'][0]['classification']=classification
        validate_occlusion_decisions(OcclusionDecisionSet.model_validate(payload),(candidate,),duration=.4)


def test_uncertainty_windows_are_three_independent_visual_options():
    target = _track('item_1','item',(_observation(0),_observation(1,confidence=.1),_observation(2)))
    summary = sm.summarize_cv_evidence(_artifact((target,)), timeline=_timeline(0,1,2))
    candidate = sm.build_cv_prompt_bundle(summary,_thresholds()).candidates[0]
    assert [(o.event_type,o.start,o.end) for o in candidate.allowed_event_intervals] == [
        ('occluded',0.,.2),('occlusion_enter',0.,.2),('occlusion_exit',0.,.2)]
    assert not summary.tracks[0].missing_intervals


def test_real_shaped_uncertainty_bridge_is_removed_before_budget_packing():
    def timestamp(frame):
        return round(frame / 30.0, 6)

    frames = tuple(range(80))
    timeline = sm.FrameTimeline(
        frames=tuple(
            sm.FrameTimestamp(
                frame_index=frame,
                timestamp_seconds=timestamp(frame),
            )
            for frame in frames
        )
    )
    observations = tuple(
        _observation(
            frame,
            timestamp_seconds=timestamp(frame),
            area_fraction={
                43: 0.0009717399691358024,
                44: 0.00023775077160493826,
                79: 0.00037229938271604937,
            }.get(frame, 0.02),
            confidence=0.9649122953414917,
        )
        for frame in (*range(45), 79)
    )
    summary = sm.summarize_cv_evidence(
        _artifact(
            (_track("yellow_ball_0", "yellow_ball", observations),),
            processed_timeline=timeline,
        ),
        timeline=timeline,
        max_observations_per_track=4,
    )

    track = summary.tracks[0]
    assert [item.frame_index for item in track.observations] == [0, 43, 44, 79]
    assert [item.source_ordinal for item in track.observations] == [0, 43, 44, 45]
    assert [
        (run.state, run.start_frame, run.end_frame)
        for run in track.visibility_runs
    ] == [
        ("visible", 0, 44),
        ("missing", 45, 78),
        ("visible", 79, 79),
    ]
    assert track.visibility_lifecycle_complete is True
    assert track.candidate_search_complete is False

    bundle = sm.build_cv_prompt_bundle(
        summary,
        _thresholds(),
        max_candidates=1,
    )
    repeated = sm.build_cv_prompt_bundle(
        summary,
        _thresholds(),
        max_candidates=1,
    )

    assert bundle == repeated
    assert bundle.candidates_complete is True
    assert bundle.truncation_codes == ("SUMMARY_CANDIDATE_SEARCH_INCOMPLETE",)
    assert len(bundle.candidates) == 1
    [candidate] = bundle.candidates
    assert (candidate.last_visible_frame, candidate.first_revisible_frame) == (44, 79)
    assert [
        (option.event_type, option.start, option.end)
        for option in candidate.allowed_event_intervals
    ] == [
        ("occlusion_enter", 1.466667, 1.5),
        ("occluded", 1.5, 2.6),
        ("occlusion_exit", 2.6, 2.633333),
    ]
    sm.validate_candidate_identity_evidence(summary, bundle.candidates)


def test_missing_runs_split_individual_and_grouped_uncertainty_support():
    target = _track(
        "item_1",
        "item",
        tuple(
            _observation(
                frame,
                confidence=0.1 if frame in (1, 2, 4, 5, 6, 8, 9) else 0.9,
            )
            for frame in (0, 1, 2, 4, 5, 6, 8, 9, 10)
        ),
    )
    summary = sm.summarize_cv_evidence(
        _artifact((target,), processed_timeline=_timeline(*range(11))),
        timeline=_timeline(*range(11)),
    )
    bundle = sm.build_cv_prompt_bundle(summary, _thresholds())

    assert [item.frame_index for item in summary.tracks[0].observations] == [
        0, 1, 2, 4, 5, 6, 8, 9, 10
    ]
    assert [
        (run.state, run.start_frame, run.end_frame)
        for run in summary.tracks[0].visibility_runs
    ] == [
        ("visible", 0, 2),
        ("missing", 3, 3),
        ("visible", 4, 6),
        ("missing", 7, 7),
        ("visible", 8, 10),
    ]
    assert [
        (candidate.last_visible_frame, candidate.first_revisible_frame)
        for candidate in bundle.candidates
    ] == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    uncertainty = [
        candidate
        for candidate in bundle.candidates
        if len(candidate.allowed_event_intervals) == 3
    ]
    assert [
        (candidate.last_visible_frame, candidate.first_revisible_frame)
        for candidate in uncertainty
    ] == [(0, 2), (4, 6), (8, 10)]
    assert all(
        len({(option.start, option.end) for option in candidate.allowed_event_intervals})
        == 1
        for candidate in uncertainty
    )


def test_sparse_processed_frames_without_missing_run_remain_uncertainty_support():
    timeline = _timeline(0, 10, 20)
    target = _track(
        "item_1",
        "item",
        (
            _observation(0),
            _observation(10, confidence=0.1),
            _observation(20),
        ),
    )
    summary = sm.summarize_cv_evidence(
        _artifact((target,), processed_timeline=timeline),
        timeline=timeline,
    )

    assert [run.state for run in summary.tracks[0].visibility_runs] == ["visible"]
    [candidate] = sm.build_cv_prompt_bundle(summary, _thresholds()).candidates
    assert (candidate.last_visible_frame, candidate.first_revisible_frame) == (0, 20)
    assert {
        (option.event_type, option.start, option.end)
        for option in candidate.allowed_event_intervals
    } == {
        ("occluded", 0.0, 2.0),
        ("occlusion_enter", 0.0, 2.0),
        ("occlusion_exit", 0.0, 2.0),
    }


def test_old_bundle_marker_cannot_be_replayed_via_bypass_model():
    bundle=gap_bundle()
    with pytest.raises(ValueError):
        bundle.model_copy(update={'schema_version':'cv_prompt_bundle_v1'}).prompt_record()


def test_fair_global_rows_and_byte_caps_with_reversed_source_order():
    tracks=tuple(_track(f'item_{i:03}', 'item', (_observation(0),)) for i in range(80))
    timeline=_timeline(0,1,2)
    def build(ts):
        summary=sm.summarize_cv_evidence(_artifact(ts,processed_timeline=timeline),timeline=timeline,
                                        max_tracks=80,max_relations=1)
        return sm.build_cv_prompt_bundle(summary,_thresholds())
    bundle=build(tracks)
    assert bundle==build(tuple(reversed(tracks)))
    cues=[c.identity_evidence for c in bundle.candidates]
    assert sum(len(c.continuation_cues)+len(c.cross_label_cues) for c in cues if c)==64
    import json
    assert sum(len(json.dumps(c.model_dump(mode='json'),ensure_ascii=False,separators=(',',':')).encode())
               for c in cues if c)<=24000
    assert cues[0].continuation_cues and cues[-1].continuation_cues
    assert all(c is None or not c.complete for c in cues)
    assert len(bundle.candidates)==80


def test_optional_identity_sections_yield_before_basic_candidates():
    import json
    other=_track('item_2','item',tuple(_observation(i) for i in (1,2,3)))
    bundle=gap_bundle(extras=(other,))
    full_size=len(json.dumps(bundle.prompt_record(),separators=(',',':'),ensure_ascii=False))
    bounded=sm.build_cv_prompt_bundle(bundle.summary,_thresholds(),max_prompt_chars=full_size-1)
    assert len(bounded.candidates)==len(bundle.candidates)
    assert any(c.identity_evidence is not None for c in bounded.candidates)
    assert sum(len(c.identity_evidence.continuation_cues)+len(c.identity_evidence.cross_label_cues)
               for c in bounded.candidates if c.identity_evidence) < sum(
        len(c.identity_evidence.continuation_cues)+len(c.identity_evidence.cross_label_cues)
        for c in bundle.candidates if c.identity_evidence)
    assert bounded.candidates_complete
    assert [(c.target_track_id,c.allowed_event_intervals,c.possible_occluders) for c in bounded.candidates]==[
        (c.target_track_id,c.allowed_event_intervals,c.possible_occluders) for c in bundle.candidates]
    assert sm.CvPromptBundle.model_validate_json(bounded.model_dump_json())==bounded


def test_sparse_continuation_uses_observed_pts_and_distinct_regions_remain_hypotheses():
    other=_track('item_2','item',(_observation(20,bbox_xyxy=(.7,.7,.9,.9)),))
    bundle=gap_bundle(visible=(0,40),frames=(0,10,20,30,40),extras=(other,))
    candidate=bundle.candidates[0]
    # The mid-gap disjoint sibling trims the entity-level gap mechanically;
    # the surviving near-side cue still reaches the adjudicator.
    assert candidate.first_revisible_frame==20
    cues=candidate.identity_evidence.continuation_cues
    assert [(c.target_frame_index,c.other_frame_index,c.target_timestamp_seconds,c.other_timestamp_seconds)
            for c in cues]==[(0,20,0.,2.)]
    assert all(c.bbox_iou==0. for c in cues)


def test_no_continuation_is_no_physical_identity_verdict():
    candidate=gap_bundle().candidates[0]
    assert not candidate.identity_evidence.continuation_cues
    assert not candidate.identity_evidence.cross_label_cues
    assert 'same_object' not in candidate.prompt_record()['identity_evidence']


def test_permanent_gap_can_supply_two_continuation_alternatives():
    extras=(_track('item_2','item',(_observation(1),)), _track('item_3','item',(_observation(1),)))
    candidate=gap_bundle(visible=(0,),frames=(0,1,2),extras=extras).candidates[0]
    assert [c.other_track_id for c in candidate.identity_evidence.continuation_cues]==['item_2','item_3']


def test_scan_caps_omit_truthfully_without_unbounded_geometry_recompute(monkeypatch):
    import percept_harness.cv.identity as identity
    count=0
    original=identity._geometry
    def counted(a,b):
        nonlocal count
        count+=1
        return original(a,b)
    monkeypatch.setattr(identity,'MAX_CONTINUATION_PROBES',1)
    monkeypatch.setattr(identity,'MAX_RELATION_PROBES',1)
    monkeypatch.setattr(identity,'_geometry',counted)
    other=_track('item_2','item',tuple(_observation(i) for i in (1,2,3)))
    bundle=gap_bundle(extras=(other,))
    # Builder plus canonical bundle authentication each run one bounded pass.
    assert count==2
    assert all(c.identity_evidence is None or not c.identity_evidence.complete for c in bundle.candidates)


@pytest.mark.parametrize('mutation', [
    lambda e: e['continuation_cues'].append(copy.deepcopy(e['continuation_cues'][0])),
    lambda e: e['continuation_cues'].reverse(),
    lambda e: e['continuation_cues'][0].update(bbox_iou='1.0'),
    lambda e: e['continuation_cues'][0].update(bbox_iou=True),
    lambda e: e['continuation_cues'][0].update(bbox_iou=float('inf')),
    lambda e: e.update(basis='mask_iou'),
    lambda e: e.update(target_scope='physical_object'),
])
def test_identity_wire_shape_cannot_smuggle_claims_or_noncanonical_rows(mutation):
    other=_track('item_2','item',tuple(_observation(i) for i in (1,2,3)))
    payload=gap_bundle(extras=(other,)).candidates[0].model_dump(mode='json')
    mutation(payload['identity_evidence'])
    with pytest.raises(ValueError):
        sm.OcclusionCandidate.model_validate(payload)


def test_registry_authenticates_retained_cues():
    from percept_harness.pipelines.output_validation import _validate_occlusion_decision_output
    from test_cv_summary import _reseal_candidate_payload
    other=_track('item_2','item',tuple(_observation(i) for i in (1,2,3)))
    bundle=gap_bundle(extras=(other,))
    candidate=bundle.candidates[0]
    # One candidate context is independently regenerated under its own bounded selection.
    bundle=sm.build_cv_prompt_bundle(bundle.summary,_thresholds(),max_candidates=1)
    candidate=bundle.candidates[0]
    context={'duration':.4,'candidates':[candidate.model_dump(mode='json')],
             'evidence_summary':bundle.summary.model_dump(mode='json')}
    positive=decisions(candidate,[candidate.allowed_event_intervals[0].model_dump()]).model_dump(mode='json')
    assert _validate_occlusion_decision_output(positive,context)==positive
    context['candidates'][0]['identity_evidence']['continuation_cues'][0]['bbox_iou']=.2
    _reseal_candidate_payload(context['candidates'][0])
    positive['decisions'][0]['candidate_id']=context['candidates'][0]['candidate_id']
    with pytest.raises(ValueError):
        _validate_occlusion_decision_output(positive,context)


def test_registry_requires_source_for_supplied_identity_cues():
    from percept_harness.pipelines.output_validation import _validate_occlusion_decision_output
    candidate=gap_bundle().candidates[0]
    positive=decisions(candidate,[candidate.allowed_event_intervals[0].model_dump()]).model_dump(mode='json')
    with pytest.raises(ValueError):
        _validate_occlusion_decision_output(positive,{'duration':.4,'candidates':[candidate.model_dump(mode='json')]})


def test_cross_label_cues_inspect_typed_missing_boundary_witnesses():
    target=_track('target_1','target',(_observation(0),_observation(4)))
    board=_track('board_1','board',tuple(_observation(i) for i in range(5)))
    alias=_track('alias_1','alias',(_observation(1),_observation(3)))
    summary=sm.summarize_cv_evidence(_artifact((target,board,alias)),timeline=_timeline(0,1,2,3,4))
    bundle=sm.build_cv_prompt_bundle(summary,_thresholds())
    candidate=next(c for c in bundle.candidates if c.target_track_id=='target_1')
    assert any(c.left_track_id=='alias_1' and c.right_track_id=='board_1' and c.frame_index in (1,3)
               for c in candidate.identity_evidence.cross_label_cues)


def test_bypass_option_cannot_escape_direct_decision_validation():
    candidate=gap_bundle().candidates[0]
    invalid=candidate.model_copy(update={'allowed_event_intervals':tuple(candidate.allowed_event_intervals)*100})
    with pytest.raises(ValueError):
        validate_occlusion_decisions(decisions(invalid,[candidate.allowed_event_intervals[0].model_dump()]),
                                     (invalid,),duration=.4)


def test_identity_context_rejects_unrelated_occluder_support_even_with_real_metrics():
    from percept_harness.cv.identity import derive_identity_evidence
    target=_track('target_1','target',(_observation(0),_observation(4)))
    board=_track('board_1','board',tuple(_observation(i) for i in range(5)))
    summary=sm.summarize_cv_evidence(_artifact((target,board)),timeline=_timeline(0,1,2,3,4))
    candidate=sm.build_cv_prompt_bundle(summary,_thresholds()).candidates[0]
    provenance=candidate.possible_occluders[0].model_copy(update={'supporting_frames':(99,)})
    forged=candidate.model_copy(update={'possible_occluders':(provenance,)})
    forged=sm._with_identity(forged,derive_identity_evidence(summary,(forged,))[0])
    with pytest.raises(ValueError,match='relation support'):
        sm.validate_candidate_identity_evidence(summary,(forged,))


def test_long_identifier_identity_payload_obeys_byte_cap_before_row_cap():
    import json
    tracks=tuple(_track('item_'+('x'*112)+f'_{i:03}', 'item', (_observation(0),)) for i in range(80))
    timeline=_timeline(0,1,2)
    summary=sm.summarize_cv_evidence(_artifact(tracks,processed_timeline=timeline),timeline=timeline,
                                    max_tracks=80,max_relations=1)
    bundle=sm.build_cv_prompt_bundle(summary,_thresholds())
    cues=[c.identity_evidence for c in bundle.candidates if c.identity_evidence is not None]
    rows=sum(len(c.continuation_cues)+len(c.cross_label_cues) for c in cues)
    size=sum(len(json.dumps(c.model_dump(mode='json'),ensure_ascii=False,separators=(',',':')).encode()) for c in cues)
    assert 0 < rows < 64
    assert 23500 < size <= 24000
    assert bundle.candidates[0].identity_evidence and bundle.candidates[-1].identity_evidence
    assert len(bundle.candidates)==80


def test_moving_geometry_handoff_exposes_metrics_without_asserting_identity():
    other=_track('item_2','item',(_observation(1,bbox_xyxy=(.21,.2,.41,.4)),))
    candidate=gap_bundle(visible=(0,),frames=(0,1,2),extras=(other,)).candidates[0]
    cue=candidate.identity_evidence.continuation_cues[0]
    assert cue.bbox_iou == pytest.approx(.19/.21)
    assert cue.area_similarity == pytest.approx(1.)
    assert cue.center_distance_fraction == pytest.approx(.01/(2**.5))


def test_missing_fields_in_bypass_nested_option_fail_at_structural_boundary():
    candidate=gap_bundle().candidates[0]
    forged=sm.AllowedEventInterval.model_construct(event_type='occluded',end=.3)
    payload={name:getattr(candidate,name) for name in sm.OcclusionCandidate.model_fields}
    payload['allowed_event_intervals']=(forged,)
    with pytest.raises(ValueError):
        sm.OcclusionCandidate.model_validate(payload)


def test_row_budget_covers_cross_label_only_candidates_before_second_rows():
    tracks=tuple(_track(f'item_{i:03}', 'item' if i<32 else f'entity_{i}', (_observation(0),)) for i in range(64))
    timeline=_timeline(0,1,2)
    summary=sm.summarize_cv_evidence(_artifact(tracks,processed_timeline=timeline),timeline=timeline,
                                    max_tracks=64,max_relations=64)
    bundle=sm.build_cv_prompt_bundle(summary,_thresholds())
    assert len(bundle.candidates)==64
    assert all(c.identity_evidence is not None and
               (c.identity_evidence.continuation_cues or c.identity_evidence.cross_label_cues)
               for c in bundle.candidates)

def test_occlusion_event_without_description_is_rejected():
    candidate = gap_bundle().candidates[0]
    payload = {'decisions': [{
        'candidate_id': candidate.candidate_id, 'classification': 'occlusion',
        'target_entity_id': candidate.target_entity_id, 'occluder_entity_id': 'unknown',
        'events': [{'event_type': 'occlusion_enter', 'start': 0., 'end': .1}],
        'visual_evidence': 'Visible partial covering and return', 'confidence': .8,
    }]}
    result = OcclusionDecisionSet.model_validate(payload)
    with pytest.raises(TemporalValidationError) as exc:
        validate_occlusion_decisions(result, (candidate,), duration=.4)
    assert {i.code for i in exc.value.issues} == {'OCCLUSION_EVENT_DESCRIPTION_MISSING'}


def test_projected_events_carry_their_own_phase_descriptions():
    from percept_harness.pipelines.occlusion import project_occlusion_events
    candidate = gap_bundle().candidates[0]
    phases = [
        {'event_type': 'occlusion_enter', 'start': 0., 'end': .1,
         'description': 'the item slips behind the cover'},
        {'event_type': 'occluded', 'start': .1, 'end': .3,
         'description': 'the item stays hidden behind the cover'},
        {'event_type': 'occlusion_exit', 'start': .3, 'end': .4,
         'description': 'the item emerges from behind the cover'},
    ]
    projected = project_occlusion_events(
        decisions(candidate, phases), (candidate,), (), [], repair_history=('initial',)
    )
    assert [row['description'] for row in projected] == [p['description'] for p in phases]


def _rows(candidates):
    return sum(len(c.identity_evidence.continuation_cues) + len(c.identity_evidence.cross_label_cues)
               for c in candidates if c.identity_evidence is not None)


def _size(record):
    return len(json.dumps(record, ensure_ascii=False, separators=(',', ':')))


def _dense_handoffs():
    timeline = _timeline(*range(120))
    tracks = [
        _track(f'item_{i}', 'item', tuple(_observation(frame, bbox_xyxy=(.1,.1,.5,.5))
                                        for frame in range(i*20, (i+1)*20)))
        for i in range(6)
    ]
    tracks.append(_track('alias_1', 'alias', tuple(_observation(frame, bbox_xyxy=(.1,.1,.5,.5))
                                                for frame in range(120))))
    tracks.extend(_track(f'context_{i}', f'context_{i}', tuple(
        _observation(frame, bbox_xyxy=(.6,.6,.9,.9)) for frame in range(120))) for i in range(8))
    return _artifact(tuple(tracks), processed_timeline=timeline), timeline


def test_default_joint_sizing_reserves_real_dense_handoff_and_alias_cues():
    artifact, timeline = _dense_handoffs()
    summary = sm.summarize_cv_evidence(artifact, timeline=timeline, thresholds=_thresholds())
    full_candidates = sm.build_occlusion_candidates(summary, _thresholds())
    bundle = sm.build_cv_prompt_bundle(summary, _thresholds())
    assert len(bundle.candidates) == 5
    assert len(summary.tracks) == len(artifact.tracks)
    assert _rows(full_candidates) > 0
    assert _rows(bundle.candidates) == _rows(full_candidates)
    assert any(c.identity_evidence.continuation_cues for c in bundle.candidates if c.identity_evidence)
    assert any(c.identity_evidence.cross_label_cues for c in bundle.candidates if c.identity_evidence)
    assert bundle.candidates_complete
    assert _size(bundle.prompt_record()) <= 200_000
    assert sum(identity_bytes(c.identity_evidence) for c in bundle.candidates) <= 24_000
    assert sm.CvPromptBundle.model_validate_json(bundle.model_dump_json()) == bundle
    sm.validate_candidate_identity_evidence(summary, bundle.candidates)


def test_tight_explicit_budget_retains_whole_rows_and_required_candidates():
    other = _track('item_2','item',tuple(_observation(i) for i in (1,2,3)))
    full = gap_bundle(extras=(other,))
    basic = full.prompt_record()
    for candidate in basic['candidates']:
        candidate['identity_evidence'] = None
    limit = _size(basic) + 400
    assert _size(full.prompt_record()) > limit
    bounded = sm.build_cv_prompt_bundle(full.summary, _thresholds(), max_prompt_chars=limit)
    assert 0 < _rows(bounded.candidates) < _rows(full.candidates)
    assert len(bounded.candidates) == len(full.candidates)
    assert bounded.candidates_complete
    assert _size(bounded.prompt_record()) <= limit
    assert all(c.identity_evidence is None or not c.identity_evidence.complete for c in bounded.candidates)
    assert [(c.target_track_id,c.allowed_event_intervals,c.possible_occluders) for c in bounded.candidates] == [
        (c.target_track_id,c.allowed_event_intervals,c.possible_occluders) for c in full.candidates]
    assert sm.CvPromptBundle.model_validate_json(bounded.model_dump_json()) == bounded
    sm.validate_candidate_identity_evidence(full.summary, bounded.candidates)


def test_tight_row_selection_is_fair_across_tracks_and_source_order():
    timeline = _timeline(0,1,2)
    tracks = tuple(_track(f'item_{i}', 'item', (_observation(0),)) for i in range(8))
    def build(source_tracks, cap=None):
        summary = sm.summarize_cv_evidence(_artifact(source_tracks,processed_timeline=timeline),timeline=timeline)
        return sm.build_cv_prompt_bundle(summary,_thresholds(),max_prompt_chars=cap)
    full = build(tracks)
    basic = full.prompt_record()
    for candidate in basic['candidates']:
        candidate['identity_evidence'] = None
    cap = _size(basic) + 750
    bounded = build(tracks, cap)
    assert bounded == build(tuple(reversed(tracks)), cap)
    assert _rows(bounded.candidates) == 2
    assert bounded.candidates[0].identity_evidence.continuation_cues
    assert bounded.candidates[-1].identity_evidence.continuation_cues
    assert len(bounded.candidates) == 8
    assert _size(bounded.prompt_record()) <= cap


def _partial_complete_source():
    other = _track('item_2','item',(_observation(1),))
    third = _track('item_3','item',(_observation(1),))
    full = gap_bundle(visible=(0,),frames=(0,1,2),extras=(other,third))
    assert full.candidates[0].identity_evidence.complete
    basic = full.prompt_record()
    for candidate in basic['candidates']:
        candidate['identity_evidence'] = None
    bounded = sm.build_cv_prompt_bundle(full.summary,_thresholds(),max_prompt_chars=_size(basic)+400)
    assert len(bounded.candidates[0].identity_evidence.continuation_cues) == 1
    assert not bounded.candidates[0].identity_evidence.complete
    return full, bounded


def test_resealed_partial_selection_cannot_claim_false_completeness():
    full, bounded = _partial_complete_source()
    candidate = bounded.candidates[0]
    forged = sm._with_identity(candidate,candidate.identity_evidence.model_copy(update={'complete':True}))
    with pytest.raises(ValueError,match='completeness'):
        sm.validate_candidate_identity_evidence(full.summary,(forged,*bounded.candidates[1:]))
    with pytest.raises(ValueError,match='canonical candidate set'):
        sm.CvPromptBundle.model_validate(bounded.model_copy(update={'candidates':(forged,*bounded.candidates[1:])}).model_dump(mode='json'))


@pytest.mark.parametrize(('field','value'), [
    ('bbox_iou',.5), ('other_track_id','foreign_1'), ('other_frame_index',2),
    ('target_timestamp_seconds',.1), ('area_similarity',.5),
])
def test_resealed_identity_subset_still_requires_exact_source_rows(field,value):
    full, bounded = _partial_complete_source()
    candidate = bounded.candidates[0]
    payload = candidate.identity_evidence.model_dump(mode='json')
    payload['continuation_cues'][0][field] = value
    forged = sm._with_identity(candidate,sm.IdentityEvidence.model_validate(payload))
    with pytest.raises(ValueError,match='authenticated'):
        sm.validate_candidate_identity_evidence(full.summary,(forged,*bounded.candidates[1:]))


def test_budget_subset_roundtrips_registry_and_renderer():
    from percept_harness.pipelines.output_validation import _validate_occlusion_decision_output
    from percept_harness.pipelines.embodied import PromptRenderer
    from percept_harness.cv.entities import NormalizedEntities
    from percept_harness.cv.contracts import EntityPrompt
    full, bounded = _partial_complete_source()
    result = {'decisions':[{
        'candidate_id':c.candidate_id,'classification':'unknown','target_entity_id':c.target_entity_id,
        'occluder_entity_id':'unknown','events':[],'visual_evidence':'identity remains uncertain','confidence':.2,
    } for c in bounded.candidates]}
    context = {'duration':.2,'candidates':[c.model_dump(mode='json') for c in bounded.candidates],
               'evidence_summary':full.summary.model_dump(mode='json')}
    assert _validate_occlusion_decision_output(result,context) == result
    entities = NormalizedEntities(entities=tuple(EntityPrompt.model_validate(e.model_dump(mode='json'))
                                               for e in full.summary.entities), omitted_count=0)
    prompt = PromptRenderer().occlusion_semantics(bounded.candidates,entities,full.summary,
                                                 video_duration=.2,frame_pts=[0.,.1,.2])
    assert '"complete":false' in prompt


def test_remaining_budget_after_required_candidate_truncation_keeps_a_real_row():
    timeline = _timeline(0,1,2)
    tracks = tuple(_track(f'item_{i}', 'item', (_observation(0),)) for i in range(8))
    summary = sm.summarize_cv_evidence(_artifact(tracks,processed_timeline=timeline),timeline=timeline)
    full = sm.build_cv_prompt_bundle(summary,_thresholds())
    prefix = full.prompt_record()
    prefix['candidates'] = prefix['candidates'][:3]
    for candidate in prefix['candidates']:
        candidate['identity_evidence'] = None
    prefix['candidates_complete'] = False
    prefix['truncation_codes'].append('CANDIDATE_PROMPT_TRUNCATED')
    limit = _size(prefix) + 400
    bounded = sm.build_cv_prompt_bundle(summary,_thresholds(),max_prompt_chars=limit)
    assert len(bounded.candidates) == 3
    assert _rows(bounded.candidates) == 1
    assert not bounded.candidates_complete
    assert _size(bounded.prompt_record()) <= limit
    sm.validate_candidate_identity_evidence(summary,bounded.candidates)


def test_irreducible_evidence_floor_does_not_drop_tracks_to_reserve_optional_rows():
    timeline = _timeline(0,1,2)
    tracks = (_track('item_1','item',(_observation(0),)),
              _track('item_2','item',(_observation(1),)),
              _track('item_3','item',(_observation(1),)))
    artifact = _artifact(tracks,processed_timeline=timeline)
    limits = dict(max_observations_per_track=3,max_relations=1,max_overlays=1)
    reference = sm.summarize_cv_evidence(artifact,timeline=timeline,**limits)
    full = sm.build_cv_prompt_bundle(reference,_thresholds())
    basic = full.prompt_record()
    for candidate in basic['candidates']:
        candidate['identity_evidence'] = None
    # Too small for a cue row but enough for the complete irreducible evidence.
    cap = _size(basic) + 100
    summary = sm.summarize_cv_evidence(artifact,timeline=timeline,thresholds=_thresholds(),
                                      max_prompt_chars=cap,**limits)
    bundle = sm.build_cv_prompt_bundle(summary,_thresholds())
    assert len(summary.tracks) == 3
    assert len(bundle.candidates) == len(full.candidates) == 3
    assert bundle.candidates_complete
    assert _size(bundle.prompt_record()) <= cap
    assert _rows(bundle.candidates) == 0

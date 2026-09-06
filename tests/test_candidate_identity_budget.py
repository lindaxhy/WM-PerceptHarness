"""Actual cue reservation and fair packing under joint evidence limits."""
import json

import pytest

import las_repro.cv.summary as sm
from las_repro.cv.identity import identity_bytes
from test_cv_summary import _artifact, _track, _observation, _timeline, _thresholds
from test_candidate_v2 import gap_bundle


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


def test_budget_subset_roundtrips_registry_renderer_and_operator():
    from las_repro.pipelines.output_validation import _validate_occlusion_decision_output
    from las_repro.pipelines.embodied import PromptRenderer
    from las_repro.cv.entities import NormalizedEntities
    from las_repro.cv.contracts import EntityPrompt
    from scripts.reverify_semantic_stages import _validate_occlusion_alignment
    full, bounded = _partial_complete_source()
    result = {'decisions':[{
        'candidate_id':c.candidate_id,'classification':'unknown','target_entity_id':c.target_entity_id,
        'occluder_entity_id':'unknown','events':[],'visual_evidence':'identity remains uncertain','confidence':.2,
    } for c in bounded.candidates]}
    context = {'duration':.2,'candidates':[c.model_dump(mode='json') for c in bounded.candidates],
               'evidence_summary':full.summary.model_dump(mode='json')}
    assert _validate_occlusion_decision_output(result,context) == result
    values = {'OCCLUSION_CANDIDATES_JSON':[c.prompt_record() for c in bounded.candidates],
              'VIDEO_DURATION_SECONDS_JSON':.2,'CV_EVIDENCE_SUMMARY_JSON':full.summary.prompt_record()}
    _validate_occlusion_alignment(values,None,context)
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

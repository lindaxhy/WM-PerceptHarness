"""Compact hints must retain exact authenticated sources without changing offers."""
import copy
import hashlib
import json
from importlib import resources

import pytest
from test_cv_summary import _artifact, _observation, _track
from test_scene_provenance import source
from las_repro.cv.summary import summarize_cv_evidence
from las_repro.pipelines.scene_choices import canonical, prepare_scene_choices
from las_repro.pipelines.embodied import PromptRenderer


def build(context):
    from las_repro.pipelines.scene_model_view import build_scene_model_view
    return build_scene_model_view(context)


def context_for(summary, segments):
    return prepare_scene_choices(summary, segments, duration=segments[-1]['end']).context()


def values_for(summary, segments):
    from scripts.reverify_semantic_stages import _extract_prompt_data
    prompt = PromptRenderer().scene_semantics(segments, video_duration=segments[-1]['end'],
                                            evidence_summary=summary)
    template = resources.files('las_repro.prompts').joinpath('scene_semantics.txt').read_text()
    values, _, suffix = _extract_prompt_data(prompt, template, 'scene_semantics')
    return prompt, values, suffix


def test_compact_view_preserves_sources_options_and_nonspatial_records():
    summary, segments = source()
    segments[0].update(actor='right_hand', description='完整描述', extra='private enrichment')
    context = context_for(summary, segments)
    original = canonical(context)
    result = build(context)
    assert result['metadata']['mode'] == 'compact'
    view = result['data']
    assert view['known_targets'] == context['known_targets']
    assert view['segments'][0] == dict(segment_index=0, start=0.0, end=1.0,
        target='item 0', actor='right_hand', description='完整描述')
    assert len(view['segments']) == len(segments)
    assert [o['option_id'] for o in view['options']['options']] == [
        o['option_id'] for o in context['spatial_options']['options']]
    assert view['entities'] == [dict(entity_id=e.entity_id, canonical_label=e.canonical_label,
        role=e.role.value) for e in summary.entities]
    assert canonical(context) == original
    assert result == build(context)
    assert result['metadata']['data_sha256'] == hashlib.sha256(canonical(view).encode()).hexdigest()


def test_independently_sorted_tracks_preserve_directed_pair_and_tie_break():
    summary = summarize_cv_evidence(_artifact((
        _track('a_track', 'z_entity', tuple(_observation(i, timestamp_seconds=float(i),
            bbox_xyxy=(0.2, 0.2, 0.4, 0.4)) for i in (0, 1, 2, 3))),
        _track('z_track', 'a_entity', tuple(_observation(i, timestamp_seconds=float(i),
            bbox_xyxy=(0.1, 0.1, 0.5, 0.5)) for i in (0, 1, 2, 3))),
    )))
    segments = [dict(segment_index=0, start=0.0, end=3.0, target='unknown')]
    context = context_for(summary, segments)
    view = build(context)['data']
    relation_index = next(i for i, o in enumerate(view['options']['options']) if o['kind'] == 'relation')
    assert view['options']['options'][relation_index]['object_ids'] == ['a_entity', 'z_entity']
    _, observations, pairs = view['evidence']['option_witnesses'][relation_index]
    assert len(observations) == 2
    assert [view['evidence']['observations'][i]['object_id'] for i in observations] == ['z_entity', 'a_entity']
    pair = view['evidence']['pairs'][pairs[0]]
    assert pair == dict(subject_object_id='z_entity', object_object_id='a_entity', time=1.0,
                       bbox_iou=0.25, subject_covered=1.0, object_covered=0.25)
    assert all(view['evidence']['observations'][i]['time'] == 1.0 for i in observations)
    assert 'a_track' not in canonical(view) and 'z_track' not in canonical(view)


def test_lifecycle_retains_all_runs_gaps_endpoints_and_null_confidence():
    summary = summarize_cv_evidence(_artifact((_track('item_track', 'item', tuple(
        _observation(i, visible=i not in (1, 2, 4), confidence=0.3 if i == 2 else 0.9)
        for i in range(5))),)))
    segments = [dict(segment_index=0, start=0.0, end=1.0, target='item')]
    track = summary.tracks[0]
    view = build(context_for(summary, segments))['data']
    lifecycle = view['lifecycle'][0]
    assert lifecycle['track_ref'] == 0
    assert lifecycle['entity_id'] == 'item'
    assert lifecycle['visibility_lifecycle_complete'] == track.visibility_lifecycle_complete
    assert lifecycle['runs'] == [{k: getattr(r, k) for k in
        ('start_time', 'end_time', 'state', 'minimum_confidence')} for r in track.visibility_runs]
    assert lifecycle['gaps'] == [{k: getattr(g, k) for k in ('last_visible_time',
        'first_missing_time', 'last_missing_time', 'first_revisible_time',
        'minimum_confidence', 'edge_departure')} for g in track.missing_intervals]
    assert lifecycle['gaps'][-1]['first_revisible_time'] is None
    assert [r['time'] for r in lifecycle['endpoints']] == [0.0, 0.4]
    assert view['completeness']['retained_lifecycle_rows_complete'] is True
    assert view['completeness']['witness_selection'] == 'representative_subset'


def test_no_cv_and_available_empty_are_distinct():
    segments = [dict(segment_index=0, start=0.0, end=1.0, target='required object')]
    no_cv = build(context_for(None, segments))['data']
    empty_summary = summarize_cv_evidence(_artifact(()))
    empty = build(context_for(empty_summary, segments))['data']
    assert no_cv['cv_available'] is False and empty['cv_available'] is True
    for view in (no_cv, empty):
        assert view['options'] == dict(options=[], options_complete=True)
        assert view['evidence']['observations'] == []
        assert view['known_targets'] == [dict(object_id='required_object', name='required object')]


def test_utf8_size_fallback_retains_full_text_same_mode_on_repair():
    segments = [dict(segment_index=0, start=0.0, end=1.0, target='target',
                     description='字' * 18000)]
    result = build(context_for(None, segments))
    assert result['data'] is None
    assert result['metadata']['mode'] == 'full_fallback'
    assert result['metadata']['reason'] == 'data_bytes'
    initial, values, _ = values_for(None, segments)
    repair = PromptRenderer().scene_semantics(segments, video_duration=1.0,
        repair={'issue_codes': ['SCENE_SEMANTICS_CHOICES_ACTOR_ENUM_VALUE']})
    repair_view = json.JSONDecoder().raw_decode(repair.split('[SCENE_MODEL_VIEW_JSON]\n', 1)[1])[0]
    assert values['SCENE_MODEL_VIEW_JSON'] == repair_view
    assert values['SEGMENTS_JSON'] == segments
    assert '字' * 18000 in initial


def test_compact_prompt_and_repair_fit_ceiling_and_operator_rejects_view_tampering():
    from scripts.reverify_semantic_stages import _validate_scene_alignment, OperatorError
    summary, segments = source()
    prompt, values, suffix = values_for(summary, segments)
    assert len(prompt.encode()) <= 65536
    assert suffix is None
    assert values['SCENE_MODEL_VIEW_JSON']['metadata']['mode'] == 'compact'
    context = context_for(summary, segments)
    _validate_scene_alignment(values, suffix, context)
    bad = copy.deepcopy(values)
    bad['SCENE_MODEL_VIEW_JSON']['evidence']['observations'][0]['time'] += 0.01
    with pytest.raises(OperatorError):
        _validate_scene_alignment(bad, suffix, context)
    context['segments'][0]['target'] = 'tampered'
    with pytest.raises(OperatorError):
        _validate_scene_alignment(values, suffix, context)


def test_available_track_cap_falls_back_without_truncating_source():
    summary, segments = source(count=33, frames=2)
    result = build(context_for(summary, segments))
    assert result['metadata']['mode'] == 'full_fallback'
    assert result['metadata']['reason'] == 'row_limits'
    _, values, suffix = values_for(summary, segments)
    assert values['SEGMENTS_JSON'] == segments
    assert suffix == summary.prompt_record()


def test_disjoint_center_witnesses_keep_their_observed_times():
    summary = summarize_cv_evidence(_artifact((
        _track('a_track', 'a', tuple(_observation(i, timestamp_seconds=float(i)) for i in (0, 1, 3))),
        _track('b_track', 'b', tuple(_observation(i, timestamp_seconds=float(i)) for i in (0, 2, 3))),
    )))
    segments = [dict(segment_index=0, start=0.0, end=3.0, target='unknown')]
    view = build(context_for(summary, segments))['data']
    index = next(i for i, o in enumerate(view['options']['options']) if o['kind'] == 'relation')
    _, obs, pairs = view['evidence']['option_witnesses'][index]
    assert [view['evidence']['observations'][i]['time'] for i in obs] == [1.0, 2.0]
    assert view['evidence']['pairs'][pairs[0]]['time'] == 0.0


def test_incomplete_search_and_null_offers_do_not_invent_witnesses():
    from las_repro.cv.summary import _summary_identity
    summary, segments = source()
    suffix = json.dumps(summary.prompt_record(), ensure_ascii=False, separators=(',', ':'))
    summary = summary.model_copy(update={'prompt_char_limit': len(suffix)})
    summary = summary.model_copy(update={'summary_id': _summary_identity(summary)})
    view = build(context_for(summary, segments))['data']
    assert view['options'] is None
    assert view['evidence']['option_witnesses'] == []
    assert view['completeness']['candidate_search_complete'] is False
    assert view['completeness']['options_complete'] is None
    assert len(view['lifecycle']) == 2


@pytest.mark.parametrize('limit', ['options', 'observations', 'pairs', 'tracks', 'endpoints', 'runs', 'gaps'])
def test_each_row_budget_falls_back_whole_instead_of_truncating(monkeypatch, limit):
    import las_repro.pipelines.scene_model_view as module
    summary = summarize_cv_evidence(_artifact(tuple(_track(f'item_{i}_track', f'item_{i}', tuple(
        _observation(f, visible=f != 1) for f in range(4))) for i in range(2))))
    segments = [dict(segment_index=0, start=0.0, end=1.0, target='unknown')]
    context = context_for(summary, segments)
    counts = build(context)['metadata']['counts']
    assert counts[limit] > 0
    monkeypatch.setitem(module.LIMITS, limit, counts[limit] - 1)
    fallback = build(context)
    assert fallback['data'] is None
    assert fallback['metadata']['reason'] == 'row_limits'
    _, values, suffix = values_for(summary, segments)
    assert values['SEGMENTS_JSON'] == segments
    assert suffix == summary.prompt_record()


def test_prompt_budget_reserves_repair_space_without_switching_mode(monkeypatch):
    import las_repro.pipelines.scene_model_view as module
    segments = [dict(segment_index=0, start=0.0, end=1.0, target='target', description='x' * 1000)]
    prompt, _, _ = values_for(None, segments)
    monkeypatch.setattr(module, 'MAX_PROMPT_BYTES', len(prompt.encode()) + module.MAX_REPAIR_BYTES - 1)
    assert build(context_for(None, segments))['data'] is not None
    initial, values, _ = values_for(None, segments)
    assert values['SCENE_MODEL_VIEW_JSON']['metadata']['reason'] == 'prompt_bytes'
    repair = PromptRenderer().scene_semantics(segments, video_duration=1.0,
        repair={'issue_codes': ['SCENE_SEMANTICS_CHOICES_ACTOR_ENUM_VALUE']})
    meta = json.JSONDecoder().raw_decode(repair.split('[SCENE_MODEL_VIEW_JSON]\n', 1)[1])[0]
    assert meta == values['SCENE_MODEL_VIEW_JSON']


def test_all_registered_repair_codes_fit_reserved_space_and_keep_identical_view():
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
    summary, segments = source()
    initial, values, _ = values_for(summary, segments)
    codes = list(DEFAULT_OUTPUT_SCHEMAS._entries['SceneSemanticsChoices'].allowed_issue_codes)
    repair = PromptRenderer().scene_semantics(segments, video_duration=segments[-1]['end'],
        evidence_summary=summary, repair={'issue_codes': codes})
    meta = json.JSONDecoder().raw_decode(repair.split('[SCENE_MODEL_VIEW_JSON]\n', 1)[1])[0]
    assert meta == values['SCENE_MODEL_VIEW_JSON']
    assert len(repair.encode()) <= 65536


def test_operator_direct_stage_rejects_tampered_compact_view_before_output(tmp_path):
    from test_semantic_reverification import _scene_stage, _Model, _valid_scene
    from scripts.reverify_semantic_stages import run_stage, OperatorError
    stage = _scene_stage(tmp_path)
    context = stage['payload']['schema_context']
    prompt = PromptRenderer().scene_semantics(context['segments'], video_duration=context['duration'])
    stage['original_template'] = resources.files('las_repro.prompts').joinpath('scene_semantics.txt').read_text()
    stage['payload']['prompt'] = prompt.replace('"representative_subset"', '"tampered"')
    output = tmp_path / 'output'; output.mkdir(mode=0o700)
    with pytest.raises(OperatorError, match='SCENE_PROMPT_CONTEXT_MISMATCH'):
        run_stage(stage, output_dir=output, model=_Model([_valid_scene()]))
    assert list(output.iterdir()) == []


def test_operator_cannot_bypass_compact_repair_text_ceiling():
    from scripts.reverify_semantic_stages import rebuild_prompt, OperatorError
    summary, segments = source()
    prompt, _, _ = values_for(summary, segments)
    template = resources.files('las_repro.prompts').joinpath('scene_semantics.txt').read_text()
    with pytest.raises(OperatorError, match='REPAIR_INVALID'):
        rebuild_prompt(prompt, template, 'scene_semantics', repair={'issue_codes': ['x' * 65536]})

"""Reference DTOs must authenticate sources and preserve the public boundary."""
import copy
import json

import pytest
from test_scene_provenance import source, scene_from, context

from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS as registry
from las_repro.pipelines.scene_provenance import scene_spatial_prompt_data


def fixture():
    from las_repro.pipelines.scene_choices import prepare_scene_choices
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


def test_projection_preserves_nonspatial_and_exact_sources_and_direction():
    from las_repro.pipelines.scene_choices import project_scene_choices
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
    from las_repro.pipelines.scene_choices import prepare_scene_choices, project_scene_choices
    draft, _ = fixture()
    _, segments = source()
    ctx = prepare_scene_choices(None, segments, duration=segments[-1]['end']).context()
    draft['locations'] = []; draft['relations'] = []
    assert project_scene_choices(draft, ctx) == draft
    for value in ({'_schema_validation': {'status': 'normalized'}, 'data': draft},
                  {'projected': draft}):
        assert '_schema_validation' in registry.sanitize('SceneSemanticsChoices', value, ctx)


def test_compiled_schema_separates_kind_enums_and_empty_branches():
    from las_repro.pipelines.scene_choices import prepare_scene_choices
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
    from las_repro.pipelines.scene_choices import project_scene_choices
    draft, ctx = fixture()
    draft['relations'].append(dict(draft['relations'][0], relation='on'))
    reversed_draft = copy.deepcopy(draft)
    reversed_draft['locations'].reverse(); reversed_draft['relations'].reverse()
    assert project_scene_choices(draft, ctx) == project_scene_choices(reversed_draft, ctx)


@pytest.mark.parametrize('branch', ['empty_choices', 'empty_offers', 'null_offers'])
def test_available_cv_fallbacks_preserve_nonspatial(branch):
    from las_repro.cv.summary import _summary_identity
    from las_repro.pipelines.scene_choices import prepare_scene_choices, project_scene_choices
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
    from las_repro.pipelines.scene_choices import SceneSemanticsChoices, prepare_scene_choices
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
    from las_repro.pipelines.scene_choices import authenticate_scene_context
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
    from las_repro.pipelines.output_validation import _scene_choice_pydantic_issue_codes
    error = ValidationError.from_exception_data('synthetic', [
        dict(type=kind, loc=path, input='private raw value', ctx={'expected': 'private message'})])
    assert _scene_choice_pydantic_issue_codes(error) == ('SCENE_SEMANTICS_CHOICES_ENUM_VALUE',)


def test_scene_unknown_requires_explicit_model_value_and_retains_supported_records():
    from las_repro.pipelines.scene_choices import project_scene_choices
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
    from las_repro.pipelines.scene_choices import project_scene_choices
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
    from las_repro.pipelines.output_validation import (
        DEFAULT_OUTPUT_SCHEMAS, NormalizedSchemaOutput,
    )
    from las_repro.pipelines.scene_choices import normalize_scene_choice_mechanics

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
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

    draft, ctx = fixture()
    draft['outcome']['status'] = 'victorious'
    flagged = dict(ctx)
    flagged['allow_scene_normalization'] = True
    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize('SceneSemanticsChoices', draft, flagged)
    assert sanitized['_schema_validation']['status'] == 'invalid'


def test_invented_option_ids_are_dropped_locally_without_a_model_call():
    """A selection referencing an unoffered option is removed, not retried."""
    from las_repro.pipelines.output_validation import (
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
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

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

"""BoundaryPlan generation, closed repair feedback, and immutable public contracts."""
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest
from pydantic import ValidationError

from las_repro.models.response_contract import ModelResponseContract
from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS as registry
from las_repro.pipelines.validators import BoundaryPlan

COARSE_EVENTS = ['idle', 'reach_and_grasp', 'lift', 'transport', 'lower_and_place',
                 'release', 'retract', 'search_or_adjust', 'unknown_action']
FINE_EVENTS = ['action_start', 'idle', 'reach_start', 'approach', 'pre_contact',
               'contact_start', 'grasp_secured', 'lift_start', 'lift_continue',
               'transport_start', 'transport_continue', 'lower_start',
               'destination_contact', 'place_continue', 'release_start',
               'release_end', 'retract_start', 'action_end', 'unknown_transition']
CODES = ['BOUNDARY_PLAN_ACTION_EVENT_TYPE_ENUM_VALUE',
         'BOUNDARY_PLAN_BOUNDARY_POINT_EVENT_TYPE_ENUM_VALUE',
         'BOUNDARY_PLAN_FINE_SEGMENT_EVENT_TYPE_ENUM_VALUE']


def boundary_fixture():
    from test_embodied_pipeline import _valid_boundary_output, _entity_candidates
    draft = _valid_boundary_output()
    coarse = dict(task_description=draft['task_description'], entity_candidates=_entity_candidates(),
                  actions=[{k: v for k, v in a.items() if k not in ('boundary_points', 'fine_segments')}
                           for a in draft['actions']])
    return draft, dict(coarse_plan=coarse, max_segment_seconds=1.0, allow_topology_fallback=False)


def assert_boundary_schema(schema):
    assert set(schema['properties']) == {'task_description', 'actions'}
    assert schema['additionalProperties'] is False
    actions = schema['properties']['actions']
    assert 'maxItems' not in actions
    action = actions['items']['properties']
    assert action['event_type']['enum'] == COARSE_EVENTS
    for field in ('boundary_points', 'fine_segments'):
        assert 'maxItems' not in action[field]
        assert action[field]['items']['additionalProperties'] is False
        assert action[field]['items']['properties']['event_type']['enum'] == FINE_EVENTS
    assert '$ref' not in json.dumps(schema)
    assert '$defs' not in schema


def test_boundary_factory_authenticates_and_compiles_exact_existing_vocabularies():
    _, context = boundary_fixture()
    snapshot = copy.deepcopy(context)
    contract = registry.model_response_contract('BoundaryPlan', context)
    assert contract is not None
    assert contract.name == 'boundary-plan-v1'
    assert contract.format()['strict'] is True
    assert_boundary_schema(contract.format()['schema'])
    assert context == snapshot
    assert registry.model_response_contract('unregistered', context) is None
    legacy = {k: v for k, v in context.items() if k != 'allow_topology_fallback'}
    assert registry.model_response_contract('BoundaryPlan', legacy) == contract
    context['allow_topology_fallback'] = True
    assert registry.model_response_contract('BoundaryPlan', context) == contract


@pytest.mark.parametrize('change', ['missing', 'extra', 'schema_injection', 'invalid_flag',
    'nan', 'zero', 'bool_max', 'string_max', 'bad_coarse_enum', 'empty_actions',
    'repeated_index', 'nonpositive', 'empty_entities', 'infinite_end'])
def test_invalid_boundary_context_cannot_compile(change):
    _, context = boundary_fixture()
    if change == 'missing': del context['coarse_plan']
    if change == 'extra': context['private_extra'] = 'private value'
    if change == 'schema_injection': context['response_schema'] = {'type': 'string'}
    if change == 'invalid_flag': context['allow_topology_fallback'] = 1
    for key, value in [('nan', float('nan')), ('zero', 0), ('bool_max', True), ('string_max', '1')]:
        if change == key: context['max_segment_seconds'] = value
    coarse = context.get('coarse_plan', {})
    if change == 'bad_coarse_enum': coarse['actions'][0]['event_type'] = 'private synonym'
    if change == 'empty_actions': coarse['actions'] = []
    if change == 'repeated_index': coarse['actions'][1]['action_index'] = 0
    if change == 'nonpositive': coarse['actions'][1]['end'] = 1.0
    if change == 'empty_entities': coarse['entity_candidates'] = []
    if change == 'infinite_end': coarse['actions'][1]['end'] = float('inf')
    with pytest.raises(ValueError): registry.model_response_contract('BoundaryPlan', context)


@pytest.mark.parametrize('attack', ['bytes', 'nodes', 'depth', 'external_ref', 'recursive_ref'])
def test_boundary_schema_compilation_is_bounded_and_local(monkeypatch, attack):
    schema = {'type': 'object'}
    if attack == 'bytes': schema['title'] = 'x' * 65537
    if attack == 'nodes': schema['required'] = ['x'] * 5000
    if attack == 'depth':
        for _ in range(33): schema = {'items': schema}
    if attack == 'external_ref': schema = {'$ref': 'https://example.invalid/private'}
    if attack == 'recursive_ref': schema = {'$defs': {'loop': {'$ref': '#/$defs/loop'}}, '$ref': '#/$defs/loop'}
    monkeypatch.setattr(BoundaryPlan, 'model_json_schema', lambda: schema)
    with pytest.raises(ValueError): registry.model_response_contract('BoundaryPlan', boundary_fixture()[1])


def test_boundary_context_preflight_rejects_nonplain_and_unbounded_input():
    class Hostile(dict):
        def items(self): raise AssertionError('must not traverse custom container')
    with pytest.raises(ValueError): registry.model_response_contract('BoundaryPlan', Hostile())
    _, context = boundary_fixture()
    context['coarse_plan']['task_description'] = 'x' * (16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError): registry.model_response_contract('BoundaryPlan', context)


def test_response_descriptor_accepts_only_two_names_with_immutable_canonical_identity():
    encoded = '{"type":"object"}'
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    for name in ('scene-spatial-choice-refs-v1', 'boundary-plan-v1'):
        contract = ModelResponseContract(name, encoded, digest)
        with pytest.raises(FrozenInstanceError): contract.name = 'external'
        for changes in ({'name': 'external'}, {'schema_sha256': '0' * 64},
                        {'schema_json': '{ "type": "object" }'}):
            with pytest.raises(ValueError): replace(contract, **changes)
        detached = contract.format()
        detached['schema']['type'] = 'string'
        assert contract.format()['schema'] == {'type': 'object'}


@pytest.mark.parametrize('field,index', [(None, 0), ('boundary_points', 1), ('fine_segments', 2)])
def test_boundary_enum_feedback_is_field_specific_static_and_nonmutating(field, index):
    draft, context = boundary_fixture()
    row = draft['actions'][0]
    if field: row = row[field][0]
    row['event_type'] = 'private file:///secret synonym'
    snapshot = copy.deepcopy(draft)
    failure = registry.sanitize('BoundaryPlan', draft, context)
    assert failure == {'_schema_validation': {'schema_name': 'BoundaryPlan', 'status': 'invalid',
                                            'issue_codes': [CODES[index]]}}
    assert registry.failure_codes('BoundaryPlan', failure) == (CODES[index],)
    assert draft == snapshot


def test_boundary_enum_codes_have_deterministic_deduplicated_order():
    draft, context = boundary_fixture()
    for action in reversed(draft['actions']):
        action['event_type'] = 'private coarse'
        for field in ('fine_segments', 'boundary_points'):
            for row in action[field]: row['event_type'] = 'private fine'
    draft['private extra'] = 'private value'
    del draft['task_description']
    codes = registry.sanitize('BoundaryPlan', draft, context)['_schema_validation']['issue_codes']
    assert codes == ['BOUNDARY_PLAN_MISSING_FIELD', 'BOUNDARY_PLAN_EXTRA_FIELD', *CODES]


@pytest.mark.parametrize('kind,path', [
    ('enum', ('private field', 2, 'event_type')), ('enum', ('actions', '0', 'event_type')),
    ('enum', ('actions', -1, 'event_type')), ('enum', ('actions', 0, 'event_type', 'extra')),
    ('enum', ('actions', 0, 'boundary_points', -1, 'event_type')),
    ('enum', ('actions', 0, 'fine_segments', 'private', 'event_type')),
    ('literal_error', ('actions', 0, 'fine_segments', 2, 'event_type')),
])
def test_unexpected_boundary_enum_paths_or_kinds_remain_generic(kind, path):
    import las_repro.pipelines.output_validation as validation
    mapper = getattr(validation, '_boundary_pydantic_issue_codes', None)
    assert callable(mapper)
    error = ValidationError.from_exception_data('synthetic', [dict(type=kind, loc=path,
        input='private raw value', ctx={'expected': 'private message'})])
    assert mapper(error) == ('BOUNDARY_PLAN_ENUM_VALUE',)


def test_explicit_unknown_transition_preserves_model_output_and_valid_schema():
    draft, context = boundary_fixture()
    draft['actions'][0]['boundary_points'][1]['event_type'] = 'unknown_transition'
    draft['actions'][0]['fine_segments'][0]['event_type'] = 'unknown_transition'
    snapshot = copy.deepcopy(draft)
    assert registry.sanitize('BoundaryPlan', draft, context) == snapshot
    assert draft == snapshot


def test_task2k_preserves_public_schemas_and_scene_response_bytes():
    from las_repro.pipelines.scene_choices import SceneSemanticsChoices, prepare_scene_choices
    for model, expected in [(BoundaryPlan, 'c072d2ff4f715ca04e2d11137e2d027a41a01be80633192ae9939df848d862ba'),
                            (SceneSemanticsChoices, '7e644b9577192c31f936fffcf9ec8d931fe0b7d76942beff86a72d6ea1a57a80')]:
        encoded = json.dumps(model.model_json_schema(), sort_keys=True, separators=(',', ':'))
        assert hashlib.sha256(encoded.encode()).hexdigest() == expected
    empty = prepare_scene_choices(None, [], duration=1.0).context()
    assert registry.model_response_contract('SceneSemanticsChoices', empty).schema_sha256 == 'ea341cf3f9675593b4193233c5ca3881351e7c70ab1fe4ff450ddc2430bc8e69'

    from test_scene_choices import fixture
    assert registry.model_response_contract('SceneSemanticsChoices', fixture()[1]).schema_sha256 == 'cce8941ac7f90ed4baf69d5789fdc5f1c518a66ad4d33b5e5949db5cac399c48'

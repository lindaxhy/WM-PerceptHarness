"""Authenticated scene reference DTOs and explicit public-scene projection.

The immutable package is private server context. Option IDs are lookup keys;
all consumers rederive the bounded offer set from the original source. Registry
acceptance uses initial provenance; final projection uses the completed attempt.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

from ..cv.summary import CvEvidenceSummary
from ..models.response_contract import (
    CHOICE_CONTRACT, MAX_SCHEMA_BYTES, MAX_SCHEMA_DEPTH, MAX_SCHEMA_NODES,
    ModelResponseContract, canonical, preflight_plain, compile_local_schema,
)
from .scene_provenance import MAX_OPTIONS, scene_spatial_prompt_data
from .scene_semantics import (
    Confidence, SceneRelation, SceneSemantics, _SceneModel, trusted_target_skeleton,
)

_CONTEXT_KEYS = {'duration', 'require_observed_content', 'required_object_ids',
                 'evidence_summary', 'segments', 'known_targets', 'spatial_options',
                 'spatial_options_sha256', 'choice_contract'}
CHOICE_CODES = tuple('SCENE_SEMANTICS_CHOICES_' + suffix for suffix in (
    'OPTION_UNKNOWN', 'OPTION_KIND', 'OBJECT_BINDING', 'DUPLICATE',
    'SELECTION_LIMIT', 'PROHIBITED_CONTENT'))


class SceneLocationChoice(_SceneModel):
    option_id: str
    location: str
    visual_evidence: str
    confidence: Confidence


class SceneRelationChoice(_SceneModel):
    option_id: str
    direction: Literal['forward', 'reverse']
    relation: SceneRelation.model_fields['relation'].annotation
    visual_evidence: str
    confidence: Confidence


class SceneSemanticsChoices(SceneSemantics):
    locations: list[SceneLocationChoice] = Field(max_length=MAX_OPTIONS)
    relations: list[SceneRelationChoice] = Field(max_length=MAX_OPTIONS)


@dataclass(frozen=True)
class SceneInputPackage:
    context_json: str

    def context(self) -> dict[str, Any]:
        return json.loads(self.context_json)


def prepare_scene_choices(summary: CvEvidenceSummary | None,
                          segments: list[dict[str, Any]], *, duration: float) -> SceneInputPackage:
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('scene duration must be finite and positive')
    preflight_plain(segments, max_bytes=8 * 1024 * 1024, max_nodes=500_000, max_depth=32)
    offers, _ = scene_spatial_prompt_data(summary, segments, duration=duration)
    targets = trusted_target_skeleton(segments)
    context = dict(duration=duration, require_observed_content=bool(targets),
                   required_object_ids=[t['object_id'] for t in targets],
                   known_targets=targets, segments=segments,
                   evidence_summary=summary.model_dump(mode='json') if summary is not None else None,
                   spatial_options=offers,
                   spatial_options_sha256=hashlib.sha256(canonical(offers).encode()).hexdigest(),
                   choice_contract=CHOICE_CONTRACT)
    return SceneInputPackage(canonical(context))


def authenticate_scene_context(context: Any) -> dict[str, Any]:
    # This failure is an internal source/context failure, never a repairable
    # model selection failure. It must happen before schema compilation/cache use.
    preflight_plain(context, max_bytes=16 * 1024 * 1024, max_nodes=1_000_000, max_depth=32)
    if type(context) is not dict or set(context) != _CONTEXT_KEYS:
        raise ValueError('scene choice context is invalid')
    if context['choice_contract'] != CHOICE_CONTRACT:
        raise ValueError('scene choice contract is invalid')
    summary = (CvEvidenceSummary.model_validate(context['evidence_summary'])
               if context['evidence_summary'] is not None else None)
    expected = prepare_scene_choices(summary, context['segments'], duration=context['duration']).context()
    if canonical(context) != canonical(expected):
        raise ValueError('scene choice source or offer binding is invalid')
    return expected


def compact_scene_options(context: dict[str, Any]) -> dict[str, Any] | None:
    offers = context['spatial_options']
    if offers is None:
        return None
    keys = ('option_id', 'kind', 'object_ids', 'object_names', 'start', 'end')
    return dict(options=[{k: o[k] for k in keys} for o in offers['options']],
                options_complete=offers['options_complete'])


def project_scene_choices(result: Any, context: Any, *,
                          repair_history: tuple[str, ...] = ('initial',)) -> dict[str, Any]:
    from .output_validation import (
        DeclaredSchemaOutputError, _model_from_json, _scene_choice_pydantic_issue_codes,
        _validate_scene_semantics_output,
    )
    from .hybrid_result import _reject_artifact_text
    from pydantic import ValidationError

    context = authenticate_scene_context(context)
    if repair_history not in (('initial',), ('initial', 'repair'), ('initial', 'repair', 'repair')):
        raise ValueError('scene repair history is invalid')
    def fail(suffix):
        raise DeclaredSchemaOutputError(('SCENE_SEMANTICS_CHOICES_' + suffix,))
    try:
        draft = _model_from_json(SceneSemanticsChoices, result)
    except ValidationError as error:
        raise DeclaredSchemaOutputError(_scene_choice_pydantic_issue_codes(error)) from None
    except (TypeError, ValueError, OverflowError, RecursionError):
        fail('SCHEMA_INVALID')
    if len(draft.locations) + len(draft.relations) > MAX_OPTIONS:
        fail('SELECTION_LIMIT')
    data = draft.model_dump(mode='json')
    try:
        _reject_artifact_text(data)
    except ValueError:
        fail('PROHIBITED_CONTENT')
    offered = context['spatial_options']
    options = {o['option_id']: o for o in offered['options']} if offered else {}
    names = {o.object_id: o.name for o in draft.objects}
    for collection, kind in (('locations', 'location'), ('relations', 'relation')):
        rows = []
        seen = set()
        for choice in data[collection]:
            option = options.get(choice['option_id'])
            if option is None:
                fail('OPTION_UNKNOWN')
            if option['kind'] != kind:
                fail('OPTION_KIND')
            key = (choice['option_id'],) if kind == 'location' else (
                choice['option_id'], choice['direction'], choice['relation'])
            if key in seen:
                fail('DUPLICATE')
            seen.add(key)
            if any(names.get(i) != n for i, n in zip(option['object_ids'], option['object_names'])):
                fail('OBJECT_BINDING')
            row = {k: option[k] for k in ('start', 'end', 'source_track_ids',
                                          'source_keyframe_ids', 'source_segment_indices')}
            row.update(branch='scene', model_stage='scene_semantics', evidence_mode='hybrid',
                       review_status='not_required', repair_history=list(repair_history),
                       visual_evidence=choice['visual_evidence'], confidence=choice['confidence'])
            if kind == 'location':
                row.update(object_id=option['object_ids'][0], location=choice['location'])
            else:
                ids = option['object_ids'] if choice['direction'] == 'forward' else option['object_ids'][::-1]
                row.update(subject_object_id=ids[0], object_object_id=ids[1], relation=choice['relation'])
            rows.append(row)
        ids = ('object_id',) if kind == 'location' else ('subject_object_id', 'object_object_id')
        # The final tie-breaker makes distinct claims on an identical interval
        # deterministic without changing the public validator's global key.
        data[collection] = sorted(
            rows, key=lambda r: (r['start'], r['end'], *(r[k] for k in ids), canonical(r)))
    public_context = {k: context[k] for k in ('duration', 'require_observed_content',
                     'required_object_ids', 'evidence_summary', 'segments')}
    return _validate_scene_semantics_output(data, public_context)


_RELATION_PREDICATES = frozenset(
    SceneRelation.model_fields['relation'].annotation.__args__
)
SCENE_NORMALIZATION_CODES = (
    'SCENE_EVENT_START_NOT_ORDERED',
    'SCENE_EVENT_INDEX_NOT_CONTIGUOUS',
    'SCENE_EVENT_UNKNOWN_OBJECT',
    'SCENE_SEMANTICS_CHOICES_RELATION_PREDICATE_ENUM_VALUE',
)


def normalize_scene_choice_mechanics(result: Any) -> tuple[Any, tuple[str, ...], int]:
    """Fix purely mechanical scene faults locally before strict validation.

    Sorting and renumbering events, downgrading dangling event targets to
    unknown, and dropping relation rows with an out-of-vocabulary predicate
    never require another model call. Anything structurally unexpected is
    left untouched for the strict validator to reject.
    """
    if type(result) is not dict:
        return result, (), 0
    fixed = json.loads(json.dumps(result))
    codes: list[str] = []
    count = 0
    events = fixed.get('semantic_events')
    if isinstance(events, list) and events and all(
        isinstance(e, dict)
        and type(e.get('start')) in (int, float)
        and math.isfinite(e.get('start'))
        for e in events
    ):
        ordered = sorted(events, key=lambda e: e['start'])
        if ordered != events:
            codes.append('SCENE_EVENT_START_NOT_ORDERED')
            count += sum(a is not b for a, b in zip(events, ordered))
        renumbered = 0
        for position, event in enumerate(ordered):
            if event.get('event_index') != position:
                event['event_index'] = position
                renumbered += 1
        if renumbered and 'SCENE_EVENT_START_NOT_ORDERED' not in codes:
            codes.append('SCENE_EVENT_INDEX_NOT_CONTIGUOUS')
        count += renumbered
        fixed['semantic_events'] = ordered
        objects = fixed.get('objects')
        if isinstance(objects, list):
            declared = {
                o.get('object_id')
                for o in objects
                if isinstance(o, dict) and isinstance(o.get('object_id'), str)
            }
            dangling = 0
            for event in ordered:
                target = event.get('target_object_id')
                if isinstance(target, str) and target != 'unknown' and target not in declared:
                    event['target_object_id'] = 'unknown'
                    dangling += 1
            if dangling:
                codes.append('SCENE_EVENT_UNKNOWN_OBJECT')
                count += dangling
    relations = fixed.get('relations')
    if isinstance(relations, list):
        kept = [
            row
            for row in relations
            if not isinstance(row, dict)
            or not isinstance(row.get('relation'), str)
            or row['relation'] in _RELATION_PREDICATES
        ]
        dropped = len(relations) - len(kept)
        if dropped:
            codes.append('SCENE_SEMANTICS_CHOICES_RELATION_PREDICATE_ENUM_VALUE')
            count += dropped
            fixed['relations'] = kept
    if not codes:
        return result, (), 0
    return fixed, tuple(codes), count


def validate_scene_choices(result: Any, context: Any) -> dict[str, Any]:
    from .output_validation import NormalizedSchemaOutput

    allow = type(context) is dict and context.get('allow_scene_normalization') is True
    if allow:
        context = {k: v for k, v in context.items() if k != 'allow_scene_normalization'}
        fixed, codes, count = normalize_scene_choice_mechanics(result)
        if codes:
            projected = project_scene_choices(fixed, context)
            SceneSemanticsChoices.model_validate(fixed)
            return NormalizedSchemaOutput(
                data=SceneSemanticsChoices.model_validate(fixed).model_dump(mode='json'),
                issue_codes=codes,
                normalized_field_count=count,
            )
    project_scene_choices(result, context)
    return SceneSemanticsChoices.model_validate(result).model_dump(mode='json')


def scene_response_contract(context: Any) -> ModelResponseContract:
    if type(context) is dict and 'allow_scene_normalization' in context:
        context = {k: v for k, v in context.items() if k != 'allow_scene_normalization'}
    context = authenticate_scene_context(context)
    source = SceneSemanticsChoices.model_json_schema()
    schema = compile_local_schema(source)
    offered = context['spatial_options']
    for field, kind in (('locations', 'location'), ('relations', 'relation')):
        ids = [o['option_id'] for o in offered['options'] if o['kind'] == kind] if offered else []
        collection = schema['properties'][field]
        collection['maxItems'] = 64 if ids else 0
        if ids:
            collection['items']['properties']['option_id']['enum'] = ids
    preflight_plain(schema, max_bytes=MAX_SCHEMA_BYTES, max_nodes=MAX_SCHEMA_NODES, max_depth=MAX_SCHEMA_DEPTH)
    encoded = canonical(schema)
    if len(encoded.encode()) > MAX_SCHEMA_BYTES:
        raise ValueError('scene response schema exceeds byte limit')
    return ModelResponseContract(CHOICE_CONTRACT, encoded, hashlib.sha256(encoded.encode()).hexdigest())

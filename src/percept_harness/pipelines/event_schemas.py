"""Strict natural event contracts; times always refer to the source video."""
from typing import Annotated, Literal
import hashlib
import math
from ..models.response_contract import ModelResponseContract, compile_local_schema, canonical
from pydantic import Field, StrictStr, StrictBool, field_validator, model_validator
from ..cv.contracts import StrictModel, Timestamp, PositiveTimestamp, Fraction
from ..cv.entities import EntityCandidate

Text = Annotated[StrictStr, Field(min_length=1, max_length=4000)]
Label = Annotated[StrictStr, Field(min_length=1, max_length=256)]


class Event(StrictModel):
    start: Timestamp
    end: PositiveTimestamp
    description: Text

    @field_validator('description')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('description must not be blank')
        return value

    @model_validator(mode='after')
    def interval(self):
        if self.end <= self.start:
            raise ValueError('event must have positive duration')
        return self


class GroundedEvents(StrictModel):
    duration: PositiveTimestamp
    events: Annotated[tuple[Event, ...], Field(max_length=2000)]

    @model_validator(mode='after')
    def bounds(self):
        if any(event.end > self.duration for event in self.events):
            raise ValueError('event outside original video')
        return self


class EntityNomination(StrictModel):
    entities: Annotated[tuple[EntityCandidate, ...], Field(max_length=64)]


class SpatialTarget(StrictModel):
    description: Label
    reference_time: Timestamp
    bbox_xyxy: tuple[Fraction, Fraction, Fraction, Fraction]

    @model_validator(mode='after')
    def valid(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        if not self.description.strip() or x1 >= x2 or y1 >= y2:
            raise ValueError('target needs a description and positive normalized box')
        return self


SAM_KINDS = frozenset(('object_identity', 'object_location', 'occlusion', 'trajectory'))


class EventUncertainty(StrictModel):
    kind: Literal['object_identity', 'object_location', 'occlusion', 'trajectory',
                  'event_boundary', 'description', 'missing_event', 'camera_motion']
    reason: Text
    start: Timestamp
    end: PositiveTimestamp
    target: SpatialTarget | None

    @model_validator(mode='after')
    def valid(self):
        if self.end <= self.start or not self.reason.strip():
            raise ValueError('invalid review concern')
        if self.kind not in SAM_KINDS and self.target is not None:
            raise ValueError('editorial and camera concerns must not request SAM')
        if self.target and not self.start <= self.target.reference_time <= self.end:
            raise ValueError('reference frame must be within concern window')
        return self


class Concern(EventUncertainty):
    event_indices: Annotated[tuple[Annotated[int, Field(ge=0, strict=True)], ...], Field(max_length=2000)]


SAM_CONFIDENCE_THRESHOLD = 0.6


class ConfidenceEvent(Event):
    confidence: Fraction
    uncertainty: EventUncertainty | None

    @model_validator(mode='after')
    def explain_low_confidence(self):
        if self.confidence <= SAM_CONFIDENCE_THRESHOLD and self.uncertainty is None:
            raise ValueError('low-confidence event requires a concrete uncertainty')
        return self


class ConfidenceDraft(GroundedEvents):
    events: Annotated[tuple[ConfidenceEvent, ...], Field(max_length=2000)]

    @model_validator(mode='after')
    def questions(self):
        for event in self.events:
            question = event.uncertainty
            if question and question.end > self.duration:
                raise ValueError('uncertainty must stay within the original video window')
        return self

    def public_annotation(self):
        return {'duration': self.duration, 'events': [
            {'start': e.start, 'end': e.end, 'description': e.description} for e in self.events]}


class SamPromptReview(StrictModel):
    accepted: StrictBool
    reason: Text
    correction: SpatialTarget | None

    @model_validator(mode='after')
    def valid(self):
        if not self.reason.strip() or (self.accepted and self.correction is not None):
            raise ValueError('accepted segmentation cannot request correction')
        return self


class EvidenceReview(StrictModel):
    motion_evidence: Annotated[StrictStr, Field(min_length=1, max_length=800, pattern=r"\S")]
    concerns: Annotated[tuple[Concern, ...], Field(max_length=32)]
    unresolved: Annotated[tuple[Text, ...], Field(max_length=32)]

    @field_validator('unresolved')
    @classmethod
    def nonblank(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError('blank unresolved concern')
        return values


def validate_stage(schema, data, context, *, discarded_targets=None):
    # Media metadata is authoritative; tolerate decimal serialization at <=1 µs,
    # never rescale a model's event timeline or accept a materially wrong duration.
    if schema in (GroundedEvents, ConfidenceDraft) and isinstance(data, dict):
        actual, reported = context['duration'], data.get('duration')
        if type(reported) in (int, float) and math.isfinite(reported) and abs(reported - actual) <= 1e-6:
            data = {**data, 'duration': actual}
            if isinstance(data.get('events'), (list, tuple)):
                events = []
                for event in data['events']:
                    if isinstance(event, dict):
                        end = event.get('end')
                        if type(end) in (int, float) and math.isfinite(end) and actual < end <= actual + 1e-6:
                            event = {**event, 'end': actual}
                    if schema is ConfidenceDraft and isinstance(event, dict):
                        question = event.get('uncertainty')
                        if isinstance(question, dict):
                            question = dict(question)
                            end = question.get('end')
                            if type(end) in (int, float) and math.isfinite(end) and actual < end <= actual + 1e-6:
                                question['end'] = actual
                            target = question.get('target')
                            if isinstance(target, dict):
                                reference = target.get('reference_time')
                                if type(reference) in (int, float) and math.isfinite(reference) and actual < reference <= actual + 1e-6:
                                    question['target'] = {**target, 'reference_time': actual}
                            event = {**event, 'uncertainty': question}
                    events.append(event)
                data['events'] = events
    if schema is ConfidenceDraft and isinstance(data, dict) and isinstance(data.get('events'), (list, tuple)):
        events = []
        for index, event in enumerate(data['events']):
            question = event.get('uncertainty') if isinstance(event, dict) else None
            if isinstance(question, dict) and question.get('target') is not None:
                # Only optional target errors are recoverable. Invalid events,
                # confidence, question text/type/window still fail closed.
                without_target = {**question, 'target': None}
                valid_question = EventUncertainty.model_validate(without_target)
                if valid_question.end > context['duration']:
                    raise ValueError('uncertainty outside original video')
                try:
                    EventUncertainty.model_validate(question)
                except ValueError:
                    if discarded_targets is not None:
                        discarded_targets.append({'event_index': index,
                            'reason': 'invalid_optional_sam_target', 'target': question['target'],
                            'question': question['reason']})
                    event = {**event, 'uncertainty': without_target}
            events.append(event)
        data = {**data, 'events': events}
    value = schema.model_validate(data)
    if isinstance(value, GroundedEvents) and value.duration != context['duration']:
        raise ValueError('duration must match original video')
    if isinstance(value, EvidenceReview):
        for concern in value.concerns:
            if concern.end > context['duration'] or any(i >= context['event_count'] for i in concern.event_indices):
                raise ValueError('review references invalid event or time')
    if isinstance(value, SamPromptReview) and value.correction:
        if not context['window_start'] <= value.correction.reference_time <= context['window_end']:
            raise ValueError('corrected reference outside requested window')
    return value.model_dump(mode='json')


def register_event_schemas(registry):
    from .output_validation import DeclaredSchemaOutputError
    from pydantic import ValidationError
    for schema in (EntityNomination, GroundedEvents, ConfidenceDraft, EvidenceReview, SamPromptReview):
        def validator(data, context, schema=schema):
            try:
                validated = validate_stage(schema, data, context or {})
                # Preserve recoverable raw target data through the job store;
                # pipeline validation records discarded targets in its audit.
                return dict(data) if schema is ConfidenceDraft else validated
            except (ValueError, TypeError, KeyError, ValidationError):
                raise DeclaredSchemaOutputError(('GROUNDED_SCHEMA_INVALID',)) from None
        def response_contract(context, schema=schema):
            encoded = canonical(compile_local_schema(schema.model_json_schema()))
            return ModelResponseContract(schema.__name__, encoded, hashlib.sha256(encoded.encode()).hexdigest())
        registry.register(schema.__name__, validator,
                          allowed_issue_codes=('GROUNDED_SCHEMA_INVALID',),
                          generic_issue_code='GROUNDED_SCHEMA_INVALID', response_contract_factory=response_contract)

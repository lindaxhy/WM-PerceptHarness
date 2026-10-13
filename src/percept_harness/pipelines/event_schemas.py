"""Strict natural event contracts; times always refer to the source video."""
from typing import Annotated
import hashlib
from ..models.response_contract import ModelResponseContract, compile_local_schema, canonical
from pydantic import Field, StrictStr, field_validator, model_validator
from ..cv.contracts import StrictModel, Timestamp, PositiveTimestamp
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


class Concern(StrictModel):
    reason: Text
    event_indices: Annotated[tuple[Annotated[int, Field(ge=0, strict=True)], ...], Field(max_length=2000)]
    start: Timestamp
    end: PositiveTimestamp
    new_entities: Annotated[tuple[Label, ...], Field(max_length=16)]

    @model_validator(mode='after')
    def valid(self):
        if self.end <= self.start or not self.reason.strip() or any(not x.strip() for x in self.new_entities):
            raise ValueError('invalid review concern')
        return self


class EvidenceReview(StrictModel):
    concerns: Annotated[tuple[Concern, ...], Field(max_length=32)]
    unresolved: Annotated[tuple[Text, ...], Field(max_length=32)]

    @field_validator('unresolved')
    @classmethod
    def nonblank(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError('blank unresolved concern')
        return values


def validate_stage(schema, data, context):
    value = schema.model_validate(data)
    if isinstance(value, GroundedEvents) and value.duration != context['duration']:
        raise ValueError('duration must match original video')
    if isinstance(value, EvidenceReview):
        for concern in value.concerns:
            if concern.end > context['duration'] or any(i >= context['event_count'] for i in concern.event_indices):
                raise ValueError('review references invalid event or time')
    return value.model_dump(mode='json')


def register_event_schemas(registry):
    from .output_validation import DeclaredSchemaOutputError
    from pydantic import ValidationError
    for schema in (EntityNomination, GroundedEvents, EvidenceReview):
        def validator(data, context, schema=schema):
            try:
                return validate_stage(schema, data, context or {})
            except (ValueError, TypeError, KeyError, ValidationError):
                raise DeclaredSchemaOutputError(('GROUNDED_SCHEMA_INVALID',)) from None
        def response_contract(context, schema=schema):
            encoded = canonical(compile_local_schema(schema.model_json_schema()))
            return ModelResponseContract(schema.__name__, encoded, hashlib.sha256(encoded.encode()).hexdigest())
        registry.register(schema.__name__, validator,
                          allowed_issue_codes=('GROUNDED_SCHEMA_INVALID',),
                          generic_issue_code='GROUNDED_SCHEMA_INVALID', response_contract_factory=response_contract)

"""Pre-persistence validation of the active grounded annotation schemas."""
from collections.abc import Mapping
from functools import cache
import json

from pydantic import ValidationError

from ..models.response_contract import ModelResponseContract, canonical, compile_local_schema, preflight_plain
from .event_schemas import SCHEMAS, ConfidenceDraft, CrossFieldValidationError, validate_stage

SCHEMA_VALIDATION_FIELD = '_schema_validation'
INVALID_CANDIDATE_FIELD = '_invalid_candidate'
ISSUE_CODE = 'GROUNDED_SCHEMA_INVALID'


def bounded_invalid_candidate(result):
    """Keep untrusted model data for repair, never as an accepted annotation."""
    try:
        preflight_plain(result, max_bytes=200_000, max_nodes=50_000, max_depth=32)
        return json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (ValueError, TypeError, RecursionError):
        return None


@cache
def response_contract(schema_name):
    schema = SCHEMAS.get(schema_name)
    if schema is None:
        return None
    return ModelResponseContract(schema_name, canonical(compile_local_schema(schema.model_json_schema())))


def _invalid(schema_name, details=None):
    record = {'schema_name': schema_name, 'status': 'invalid', 'issue_codes': [ISSUE_CODE]}
    if details is not None:
        record['details'] = details
    return {SCHEMA_VALIDATION_FIELD: record}


def _error_details(error):
    if isinstance(error, ValidationError):
        return [{'location': list(e['loc']), 'type': e['type'], 'message': e['msg'][:500]}
                for e in error.errors(include_input=False, include_context=False, include_url=False)[:16]]
    if isinstance(error, CrossFieldValidationError):
        return [{'location': [], 'type': 'ValueError', 'message': message[:500]} for message in error.issues]
    return [{'location': [], 'type': type(error).__name__, 'message': str(error)[:500]}]


def sanitize(schema_name, result, validation_context=None):
    """Validate a model result, or return an invalid-output envelope for repair."""
    schema = SCHEMAS.get(schema_name)
    if schema is None:
        return dict(result)
    try:
        validated = validate_stage(schema, result, validation_context or {})
        # Preserve recoverable raw target data through stage execution;
        # pipeline validation records discarded targets in its audit.
        return dict(result) if schema is ConfidenceDraft else validated
    except (ValueError, TypeError, KeyError) as error:
        envelope = _invalid(schema_name, _error_details(error))
    candidate = bounded_invalid_candidate(result)
    if candidate is not None:
        envelope[INVALID_CANDIDATE_FIELD] = candidate
    return envelope


def failure_codes(schema_name, result):
    if schema_name not in SCHEMAS:
        return None
    envelope = result.get(SCHEMA_VALIDATION_FIELD)
    if isinstance(envelope, Mapping) and envelope.get('status') == 'invalid':
        return tuple(envelope.get('issue_codes', ()))
    return None


def model_output_failure(schema_name):
    return None if schema_name not in SCHEMAS else _invalid(schema_name)

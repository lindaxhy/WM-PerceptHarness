"""Pre-persistence validation of the active grounded annotation schemas."""
from dataclasses import dataclass
from collections.abc import Mapping

SCHEMA_VALIDATION_FIELD = '_schema_validation'


class DeclaredSchemaOutputError(ValueError):
    def __init__(self, issue_codes):
        self.issue_codes = tuple(issue_codes)
        super().__init__('declared model output schema is invalid')


@dataclass(frozen=True)
class _SchemaEntry:
    validator: object
    allowed_issue_codes: tuple[str, ...]
    generic_issue_code: str
    response_contract_factory: object


class OutputSchemaRegistry:
    def __init__(self):
        self._entries = {}

    def register(self, schema_name, validator, *, allowed_issue_codes, generic_issue_code,
                 response_contract_factory=None):
        if schema_name in self._entries:
            raise ValueError('duplicate output schema')
        self._entries[schema_name] = _SchemaEntry(validator, allowed_issue_codes, generic_issue_code, response_contract_factory)

    def model_response_contract(self, schema_name, authenticated_context):
        entry = self._entries.get(schema_name)
        return entry.response_contract_factory(authenticated_context) if entry and entry.response_contract_factory else None

    def sanitize(self, schema_name, result, validation_context=None):
        entry = self._entries.get(schema_name)
        if entry is None:
            return dict(result)
        try:
            return entry.validator(result, validation_context)
        except DeclaredSchemaOutputError as error:
            codes = [code for code in error.issue_codes if code in entry.allowed_issue_codes]
            return {SCHEMA_VALIDATION_FIELD: {'schema_name': schema_name, 'status': 'invalid',
                    'issue_codes': codes or [entry.generic_issue_code]}}

    def failure_codes(self, schema_name, result):
        if schema_name not in self._entries:
            return None
        envelope = result.get(SCHEMA_VALIDATION_FIELD)
        if isinstance(envelope, Mapping) and envelope.get('status') == 'invalid':
            return tuple(envelope.get('issue_codes', ()))
        return None

    def model_output_failure(self, schema_name):
        entry = self._entries.get(schema_name)
        if entry is None:
            return None
        return {SCHEMA_VALIDATION_FIELD: {'schema_name': schema_name, 'status': 'invalid',
                'issue_codes': [entry.generic_issue_code]}}


DEFAULT_OUTPUT_SCHEMAS = OutputSchemaRegistry()
from .event_schemas import register_event_schemas
register_event_schemas(DEFAULT_OUTPUT_SCHEMAS)

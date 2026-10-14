"""Bounded immutable internal descriptor; only registered factories supply it."""
from dataclasses import dataclass
import json
import math
from typing import Any

MAX_SCHEMA_BYTES = 65_536
MAX_SCHEMA_DEPTH = 32
MAX_SCHEMA_NODES = 4_096


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def preflight_plain(value: Any, *, max_bytes: int, max_nodes: int, max_depth: int) -> None:
    """Bound plain JSON before copying, encoding, parsing models or expanding refs.

    Containers and scalar strings are counted without constructing a second tree.
    Ancestor cycles fail by depth. UTF-8 encoding happens only after character
    length is bounded. Final canonical byte length is checked by callers.
    """
    nodes = size = 0
    def visit(item, depth):
        nonlocal nodes, size
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            raise ValueError('scene structure exceeds complexity limit')
        if type(item) is dict:
            size += 2 + len(item)
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError('scene structure must have string keys')
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif type(item) is list:
            size += 2 + len(item)
            for child in item:
                visit(child, depth + 1)
        elif type(item) is str:
            if len(item) > max_bytes:
                raise ValueError('scene string exceeds size limit')
            size += len(item.encode('utf-8')) + 2
        elif item is None or type(item) is bool:
            size += 5
        elif type(item) in (int, float):
            if not math.isfinite(item):
                raise ValueError('scene number must be finite')
            size += 24
        else:
            raise ValueError('scene structure must be plain JSON')
        if size > max_bytes:
            raise ValueError('scene structure exceeds size limit')
    visit(value, 0)


def compile_local_schema(source: dict[str, Any]) -> dict[str, Any]:
    """Compile bounded server-generated schemas using only known local refs."""
    preflight_plain(source, max_bytes=MAX_SCHEMA_BYTES, max_nodes=MAX_SCHEMA_NODES, max_depth=MAX_SCHEMA_DEPTH)
    definitions = source.get('$defs', {})
    allowed = {'type', 'properties', 'items', 'required', 'additionalProperties',
               'enum', 'minimum', 'maximum', 'maxItems', 'minItems', 'pattern', 'anyOf'}
    # Only known generated local refs are resolved. Expansion is bounded as it
    # happens, so a compact recursively referenced schema cannot expand freely.
    count = 0
    def expand(node, depth=0):
        nonlocal count
        count += 1
        if count > MAX_SCHEMA_NODES or depth > MAX_SCHEMA_DEPTH:
            raise ValueError('scene response schema exceeds complexity limit')
        if isinstance(node, list):
            return [expand(v, depth + 1) for v in node]
        if not isinstance(node, dict):
            return node
        if '$ref' in node:
            ref = node['$ref']
            if not ref.startswith('#/$defs/') or ref[8:] not in definitions:
                raise ValueError('scene response schema reference is invalid')
            return expand(definitions[ref[8:]], depth + 1)
        result = {}
        for key, value in node.items():
            if key == 'properties':
                result[key] = {k: expand(v, depth + 1) for k, v in value.items()}
            elif key == 'prefixItems':
                compiled = [expand(v, depth + 1) for v in value]
                if not compiled or any(v != compiled[0] for v in compiled[1:]):
                    raise ValueError('heterogeneous tuple response schema is unsupported')
                result['items'] = compiled[0]
            elif key in allowed:
                result[key] = expand(value, depth + 1)
        return result
    return expand(source)


@dataclass(frozen=True)
class ModelResponseContract:
    name: str
    schema_json: str

    def __post_init__(self) -> None:
        if (type(self.name) is not str or self.name not in ('GroundedEvents', 'ConfidenceDraft', 'EvidenceReview', 'SamPromptReview')
                or type(self.schema_json) is not str
                or len(self.schema_json) > MAX_SCHEMA_BYTES
                or len(self.schema_json.encode()) > MAX_SCHEMA_BYTES):
            raise ValueError('scene response descriptor is invalid')
        value = json.loads(self.schema_json)
        preflight_plain(value, max_bytes=MAX_SCHEMA_BYTES,
                        max_nodes=MAX_SCHEMA_NODES, max_depth=MAX_SCHEMA_DEPTH)
        if canonical(value) != self.schema_json:
            raise ValueError('scene response descriptor identity is invalid')

    def format(self) -> dict[str, Any]:
        return dict(type='json_schema', name=self.name, schema=json.loads(self.schema_json), strict=True)

    def cache_identity(self) -> dict[str, Any]:
        return dict(type='json_schema', name=self.name, strict=True, schema=json.loads(self.schema_json))

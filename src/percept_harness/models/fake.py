"""Deterministic, offline visual-model fixture for local tests."""

from __future__ import annotations

import copy
import json
import re
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from .base import ModelOutputError, ModelRequest, ModelRequestError, VideoModel


class UnknownModelStageError(ModelRequestError):
    """The requested stage has no deterministic fake fixture."""


ScriptedResult = Mapping[str, Any] | BaseException


class FakeVideoModel(VideoModel):
    """Return stable visual-only fixtures without loading weights or using a GPU.

    ``failure_script`` is a per-stage FIFO.  Each scripted response is consumed
    exactly once, allowing repair-path tests to provide an invalid structured
    response followed by a valid one.  Script inputs and returned mappings are
    copied so callers cannot alter later results by mutation.
    """

    _STAGES = frozenset({'confidence_draft', 'grounded_draft', 'evidence_review', 'grounded_revision', 'sam_prompt_review'})

    def __init__(
        self,
        *,
        failure_script: Mapping[str, Sequence[ScriptedResult]] | None = None,
    ) -> None:
        self._failure_script = {
            stage: deque(copy.deepcopy(list(results)))
            for stage, results in (failure_script or {}).items()
        }

    def generate(self, request: ModelRequest) -> dict[str, Any]:
        """Return the next scripted result or the stable fixture for ``stage``."""
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        if request.stage not in self._STAGES:
            raise UnknownModelStageError(
                f"unsupported fake model stage: {request.stage}"
            )

        scripted = self._failure_script.get(request.stage)
        if scripted:
            result = scripted.popleft()
            if isinstance(result, BaseException):
                raise result
            if not isinstance(result, Mapping):
                raise ModelOutputError("scripted fake result must be a JSON object")
            return copy.deepcopy(dict(result))

        return copy.deepcopy(self._fixture(request))

    def _fixture(self, request: ModelRequest) -> dict[str, Any]:
        if request.stage == 'confidence_draft':
            return {'duration': request.span.end, 'events': [
                {'start': request.span.start, 'end': request.span.end,
                 'description': 'A hand moves the red container.',
                 'confidence': 0.9, 'uncertainty': None}]}
        if request.stage == 'sam_prompt_review':
            return {'accepted': True, 'reason': 'Synthetic fixture mask matches target.', 'correction': None}
        if request.stage == 'evidence_review':
            return {'motion_evidence': 'Synthetic fixture: no ambiguous scene-relative motion.', 'concerns': [], 'unresolved': []}
        return {'duration': request.span.end, 'events': [
            {'start': request.span.start, 'end': request.span.end,
             'description': 'A hand moves the red container.'}]}

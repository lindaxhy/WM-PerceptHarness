"""Shared embodied-pipeline fixtures used across test modules."""

from __future__ import annotations

from typing import Any


def _entity_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "right hand",
            "aliases": ["hand"],
            "role": "actor",
        },
        {
            "name": "red container",
            "aliases": ["container"],
            "role": "manipulated_object",
        },
    ]


def _valid_boundary_output() -> dict[str, Any]:
    """Literal two-action Pass-B fixture independent of production builders."""
    return {
        "task_description": "move the red container",
        "actions": [
            {
                "action_index": 0,
                "start": 0.0,
                "end": 1.0,
                "description": "right hand reaches toward red container",
                "event_type": "reach_and_grasp",
                "boundary_points": [
                    {
                        "boundary_id": "a0-b0",
                        "time": 0.0,
                        "event_type": "action_start",
                        "visual_evidence": "right hand begins moving toward container",
                    },
                    {
                        "boundary_id": "a0-b1",
                        "time": 0.5,
                        "event_type": "approach",
                        "visual_evidence": "right hand visibly approaches container",
                    },
                    {
                        "boundary_id": "a0-b2",
                        "time": 1.0,
                        "event_type": "action_end",
                        "visual_evidence": "right hand reaches the container",
                    },
                ],
                "fine_segments": [
                    {
                        "segment_index": 0,
                        "start": 0.0,
                        "end": 0.5,
                        "description": "right hand approaches red container",
                        "event_type": "approach",
                        "start_boundary_id": "a0-b0",
                        "end_boundary_id": "a0-b1",
                    },
                    {
                        "segment_index": 1,
                        "start": 0.5,
                        "end": 1.0,
                        "description": "right hand reaches red container",
                        "event_type": "contact_start",
                        "start_boundary_id": "a0-b1",
                        "end_boundary_id": "a0-b2",
                    },
                ],
            },
            {
                "action_index": 1,
                "start": 1.0,
                "end": 2.0,
                "description": "right hand moves red container",
                "event_type": "transport",
                "boundary_points": [
                    {
                        "boundary_id": "a1-b0",
                        "time": 1.0,
                        "event_type": "action_start",
                        "visual_evidence": "right hand starts moving held container",
                    },
                    {
                        "boundary_id": "a1-b1",
                        "time": 1.5,
                        "event_type": "transport_continue",
                        "visual_evidence": "container visibly continues moving",
                    },
                    {
                        "boundary_id": "a1-b2",
                        "time": 2.0,
                        "event_type": "action_end",
                        "visual_evidence": "container motion visibly stops",
                    },
                ],
                "fine_segments": [
                    {
                        "segment_index": 2,
                        "start": 1.0,
                        "end": 1.5,
                        "description": "right hand transports red container",
                        "event_type": "transport_start",
                        "start_boundary_id": "a1-b0",
                        "end_boundary_id": "a1-b1",
                    },
                    {
                        "segment_index": 3,
                        "start": 1.5,
                        "end": 2.0,
                        "description": "right hand continues transporting container",
                        "event_type": "transport_continue",
                        "start_boundary_id": "a1-b1",
                        "end_boundary_id": "a1-b2",
                    },
                ],
            },
        ],
    }

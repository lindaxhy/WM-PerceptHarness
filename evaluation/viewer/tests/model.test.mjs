import test from "node:test";
import assert from "node:assert/strict";

import {
  formatTime,
  isEventActive,
  normalizeLas,
  normalizeLocal,
  packLanes,
  validateRelativeAssetPath,
} from "../js/model.js";


test("formatTime emits a stable minute-second label", () => {
  assert.equal(formatTime(65.23), "01:05.23");
  assert.equal(formatTime(0), "00:00.00");
  assert.equal(formatTime(Number.NaN), "--:--.--");
});


test("event activity is start-inclusive and end-exclusive", () => {
  const event = { start: 1, end: 2 };
  assert.equal(isEventActive(event, 0.999), false);
  assert.equal(isEventActive(event, 1), true);
  assert.equal(isEventActive(event, 1.999), true);
  assert.equal(isEventActive(event, 2), false);
});


test("overlapping events receive deterministic reusable lanes", () => {
  const events = [
    { id: "a", start: 0, end: 2 },
    { id: "b", start: 0.5, end: 1 },
    { id: "c", start: 2, end: 3 },
    { id: "d", start: 1, end: 2.5 },
  ];

  const packed = packLanes(events);

  assert.deepEqual(
    packed.map((event) => [event.id, event.lane]),
    [
      ["a", 0],
      ["b", 1],
      ["c", 0],
      ["d", 1],
    ],
  );
  assert.equal("lane" in events[0], false);
});


test("LAS events normalize to the shared display model", () => {
  const source = {
    objects: [
      { id: "obj_001", name: "yellow ball" },
      { id: "obj_002", name: "wooden board" },
    ],
    semantic_events: [
      {
        event_id: "evt_001",
        start_s: 0,
        end_s: 1.25,
        type: "move",
        actor: "right hand",
        object_ids: ["obj_002"],
        description: "Right hand moves the board.",
        confidence: 0.98,
      },
    ],
  };

  assert.deepEqual(normalizeLas(source, 2), [
    {
      id: "evt_001",
      start: 0,
      end: 1.25,
      type: "move",
      actor: "right hand",
      target: "wooden board",
      description: "Right hand moves the board.",
      confidence: 0.98,
      source: "las",
      lane: 0,
    },
  ]);
});


test("all three local layers normalize without changing source values", () => {
  const common = {
    start: 0,
    end: 1,
    actor: "right_hand",
    description: "right hand reaches for block",
    confidence: 0.9,
  };
  const source = {
    schema_version: "comparison_viewer_local_v1",
    sample_id: "demo_0001",
    duration_seconds: 2,
    fine_segments: [
      {
        ...common,
        segment_index: 0,
        action_index: 0,
        skill: "reach",
        target: "block",
        actor_state: "reaching",
        visual_motion_state: "active",
        event_type: "pre_contact",
      },
    ],
    grouped_events: [
      {
        ...common,
        event_index: 0,
        action: "reach",
        target: "block",
        source_segment_indices: [0],
      },
    ],
    scene_events: [
      {
        ...common,
        event_index: 0,
        event_type: "reach",
        target_object_id: "block",
      },
    ],
  };

  const result = normalizeLocal(source, 2, "demo_0001");

  assert.deepEqual(Object.keys(result), ["grouped", "fine", "scene"]);
  assert.deepEqual(
    [result.grouped[0].id, result.grouped[0].type, result.grouped[0].target],
    ["grouped-0", "reach", "block"],
  );
  assert.deepEqual(
    [result.fine[0].id, result.fine[0].type, result.fine[0].target],
    ["fine-0", "reach", "block"],
  );
  assert.deepEqual(
    [result.scene[0].id, result.scene[0].type, result.scene[0].target],
    ["scene-0", "reach", "block"],
  );
  assert.equal("lane" in source.grouped_events[0], false);
});


test("local normalization requires the selected manifest sample ID", () => {
  const source = {
    schema_version: "comparison_viewer_local_v1",
    sample_id: "wrong_sample",
    duration_seconds: 2,
    fine_segments: [],
    grouped_events: [],
    scene_events: [],
  };

  assert.throws(() => normalizeLocal(source, 2, "demo_0001"), /INVALID_LOCAL_DATA/);
  assert.throws(() => normalizeLocal(source, 2), /INVALID_LOCAL_DATA/);
});


test("normalization rejects invalid event intervals", () => {
  const makeReference = (start, end) => ({
    objects: [],
    semantic_events: [
      {
        event_id: "evt_001",
        start_s: start,
        end_s: end,
        type: "move",
        actor: "right hand",
        object_ids: [],
        description: "move",
        confidence: 0.9,
      },
    ],
  });
  for (const [start, end] of [
    [-0.1, 1],
    [0, 2.1],
    [1, 1],
    [1.1, 1],
    [Number.NaN, 1],
  ]) {
    assert.throws(() => normalizeLas(makeReference(start, end), 2), /INVALID_INTERVAL/);
  }
});


test("asset paths remain inside the served repository", () => {
  assert.equal(
    validateRelativeAssetPath("evaluation/viewer/data/demo-manifest.json"),
    "evaluation/viewer/data/demo-manifest.json",
  );
  for (const path of [
    "/private/video.mp4",
    "../video.mp4",
    "evaluation/../video.mp4",
    "https://example.com/video.mp4",
    "file:///video.mp4",
    "//example.com/video.mp4",
    "evaluation\\video.mp4",
  ]) {
    assert.throws(() => validateRelativeAssetPath(path), /INVALID_ASSET_PATH/);
  }
});

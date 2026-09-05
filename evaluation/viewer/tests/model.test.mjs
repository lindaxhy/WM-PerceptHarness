import test from "node:test";
import assert from "node:assert/strict";

import {
  formatTime,
  isEventActive,
  normalizeLas,
  normalizeLocal,
  normalizeHybrid,
  validateManifest,
  localVariants,
  preferredVariant,
  normalizeVariant,
  readVerifiedBytes,
  validateOverlayPng,
  packLanes,
  validateRelativeAssetPath,
} from "../js/model.js";

function hybridFixture() {
  const event = {
    id: "action_0", event_index: 0, start: 0, end: 1, actor: "right_hand",
    action: "motion", target: "cup", description: "Hand moves cup", confidence: 0.8,
    branch: "action", model_stage: "embodied_enrichment", evidence_mode: "hybrid",
    source_segment_indices: [0], source_track_ids: ["cup_1"], source_keyframe_ids: ["frame_0002"],
    repair_history: ["initial"], review_status: "not_required", review: null,
  };
  return {
    schema_version: "comparison_viewer_hybrid_v1",
    sample: { sample_id: "full_0001", duration_seconds: 1 },
    layers: {
      action_events: { status: "available", events: [event] },
      occlusion_events: { status: "available", events: [{
        ...event, id: "occlusion_0", branch: "occlusion", model_stage: "occlusion_semantics",
        action: undefined, actor: undefined, target: undefined,
        event_type: "occluded", target_entity_id: "cup", occluder_entity_id: "board",
        source_candidate_id: "occ_aaaaaaaaaaaa_0000", review_status: "unreviewed",
      }].map(({action, actor, target, ...rest}) => rest) },
      scene_facts: { status: "available", events: [], locations: [], relations: [],
        objects: [{ object_id: "cup", name: "cup", description: "visible cup" }],
        initial_state: [], final_state: [], outcome: { status: "unknown", description: "unknown", confidence: 0 } },
    },
    provenance: {
      source_video_sha256: "a".repeat(64), source_result_sha256: "b".repeat(64),
      canonical_result_sha256: "c".repeat(64), model_identity: "doubao-seed-2-1-pro-260628",
      cv_evidence: { status: "available", artifact_key: "d".repeat(64), manifest_sha256: "e".repeat(64), cache_hit: false },
      review_sha256: null, performance: { total_seconds: 5, repair_count: 0, degradation_count: 0 },
      overlays: [{ keyframe_id: "frame_0002", track_id: "cup_1", frame_index: 2, timestamp_seconds: 0.2,
        path: `evaluation/viewer/data/hybrid/overlays/${"f".repeat(64)}.png`, sha256: "f".repeat(64), size_bytes: 100 }],
    }, warnings: [],
  };
}

test("hybrid normalization retains evidence, statuses and provenance without mutating input", () => {
  const data = hybridFixture();
  const before = structuredClone(data);
  const result = normalizeHybrid(data, 1, "full_0001");
  assert.deepEqual(result.availableModes, ["grouped", "occlusion", "scene"]);
  assert.equal(result.grouped[0].type, "motion");
  assert.equal(result.grouped[0].lane, 0);
  assert.equal(result.occlusion[0].type, "occluded");
  assert.equal(result.occlusion[0].meta.reviewStatus, "unreviewed");
  assert.deepEqual(result.grouped[0].meta.sourceTracks, ["cup_1"]);
  assert.equal(result.grouped[0].meta.overlays[0].keyframe_id, "frame_0002");
  assert.equal(result.provenance.source_result_sha256, "b".repeat(64));
  assert.deepEqual(result.fine, []);
  assert.deepEqual(data, before);
});

test("hybrid fine evidence and unavailable modes remain separate", () => {
  const data = hybridFixture();
  data.layers.occlusion_events = { status: "unavailable", events: [] };
  data.warnings = [{ code: "OCCLUSION_UNAVAILABLE" }];
  data.provenance.performance.degradation_count = 1;
  data.layers.fine_segments = { status: "available", events: [{
    id: "fine_0", segment_index: 0, start: 0, end: 1, actor: "right_hand", skill: "move", target: "cup",
    description: "move", confidence: 0.8, branch: "fine", model_stage: "embodied_enrichment", evidence_mode: "vlm_only",
    source_segment_indices: [0], source_track_ids: [], source_keyframe_ids: [], repair_history: ["initial"],
    review_status: "not_required", review: null,
  }] };
  const result = normalizeHybrid(data, 1, "full_0001");
  assert.deepEqual(result.availableModes, ["grouped", "scene", "fine"]);
  assert.deepEqual(result.occlusion, []);
  assert.equal(result.fine[0].id, "fine_0");
  assert.equal(result.fine[0].type, "move");
});

test("hybrid overlapping scene facts receive deterministic lanes and retain objects", () => {
  const data = hybridFixture();
  const { id, event_index, action, actor, target, description, review, ...provenance } = data.layers.action_events.events[0];
  data.layers.scene_facts.locations = [{ ...provenance, branch: "scene", model_stage: "scene_semantics",
    object_id: "cup", location: "center", visual_evidence: "Cup visibly centered" }];
  data.layers.scene_facts.relations = [{ ...provenance, branch: "scene", model_stage: "scene_semantics",
    subject_object_id: "cup", object_object_id: "cup", relation: "near", visual_evidence: "Visible relation" }];
  const result = normalizeHybrid(data, 1, "full_0001");
  assert.deepEqual(result.scene.map(event => event.lane), [0, 1]);
  assert.equal(result.scene[0].type, "location");
  assert.match(result.scene[0].description, /center/);
  assert.equal(result.context.objects[0].name, "cup");
});

test("hybrid scene targets accept the canonical unknown sentinel without weakening object references", () => {
  const fixture = target => {
    const data = hybridFixture();
    const {action, target: ignoredTarget, ...event} = data.layers.action_events.events[0];
    data.layers.scene_facts.events = [{...event, id: "scene_0", branch: "scene",
      model_stage: "scene_semantics", event_type: "move", target_object_id: target}];
    return data;
  };
  for (const target of ["cup", "unknown"]) {
    const data = fixture(target);
    if (target === "unknown") data.layers.scene_facts.objects = [];
    const before = structuredClone(data);
    assert.equal(normalizeHybrid(data, 1, "full_0001").scene[0].target, target);
    assert.deepEqual(data, before);
  }
  for (const target of ["foreign", null, ""]) {
    assert.throws(() => normalizeHybrid(fixture(target), 1, "full_0001"));
  }
  for (const target of ["unknown", "foreign", null]) {
    const data = hybridFixture();
    const {id, event_index, action, actor, target: ignoredTarget, description, review, ...provenance} = data.layers.action_events.events[0];
    data.layers.scene_facts.locations = [{...provenance, branch: "scene", model_stage: "scene_semantics",
      object_id: target, location: "center", visual_evidence: "Visible location"}];
    assert.throws(() => normalizeHybrid(data, 1, "full_0001"));
    data.layers.scene_facts.locations = [];
    for (const field of ["subject_object_id", "object_object_id"]) {
      data.layers.scene_facts.relations = [{...provenance, branch: "scene", model_stage: "scene_semantics",
        subject_object_id: "cup", object_object_id: "cup", relation: "near", visual_evidence: "Visible relation", [field]: target}];
      assert.throws(() => normalizeHybrid(data, 1, "full_0001"));
    }
  }
});

test("hybrid review verdicts require matching provenance and human evidence", () => {
  const data = hybridFixture();
  const event = data.layers.occlusion_events.events[0];
  event.review_status = "supported";
  event.review = { reviewer: "human-reviewer", visual_reason: "Board visibly covers cup" };
  assert.throws(() => normalizeHybrid(data, 1, "full_0001"));
  data.provenance.review_sha256 = "1".repeat(64);
  assert.equal(normalizeHybrid(data, 1, "full_0001").occlusion[0].meta.reviewStatus, "supported");
  event.review_status = "unsupported";
  assert.equal(normalizeHybrid(data, 1, "full_0001").occlusion[0].meta.reviewStatus, "unsupported");
});

test("hybrid parser rejects invalid identity, intervals, raw masks, paths and provenance", () => {
  const mutations = [
    d => d.schema_version = "unknown", d => d.sample.sample_id = "other", d => d.sample.duration_seconds = 2,
    d => d.extra = {}, d => d.layers.action_events.events[0].start = -1,
    d => d.layers.action_events.events[0].end = 2, d => d.layers.action_events.events[0].confidence = 2,
    d => d.layers.action_events.events[0].raw_mask = [[1]], d => d.layers.action_events.events[0].source_track_ids = ["../mask"],
    d => d.provenance.source_video_sha256 = "bad", d => d.provenance.cv_evidence.manifest_sha256 = "bad",
    d => d.provenance.overlays[0].path = "../frame.png", d => d.provenance.overlays[0].path = "https://example.com/x.png",
    d => d.provenance.overlays[0].path = "evaluation/%2e%2e/x.png", d => d.provenance.overlays[0].path = "evaluation/x.npz",
    d => d.provenance.overlays[0].keyframe_id = "foreign", d => d.provenance.overlays[0].sha256 = "0".repeat(64),
    d => d.provenance.overlays[0].frame_index = 0.5, d => d.provenance.overlays[0].size_bytes = 0,
    d => d.provenance.overlays[0].timestamp_seconds = 1, d => d.layers.occlusion_events.status = "unavailable",
    d => d.layers.action_events.events[0].repair_history = ["repair"], d => d.layers.action_events.events[0].evidence_mode = "vlm_only",
    d => d.layers.scene_facts.objects.push({...d.layers.scene_facts.objects[0]}),
    d => d.warnings = [{code: "UNKNOWN", error: "private"}],
  ];
  for (const mutate of mutations) {
    const data = hybridFixture(); mutate(data);
    assert.throws(() => normalizeHybrid(data, 1, "full_0001"), undefined, String(mutate));
  }
});

test("hybrid parser rejects textual binary masks but retains ordinary mask discussion", () => {
  for (const separator of ["\n", "\r\n", "\r", "\u0085", "\u2028", "\u2029"]) {
    const data = hybridFixture();
    data.layers.action_events.events[0].description = ["MASK", " 0, 1,0 ", "1,0, 1"].join(separator);
    assert.throws(() => normalizeHybrid(data, 1, "full_0001"), /INVALID_HYBRID_TEXT/);
  }
  for (const description of ["Hand lifts a mask", "mask\n0,1\nVisible hand", "mask\n10,11\n11,10"]) {
    const data = hybridFixture();
    data.layers.action_events.events[0].description = description;
    assert.equal(normalizeHybrid(data, 1, "full_0001").grouped[0].description, description);
  }
});

function manifestFixture(version = 2) {
  const sample = { sample_id: "full_0001", duration_seconds: 1, media_path: "evaluation/media/s.mp4",
    las_path: "evaluation/reference/s.json", caveat: "" };
  if (version === 1) sample.local_path = "evaluation/local/s.json";
  else sample.local_variants = [
    { id: "qwen_only", label: "Qwen-only", path: "evaluation/local/s.json", format: "comparison_viewer_local_v1" },
    { id: "doubao_only", label: "Doubao-only", path: "evaluation/doubao/s.json", format: "comparison_viewer_hybrid_v1" },
    { id: "doubao_sam31", label: "Doubao + SAM3.1", path: "evaluation/hybrid/s.json", format: "comparison_viewer_hybrid_v1" },
  ];
  return { schema_version: `comparison_viewer_manifest_v${version}`, reference_set_id: "english", samples: [sample] };
}

test("manifest v1 compatibility and v2 honest deterministic model selection", () => {
  const legacy = validateManifest(manifestFixture(1)).samples[0];
  assert.equal(preferredVariant(localVariants(legacy)).id, "qwen_only");
  const modern = validateManifest(manifestFixture()).samples[0];
  assert.equal(preferredVariant(localVariants(modern)).id, "doubao_sam31");
  modern.local_variants.pop();
  assert.equal(preferredVariant(localVariants(modern)).id, "doubao_only");
  modern.local_variants.push({ id: "qwen_sam31", label: "Qwen + SAM3.1", path: "evaluation/qwen-sam/s.json", format: "comparison_viewer_hybrid_v1" });
  assert.equal(preferredVariant(localVariants(modern)).id, "qwen_sam31");
});

test("manifest rejects duplicates, foreign variants, misleading labels and unsafe paths", () => {
  for (const mutate of [
    m => m.samples[0].local_variants.push(m.samples[0].local_variants[0]),
    m => m.samples[0].local_variants.shift(), m => m.samples[0].local_variants[1].id = "unknown",
    m => m.samples[0].local_variants[1].label = "Qwen-only", m => m.samples[0].local_variants[1].format = "raw",
    m => m.samples[0].local_variants[1].path = "evaluation/%2fprivate.json",
    m => m.samples[0].local_variants[1].sha256 = "bad", m => m.samples.push(m.samples[0]),
    m => m.samples[0].duration_seconds = NaN, m => m.samples[0].local_path = "evaluation/stale.json",
  ]) {
    const manifest = manifestFixture(); mutate(manifest);
    assert.throws(() => validateManifest(manifest), undefined, String(mutate));
  }
});

test("variant binding rejects relabeling, source swaps and format confusion", () => {
  const data = hybridFixture();
  const sample = manifestFixture().samples[0];
  sample.source_video_sha256 = "a".repeat(64);
  const variant = sample.local_variants[2];
  variant.source_result_sha256 = "b".repeat(64);
  variant.model_identity = "doubao-seed-2-1-pro-260628";
  assert.equal(normalizeVariant(data, sample, variant).grouped[0].type, "motion");
  for (const change of [
    v => v.source_result_sha256 = "9".repeat(64), v => v.model_identity = "other",
    v => v.id = "qwen_sam31", v => v.id = "doubao_only", v => v.format = "comparison_viewer_local_v1",
  ]) {
    const bad = structuredClone(variant); change(bad);
    assert.throws(() => normalizeVariant(data, sample, bad));
  }
  sample.source_video_sha256 = "9".repeat(64);
  assert.throws(() => normalizeVariant(data, sample, variant));
});

test("asset bytes are bounded and hashed before use", async () => {
  const bytes = new TextEncoder().encode("abc");
  const sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  assert.deepEqual(await readVerifiedBytes(new Response(bytes), sha, 3), bytes);
  await assert.rejects(readVerifiedBytes(new Response(bytes), "0".repeat(64), 3));
  await assert.rejects(readVerifiedBytes(new Response(bytes), sha, 2));
  await assert.rejects(readVerifiedBytes(new Response(bytes, {status: 404}), sha, 3));
});

test("overlay image header prevents non-PNG and excessive decoded dimensions", () => {
  const bytes = new Uint8Array(24);
  bytes.set([137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82]);
  const view = new DataView(bytes.buffer);
  view.setUint32(16, 1280); view.setUint32(20, 720);
  assert.equal(validateOverlayPng(bytes), bytes);
  view.setUint32(16, 100000);
  assert.throws(() => validateOverlayPng(bytes));
  assert.throws(() => validateOverlayPng(new Uint8Array([0])));
});


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

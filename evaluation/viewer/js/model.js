const INVALID_INTERVAL = "INVALID_INTERVAL";


function finiteNumber(value, code = INVALID_INTERVAL) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(code);
  }
  return value;
}


function validateDuration(duration) {
  if (finiteNumber(duration, "INVALID_DURATION") <= 0) {
    throw new Error("INVALID_DURATION");
  }
  return duration;
}


function validateInterval(start, end, duration) {
  const left = finiteNumber(start);
  const right = finiteNumber(end);
  if (!(0 <= left && left < right && right <= duration)) {
    throw new Error(INVALID_INTERVAL);
  }
  return [left, right];
}


function array(value, code) {
  if (!Array.isArray(value)) {
    throw new Error(code);
  }
  return value;
}


function text(value, code) {
  if (typeof value !== "string") {
    throw new Error(code);
  }
  return value;
}


export function formatTime(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) {
    return "--:--.--";
  }
  const centiseconds = Math.max(0, Math.round(seconds * 100));
  const minutes = Math.floor(centiseconds / 6000);
  const remainder = (centiseconds % 6000) / 100;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}


export function isEventActive(event, currentTime) {
  return event.start <= currentTime && currentTime < event.end;
}


export function packLanes(events) {
  const indexed = events.map((event, index) => ({ event, index }));
  indexed.sort(
    (left, right) =>
      left.event.start - right.event.start ||
      left.event.end - right.event.end ||
      left.index - right.index,
  );
  const laneEnds = [];
  const lanes = new Array(events.length);
  for (const { event, index } of indexed) {
    let lane = laneEnds.findIndex((end) => event.start >= end);
    if (lane === -1) {
      lane = laneEnds.length;
      laneEnds.push(event.end);
    } else {
      laneEnds[lane] = event.end;
    }
    lanes[index] = lane;
  }
  return events.map((event, index) => ({ ...event, lane: lanes[index] }));
}


export function validateRelativeAssetPath(path) {
  if (
    typeof path !== "string" ||
    path.length === 0 ||
    path.startsWith("/") ||
    path.startsWith("//") ||
    path.includes("\\") ||
    /[%?#\s\u0000-\u001f]/.test(path) ||
    /^[a-z][a-z0-9+.-]*:/i.test(path)
  ) {
    throw new Error("INVALID_ASSET_PATH");
  }
  const parts = path.split("/");
  if (parts.some((part) => part === "" || part === "." || part === "..")) {
    throw new Error("INVALID_ASSET_PATH");
  }
  return path;
}


export function normalizeLas(reference, duration) {
  validateDuration(duration);
  if (reference === null || typeof reference !== "object") {
    throw new Error("INVALID_LAS_REFERENCE");
  }
  const objectNames = new Map(
    array(reference.objects, "INVALID_LAS_OBJECTS").map((object) => [
      text(object.id, "INVALID_LAS_OBJECT"),
      text(object.name, "INVALID_LAS_OBJECT"),
    ]),
  );
  const events = array(reference.semantic_events, "INVALID_LAS_EVENTS").map(
    (event) => {
      const [start, end] = validateInterval(event.start_s, event.end_s, duration);
      const objectIds = array(event.object_ids, "INVALID_LAS_OBJECT_IDS");
      return {
        id: text(event.event_id, "INVALID_LAS_EVENT"),
        start,
        end,
        type: text(event.type, "INVALID_LAS_EVENT"),
        actor: text(event.actor, "INVALID_LAS_EVENT"),
        target: objectIds.map((id) => objectNames.get(id) ?? String(id)).join(", "),
        description: text(event.description, "INVALID_LAS_EVENT"),
        confidence: finiteNumber(event.confidence, "INVALID_CONFIDENCE"),
        source: "las",
      };
    },
  );
  return packLanes(events);
}


function normalizeLocalLayer(events, duration, mode) {
  return packLanes(
    array(events, `INVALID_LOCAL_${mode.toUpperCase()}_EVENTS`).map((event) => {
      const [start, end] = validateInterval(event.start, event.end, duration);
      if (mode === "grouped") {
        return {
          id: `grouped-${event.event_index}`,
          start,
          end,
          type: text(event.action, "INVALID_LOCAL_EVENT"),
          actor: text(event.actor, "INVALID_LOCAL_EVENT"),
          target: text(event.target, "INVALID_LOCAL_EVENT"),
          description: text(event.description, "INVALID_LOCAL_EVENT"),
          confidence: finiteNumber(event.confidence, "INVALID_CONFIDENCE"),
          source: "local",
          meta: { sourceSegments: [...array(event.source_segment_indices, "INVALID_SOURCE_SEGMENTS")] },
        };
      }
      if (mode === "fine") {
        return {
          id: `fine-${event.segment_index}`,
          start,
          end,
          type: text(event.skill, "INVALID_LOCAL_EVENT"),
          actor: text(event.actor, "INVALID_LOCAL_EVENT"),
          target: text(event.target, "INVALID_LOCAL_EVENT"),
          description: text(event.description, "INVALID_LOCAL_EVENT"),
          confidence: finiteNumber(event.confidence, "INVALID_CONFIDENCE"),
          source: "local",
          meta: {
            eventType: text(event.event_type, "INVALID_LOCAL_EVENT"),
            actorState: text(event.actor_state, "INVALID_LOCAL_EVENT"),
            motionState: text(event.visual_motion_state, "INVALID_LOCAL_EVENT"),
          },
        };
      }
      return {
        id: `scene-${event.event_index}`,
        start,
        end,
        type: text(event.event_type, "INVALID_LOCAL_EVENT"),
        actor: text(event.actor, "INVALID_LOCAL_EVENT"),
        target: text(event.target_object_id, "INVALID_LOCAL_EVENT"),
        description: text(event.description, "INVALID_LOCAL_EVENT"),
        confidence: finiteNumber(event.confidence, "INVALID_CONFIDENCE"),
        source: "local",
      };
    }),
  );
}


export function normalizeLocal(displayData, duration, expectedSampleId) {
  validateDuration(duration);
  if (displayData === null || typeof displayData !== "object") {
    throw new Error("INVALID_LOCAL_DATA");
  }
  if (
    displayData.schema_version !== "comparison_viewer_local_v1" ||
    typeof expectedSampleId !== "string" ||
    expectedSampleId.length === 0 ||
    displayData.sample_id !== expectedSampleId ||
    displayData.duration_seconds !== duration
  ) {
    throw new Error("INVALID_LOCAL_DATA");
  }
  return {
    grouped: normalizeLocalLayer(displayData.grouped_events, duration, "grouped"),
    fine: normalizeLocalLayer(displayData.fine_segments, duration, "fine"),
    scene: normalizeLocalLayer(displayData.scene_events, duration, "scene"),
  };
}


function exact(value, keys, optional = []) {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      keys.some(key => !Object.hasOwn(value, key)) ||
      Object.keys(value).some(key => !keys.includes(key) && !optional.includes(key))) {
    throw new Error("INVALID_HYBRID_FIELDS");
  }
  return value;
}

function safeText(value, limit = 4096) {
  if (typeof value !== "string" || !value.trim() || value.length > limit ||
      /[{}\[\]\\/\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value) || /\.(npz|npy|mask)\b/i.test(value)) {
    throw new Error("INVALID_HYBRID_TEXT");
  }
  return value;
}

function sha256(value) {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) throw new Error("INVALID_DIGEST");
  return value;
}

function integer(value) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error("INVALID_HYBRID_INDEX");
  return value;
}

function confidence(value) {
  if (!(finiteNumber(value) >= 0 && value <= 1)) throw new Error("INVALID_CONFIDENCE");
  return value;
}

function unique(values, validate) {
  array(values, "INVALID_HYBRID_IDS").forEach(value => validate(value));
  if (new Set(values).size !== values.length) throw new Error("DUPLICATE_HYBRID_ID");
  return values;
}

const VARIANTS = {
  qwen_only: ["Qwen-only", "comparison_viewer_local_v1"],
  qwen_sam31: ["Qwen + SAM3.1", "comparison_viewer_hybrid_v1"],
  doubao_only: ["Doubao-only", "comparison_viewer_hybrid_v1"],
  doubao_sam31: ["Doubao + SAM3.1", "comparison_viewer_hybrid_v1"],
};

export function localVariants(sample) {
  return sample.local_variants ?? [{ id: "qwen_only", label: "Qwen-only", path: sample.local_path, format: "comparison_viewer_local_v1" }];
}

export function preferredVariant(variants) {
  for (const id of ["doubao_sam31", "qwen_sam31", "doubao_only", "qwen_only"]) {
    const variant = variants.find(item => item.id === id);
    if (variant) return variant;
  }
  throw new Error("NO_LOCAL_VARIANT");
}

export function validateManifest(manifest) {
  exact(manifest, ["schema_version", "reference_set_id", "samples"]);
  if (!["comparison_viewer_manifest_v1", "comparison_viewer_manifest_v2"].includes(manifest.schema_version)) throw new Error("INVALID_MANIFEST");
  safeText(manifest.reference_set_id);
  const samples = array(manifest.samples, "INVALID_MANIFEST");
  if (!samples.length) throw new Error("INVALID_MANIFEST");
  const seen = new Set();
  for (const sample of samples) {
    const modern = manifest.schema_version.endsWith("v2");
    exact(sample, ["sample_id", "duration_seconds", "media_path", "las_path", modern ? "local_variants" : "local_path"],
      ["caveat", "source_video_sha256", "las_sha256"]);
    if (typeof sample.sample_id !== "string" || !/^[A-Za-z0-9_-]{1,128}$/.test(sample.sample_id) || seen.has(sample.sample_id)) throw new Error("INVALID_SAMPLE_ID");
    seen.add(sample.sample_id);
    validateDuration(sample.duration_seconds);
    validateRelativeAssetPath(sample.media_path);
    validateRelativeAssetPath(sample.las_path);
    if (sample.caveat !== undefined && typeof sample.caveat !== "string") throw new Error("INVALID_CAVEAT");
    for (const name of ["source_video_sha256", "las_sha256"]) if (sample[name] !== undefined) sha256(sample[name]);
    const variants = array(localVariants(sample), "INVALID_VARIANTS");
    const ids = new Set();
    for (const variant of variants) {
      exact(variant, ["id", "label", "path", "format"], ["sha256", "source_result_sha256", "model_identity"]);
      if (!Object.hasOwn(VARIANTS, variant.id) || ids.has(variant.id) || variant.label !== VARIANTS[variant.id][0] || variant.format !== VARIANTS[variant.id][1]) throw new Error("INVALID_VARIANT");
      ids.add(variant.id);
      validateRelativeAssetPath(variant.path);
      for (const key of ["sha256", "source_result_sha256"]) if (variant[key] !== undefined) sha256(variant[key]);
      if (variant.model_identity !== undefined) safeText(variant.model_identity, 256);
    }
    if (!ids.has("qwen_only")) throw new Error("MISSING_FROZEN_QWEN");
  }
  return manifest;
}

const COMMON_PROVENANCE = ["branch", "model_stage", "evidence_mode", "source_segment_indices", "source_track_ids", "source_keyframe_ids", "repair_history", "review_status"];
const EVENT_BASE = ["id", "start", "end", "confidence", ...COMMON_PROVENANCE, "review"];
const IDENTIFIER = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/;
function evidenceId(value) {
  if (typeof value !== "string" || value.length > 128 || !IDENTIFIER.test(value)) throw new Error("INVALID_EVIDENCE_ID");
}

function keyframeId(value) {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,498}$/.test(value)) throw new Error("INVALID_KEYFRAME_ID");
}

function validateWarning(warning) {
  const fields = {
    SCENE_SEMANTICS_UNAVAILABLE: [], CV_EVIDENCE_UNAVAILABLE: [], OCCLUSION_UNAVAILABLE: [],
    ENTITY_ALIASES_TRUNCATED: ["omitted_count"], CV_ENTITY_LIMIT_APPLIED: ["omitted_count", "limit", "message"],
    ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN: ["fields", "count"], BOUNDARY_TOPOLOGY_NORMALIZED: ["issue_codes", "count"],
  };
  if (!warning || !Object.hasOwn(fields, warning.code)) throw new Error("INVALID_WARNING");
  exact(warning, ["code", ...fields[warning.code]]);
  for (const [key, value] of Object.entries(warning)) {
    if (["count", "limit", "omitted_count"].includes(key)) integer(value);
    else if (Array.isArray(value)) unique(value, item => safeText(item));
    else safeText(value);
  }
}

export function normalizeHybrid(data, duration, expectedSampleId) {
  validateDuration(duration);
  exact(data, ["schema_version", "sample", "layers", "provenance", "warnings"]);
  exact(data.sample, ["sample_id", "duration_seconds"]);
  if (data.schema_version !== "comparison_viewer_hybrid_v1" || data.sample.sample_id !== expectedSampleId ||
      typeof expectedSampleId !== "string" || !/^[A-Za-z0-9_-]{1,128}$/.test(expectedSampleId) || data.sample.duration_seconds !== duration) throw new Error("INVALID_HYBRID_SAMPLE");
  exact(data.layers, ["action_events", "occlusion_events", "scene_facts"], ["fine_segments"]);
  const provenance = data.provenance;
  exact(provenance, ["source_video_sha256", "source_result_sha256", "canonical_result_sha256", "model_identity", "cv_evidence", "review_sha256", "performance", "overlays"]);
  for (const key of ["source_video_sha256", "source_result_sha256", "canonical_result_sha256"]) sha256(provenance[key]);
  if (provenance.model_identity !== null) safeText(provenance.model_identity, 256);
  if (provenance.review_sha256 !== null) sha256(provenance.review_sha256);
  const cv = provenance.cv_evidence;
  exact(cv, cv?.status === "available" ? ["status", "artifact_key", "manifest_sha256", "cache_hit"] : ["status"]);
  if (!["available", "unavailable", "disabled"].includes(cv.status)) throw new Error("INVALID_CV_STATUS");
  if (cv.status === "available") {
    sha256(cv.artifact_key); sha256(cv.manifest_sha256);
    if (typeof cv.cache_hit !== "boolean") throw new Error("INVALID_CACHE_HIT");
  }
  exact(provenance.performance, ["total_seconds", "repair_count", "degradation_count"]);
  if (finiteNumber(provenance.performance.total_seconds) < 0) throw new Error("INVALID_PERFORMANCE");
  integer(provenance.performance.repair_count); integer(provenance.performance.degradation_count);
  unique(array(data.warnings, "INVALID_WARNINGS").map(w => { validateWarning(w); return w.code; }), safeText);
  const scene = data.layers.scene_facts;
  exact(scene, ["status", "events", "objects", "initial_state", "final_state", "outcome", "locations", "relations"]);
  const objects = new Map();
  for (const object of array(scene.objects, "INVALID_OBJECTS")) {
    exact(object, ["object_id", "name", "description"]);
    safeText(object.object_id); safeText(object.name); safeText(object.description);
    if (objects.has(object.object_id)) throw new Error("DUPLICATE_OBJECT");
    objects.set(object.object_id, object.name);
  }
  for (const name of ["initial_state", "final_state"]) for (const row of array(scene[name], "INVALID_STATES")) {
    exact(row, ["object_id", "state", "visual_evidence", "confidence"]);
    if (!objects.has(row.object_id)) throw new Error("FOREIGN_OBJECT");
    safeText(row.state); safeText(row.visual_evidence); confidence(row.confidence);
  }
  exact(scene.outcome, ["status", "description", "confidence"]);
  if (!["success", "failure", "partial", "unknown"].includes(scene.outcome.status)) throw new Error("INVALID_OUTCOME");
  safeText(scene.outcome.description); confidence(scene.outcome.confidence);
  const sourceEvents = [];
  let positiveCount = 0, reviewedCount = 0;
  function normalizeEvent(event, mode, index) {
    const additions = {
      grouped: ["event_index", "actor", "action", "target", "description"],
      occlusion: ["event_index", "event_type", "target_entity_id", "occluder_entity_id", "source_candidate_id", "description"],
      scene: ["event_index", "event_type", "actor", "target_object_id", "description"],
      fine: ["segment_index", "actor", "skill", "target", "description"],
      location: ["object_id", "location", "visual_evidence"],
      relation: ["subject_object_id", "object_object_id", "relation", "visual_evidence"],
    };
    const spatial = ["location", "relation"].includes(mode);
    exact(event, [...(spatial ? ["start", "end", "confidence", ...COMMON_PROVENANCE] : EVENT_BASE), ...additions[mode]],
      mode === "fine" ? ["action_index", "actor_state", "visual_motion_state", "event_type"] : []);
    validateInterval(event.start, event.end, duration); confidence(event.confidence);
    const branch = {grouped: "action", occlusion: "occlusion", scene: "scene", fine: "fine", location: "scene", relation: "scene"}[mode];
    const stage = branch === "occlusion" ? "occlusion_semantics" : branch === "scene" ? "scene_semantics" : "embodied_enrichment";
    if (event.branch !== branch || event.model_stage !== stage) throw new Error("INVALID_EVENT_BRANCH");
    unique(event.source_segment_indices, integer); unique(event.source_track_ids, evidenceId); unique(event.source_keyframe_ids, keyframeId);
    const history = JSON.stringify(event.repair_history);
    if (![ '["initial"]', '["initial","repair"]' ].includes(history)) throw new Error("INVALID_REPAIR_HISTORY");
    if (event.evidence_mode !== (event.source_track_ids.length ? "hybrid" : "vlm_only") ||
        (event.source_keyframe_ids.length && !event.source_track_ids.length) ||
        (cv.status !== "available" && (event.source_track_ids.length || event.source_keyframe_ids.length)) ||
        ((spatial || mode === "occlusion") && event.evidence_mode !== "hybrid")) throw new Error("INVALID_EVIDENCE_MODE");
    let id = `${mode}_${index}`;
    if (!spatial) {
      const eventIndex = integer(mode === "fine" ? event.segment_index : event.event_index);
      id = `${branch}_${eventIndex}`;
      if (event.id !== id) throw new Error("INVALID_EVENT_ID");
    }
    for (const key of additions[mode]) if (!key.endsWith("index")) safeText(event[key]);
    if (mode === "fine") {
      if (event.action_index !== undefined) integer(event.action_index);
      for (const key of ["actor_state", "visual_motion_state", "event_type"]) if (event[key] !== undefined) safeText(event[key]);
    }
    if (mode === "occlusion") {
      positiveCount++;
      evidenceId(event.target_entity_id); evidenceId(event.occluder_entity_id);
      if (!/^occ_[0-9a-f]{12}_[0-9]{4}$/.test(event.source_candidate_id) || !["occlusion_enter", "occluded", "occlusion_exit"].includes(event.event_type)) throw new Error("INVALID_OCCLUSION");
      if (!["unreviewed", "supported", "unsupported"].includes(event.review_status)) throw new Error("INVALID_REVIEW_STATUS");
      if (event.review_status !== "unreviewed") {
        reviewedCount++;
        if (!provenance.review_sha256) throw new Error("MISSING_REVIEW_DIGEST");
        exact(event.review, ["reviewer", "visual_reason"]);
        safeText(event.review.reviewer, 128); safeText(event.review.visual_reason, 1024);
      } else if (event.review !== null) throw new Error("INVALID_REVIEW");
    } else if (event.review_status !== "not_required" || (!spatial && event.review !== null)) throw new Error("INVALID_REVIEW_STATUS");
    for (const key of ["target_object_id", "object_id", "subject_object_id", "object_object_id"]) {
      if (event[key] !== undefined && !objects.has(event[key])) throw new Error("FOREIGN_OBJECT");
    }
    if (mode === "relation" && !["left_of", "right_of", "above", "below", "inside", "on", "overlapping", "near", "occluding", "unknown"].includes(event.relation)) throw new Error("INVALID_RELATION");
    sourceEvents.push(event);
    return {
      id, start: event.start, end: event.end, confidence: event.confidence, source: "local",
      type: spatial ? mode : event.action ?? event.skill ?? event.event_type,
      actor: event.actor ?? (mode === "relation" ? objects.get(event.subject_object_id) : "unknown"),
      target: event.target ?? event.target_entity_id ?? objects.get(event.target_object_id ?? event.object_id ?? event.object_object_id),
      description: spatial ? `${event.location ?? event.relation}: ${event.visual_evidence}` : event.description,
      meta: { branch, evidenceMode: event.evidence_mode, modelStage: event.model_stage,
        sourceSegments: [...event.source_segment_indices], sourceTracks: [...event.source_track_ids],
        sourceKeyframes: [...event.source_keyframe_ids], sourceCandidate: event.source_candidate_id ?? null,
        occluder: event.occluder_entity_id ?? null, repairHistory: [...event.repair_history],
        reviewStatus: event.review_status, review: event.review ? {...event.review} : null, overlays: [] },
    };
  }
  const result = { grouped: [], occlusion: [], scene: [], fine: [], availableModes: [], statuses: {},
    provenance: structuredClone(provenance), warnings: structuredClone(data.warnings), context: structuredClone(scene) };
  for (const [mode, key] of [["grouped", "action_events"], ["occlusion", "occlusion_events"], ["scene", "scene_facts"], ["fine", "fine_segments"]]) {
    const layer = data.layers[key];
    if (layer === undefined) { result.statuses[mode] = "disabled"; continue; }
    if (mode !== "scene") exact(layer, ["status", "events"]);
    if (!["available", "disabled", "unavailable"].includes(layer.status)) throw new Error("INVALID_LAYER_STATUS");
    const rows = array(layer.events, "INVALID_LAYER_EVENTS");
    if (layer.status !== "available" && (rows.length || (mode === "scene" && [scene.objects, scene.initial_state, scene.final_state, scene.locations, scene.relations].some(values => values.length)))) throw new Error("UNAVAILABLE_LAYER_HAS_EVENTS");
    result.statuses[mode] = layer.status;
    if (layer.status === "available") result.availableModes.push(mode);
    result[mode] = rows.map((event, index) => normalizeEvent(event, mode, index));
    if (mode === "scene") for (const [name, kind] of [["locations", "location"], ["relations", "relation"]]) {
      result.scene.push(...array(scene[name], "INVALID_SPATIAL_FACTS").map((row, index) => normalizeEvent(row, kind, index)));
    }
    unique(result[mode].map(event => event.id), safeText);
    result[mode] = packLanes(result[mode]);
  }
  if ((provenance.review_sha256 && reviewedCount !== positiveCount) || (!provenance.review_sha256 && reviewedCount)) throw new Error("INCOMPLETE_REVIEW");
  if (result.statuses.grouped !== "available" || (cv.status !== "available" && result.statuses.occlusion !== cv.status)) throw new Error("INCONSISTENT_LAYER_STATUS");
  const degraded = [cv.status, result.statuses.scene, result.statuses.occlusion].filter(status => status === "unavailable").length;
  if (provenance.performance.degradation_count !== degraded) throw new Error("INVALID_DEGRADATION_COUNT");
  for (const [status, code] of [[cv.status, "CV_EVIDENCE_UNAVAILABLE"], [result.statuses.scene, "SCENE_SEMANTICS_UNAVAILABLE"], [result.statuses.occlusion, "OCCLUSION_UNAVAILABLE"]]) {
    if ((status === "unavailable") !== data.warnings.some(warning => warning.code === code)) throw new Error("INCONSISTENT_WARNING");
  }
  const overlays = array(provenance.overlays, "INVALID_OVERLAYS");
  if (overlays.length > 24) throw new Error("OVERLAY_LIMIT");
  let totalBytes = 0;
  unique(overlays.map(overlay => {
    exact(overlay, ["keyframe_id", "track_id", "frame_index", "timestamp_seconds", "path", "sha256", "size_bytes"]);
    keyframeId(overlay.keyframe_id); evidenceId(overlay.track_id); integer(overlay.frame_index); integer(overlay.size_bytes);
    sha256(overlay.sha256); validateRelativeAssetPath(overlay.path);
    if (!/^(?:[A-Za-z0-9_-]+\/)+[0-9a-f]{64}\.png$/.test(overlay.path) || !overlay.path.endsWith(`/${overlay.sha256}.png`) ||
        !(0 <= finiteNumber(overlay.timestamp_seconds) && overlay.timestamp_seconds < duration) || !overlay.size_bytes) throw new Error("INVALID_OVERLAY");
    totalBytes += overlay.size_bytes;
    const consumers = sourceEvents.filter(event => event.source_keyframe_ids.includes(overlay.keyframe_id));
    if (!consumers.length || consumers.some(event => !event.source_track_ids.includes(overlay.track_id))) throw new Error("FOREIGN_OVERLAY");
    return overlay.keyframe_id;
  }), keyframeId);
  if (totalBytes > 64 * 1024 * 1024) throw new Error("OVERLAY_LIMIT");
  for (const mode of ["grouped", "occlusion", "scene", "fine"]) for (const event of result[mode]) {
    event.meta.overlays = overlays.filter(overlay => event.meta.sourceKeyframes.includes(overlay.keyframe_id)).map(overlay => ({...overlay}));
  }
  return result;
}

export function normalizeVariant(data, sample, variant) {
  if (!Object.hasOwn(VARIANTS, variant.id) || variant.label !== VARIANTS[variant.id][0] || variant.format !== VARIANTS[variant.id][1] || data.schema_version !== variant.format) throw new Error("VARIANT_FORMAT_MISMATCH");
  if (variant.format === "comparison_viewer_local_v1") {
    const layers = normalizeLocal(data, sample.duration_seconds, sample.sample_id);
    sha256(data.source_result_sha256);
    if (variant.source_result_sha256 && data.source_result_sha256 !== variant.source_result_sha256) throw new Error("RESULT_DIGEST_MISMATCH");
    return {...layers, occlusion: [], availableModes: ["grouped", "scene", "fine"], statuses: {grouped: "available", scene: "available", fine: "available", occlusion: "disabled"}, context: data, provenance: null, warnings: []};
  }
  const result = normalizeHybrid(data, sample.duration_seconds, sample.sample_id);
  const p = result.provenance;
  if ((sample.source_video_sha256 && p.source_video_sha256 !== sample.source_video_sha256) ||
      (variant.source_result_sha256 && p.source_result_sha256 !== variant.source_result_sha256) ||
      (variant.model_identity && p.model_identity !== variant.model_identity)) throw new Error("VARIANT_PROVENANCE_MISMATCH");
  if ((variant.id.startsWith("doubao") && p.model_identity !== "doubao-seed-2-1-pro-260628") ||
      (variant.id === "qwen_sam31" && !/qwen/i.test(p.model_identity ?? "")) ||
      (variant.id.endsWith("sam31") && p.cv_evidence.status === "disabled") ||
      (variant.id === "doubao_only" && p.cv_evidence.status !== "disabled")) throw new Error("VARIANT_MODEL_MISMATCH");
  return result;
}

export async function readVerifiedBytes(response, expectedSha, maxBytes) {
  if (!response.ok || !response.body || !Number.isSafeInteger(maxBytes) || maxBytes <= 0) throw new Error("ASSET_READ_FAILED");
  if (expectedSha !== undefined && expectedSha !== null) sha256(expectedSha);
  const reader = response.body.getReader();
  const chunks = [];
  let size = 0;
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maxBytes) { await reader.cancel(); throw new Error("ASSET_SIZE_LIMIT"); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  if (expectedSha) {
    const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    const actual = [...hash].map(byte => byte.toString(16).padStart(2, "0")).join("");
    if (actual !== expectedSha) throw new Error("ASSET_DIGEST_MISMATCH");
  }
  return bytes;
}

export function validateOverlayPng(bytes) {
  const header = [137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82];
  if (!(bytes instanceof Uint8Array) || bytes.byteLength < 24 || header.some((byte, index) => bytes[index] !== byte)) throw new Error("INVALID_OVERLAY_PNG");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const width = view.getUint32(16), height = view.getUint32(20);
  if (!width || !height || width > 4096 || height > 4096) throw new Error("OVERLAY_DIMENSION_LIMIT");
  return bytes;
}

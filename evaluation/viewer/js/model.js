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

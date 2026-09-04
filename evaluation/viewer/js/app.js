import {
  formatTime,
  isEventActive,
  normalizeLas,
  normalizeLocal,
  validateRelativeAssetPath,
} from "./model.js";


const MANIFEST_PATH = "evaluation/viewer/data/demo-manifest.json";
const LOCAL_MODES = new Set(["grouped", "fine", "scene"]);

const elements = {
  workspace: document.querySelector("#comparison-workspace"),
  sampleNav: document.querySelector("#sample-nav"),
  sampleDuration: document.querySelector("#sample-duration"),
  eventTotal: document.querySelector("#event-total"),
  lasCount: document.querySelector("#las-count"),
  localCount: document.querySelector("#local-count"),
  lasEvents: document.querySelector("#las-events"),
  localEvents: document.querySelector("#local-events"),
  lasContext: document.querySelector("#las-context"),
  localContext: document.querySelector("#local-context"),
  localModes: document.querySelector("#local-modes"),
  video: document.querySelector("#demo-video"),
  videoMissing: document.querySelector("#video-missing"),
  expectedVideo: document.querySelector("#expected-video"),
  videoFile: document.querySelector("#video-file"),
  playToggle: document.querySelector("#play-toggle"),
  playLabel: document.querySelector("#play-label"),
  currentTime: document.querySelector("#current-time"),
  totalTime: document.querySelector("#total-time"),
  scrubber: document.querySelector("#scrubber"),
  caveat: document.querySelector("#sample-caveat"),
  timeTicks: document.querySelector("#time-ticks"),
  timelineTracks: document.querySelector("#timeline-tracks"),
  playhead: document.querySelector("#playhead"),
  status: document.querySelector("#status"),
  eventTemplate: document.querySelector("#event-card-template"),
};

const state = {
  manifest: null,
  sample: null,
  lasReference: null,
  localData: null,
  lasEvents: [],
  localLayers: { grouped: [], fine: [], scene: [] },
  localMode: "grouped",
  localObjectUrl: null,
  loadSequence: 0,
};


function assetUrl(path) {
  const safePath = validateRelativeAssetPath(path);
  return new URL(`/${safePath}`, window.location.origin).href;
}


async function loadJson(path) {
  const safePath = validateRelativeAssetPath(path);
  const response = await fetch(assetUrl(safePath), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`无法读取 ${safePath}（HTTP ${response.status}）`);
  }
  try {
    return await response.json();
  } catch {
    throw new Error(`JSON 格式无效：${safePath}`);
  }
}


function validateManifest(manifest) {
  if (
    manifest === null ||
    typeof manifest !== "object" ||
    manifest.schema_version !== "comparison_viewer_manifest_v1" ||
    !Array.isArray(manifest.samples) ||
    manifest.samples.length === 0
  ) {
    throw new Error("demo manifest 格式无效");
  }
  const seen = new Set();
  for (const sample of manifest.samples) {
    if (
      sample === null ||
      typeof sample !== "object" ||
      typeof sample.sample_id !== "string" ||
      sample.sample_id.length === 0 ||
      seen.has(sample.sample_id) ||
      typeof sample.duration_seconds !== "number" ||
      !Number.isFinite(sample.duration_seconds) ||
      sample.duration_seconds <= 0
    ) {
      throw new Error("demo manifest 的样本定义无效");
    }
    validateRelativeAssetPath(sample.media_path);
    validateRelativeAssetPath(sample.las_path);
    validateRelativeAssetPath(sample.local_path);
    seen.add(sample.sample_id);
  }
  return manifest;
}


function showStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("is-error", isError);
  elements.status.hidden = false;
}


function hideStatus() {
  elements.status.hidden = true;
  elements.status.classList.remove("is-error");
}


function sampleById(sampleId) {
  return state.manifest.samples.find((sample) => sample.sample_id === sampleId);
}


function updateQuery(sampleId) {
  const url = new URL(window.location.href);
  url.searchParams.set("sample", sampleId);
  history.replaceState(null, "", url);
}


function renderSampleNav() {
  elements.sampleNav.replaceChildren();
  for (const sample of state.manifest.samples) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample-button";
    button.textContent = sample.sample_id;
    button.setAttribute("aria-current", String(sample.sample_id === state.sample?.sample_id));
    button.addEventListener("click", () => loadSample(sample.sample_id));
    elements.sampleNav.append(button);
  }
}


function eventKey(source, mode, event) {
  return `${source}:${mode}:${event.id}`;
}


function seekTo(seconds) {
  if (!state.sample) return;
  const clamped = Math.min(state.sample.duration_seconds, Math.max(0, seconds));
  if (Number.isFinite(elements.video.duration)) {
    elements.video.currentTime = Math.min(clamped, elements.video.duration);
  }
  syncClock(clamped);
}


function renderEventList(container, events, source, mode) {
  container.replaceChildren();
  if (events.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = mode === "scene" ? "该样本没有通过校验的 Scene events。" : "该层没有事件。";
    container.append(empty);
    return;
  }

  events.forEach((event, index) => {
    const fragment = elements.eventTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".event-card");
    const key = eventKey(source, mode, event);
    card.dataset.eventKey = key;
    card.setAttribute("aria-label", `${event.type}, ${formatTime(event.start)} 至 ${formatTime(event.end)}`);
    fragment.querySelector(".event-index").textContent = String(index + 1).padStart(2, "0");
    fragment.querySelector(".event-type").textContent = event.type;
    fragment.querySelector(".event-time").textContent = `${formatTime(event.start)}—${formatTime(event.end)}`;
    fragment.querySelector(".event-description").textContent = event.description;
    fragment.querySelector(".event-confidence").textContent = `${Math.round(event.confidence * 100)}%`;
    const metadata = fragment.querySelector(".event-meta");
    for (const [className, value] of [
      ["actor", event.actor],
      ["target", event.target || "—"],
    ]) {
      const item = document.createElement("span");
      item.className = className;
      item.textContent = value;
      metadata.append(item);
    }
    card.addEventListener("click", () => seekTo(event.start));
    container.append(fragment);
  });
}


function addContextGroup(container, label, values) {
  const group = document.createElement("section");
  group.className = "context-group";
  const heading = document.createElement("span");
  heading.className = "context-label";
  heading.textContent = label;
  group.append(heading);
  if (values.length === 0) {
    const empty = document.createElement("p");
    empty.className = "context-copy";
    empty.textContent = "未生成";
    group.append(empty);
  } else {
    for (const value of values) {
      const item = document.createElement("p");
      item.className = "context-item";
      if (value.title) {
        const strong = document.createElement("strong");
        strong.textContent = `${value.title} `;
        item.append(strong);
      }
      item.append(document.createTextNode(value.copy));
      group.append(item);
    }
  }
  container.append(group);
}


function renderLasContext(reference) {
  elements.lasContext.replaceChildren();
  addContextGroup(
    elements.lasContext,
    "Objects",
    reference.objects.map((object) => ({ title: object.name, copy: object.role })),
  );
  addContextGroup(elements.lasContext, "Initial", [{ copy: reference.initial_state }]);
  addContextGroup(elements.lasContext, "Final", [{ copy: reference.final_state }]);
  addContextGroup(elements.lasContext, "Outcome", [
    { title: reference.outcome.status, copy: reference.outcome.reason },
  ]);
}


function renderLocalContext(localData) {
  elements.localContext.replaceChildren();
  addContextGroup(
    elements.localContext,
    "Objects",
    localData.objects.map((object) => ({ title: object.name, copy: object.description })),
  );
  addContextGroup(
    elements.localContext,
    "Initial",
    localData.initial_state.map((item) => ({ title: item.object_id, copy: item.state })),
  );
  addContextGroup(
    elements.localContext,
    "Final",
    localData.final_state.map((item) => ({ title: item.object_id, copy: item.state })),
  );
  addContextGroup(elements.localContext, "Outcome", [
    { title: localData.outcome.status, copy: localData.outcome.description },
  ]);
}


function renderTicks(duration) {
  elements.timeTicks.replaceChildren();
  for (let index = 0; index <= 4; index += 1) {
    const tick = document.createElement("span");
    tick.className = "time-tick";
    tick.style.left = `${index * 25}%`;
    tick.textContent = formatTime((duration * index) / 4);
    elements.timeTicks.append(tick);
  }
}


function renderTimelineSource(label, events, source, mode, duration) {
  const track = document.createElement("div");
  track.className = `timeline-source ${source}`;
  const lanes = events.length ? Math.max(...events.map((event) => event.lane)) + 1 : 1;
  track.style.height = `${Math.max(42, lanes * 20 + 8)}px`;
  const heading = document.createElement("span");
  heading.className = "track-label";
  heading.textContent = label;
  track.append(heading);
  for (const event of events) {
    const bar = document.createElement("button");
    bar.type = "button";
    bar.className = "timeline-bar";
    bar.dataset.eventKey = eventKey(source, mode, event);
    bar.style.left = `${(event.start / duration) * 100}%`;
    bar.style.width = `${Math.max(((event.end - event.start) / duration) * 100, 0.5)}%`;
    bar.style.top = `${event.lane * 20 + 1}px`;
    bar.title = `${event.type} · ${formatTime(event.start)}—${formatTime(event.end)}`;
    bar.setAttribute("aria-label", bar.title);
    bar.addEventListener("click", () => seekTo(event.start));
    track.append(bar);
  }
  return track;
}


function renderTimeline() {
  const duration = state.sample.duration_seconds;
  const localEvents = state.localLayers[state.localMode];
  renderTicks(duration);
  elements.timelineTracks.replaceChildren(
    renderTimelineSource("LAS", state.lasEvents, "las", "events", duration),
    renderTimelineSource("LOCAL", localEvents, "local", state.localMode, duration),
    elements.playhead,
  );
}


function renderPanels() {
  const localEvents = state.localLayers[state.localMode];
  elements.lasCount.textContent = String(state.lasEvents.length);
  elements.localCount.textContent = String(localEvents.length);
  elements.eventTotal.textContent = `${state.lasEvents.length} / ${localEvents.length} events`;
  renderEventList(elements.lasEvents, state.lasEvents, "las", "events");
  renderEventList(elements.localEvents, localEvents, "local", state.localMode);
  renderLasContext(state.lasReference);
  renderLocalContext(state.localData);
  renderTimeline();
  syncClock(elements.video.currentTime || 0);
}


function syncClock(time = elements.video.currentTime || 0) {
  if (!state.sample) return;
  const duration = state.sample.duration_seconds;
  const current = Math.min(duration, Math.max(0, Number.isFinite(time) ? time : 0));
  elements.currentTime.textContent = formatTime(current);
  elements.scrubber.value = String(current);
  elements.playhead.style.setProperty("--progress", `${(current / duration) * 100}%`);
  for (const node of document.querySelectorAll("[data-event-key]")) {
    const [source, mode, id] = node.dataset.eventKey.split(":");
    const events = source === "las" ? state.lasEvents : state.localLayers[mode];
    const event = events?.find((candidate) => candidate.id === id);
    node.classList.toggle("is-active", Boolean(event && isEventActive(event, current)));
  }
}


function setPlayingState() {
  const playing = !elements.video.paused && !elements.video.ended;
  elements.playToggle.classList.toggle("is-playing", playing);
  elements.playToggle.setAttribute("aria-label", playing ? "暂停视频" : "播放视频");
  elements.playLabel.textContent = playing ? "暂停" : "播放";
}


function animateClock() {
  syncClock();
  if (!elements.video.paused && !elements.video.ended) {
    requestAnimationFrame(animateClock);
  }
}


function releaseLocalVideo() {
  if (state.localObjectUrl) {
    URL.revokeObjectURL(state.localObjectUrl);
    state.localObjectUrl = null;
  }
}


function loadConfiguredVideo(sample) {
  releaseLocalVideo();
  elements.video.pause();
  elements.videoMissing.hidden = true;
  elements.expectedVideo.textContent = `预期文件：${sample.media_path}`;
  elements.video.src = assetUrl(sample.media_path);
  elements.video.load();
}


async function loadSample(sampleId) {
  const sample = sampleById(sampleId);
  if (!sample || sample.sample_id === state.sample?.sample_id) return;
  const sequence = ++state.loadSequence;
  elements.workspace.setAttribute("aria-busy", "true");
  showStatus(`正在载入 ${sample.sample_id}…`);
  try {
    const [lasReference, localData] = await Promise.all([
      loadJson(sample.las_path),
      loadJson(sample.local_path),
    ]);
    const lasEvents = normalizeLas(lasReference, sample.duration_seconds);
    const localLayers = normalizeLocal(localData, sample.duration_seconds);
    if (sequence !== state.loadSequence) return;
    state.sample = sample;
    state.lasReference = lasReference;
    state.localData = localData;
    state.lasEvents = lasEvents;
    state.localLayers = localLayers;
    elements.sampleDuration.textContent = formatTime(sample.duration_seconds);
    elements.totalTime.textContent = formatTime(sample.duration_seconds);
    elements.scrubber.max = String(sample.duration_seconds);
    elements.scrubber.value = "0";
    elements.caveat.textContent = sample.caveat || "";
    elements.caveat.hidden = !sample.caveat;
    updateQuery(sample.sample_id);
    renderSampleNav();
    renderPanels();
    loadConfiguredVideo(sample);
    hideStatus();
  } catch (error) {
    if (sequence !== state.loadSequence) return;
    const message = error instanceof Error ? error.message : "未知错误";
    showStatus(`载入失败：${message}`, true);
  } finally {
    if (sequence === state.loadSequence) {
      elements.workspace.setAttribute("aria-busy", "false");
    }
  }
}


function bindControls() {
  elements.localModes.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (!button || !LOCAL_MODES.has(button.dataset.mode) || !state.sample) return;
    state.localMode = button.dataset.mode;
    for (const candidate of elements.localModes.querySelectorAll("button[data-mode]")) {
      candidate.setAttribute("aria-pressed", String(candidate === button));
    }
    renderPanels();
  });
  elements.scrubber.addEventListener("input", () => seekTo(Number(elements.scrubber.value)));
  elements.playToggle.addEventListener("click", async () => {
    if (elements.video.paused) {
      try {
        await elements.video.play();
      } catch {
        elements.videoMissing.hidden = false;
      }
    } else {
      elements.video.pause();
    }
  });
  elements.video.addEventListener("play", () => {
    setPlayingState();
    requestAnimationFrame(animateClock);
  });
  elements.video.addEventListener("pause", setPlayingState);
  elements.video.addEventListener("ended", setPlayingState);
  elements.video.addEventListener("timeupdate", () => syncClock());
  elements.video.addEventListener("loadedmetadata", () => {
    elements.videoMissing.hidden = true;
    syncClock();
  });
  elements.video.addEventListener("error", () => {
    if (state.sample) elements.videoMissing.hidden = false;
    setPlayingState();
  });
  elements.videoFile.addEventListener("change", () => {
    const [file] = elements.videoFile.files;
    if (!file) return;
    releaseLocalVideo();
    state.localObjectUrl = URL.createObjectURL(file);
    elements.video.src = state.localObjectUrl;
    elements.video.load();
    elements.videoMissing.hidden = true;
  });
  window.addEventListener("beforeunload", releaseLocalVideo);
}


async function initialize() {
  bindControls();
  try {
    state.manifest = validateManifest(await loadJson(MANIFEST_PATH));
    renderSampleNav();
    const requested = new URL(window.location.href).searchParams.get("sample");
    const initial = sampleById(requested) ?? state.manifest.samples[0];
    await loadSample(initial.sample_id);
  } catch (error) {
    elements.workspace.setAttribute("aria-busy", "false");
    const message = error instanceof Error ? error.message : "未知错误";
    showStatus(`初始化失败：${message}`, true);
  }
}


initialize();

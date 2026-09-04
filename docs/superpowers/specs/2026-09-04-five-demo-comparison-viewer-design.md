# Five-demo LAS/local comparison viewer design

## Purpose

Build a local, static comparison workspace for the five frozen diagnostic
videos. During playback, the viewer shows official English LAS annotations on
the left and the repaired local implementation annotations on the right. It is
for research review and debugging, not annotation editing or public hosting.

## Constraints

- Keep every source video out of Git. The existing global `*.mp4` ignore rule
  remains authoritative.
- Keep annotations and presentation code separate so either can be updated
  independently.
- Use browser-native HTML, CSS, and JavaScript only. Do not add a package
  manager, framework, build step, CDN, font download, or third-party runtime.
- Serve the repository root with a local HTTP server; direct `file://` use is
  not supported because the viewer loads JSON with `fetch`.
- Preserve LAS and local model values. The display-data exporter may select and
  rename fields but must not rewrite event semantics, intervals, or confidence.
- Exclude task identifiers, service URLs, credentials, raw requests, and model
  response envelopes from viewer data.

## User experience

The first desktop viewport is a three-column working surface:

1. The left column lists LAS events and the LAS object inventory.
2. The center contains the video, playback time, a shared scrubber, and aligned
   colored timeline lanes.
3. The right column lists local events. It defaults to grouped events and offers
   `Grouped`, `Fine`, and `Scene` switches.

A compact header switches among `full_0001`, `full_0002`, `full_0024`,
`full_0021`, and `full_0004`, and shows duration plus event counts. The active
sample is represented in `?sample=` so a view can be bookmarked.

Playback time drives all displays. An event is active when
`start <= currentTime < end`; active cards and timeline bars receive the same
highlight. Clicking any event seeks to its start. Clicking or dragging the
shared ruler seeks the video. Overlapping LAS events occupy stable lanes rather
than covering one another.

On narrow screens the video appears first, followed by LAS and local panels.
Keyboard-focusable sample, mode, event, play/pause, and seek controls are
required. Motion is subtle and disabled under `prefers-reduced-motion`.

The visual thesis is a compact research instrument: neutral graphite surfaces,
warm LAS amber, cool local cyan, tabular time labels, restrained borders, and
high information density. It must not resemble a marketing landing page.

## Data model and update flow

`evaluation/viewer/data/demo-manifest.json` is the single page entry point. For
each sample it declares:

- stable sample ID and duration;
- ignored media path `evaluation/viewer/media/<sample_id>.mp4`;
- path to the committed official English LAS reference;
- path to a committed, sanitized local display JSON;
- an optional input caveat, including the `full_0002` LAS-transcode note.

The LAS files remain the canonical files under
`evaluation/references/las_official_english_2026-09-04/references/`; the viewer
does not duplicate them.

`scripts/build_comparison_viewer_data.py` accepts a directory containing local
result JSON files and writes one deterministic display JSON per sample. Each
output includes only:

- `sample_id`, `duration_seconds`, and source-result SHA-256;
- `fine_segments`, preserving interval, description, actor, skill, target,
  actor state, visual motion state, event type, and confidence;
- `grouped_events`, preserving the existing grouped projection;
- `scene_events`, preserving overlap-capable local scene semantics;
- local object, initial state, final state, and outcome fields when present.

The exporter validates sample membership, finite bounded intervals, array
shape, and duration consistency before atomically replacing generated files.
It rejects unexpected samples and does not copy warnings or request metadata.
Re-running it with unchanged inputs produces byte-identical output.

Updating the comparison is therefore:

1. Replace or regenerate LAS reference JSON if the reference set changes.
2. Run the exporter against the new local result directory.
3. Update manifest durations or caveats only if sample inputs changed.
4. Refresh the page; no JavaScript edit is required.

## Components

- `evaluation/viewer/index.html`: accessible semantic shell and loading/error
  regions.
- `evaluation/viewer/styles.css`: responsive grid, event cards, timeline lanes,
  active states, and reduced-motion behavior.
- `evaluation/viewer/js/model.js`: pure validation, interval activity, lane
  packing, time formatting, and normalized view-model functions.
- `evaluation/viewer/js/app.js`: fetches manifest/data, owns playback state,
  renders sample controls/cards/timelines, and handles seek/file selection.
- `evaluation/viewer/data/demo-manifest.json`: the five-sample registry.
- `evaluation/viewer/data/local/*.json`: deterministic sanitized local outputs.
- `evaluation/viewer/media/README.md`: exact naming and local-serving guidance;
  video files remain ignored.
- `evaluation/viewer/README.md`: start, update, and troubleshooting commands.
- `scripts/build_comparison_viewer_data.py`: deterministic local-data exporter.

## Error handling

- If the manifest or annotation JSON fails to load or validate, the main region
  displays the failing path and a concise recovery instruction.
- If the configured video is missing, annotations remain usable and the center
  panel offers a native file picker. The selected file is session-only and is
  never uploaded or persisted.
- A sample with no events in a local layer shows an explicit empty state rather
  than an empty panel.
- Seeking clamps to `[0, duration]`; malformed, negative, non-finite, or
  out-of-range intervals are rejected during data load.

## Testing

Follow red-green-refactor for executable behavior.

- Python tests exercise the exporter with real temporary JSON files: valid
  projection, byte-identical rerun, rejection of wrong sample sets, rejection
  of invalid intervals, and exclusion of prohibited metadata.
- Node's built-in test runner exercises pure browser model functions without a
  dependency install: active interval boundaries, deterministic overlapping
  lane packing, time formatting, path validation, and malformed data rejection.
- A repository smoke test serves the repo locally and confirms that the viewer,
  manifest, LAS references, and generated local files resolve without external
  network access. Missing videos are expected and use the file-picker state.
- Final verification runs the focused exporter/JavaScript tests, the full Python
  suite, `git diff --check`, JSON parsing, and a sensitive-value scan.

## Non-goals

- No public deployment, authentication, uploads, annotation editing, or saved
  review decisions.
- No automatic semantic scoring inside the browser; committed comparison
  metrics remain the source for aggregate scores.
- No video transcoding or copying in the exporter.
- No attempt to make machine-only LAS outputs into human ground truth.

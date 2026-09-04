# Five-demo comparison viewer implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dependency-free local video player that synchronizes the five official English LAS references with the repaired local annotations in side-by-side panels and aligned timeline lanes.

**Architecture:** A static viewer fetches a small manifest, the existing LAS JSON, and deterministic sanitized local JSON. Pure ES-module functions validate and normalize events, calculate active intervals, and pack overlapping timeline lanes; a thin DOM controller owns video playback and rendering. A Python exporter converts private local result envelopes into the committed display schema without copying operational metadata.

**Tech Stack:** Python 3.12 standard library, browser-native HTML/CSS/ES modules, Node built-in test runner, pytest.

## Global Constraints

- Source videos remain outside Git under the existing global `*.mp4` ignore rule.
- Add no package manager, framework, CDN resource, external font, build step, or runtime dependency.
- Preserve source annotation semantics, intervals, descriptions, and confidence values exactly.
- Viewer data must contain no task identifiers, credentials, service URLs, raw requests, warnings, or response envelopes.
- The right panel defaults to grouped events and supports grouped, fine, and scene modes.
- Missing video files must leave annotations usable and expose a session-only local file picker.

---

### Task 1: Deterministic local-result exporter

**Files:**
- Create: `tests/test_comparison_viewer_export.py`
- Create: `scripts/build_comparison_viewer_data.py`

**Interfaces:**
- Consumes: `main(argv: Sequence[str] | None = None) -> int`, with `--input-dir`, `--output-dir`, and `--manifest` paths.
- Produces: `project_local_result(sample_id: str, duration: float, source: Mapping[str, object]) -> dict[str, object]` and one canonical UTF-8 JSON file per manifest sample.

- [ ] **Step 1: Write the failing projection test**

Create a real temporary manifest and local result containing fine segments,
grouped events, scene semantics, and prohibited operational fields. Invoke the
CLI and assert the literal projected keys and values, source SHA-256, mode
arrays, and absence of `warnings`, `task_id`, `request`, and `video_url`.

- [ ] **Step 2: Run the projection test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_comparison_viewer_export.py::test_export_projects_only_display_fields -q`

Expected: FAIL because `scripts.build_comparison_viewer_data` does not exist.

- [ ] **Step 3: Implement the minimal exporter**

Implement strict manifest parsing, exact expected-sample membership, finite
interval validation against manifest duration, source SHA-256, allowlisted
field projection, canonical `json.dumps(..., sort_keys=True,
separators=(",", ":"))`, owner-only temporary files, `fsync`, and atomic
`os.replace` publication.

- [ ] **Step 4: Add failure and determinism tests**

Add tests with literal expected errors for an unexpected sample, missing sample,
negative/end-out-of-range interval, malformed array, and a second unchanged run.
The unchanged run must produce byte-identical files.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_comparison_viewer_export.py -q`

Expected: all exporter tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_comparison_viewer_export.py scripts/build_comparison_viewer_data.py
git commit -m "feat: add comparison viewer data exporter"
```

### Task 2: Five-sample manifest and local display data

**Files:**
- Create: `evaluation/viewer/data/demo-manifest.json`
- Create: `evaluation/viewer/data/local/full_0001.json`
- Create: `evaluation/viewer/data/local/full_0002.json`
- Create: `evaluation/viewer/data/local/full_0004.json`
- Create: `evaluation/viewer/data/local/full_0021.json`
- Create: `evaluation/viewer/data/local/full_0024.json`

**Interfaces:**
- Consumes: the exporter from Task 1 and the exact five result files whose hashes are frozen in `comparison_with_postfix_local.json`.
- Produces: viewer manifest schema `comparison_viewer_manifest_v1` and local display schema `comparison_viewer_local_v1`.

- [ ] **Step 1: Write the failing real-data integration test**

Extend `tests/test_comparison_viewer_export.py` to load the repository manifest,
resolve every LAS/local path relative to the repo root, assert the literal five
sample IDs and durations, and validate that every local source hash matches
`comparison_with_postfix_local.json`.

- [ ] **Step 2: Run the integration test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_comparison_viewer_export.py::test_repository_viewer_data_is_complete -q`

Expected: FAIL because `demo-manifest.json` is absent.

- [ ] **Step 3: Add the manifest and generate local JSON**

Create the manifest with repo-root-relative annotation paths and
`evaluation/viewer/media/<sample>.mp4` media paths. Record the `full_0002` LAS
transcode caveat. Securely copy the five existing result files to a temporary
directory, run the exporter, and commit only the generated projection.

- [ ] **Step 4: Run focused integration and JSON checks**

Run:

```bash
.venv/bin/python -m pytest tests/test_comparison_viewer_export.py -q
.venv/bin/python -m json.tool evaluation/viewer/data/demo-manifest.json >/dev/null
```

Expected: tests and JSON parsing pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/viewer/data tests/test_comparison_viewer_export.py
git commit -m "data: add five-demo viewer annotations"
```

### Task 3: Pure timeline and data-model behavior

**Files:**
- Create: `evaluation/viewer/tests/model.test.mjs`
- Create: `evaluation/viewer/js/model.js`

**Interfaces:**
- Produces: `formatTime(seconds)`, `isEventActive(event, time)`, `packLanes(events)`, `normalizeLas(reference, duration)`, and `normalizeLocal(displayData, duration)`.
- Consumes: LAS reference schema and generated local display schema.

- [ ] **Step 1: Write failing Node tests**

Use literal fixtures to assert:

- `formatTime(65.23) === "01:05.23"`;
- start-inclusive/end-exclusive event activity;
- deterministic lane indices for three overlapping/non-overlapping events;
- LAS `start_s/end_s/type/actor/object_ids` normalization;
- local grouped/fine/scene normalization;
- rejection of non-finite, negative, reversed, and out-of-duration intervals;
- rejection of absolute annotation/media paths and unsupported URL schemes.

- [ ] **Step 2: Run tests and verify RED**

Run: `node --test evaluation/viewer/tests/model.test.mjs`

Expected: FAIL because `evaluation/viewer/js/model.js` is absent.

- [ ] **Step 3: Implement the pure model**

Use stable input order as the final lane-packing tie-breaker, preserve source
strings/numbers, and return new objects rather than mutating loaded JSON.
Path validation accepts only repo-root-relative paths without `..` components.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `node --test evaluation/viewer/tests/model.test.mjs`

Expected: all model tests pass with no warnings.

- [ ] **Step 5: Commit**

```bash
git add evaluation/viewer/js/model.js evaluation/viewer/tests/model.test.mjs
git commit -m "feat: add viewer timeline model"
```

### Task 4: Static comparison workspace

**Files:**
- Create: `tests/test_comparison_viewer_static.py`
- Create: `evaluation/viewer/index.html`
- Create: `evaluation/viewer/styles.css`
- Create: `evaluation/viewer/js/app.js`

**Interfaces:**
- Consumes: Task 2 manifest/data and Task 3 model functions.
- Produces: a single accessible local route at `/evaluation/viewer/`.

- [ ] **Step 1: Write the failing HTTP smoke test**

Start `ThreadingHTTPServer` on loopback with the repository root as its
directory. Request the viewer route, stylesheet, app/model modules, manifest,
and each LAS/local annotation. Assert HTTP 200, local-only asset references,
semantic `<main>`, labelled sample/mode controls, video element, file input,
left LAS region, right local region, timeline region, and status alert.

- [ ] **Step 2: Run smoke test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_comparison_viewer_static.py -q`

Expected: FAIL because the viewer route is absent.

- [ ] **Step 3: Implement the semantic HTML shell and visual system**

Create the three-column desktop grid and video-first narrow layout. Use system
fonts, CSS custom properties, amber/cyan source colors, visible focus styles,
tabular time, reduced-motion rules, a sticky compact header, and scrollable
annotation panels. Do not use external assets.

- [ ] **Step 4: Implement playback and rendering controller**

Fetch and validate manifest/annotations, keep one state object for sample,
local mode, current time, duration, and selected file URL, and render controls,
cards, metadata, shared ruler, packed lanes, playhead, counts, caveat, loading,
empty, missing-media, and fatal-error states. Revoke replaced object URLs.

- [ ] **Step 5: Run focused tests and syntax checks**

Run:

```bash
.venv/bin/python -m pytest tests/test_comparison_viewer_static.py -q
node --test evaluation/viewer/tests/model.test.mjs
node --check evaluation/viewer/js/app.js
```

Expected: all checks pass.

- [ ] **Step 6: Commit**

```bash
git add evaluation/viewer/index.html evaluation/viewer/styles.css evaluation/viewer/js/app.js tests/test_comparison_viewer_static.py
git commit -m "feat: add synchronized comparison viewer"
```

### Task 5: Local media setup and maintenance documentation

**Files:**
- Create: `evaluation/viewer/README.md`
- Create: `evaluation/viewer/media/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: repository-root HTTP serving and fixed manifest media names.
- Produces: exact start, media-copy, data-refresh, and troubleshooting instructions.

- [ ] **Step 1: Place ignored local media for the current workspace**

Copy the five frozen videos from the authorized GPU/data source to
`evaluation/viewer/media/<sample>.mp4`, verify each source SHA-256 against the
English-reference manifest, and confirm `git status` does not list video files.

- [ ] **Step 2: Document operation and updates**

Document:

```bash
.venv/bin/python -m http.server 8000
```

from the repository root and the route
`http://127.0.0.1:8000/evaluation/viewer/`. Include the exact exporter command,
expected video names, the file-picker fallback, and the privacy boundary.
Link this guide from the root README.

- [ ] **Step 3: Run complete verification**

Run:

```bash
.venv/bin/python -m pytest
node --test evaluation/viewer/tests/model.test.mjs
node --check evaluation/viewer/js/app.js
git diff --check
```

Also parse all committed viewer/reference JSON, scan the changed paths for
credential/token/task/temporary-URL patterns, and verify only `uv.lock` remains
untracked outside ignored local media.

- [ ] **Step 4: Commit and update PR**

```bash
git add README.md evaluation/viewer
git commit -m "docs: add comparison viewer guide"
git push origin feat/las-alignment
```

Update PR #2 with viewer scope, start command, tests, and the video privacy
boundary. Keep the worktree for review.


# Five-demo LAS/local comparison viewer

This dependency-free local viewer plays one frozen demo video while showing the
official English LAS reference on the left and this repository's repaired local
annotation on the right. Both panels and the shared timeline follow the same
video clock. The local panel defaults to grouped events and can switch to fine
segments or scene events.

## Start the viewer

Run the server from the repository root:

```bash
.venv/bin/python -m http.server 8000
```

Then open
<http://127.0.0.1:8000/evaluation/viewer/>. Serving the repository root is
required because the manifest links to reference JSON outside this directory.
Opening `index.html` directly with a `file://` URL will not work.

The five expected local files are:

- `evaluation/viewer/media/full_0001.mp4`
- `evaluation/viewer/media/full_0002.mp4`
- `evaluation/viewer/media/full_0024.mp4`
- `evaluation/viewer/media/full_0021.mp4`
- `evaluation/viewer/media/full_0004.mp4`

If a configured file is absent, annotations still load and the video area
offers a session-only file picker. A selected file stays in the browser and is
not uploaded.

## Refresh the comparison data

Place the exact five complete local result JSON files in a private input
directory, using `<sample_id>.json` names, then run:

```bash
.venv/bin/python scripts/build_comparison_viewer_data.py \
  --input-dir /absolute/path/to/exact-results \
  --output-dir evaluation/viewer/data/local \
  --manifest evaluation/viewer/data/demo-manifest.json
```

The exporter validates all five inputs before publishing output, retains only
display fields, and writes deterministic JSON. After changing the reference
set, durations, caveats, or sample membership, update
`evaluation/viewer/data/demo-manifest.json` as well.

Run the viewer checks after every refresh:

```bash
.venv/bin/python -m pytest \
  tests/test_comparison_viewer_export.py \
  tests/test_comparison_viewer_static.py -q
node --test evaluation/viewer/tests/model.test.mjs
node --check evaluation/viewer/js/app.js
```

## Privacy boundary

Video files are ignored by Git and must not be committed. Generated local JSON
must never contain credentials, task IDs, service URLs, request envelopes, or
temporary media URLs. See [media/README.md](media/README.md) for the frozen
video hashes.

## Troubleshooting

- A directory listing means the URL is missing `/evaluation/viewer/`.
- A JSON or module 404 usually means the server was started below the repository
  root.
- `VIDEO FILE NOT FOUND` means the expected ignored MP4 is absent or named
  incorrectly; copy it into `evaluation/viewer/media/` or use the file picker.
- The `full_0002` LAS reference used a complete 720p video-only transcode, while
  the viewer intentionally plays the frozen original used by the local run. The
  page displays this caveat next to that sample.

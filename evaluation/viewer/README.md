# Five-demo LAS/local comparison viewer

This dependency-free local viewer plays one frozen demo video while showing the
official English LAS reference on the left and this repository's repaired local
annotation on the right. Both panels and the shared timeline follow the same
video clock. The right panel selects frozen Qwen-only, Doubao-only, or a
SAM3.1-assisted variant when published. Action, Occlusion, Scene, and optional
Fine layers remain separate; unavailable layers are disabled and failed loads
clear the previous local result. The committed version-1 manifest still selects
only the frozen Qwen data until real five-demo acceptance publishes version 2.

## Start the viewer

Run the server from the repository root:

```bash
.venv/bin/python -m http.server 8000 --bind 127.0.0.1
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

## Export a verified hybrid variant

Use the exact frozen Qwen inputs for the unchanged required arguments. Hybrid
mode verifies their digests but does not rewrite the old local output or the
demo manifest. All new results must use canonical branches and a Task 12
`las_evaluation_run_v1` metadata sidecar. Source MP4s, configuration, artifact
manifests and files are authenticated through the same evaluator used for the
quantitative gates; production export rejects Fake CV artifacts.

```bash
.venv/bin/python scripts/build_comparison_viewer_data.py \
  --input-dir outputs/five-demo/qwen-only \
  --output-dir evaluation/viewer/data/local \
  --manifest evaluation/viewer/data/demo-manifest.json \
  --hybrid-input-dir outputs/five-demo/doubao-sam31 \
  --hybrid-output-dir evaluation/viewer/data/hybrid \
  --hybrid-metadata outputs/five-demo/doubao-sam31-metadata.json \
  --artifact-root outputs/five-demo/cv-artifacts \
  --review outputs/five-demo/occlusion-review.json \
  --include-fine-segments
```

The default variant is `doubao_sam31`. Use `--hybrid-variant doubao_only` with
the corresponding CV-disabled inputs/metadata and a distinct output directory
for the same-model control, or `qwen_sam31` for the optional original model.
`--media-dir`, `--reference-manifest`, and `--mapping` default to the frozen
repository assets; references and mapping retain their evaluator digest checks.

The output must name a directory below `evaluation/viewer/data/`; frozen
`data/local`, inputs and their ancestors, symlinks, and broadly scoped targets
are protected. All five projections and referenced PNGs are staged in an
owner-only tree. Only the named dataset is switched, with rollback if publication
fails. A replacement briefly removes the old directory name while moving it
aside; serve/export during a local maintenance pause, not concurrent review.
If rollback itself fails, the previous tree is retained in a private
`.hybrid-previous-*` sibling for recovery. No mask arrays are published.

The generated `variant-manifest.json` contains per-sample relative paths,
display-file digests, original result digests and model identities. At Task 15
publication, combine its `variant` entries with the frozen Qwen entry in each
version-2 sample's `local_variants`. Keep LAS/media paths and caveats, add the
verified `source_video_sha256` and `las_sha256`, and remove `local_path`.
Every published variant should carry `sha256`, `source_result_sha256`, and its
actual `model_identity`. The reader remains compatible with the version-1
manifest and the minimal four-field version-2 variant descriptor.

The projector retains distinct `canonical_result_sha256` and original-byte
`source_result_sha256`; human reviews bind the latter. Omitted review files
leave positives explicitly `unreviewed` for review preparation, not acceptance.
A supplied `las_review_set_v1` must cover all five samples and every positive
claim with a human reviewer, matching identities/times and a visual reason.
Supported/unsupported labels never change the underlying model annotation.

Event cards expose source segments, tracks, candidates, keyframes, confidence,
repair history and review verdicts. Source/run warnings remain in the context
drawer. PNG previews use registered, digest-named assets only, check size/hash
before display, cap decoded dimensions at 4096×4096, and revoke temporary object
URLs on close or selection change. Video selection still stays entirely local.

Run the viewer checks after every refresh:

```bash
.venv/bin/python -m pytest \
  tests/test_comparison_viewer_export.py \
  tests/test_viewer_projection.py \
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

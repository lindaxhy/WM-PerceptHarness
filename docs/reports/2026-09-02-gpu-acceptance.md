# LAS GPU acceptance evidence

Date: 2026-09-02

Accepted release snapshot: sanitized local GPU acceptance evidence

This document records the sanitized GPU acceptance result. It contains only
aggregate measurements, public fixture metadata, and placeholder command
shapes. Service coordinates, credentials, identifiers, private filesystem
locations, request bodies, and raw model responses are intentionally omitted.

## Test gates

| Gate | Result |
| --- | --- |
| Server full suite | 591 passed in 26.17 seconds |
| Local branch-aware suite | 591 passed; 85.99% coverage |

The local coverage gate required at least 85%. Both suites ran against the
accepted release snapshot above.

## Runtime and model identity

| Component | Accepted value |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 4.57.6 |
| qwen-vl-utils | 0.0.14 |
| Model | `Qwen/Qwen3-VL-8B-Instruct` |
| Revision | `main` |
| Manifest | 16 files; 17,545,915,883 bytes |

The transferred model manifest was verified again before inference. The pip
build used the deployment environment's reachable package mirror; real model
inference itself ran with Hugging Face and Transformers offline controls.

## GPU isolation smoke test

The host exposed four NVIDIA RTX 5090 devices, each reporting 32,607 MiB. Each
smoke process was assigned one device and observed that same device. Every
device passed.

These isolated smoke measurements were collected during a prior internal
validation pass. The Qwen generation, vision-processing, and device-selection
paths were unchanged for the accepted release snapshot. A fresh real wrist-video
end-to-end run completed successfully for that snapshot.

| GPU | Latency (seconds) | Peak bytes | Assignment check | Result |
| --- | ---: | ---: | --- | --- |
| 0 | 7.475271535 | 17,581,941,760 | assigned=observed | pass |
| 1 | 7.458510704 | 17,581,941,760 | assigned=observed | pass |
| 2 | 7.499741442 | 17,581,941,760 | assigned=observed | pass |
| 3 | 7.403160664 | 17,581,941,760 | assigned=observed | pass |

After the restart check, all four cards reported 17,276 MiB and an idle,
stable state.

## Visual-only fixtures

| Fixture | SHA-256 | Duration | Video streams | Audio streams |
| --- | --- | ---: | ---: | ---: |
| `scene-change.mp4` | `1ca7a27903d446be3bea971b457ec739adb61ddb4aa2a5614fc0aa83f96fc714` | 3.0 seconds | 1 | 0 |
| `long-general.mp4` | `9935cdbec22d8d1eb79a7187e675753dba24bbd9999c3466989b25e1e30790ab` | 5.0 seconds | 1 | 0 |
| Public wrist manipulation sample | `69f9e0e913318d91d1b0601ff893699fd42b7413480b0462e4eb52d38a8aad3b` | 10.0333 seconds | 1 | 0 |

The wrist sample came from the public
[ExylosAi pick-and-place sample dataset](https://huggingface.co/datasets/ExylosAi/pick_and_place_sample).
No fixture contained an audio stream, and no audio or ASR extraction path ran.

## End-to-end acceptance

Both configured operator identities were exercised without recording their
values. The real-model results met these checks:

- Query-only general captioning completed with three ordered timeline records.
- Explicit general captioning completed with three ordered timeline records.
- Active-object extraction completed.
- Embodied action captioning completed with 12 contiguous segments covering
  exactly `[0, 10.0333]`; the maximum segment duration was `0.8361083333`
  seconds. Global segment indices were exactly `0` through `11`.
- Every embodied segment contained all six enrichment attributes, and every
  value passed the strict declared schema.
- The JSONL export contained 12 rows. Frame ranges covered `[0, 301)` without a
  gap, schema parse was `true` for every row, and the file mode was `0600`.
- A restart poll returned the same completed embodied result: 12 segments with
  final end time `10.0333`.
- Database inspection found zero cloud-field hits and no nonterminal lease.
- The default local model alias, `qwen3-vl-8b-instruct`, completed the
  12-segment embodied action run. A filesystem-shaped model alias was rejected
  with HTTP 422 using generic wording, and no task was created.
- An in-place legacy-database migration for the accepted release snapshot reported the
  sanitized checks `model_column=true` and `routing_mismatches=0`.

During real inference, observed network connections were limited to SSH
management traffic. There was no outbound Ark, model API, Hugging Face, or
Transformers retrieval. No paid Ark request was made.

## Failures and corrective iterations

Acceptance included the following failed attempts; none was hidden or treated
as a pass:

- A solid-color general fixture was unsuitable because it supplied no visual
  evidence. Evidence-bearing scene-change and long-form fixtures replaced it.
- The embodied pipeline first exposed an exact full-duration omission, then
  Pass B cardinality and hard-cap construction gaps, and finally enrichment
  cardinality gaps. The prompts were corrected with trusted numeric
  constraints, binary64-safe feasibility windows, deterministic topology, and
  a complete conservative enrichment skeleton. Strict validators were not
  relaxed, and generated timestamps were not repaired locally.
- The first API launch used a token hash that retained a trailing newline,
  producing a local HTTP 401 before any work item was created. Hashing the
  stripped token corrected the configuration.
- A fully offline editable-install attempt could not resolve the absent
  `hatchling` build dependency. The deployment instead used a fixed editable
  source-path import, verified the imported source content, and then passed
  the complete server suite. This is not evidence of a successful offline
  reinstall.

## Final audit status

The final sanitized GPU acceptance audit is **READY**.

## Sanitized reproduction shapes

Run these from a checked-out copy of the accepted release snapshot. Angle-bracketed
values are operator-supplied placeholders and must not be committed.

```bash
cd <repository-root>
python -m pytest -q
python -m pytest -q --cov=las_repro --cov-report=term-missing
```

Verify that each fixture is visual-only before use:

```bash
ffprobe -v error -show_entries format=duration -show_streams \
  <public-silent-video>
sha256sum <public-silent-video>
```

Recheck the transferred manifest and run the isolated four-device smoke test:

```bash
python - <verified-local-model-directory> <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "sha256-manifest.json").read_text())
for item in manifest["files"]:
    path = root / item["path"]
    assert path.is_file() and path.stat().st_size == item["size"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == item["sha256"]
print(f"verified {len(manifest['files'])} files")
PY

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  python scripts/gpu_smoke.py \
  --model-dir <verified-local-model-directory> \
  --video <public-silent-video> \
  --devices 0,1,2,3
```

Start one persistent model worker per device, preserving one-model/one-device
isolation. Run each command below in its own terminal or supervised process;
each process receives the same JSON alias-to-local-directory registry and one
distinct configured device:

```bash
# Terminal/supervised process 0
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 LAS_GPU_DEVICES=0,1,2,3 \
  LAS_MODEL_REGISTRY='{"qwen3-vl-8b-instruct":"<verified-local-model-directory>"}' \
  las-repro gpu-worker --device 0 --worker-id gpu-0

# Terminal/supervised process 1
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 LAS_GPU_DEVICES=0,1,2,3 \
  LAS_MODEL_REGISTRY='{"qwen3-vl-8b-instruct":"<verified-local-model-directory>"}' \
  las-repro gpu-worker --device 1 --worker-id gpu-1

# Terminal/supervised process 2
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 LAS_GPU_DEVICES=0,1,2,3 \
  LAS_MODEL_REGISTRY='{"qwen3-vl-8b-instruct":"<verified-local-model-directory>"}' \
  las-repro gpu-worker --device 2 --worker-id gpu-2

# Terminal/supervised process 3
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 LAS_GPU_DEVICES=0,1,2,3 \
  LAS_MODEL_REGISTRY='{"qwen3-vl-8b-instruct":"<verified-local-model-directory>"}' \
  las-repro gpu-worker --device 3 --worker-id gpu-3
```

Exercise Submit and Poll with POST requests and JSON bodies. Before running
these commands, have approved secret tooling create three owner-only (`0600`)
files and set `LAS_HEADER_FILE`, `LAS_SUBMIT_JSON_FILE`, and
`LAS_POLL_JSON_FILE` to their paths without typing credentials into shell
history. The header file contains one `Authorization: Bearer ...` line. The
request files have these shapes (replace placeholders only in the protected
files):

```json
{
  "operator_id": "las_video_understanding",
  "operator_version": "v1",
  "data": {
    "video_url": "<allowed-silent-video>",
    "task_template": "general_video_captioning"
  }
}
```

```json
{
  "operator_id": "las_video_understanding",
  "operator_version": "v1",
  "task_id": "<returned-task-id>"
}
```

`curl --header "@${LAS_HEADER_FILE}"` reads headers from the protected file;
it does not expand the bearer value into the command line. Submit and Poll are:

```bash
curl --fail-with-body --silent --show-error --request POST \
  --header "@${LAS_HEADER_FILE}" \
  --header 'Content-Type: application/json' \
  --data-binary "@${LAS_SUBMIT_JSON_FILE}" \
  "${LAS_BASE_URL}/api/v1/submit"

curl --fail-with-body --silent --show-error --request POST \
  --header "@${LAS_HEADER_FILE}" \
  --header 'Content-Type: application/json' \
  --data-binary "@${LAS_POLL_JSON_FILE}" \
  "${LAS_BASE_URL}/api/v1/poll"
```

Finally, verify the export schema, frame continuity, restrictive file mode,
terminal database state, cloud-field absence, idle GPU memory, and offline
network policy using the deployment's approved local inspection tools. Store
only summarized pass/fail evidence.

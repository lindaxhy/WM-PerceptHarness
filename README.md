# WM-PerceptHarness

**A Perception Evaluation Harness for World Models and Multimodal Agents.**

WM-PerceptHarness is a self-hosted evaluation foundation for visual agents.
The first shipped adapter is `las_repro`, a visual-only video-understanding
service with local Qwen3-VL inference, durable Submit/Poll execution, strict
temporal schemas, conservative repair, and deterministic JSONL export.

## What is shipped

- Local, offline-capable Qwen3-VL inference; no Ark or remote model forwarding.
- General video captioning, active-object detection, and embodied action timelines.
- Persistent SQLite task/job coordination with leases and per-GPU workers.
- Strict structured-output validation and auditable repair-only normalization.
- Visual-only processing: no audio extraction, ASR, or transcript evidence.
- Reproducible comparison methodology and sanitized GPU acceptance evidence.

## Installation

Python 3.12 and FFmpeg/FFprobe are required.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Use `python -m pip install -e '.[gpu]'` only on a compatible CUDA host.

## Documentation

- [Architecture](docs/architecture/las-video-understanding-design.md)
- [GPU acceptance](docs/reports/2026-09-02-gpu-acceptance.md)
- [LAS/local implementation and annotation comparison](docs/reports/2026-09-03-las-vs-local-implementation-report.md)

## Development

```bash
python -m pytest -q --cov=las_repro --cov-report=term-missing
uv build --offline --wheel
```

The branch-coverage gate is 85%.

## Direction

Future releases may add composable CV tools, additional local VLM adapters,
video evaluators, agent-trajectory evaluation, and benchmark plugins. These
items describe direction, not capabilities in the current release.

## License status

This private repository does not yet grant an open-source license. Choose and
add a license before changing the repository to Public.

## LAS-compatible video adapter

This repository runs a self-hosted, asynchronous video-understanding service.
Its public surface is exactly `POST /api/v1/submit` and `POST /api/v1/poll`.
Inference is visual-only: the service neither extracts audio nor invokes ASR,
Ark, remote LAS, or another model API. `run-fake` is a deterministic local
development stack; `gpu-worker` is the only process role that imports and loads
the optional Qwen/PyTorch runtime.

The local implementation supports operator IDs `las_long_video_understand` and
`las_video_understanding`, both at version `v1`, and these templates:

- `general_video_captioning`
- `embodied_active_object_detection`
- `embodied_action_captioning`

## Install and configure

Python 3.12 and FFmpeg/FFprobe are required. For local development:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Install the offline GPU runtime only on a compatible machine:

```bash
python -m pip install -e '.[gpu]'
```

Copy `.env.example` to an ignored `.env`, edit its local paths, and explicitly
load it before every process:

```bash
cp .env.example .env
set -a
. ./.env
set +a
```

Create a bearer key at runtime and store only its SHA-256 digest in `.env`:

```bash
read -rsp 'Local LAS API key: ' LOCAL_LAS_API_KEY; echo
export LOCAL_LAS_API_KEY
python - <<'PY'
import hashlib, os
print(hashlib.sha256(os.environ["LOCAL_LAS_API_KEY"].encode()).hexdigest())
PY
```

Paste only that digest into `LAS_API_KEY_SHA256`. The `.env.example` fields are:

| Field | Purpose |
|---|---|
| `LAS_API_HOST`, `LAS_API_PORT` | Control-plane listen address. Loopback is the safe default. |
| `LAS_DATABASE_PATH` | Persistent SQLite/WAL task database inside a pre-created trusted directory. |
| `LAS_WORK_ROOT` | Per-task temporary media/frames; terminal tasks are cleaned. |
| `LAS_ALLOWED_MEDIA_ROOTS` | Comma-separated trusted local source directories. |
| `LAS_MAX_DOWNLOAD_BYTES` | Hard limit for HTTP(S) and TOS downloads. |
| `LAS_MODEL_REGISTRY` | JSON alias-to-local-directory allowlist; never a remote model ID. |
| `LAS_BACKEND` | `qwen3_vl` for production GPU workers. `run-fake` ignores it safely. |
| `LAS_GPU_DEVICES` | Comma-separated device IDs a `gpu-worker` may claim. |
| `LAS_MAX_MODEL_OUTPUT_CHARS` | Strict structured-output size limit. |
| `LAS_SEGMENT_SECONDS`, `LAS_SEGMENT_OVERLAP_SECONDS` | General-video split and overlap. |
| `LAS_MAX_FINE_SEGMENT_SECONDS` | Maximum embodied fine-segment duration. |
| `LAS_LEASE_SECONDS` | Recoverable coordinator/inference claim lease. |
| `LAS_TOS_ENDPOINT`, `LAS_TOS_REGION`, `LAS_TOS_ACCESS_KEY`, `LAS_TOS_SECRET_KEY` | Optional TOS access, supplied only at runtime. |

Create the database directory as the dedicated service account, then initialize
the database once:

```bash
install -d -m 0700 data
install -d -m 0700 work
install -d -m 0750 media
las-repro init-db
```

### SQLite runtime-directory trust boundary

The hardened multi-process storage runtime requires POSIX semantics (Linux or
macOS); Windows is not a supported service deployment target.

On POSIX, the immediate parent of `LAS_DATABASE_PATH` must already be a real
directory (not a symlink), owned by the process effective UID, with no group or
world write bits. The database and any existing `-wal`/`-shm` sidecars must be
regular, non-symlink files owned by that UID with mode `0600`. Every CLI role
enforces this before connecting to or mutating SQLite. A missing or unsafe
directory is rejected; the service does not silently create or chmod it.
Every component of the absolute directory ancestry must also be a real
directory owned by either root or the service UID. A group/world-writable
ancestor is accepted only when its POSIX sticky bit protects each root/service-
owned child, as in the conventional Linux `/tmp` hierarchy. Non-sticky writable
ancestors and all intermediate symlinks are rejected so an unrelated UID cannot
rename the protected parent.

Run all LAS roles under one dedicated service user and prefer `0700` database
and work directories. Processes with that same UID and write access to the
database path ancestry are part of the trusted runtime: Python's stdlib
`sqlite3.connect()` does not expose the `SQLITE_OPEN_NOFOLLOW` flag defined by
[SQLite's primary C API](https://www.sqlite.org/c3ref/c_open_autoproxy.html), so
stdlib code cannot exclude a malicious same-UID pathname ABA between Python's
checks and SQLite's internal open. The store retains directory/database guard
descriptors and pre/post inode checks to reject ordinary replacements, but does
not claim isolation from another trusted process deliberately replacing and
restoring names in that interval. Root is likewise inside the trusted runtime
boundary.

Keep `LAS_ALLOWED_MEDIA_ROOTS` and requester-controlled media outside the
database directory; a separate untrusted media tree must never grant write
access to the database parent. Private-copy replacement is used for a new
database, but is intentionally not used for existing-database migrations:
copying a live WAL database can omit committed WAL frames, while replacing it
would split already-running workers across different inodes. Existing
migrations therefore remain transactional in place and all normal workers may
share the one trusted directory.

## Fake-mode development

`run-fake` starts the real API, coordinator, SQLite store, all three pipelines,
and one deterministic fake inference worker in a single local process. It does
not load model weights, require a GPU, or need TOS credentials:

```bash
las-repro run-fake
```

`las-repro run-fake --once` does not open an HTTP listener. It drains all tasks
currently claimable in the configured database, waits for their fake inference
jobs, checkpoints the store, and exits. This is useful for deterministic tests
and resuming queued local work. `coordinator --once` and `gpu-worker --once`
each claim at most one record.

The examples below assume the default loopback listener. They never send Ark
credentials. Set the client key separately from the stored hash:

```bash
export LAS_BASE_URL=http://127.0.0.1:8000
export AUTH_HEADER="Authorization: Bearer ${LOCAL_LAS_API_KEY}"
export VIDEO_PATH=/absolute/path/inside/one/allowed/media/root/silent.mp4
```

Query-only general captioning uses and persists the effective local template
`general_video_captioning`:

```bash
curl --fail-with-body -sS "$LAS_BASE_URL/api/v1/submit" \
  -H "$AUTH_HEADER" -H 'Content-Type: application/json' \
  --data "{\"operator_id\":\"las_video_understanding\",\"operator_version\":\"v1\",\"data\":{\"video_url\":\"$VIDEO_PATH\",\"query\":\"describe visible actions in time order\"}}"
```

The second operator identity is independent and is preserved for Poll:

```bash
curl --fail-with-body -sS "$LAS_BASE_URL/api/v1/submit" \
  -H "$AUTH_HEADER" -H 'Content-Type: application/json' \
  --data "{\"operator_id\":\"las_long_video_understand\",\"operator_version\":\"v1\",\"data\":{\"video_url\":\"$VIDEO_PATH\",\"task_template\":\"general_video_captioning\"}}"
```

Submit each explicit template by changing only `TEMPLATE`:

```bash
for TEMPLATE in general_video_captioning embodied_active_object_detection embodied_action_captioning; do
  curl --fail-with-body -sS "$LAS_BASE_URL/api/v1/submit" \
    -H "$AUTH_HEADER" -H 'Content-Type: application/json' \
    --data "{\"operator_id\":\"las_video_understanding\",\"operator_version\":\"v1\",\"data\":{\"video_url\":\"$VIDEO_PATH\",\"task_template\":\"$TEMPLATE\"}}"
done
```

Poll with the exact operator identity used at Submit:

```bash
export TASK_ID=the-returned-task-id
curl --fail-with-body -sS "$LAS_BASE_URL/api/v1/poll" \
  -H "$AUTH_HEADER" -H 'Content-Type: application/json' \
  --data "{\"operator_id\":\"las_video_understanding\",\"operator_version\":\"v1\",\"task_id\":\"$TASK_ID\"}"
```

`query` is persisted for every template and is included in general-captioning
prompts; embodied templates retain it only for request compatibility. Finite
positive `fps` and optional `media_resolution`/`reasoning_effort`/`clip_context`
(`low`, `medium`, or `high`) are persisted and passed to local inference.
`start` and `end` must be provided together, must be finite, and must satisfy
`0 <= start < end`. General captioning applies those bounds and requires the
requested end to fit the probed video. Embodied templates persist paired bounds
for request compatibility but always process the complete probed video. An
explicit supported template or a non-blank query is required. The five
compatibility fields `ark_api_key`, `ark_endpoint_id`, `use_responses_api`,
`previous_response_ids`, and `expire_in` are accepted only to be discarded
before logging or persistence, with warnings.

### Interruption-safe local submission

Use `las_repro.local_client.stateful_submit` around the Submit call. Its identity
hash covers the normalized localhost service URL, operator, version, and
sanitized local data, but never the bearer key or an Ark field. It atomically
writes `SUBMITTING` before HTTP. If the call is interrupted, the response is
malformed, or no task ID arrives, it writes `SUBMIT_UNKNOWN`; the same invocation
then refuses to resubmit. Once a task ID is durable, repeating the same
invocation returns that ID for Poll instead of creating another task. A different
service deployment or request cannot reuse the same state file.

```python
import os
from pathlib import Path
import httpx
from las_repro.local_client import stateful_submit

base_url = os.environ["LAS_BASE_URL"].rstrip("/")
api_key = os.environ["LOCAL_LAS_API_KEY"]
submission = {
    "operator_id": "las_video_understanding",
    "operator_version": "v1",
    "data": {
        "video_url": os.environ["VIDEO_PATH"],
        "query": "describe visible actions in time order",
    },
}

def post(payload):
    response = httpx.post(
        base_url + "/api/v1/submit",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()

task_id = stateful_submit(
    Path("las_runs/local-demo/state.json"),
    submission,
    post,
    service_identity=base_url,
)
print(task_id)  # Poll this ID; never delete an unknown state and blindly retry.
```

## Media policy and optional TOS

Local sources are resolved through symlinks and must be regular files contained
by a resolved `LAS_ALLOWED_MEDIA_ROOTS` directory. Keep requesters from writing
to those trusted roots. HTTP(S) downloads reject loopback, link-local, private,
multicast, and unspecified addresses before every redirect; redirects are capped
at three, connect/read timeouts are bounded, and both Content-Length and streamed
bytes are checked against `LAS_MAX_DOWNLOAD_BYTES`. Downloads use task-local
partial files and atomic publication.

For `tos://bucket/non-empty/key`, install `.[tos]` and set all four `LAS_TOS_*`
fields at runtime. The SDK is lazy-loaded. Leaving them empty is the recommended
Fake-mode path: local allowed files still work, while a TOS request fails clearly
without logging credentials.

Generated databases, models, `data/`, `work/`, `outputs/`, videos, audio files,
caches, weights, `.env`, and `.superpowers/` evidence are excluded by Git. Keep
source media, model snapshots, the database, and work directories on separately
managed storage.

## Two-video embodied workflow

First submit `embodied_active_object_detection` with the silent wrist-view video.
Poll it to `COMPLETED`, review `data.objects`, and turn only the confirmed stable
instance names into a short naming hint such as `visible interacted object: red
container`. Then submit the silent main-view video with
`embodied_action_captioning` and:

```json
"task_context": {"prompt_context": "visible interacted object: red container"}
```

The hint is naming context, not an action SOP: an object absent from visible
interaction must not be forced into output. The main-view flow runs 0805 Pass A,
Pass B boundary fine segments, and six-field enrichment while local validators
own the final timestamps.

## Production process roles and offline GPU setup

Run each role as a separate supervised process. The API never preprocesses or
loads a model; the coordinator never imports GPU optional dependencies; each GPU
worker loads one model on exactly one configured device:

```bash
las-repro api
las-repro coordinator --worker-id coordinator-0
las-repro gpu-worker --device 0 --worker-id gpu-0
las-repro gpu-worker --device 1 --worker-id gpu-1
las-repro gpu-worker --device 2 --worker-id gpu-2
las-repro gpu-worker --device 3 --worker-id gpu-3
```

On SIGTERM/SIGINT, a worker stops making new claims, finishes its current
synchronous claim when possible, or fences an interrupted owner/generation by
making its finite SQLite lease immediately recoverable. It then checkpoints its
short-lived store connections and exits. Supervise the six processes and restart
failed processes; do not run multiple workers with the same worker ID.

On a connected staging machine, export the allowlisted snapshot and manifest:

```bash
export MODEL_EXPORT=/path/with/enough/space/qwen3-vl-8b-instruct
python scripts/download_model.py --destination "$MODEL_EXPORT"
```

Transfer the repository, an offline wheelhouse, the model directory, and test
media using operator-supplied environment variables. No address or credential is
stored in this repository:

```bash
python -m pip download --dest "$WHEELHOUSE" '.[gpu]'
scp -r "$MODEL_EXPORT" "$GPU_DESTINATION"
scp -r "$WHEELHOUSE" "$GPU_DESTINATION"
```

After transfer, verify every file against `sha256-manifest.json` before loading:

```bash
python - "$MODEL_DIRECTORY" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "sha256-manifest.json").read_text())
for item in manifest["files"]:
    path = root / item["path"]
    assert path.is_file() and path.stat().st_size == item["size"]
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
print(f"verified {len(manifest['files'])} files")
PY
```

Point `LAS_MODEL_REGISTRY` at that verified local directory, install with
`--no-index --find-links "$WHEELHOUSE"`, then run
`scripts/gpu_smoke.py --model-dir "$MODEL_DIRECTORY" --video "$SILENT_VIDEO"
--devices 0,1,2,3`. The model loader enforces `local_files_only=True`, disables
remote code, and assigns the full model to the worker's one `cuda:N` device.

## Failures, logs, cleanup, and backup

Authentication failures are HTTP 401. Invalid Submit/Poll contracts are a
sanitized HTTP 422. Poll returns HTTP 404/`TASK_NOT_FOUND` for an unknown ID and
HTTP 409 with `OPERATOR_MISMATCH` or `OPERATOR_VERSION_MISMATCH` for identity
errors. Execution failures remain HTTP 200 with task status `FAILED`, business
code `TASK_FAILED`, and a stable sanitized message. Worker diagnostics are
reduced to stable summaries; never add request bodies, bearer headers, model
prompts, or environment dumps to service logs.

Task-local temporary files under `LAS_WORK_ROOT` are removed after terminal work;
the database and completed result remain. For an online SQLite backup, use the
SQLite backup API or CLI rather than copying only the main file while WAL writers
are active:

```bash
sqlite3 "$LAS_DATABASE_PATH" ".backup '$BACKUP_PATH'"
sqlite3 "$BACKUP_PATH" 'PRAGMA integrity_check;'
```

Restore only while all roles are stopped. Keep backup paths out of the repository.
After restoring on POSIX, set the database to mode `0600` and ensure its parent
still satisfies the trusted-directory policy before starting any role.

## Verify visual-only, local operation

The deterministic suite uses a generated video with no audio stream:

```bash
python -m pytest -q --cov=las_repro --cov-report=term-missing
python -m pytest tests/test_repository_policy.py -q
ffprobe -v error -select_streams a -show_entries stream=codec_type \
  -of default=noprint_wrappers=1 "$SILENT_VIDEO"
```

The last command must print nothing. During Fake and GPU acceptance, run with
non-loopback egress denied or observe `connect(2)`/firewall logs: Fake mode should
show only the explicit localhost client connection, and Qwen mode must show no
outbound Ark, Hugging Face, LAS, or other model-API connection. Inspect persisted
payloads with SQLite and verify none of the five cloud-only field names or values
exists.

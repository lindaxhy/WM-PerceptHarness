# SAM3.1 runtime and one-video acceptance

SAM3.1 is an opt-in local CV evidence provider. Core installation, disabled
mode, and Fake mode neither install nor import SAM packages. Production uses a
dedicated SAM environment and physical GPU 3; ARK is the primary remote semantic
worker. ARK never runs locally on GPUs 0–2. Three local Qwen workers on GPUs 0–2
remain the explicit rollback/alternate semantic recipe.

## Immutable inputs and prerequisites

Use Python 3.12+, PyTorch 2.7+, and CUDA 12.6+ as upstream prerequisites. The
accepted host uses Python 3.12.13 and PyTorch 2.10.0+cu128 with CUDA 12.8. Keep
SAM and semantic environments separate; do not upgrade the Qwen environment or
change global packages to satisfy SAM.

```bash
export SAM_SOURCE=/srv/las/models/sam3
git clone https://github.com/facebookresearch/sam3.git "$SAM_SOURCE"
git -C "$SAM_SOURCE" checkout --detach 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
test "$(git -C "$SAM_SOURCE" rev-parse HEAD)" = 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
```

The accepted checkpoint is ModelScope `facebook/sam3.1`, revision
`616acbee0b9ed4177f1f389e3c13594a0a1f6398`, file
`sam3.1_multiplex.pt`, size `3502755717`, SHA-256
`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
ModelScope without Hugging Face credentials is a provisioning-time alternative,
not a worker download or runtime fallback.

```bash
export SAM_CHECKPOINT=/srv/las/models/sam3.1/sam3.1_multiplex.pt
export SAM_CHECKPOINT_SHA256="$(sha256sum "$SAM_CHECKPOINT" | awk '{print $1}')"
test "$SAM_CHECKPOINT_SHA256" = 0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6
test "$(stat -c %s "$SAM_CHECKPOINT")" = 3502755717
```

Install a transferred harness wheel, dependencies, and SAM into a clean dedicated
environment from the controlled wheelhouse:

```bash
export LAS_WHEEL=/srv/las/releases/percept_harness-0.1.0-py3-none-any.whl
export WHEELHOUSE=/srv/las/releases/wheelhouse
export HARNESS_SOURCE=/srv/las/releases/wm-percept-harness
python3.12 -m venv /srv/las/venvs/sam31
/srv/las/venvs/sam31/bin/python -m pip install --no-index --find-links "$WHEELHOUSE" "$LAS_WHEEL"
/srv/las/venvs/sam31/bin/python -m pip install --no-index --find-links "$WHEELHOUSE" 'torch>=2.7' numpy==1.26.4 ftfy==6.1.1 iopath==0.1.10 pycocotools==2.0.11 imageio-ffmpeg==0.6.0
/srv/las/venvs/sam31/bin/python -m pip install --no-index --find-links "$WHEELHOUSE" --no-deps -e "$SAM_SOURCE"
/srv/las/venvs/sam31/bin/python -m pip check
```

A `--system-site-packages` SAM venv can reuse an accepted Torch/CUDA stack, but
its whole-environment `pip check` may report unrelated inherited conflicts. Do
not call that environment clean; verify the pinned SAM import and final smoke
independently. The accepted compatibility environment adds NumPy 1.26.4, ftfy
6.1.1, iopath 0.1.10, and pycocotools 2.0.11 locally. Its PATH puts
imageio-ffmpeg 0.6.0's FFmpeg 7.0.2-static before system FFmpeg 4.4 (which lacks
`-fps_mode:v`), while system FFprobe supplies exact source PTS.

## Offline runtime configuration

Create owner-controlled state first. Keep source media separate from the
database/cache trees and prevent media writers from writing their parents.

```bash
set -euo pipefail
install -d -m 0700 /srv/las/data /srv/las/work /srv/las/cv-cache /srv/las/run /srv/las/log
install -d -m 0750 /srv/las/media
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LAS_DATABASE_PATH=/srv/las/data/tasks.sqlite3 LAS_WORK_ROOT=/srv/las/work
export LAS_ALLOWED_MEDIA_ROOTS=/srv/las/media
export LAS_BACKEND=ark
export LAS_GPU_DEVICES=0,1,2
export LAS_MODEL_REGISTRY='{"qwen3-vl-8b-instruct":"/srv/las/models/qwen3-vl-8b-instruct"}'
export LAS_ARK_MODEL_REGISTRY='{"doubao-pro":"doubao-seed-2-1-pro-260628"}'
: "${LAS_ARK_API_KEY:?inject LAS_ARK_API_KEY from the service secret store}"
: "${LAS_API_KEY_SHA256:?inject the service bearer-key SHA-256}"
export LAS_CV_PROVIDER=sam31 LAS_CV_DEVICE=3 LAS_CV_MODEL_ALIAS=sam3.1
export LAS_CV_REPOSITORY_PATH="$SAM_SOURCE" LAS_CV_CHECKPOINT_PATH="$SAM_CHECKPOINT"
export LAS_CV_BPE_PATH="$SAM_SOURCE/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
export LAS_CV_CHECKPOINT_SHA256="$SAM_CHECKPOINT_SHA256"
export LAS_CV_CACHE_ROOT=/srv/las/cv-cache
export SAM_FFMPEG_BIN="$(/srv/las/venvs/sam31/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
test -x "$SAM_FFMPEG_BIN"
export SAM_RUNTIME_BIN=/srv/las/sam-runtime-bin
install -d -m 0700 "$SAM_RUNTIME_BIN"
test -O "$SAM_RUNTIME_BIN"
test "$(stat -c %a "$SAM_RUNTIME_BIN")" = 700
if test -L "$SAM_RUNTIME_BIN/ffmpeg"; then
  test "$(readlink -f "$SAM_RUNTIME_BIN/ffmpeg")" = "$(readlink -f "$SAM_FFMPEG_BIN")"
else
  test ! -e "$SAM_RUNTIME_BIN/ffmpeg"
  ln -s "$SAM_FFMPEG_BIN" "$SAM_RUNTIME_BIN/ffmpeg"
fi
test "$(PATH="$SAM_RUNTIME_BIN:/usr/bin" command -v ffmpeg)" = "$SAM_RUNTIME_BIN/ffmpeg"
SAM_FFMPEG_VERSION="$(PATH="$SAM_RUNTIME_BIN:/usr/bin" ffmpeg -version)"
case "$SAM_FFMPEG_VERSION" in 'ffmpeg version 7.0.2-static'*) ;; *) exit 1 ;; esac
/srv/las/venvs/ark/bin/las-repro init-db
```

Run initialization and launch from the same service shell, or persist only these
non-secret values in an owner-readable environment file and inject
`LAS_ARK_API_KEY` from the service secret store. Do not assume `.env` has been
loaded. `LAS_API_KEY_SHA256` is the required service-side bearer-key digest; the
raw service bearer key is never written to this environment.

The service owns published cache entries. Consumers use authenticated handles
and validated manifests, never mask paths. Video/checkpoint/model identity,
entities, sampling, and thresholds participate in cache identity. Quarantine
cache state after any pin or evidence-setting change.

The BPE vocabulary must be byte-identical to
`sam3/assets/bpe_simple_vocab_16e6.txt.gz` in the immutable pinned SAM revision.
The runtime copies the configured local file into private storage and verifies
that copy against the pinned Git blob before constructing the predictor. An
identical relocated copy is accepted; modified bytes and mutable-worktree
substitutions are rejected.

Occlusion-candidate discovery uses a fixed positive-overlap rule. At a candidate
evidence frame, another track is eligible only when the maximum of bounding-box
IoU, target-box covered fraction, and other-box covered fraction is strictly
greater than zero. There is no configurable overlap cutoff.

`cv_overlap_threshold` and `LAS_CV_OVERLAP_THRESHOLD` were obsolete no-op
options and are no longer documented runtime controls. Remove the environment
entry from managed service configuration. Direct `Settings(...)` callers must
remove the keyword; it is now rejected as an extra field. Existing process
environment loading behavior is otherwise unchanged.

## Final one-video smoke

Choose one regular, non-symlink video beneath the allowed root. The smoke rejects
remote inputs and devices other than physical GPU 3 before GPU import. It probes
exact decoder-order PTS, sends a fixed two-entity request through the real
adapter, publishes/reloads through the real store, requires an observation, and
closes provider/store on failure. Stdout is one canonical sanitized JSON record.

```bash
export SMOKE_VIDEO=/srv/las/media/acceptance/full_0024.mp4
cd "$HARNESS_SOURCE"
test -f "$HARNESS_SOURCE/scripts/sam31_smoke.py"
PATH="$SAM_RUNTIME_BIN:/srv/las/venvs/sam31/bin:/usr/bin" \
  /srv/las/venvs/sam31/bin/python "$HARNESS_SOURCE/scripts/sam31_smoke.py" \
  --repository "$SAM_SOURCE" --checkpoint "$SAM_CHECKPOINT" \
  --checkpoint-sha256 "$SAM_CHECKPOINT_SHA256" --video "$SMOKE_VIDEO" \
  --allowed-media-root /srv/las/media --cache-root /srv/las/cv-cache \
  --device 3 | tee /srv/las/log/sam31-smoke.json
test "$(/srv/las/venvs/sam31/bin/python -c 'import json; print(json.load(open("/srv/las/log/sam31-smoke.json"))["pass"])')" = True
```

The prior formal no-override preflight for `full_0024` observed 137 exact frames,
137 observations, one track, two prompts, 56.921 seconds, and 16,419,964,928
peak Torch bytes; publish/reload passed. That is historical evidence, not final
acceptance. Unit tests inject the GPU boundary and are not GPU acceptance.

## Start, stop, restart, and rollback

`LAS_CV_TIMEOUT_SECONDS` bounds how long the coordinator waits for CV evidence;
it is not a provider execution deadline. A timed-out caller degrades to
unavailable CV, but a still-running provider can continue occupying the worker
and GPU. There is no automatic process watchdog or restart. TERM/INT requests
unwind Python worker execution and expire its owned lease when the signal
handler can run; an indefinitely blocked native operation may delay that
handler. The graceful shutdown sequence below is not a bounded recovery
guarantee for such a hang.

Operators must supervise shutdown, verify the exact owned process before any
escalation, and confirm process exit and GPU release before starting a
replacement. After an abrupt exit, allow the old lease to expire and check the
cache/staging state; do not assume Python cleanup ran, delete a shared cache,
or run two replacement workers on the same occupied GPU. Record any escalation
and recovery separately from ordinary task timing.

Primary ARK+SAM launch (ARK uses its server-owned remote credential; SAM alone
occupies physical GPU 3). Run this in the configured service shell above; the
commands deliberately fail earlier if the ARK secret or API-key hash is absent:

```bash
nohup /srv/las/venvs/ark/bin/las-repro api > /srv/las/log/api.log 2>&1 & echo $! > /srv/las/run/api.pid
nohup /srv/las/venvs/ark/bin/las-repro coordinator --worker-id coordinator-0 > /srv/las/log/coordinator.log 2>&1 & echo $! > /srv/las/run/coordinator.pid
nohup /srv/las/venvs/ark/bin/las-repro ark-worker --model-name doubao-pro --worker-id ark-0 > /srv/las/log/ark-0.log 2>&1 & echo $! > /srv/las/run/ark-0.pid
PATH="$SAM_RUNTIME_BIN:/srv/las/venvs/sam31/bin:/usr/bin" nohup /srv/las/venvs/sam31/bin/las-repro cv-worker --provider sam31 --device 3 --worker-id cv-sam31-3 > /srv/las/log/cv-sam31-3.log 2>&1 & echo $! > /srv/las/run/cv-sam31-3.pid
```

Stop cleanly and verify GPU 3 is idle:

```bash
for role in cv-sam31-3 ark-0 coordinator api; do kill -TERM "$(cat "/srv/las/run/$role.pid")"; done
for role in cv-sam31-3 ark-0 coordinator api; do while kill -0 "$(cat "/srv/las/run/$role.pid")" 2>/dev/null; do sleep 1; done; rm "/srv/las/run/$role.pid"; done
nvidia-smi --id=3 --query-compute-apps=pid,used_memory --format=csv,noheader
```

The last command must print no compute processes. Restart by rerunning the launch
block after the idle check and smoke pass. Quarantine cache while stopped:

```bash
export QUARANTINE_SUFFIX="$(date -u +%Y%m%dT%H%M%SZ)"
mv /srv/las/cv-cache "/srv/las/cv-cache.quarantine.$QUARANTINE_SUFFIX"
install -d -m 0700 /srv/las/cv-cache
```

For semantic rollback, stop `ark-0`, direct new submissions to the Qwen alias,
then launch exactly from the Qwen environment:

```bash
export LAS_BACKEND=qwen3_vl LAS_GPU_DEVICES=0,1,2
export LAS_MODEL_REGISTRY='{"qwen3-vl-8b-instruct":"/srv/las/models/qwen3-vl-8b-instruct"}'
nohup /srv/las/venvs/qwen/bin/las-repro gpu-worker --device 0 --worker-id gpu-qwen-0 > /srv/las/log/gpu-qwen-0.log 2>&1 & echo $! > /srv/las/run/gpu-qwen-0.pid
nohup /srv/las/venvs/qwen/bin/las-repro gpu-worker --device 1 --worker-id gpu-qwen-1 > /srv/las/log/gpu-qwen-1.log 2>&1 & echo $! > /srv/las/run/gpu-qwen-1.pid
nohup /srv/las/venvs/qwen/bin/las-repro gpu-worker --device 2 --worker-id gpu-qwen-2 > /srv/las/log/gpu-qwen-2.log 2>&1 & echo $! > /srv/las/run/gpu-qwen-2.pid
```

There is no silent ARK-to-Qwen or SAM-to-Fake fallback and no relabeling. To
roll back SAM, stop the CV worker, set `LAS_CV_PROVIDER=disabled`, and restart
the remaining roles. Preserve cache quarantine, database, media, and independent
source inputs for recovery.

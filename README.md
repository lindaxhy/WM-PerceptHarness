# WM-PerceptHarness

**A perception evaluation harness for world models and embodied agents.**

Point it at a folder of videos and get structured, temporally-grounded
annotations: general captions, active-object inventories, and embodied action
timelines. Inference runs on any OpenAI-compatible VLM endpoint you configure
once — Doubao ARK, DashScope, OpenAI, or a local model served by vLLM — with a
base URL, a model id, and an API key. Processing is visual-only: no audio is
extracted and no ASR is invoked.

```bash
percept eval --videos ./my_videos --template embodied_action_captioning \
  --backend openai --output results/
```

## Install

Python 3.12 and FFmpeg/FFprobe are required.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Add `-e '.[gpu]'` instead only on a CUDA host if you plan to run the local
Qwen backend.

## Configure a backend (once)

```bash
cp .env.example .env   # then edit
set -a; . ./.env; set +a
```

Pick one backend:

| Backend | What you need |
|---|---|
| `openai` | `PERCEPT_OPENAI_BASE_URL` + `PERCEPT_OPENAI_API_KEY` + `PERCEPT_OPENAI_MODEL`. Any OpenAI-compatible chat completions endpoint: Doubao ARK, DashScope, OpenAI, Gemini's compatibility layer, or a local vLLM/SGLang server. `.env.example` lists provider presets. |
| `fake` | Nothing. Deterministic CPU stub for development and CI. |
| `doubao` *(deprecated)* | `PERCEPT_ARK_API_KEY` — the legacy ARK Responses adapter; prefer `openai` with the Doubao preset. |
| `qwen` *(deprecated)* | `PERCEPT_MODEL_REGISTRY` + `PERCEPT_GPU_DEVICES` for in-process GPU inference; prefer `openai` against a local vLLM server. |

Optionally add SAM3.1 visual evidence (`--cv sam31`) for occlusion events and
CV-grounded scene facts: set the `PERCEPT_CV_*` paths to a local sam3 checkout
and checkpoint. SAM3.1 frame extraction requires FFmpeg 5.1+ (`-fps_mode`); the
plain eval path works with any FFmpeg.

That is the whole setup. Keys live only in the backend environment; nothing
else has to be provisioned.

## Evaluate

```bash
percept eval \
  --videos ./my_videos \                 # a directory, or one or more files
  --template embodied_action_captioning \
  --backend openai \
  --output results/
```

Templates:

| Template | Output |
|---|---|
| `general_video_captioning` | Whole-video summary plus a timestamped event timeline. |
| `embodied_active_object_detection` | Inventory of visibly-interacted object instances. |
| `embodied_action_captioning` | Task summary plus time-bounded action segments with enrichment fields. |

Results land in `--output` as one `<video_stem>.json` per video plus an
aggregate `results.jsonl`. Re-running the same command skips videos that
already completed, so an interrupted batch resumes where it left off.

### Recommended two-stage embodied flow

For UMI / wrist-camera data, first run object detection on the wrist view,
then pass the confirmed object names as naming context for the main view:

```bash
percept eval --videos wrist.mp4 --template embodied_active_object_detection \
  --backend openai --output stage1/

percept eval --videos main.mp4 --template embodied_action_captioning \
  --backend openai --output stage2/ \
  --prompt-context "visible interacted object: red container"
```

The context is naming guidance only, never an action script.

## Output schema

Every result is schema-validated before it is written; malformed model output
is conservatively repaired or the video is marked failed — never silently
dropped or relabeled. Action segments carry `start`/`end` in seconds on the
original video timeline.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
percept eval --videos tests/fixtures --template general_video_captioning \
  --backend fake --output /tmp/percept-smoke
```

## History

Earlier versions of this repository shipped a self-hosted, LAS-compatible
Submit/Poll service with multi-process GPU workers. That architecture is
archived intact at the tag `legacy-las-service` (branch `legacy/las-service`)
together with its deployment documentation. Videos can still be annotated by
the official Volcengine LAS operator directly; this repository no longer
reimplements its API surface.

## License status

This private repository does not yet grant an open-source license. Choose and
add a license before changing the repository to Public.

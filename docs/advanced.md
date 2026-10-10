# Advanced annotation configuration

The annotation client speaks the OpenAI-compatible chat completions protocol. Set `PERCEPT_OPENAI_BASE_URL`, `PERCEPT_OPENAI_MODEL`, and `PERCEPT_OPENAI_API_KEY` in the shell where you run `percept`. Provider examples and optional request settings are in [`.env.example`](../.env.example).

If you prefer a local environment file:

```bash
cp .env.example .env
# Edit .env with your endpoint, model and key, then load it:
set -a
. ./.env
set +a
```

The application does not automatically load `.env`. Keep keys outside committed configuration files. `--model` currently selects an alias from the configured registry; it is not an arbitrary provider model ID. For a single model, set `PERCEPT_OPENAI_MODEL` and omit `--model`. Multi-model runs can use `PERCEPT_OPENAI_MODEL_REGISTRY` instead; those two settings are mutually exclusive.

## Optional SAM3.1 evidence

Add `--cv sam31` after configuring a compatible local SAM checkout, checkpoint, tokenizer assets and GPU runtime. This adds CV evidence for supported occlusion and scene semantics. Configure these settings for your environment:

| Environment variable | Value |
|---|---|
| `PERCEPT_CV_REPOSITORY_PATH` | Local SAM source checkout |
| `PERCEPT_CV_CHECKPOINT_PATH` | Local checkpoint file |
| `PERCEPT_CV_CHECKPOINT_SHA256` | SHA-256 digest of that checkpoint |
| `PERCEPT_CV_BPE_PATH` | Local tokenizer vocabulary file |
| `PERCEPT_CV_DEVICE` | GPU device index for your runtime |
| `PERCEPT_CV_CACHE_ROOT` | Writable local artifact cache directory |

The core installation does not install the GPU runtime or download weights. The old server-specific deployment procedure and its fixed-GPU smoke script are preserved on the [viewer archive branch](https://github.com/lindaxhy/WM-PerceptHarness/tree/archive/las-comparison-viewer/docs/deployment/sam31-runtime.md); they are not a required setup step for ordinary annotation. Run a small annotation with your own video and `--cv sam31` to validate an optional CV installation, and inspect the result's CV status and warnings rather than assuming that a completed annotation proves CV succeeded.

## Historical architecture

The [LAS service design](architecture/las-video-understanding-design.md) describes the archived Submit/Poll service. It is not the current execution architecture. The current pipeline runs in process and uses a configured VLM endpoint. The [OpenAI-compatible backend design](architecture/2026-09-18-openai-compat-backend.md) documents that transition.

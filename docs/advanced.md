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

Add `--cv sam31` after configuring the local checkout, checkpoints and `PERCEPT_CV_*` settings. This adds CV evidence for supported occlusion and scene semantics; it introduces a separate GPU runtime. See [SAM3.1 deployment](deployment/sam31-runtime.md) for existing details. Deployment examples may describe the original research server and should be adapted to your environment.

## Historical architecture

The [LAS service design](architecture/las-video-understanding-design.md) describes the archived Submit/Poll service. It is not the current execution architecture. The current pipeline runs in process and uses a configured VLM endpoint. The [OpenAI-compatible backend design](architecture/2026-09-18-openai-compat-backend.md) documents that transition.

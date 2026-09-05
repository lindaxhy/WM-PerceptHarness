# Doubao Semantic Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement and review this bounded addition.

**Goal:** Run the existing validated semantic pipeline with the user-selected Doubao model through ARK Responses.

**Architecture:** One explicit remote semantic worker implements `VideoModel` and reuses the existing durable leased worker. Server configuration separates remote model IDs from local checkpoints. The API accepts the union of configured aliases.

**Tech Stack:** Python 3.12, existing httpx/Pydantic/FFmpeg, pytest; no additional GPU or SDK dependency.

## Global Constraints

- Follow `docs/superpowers/specs/2026-09-05-doubao-semantic-backend-design.md`.
- Use branch `feat/sam31-evidence-integration`; preserve frozen Qwen and LAS artifacts.
- User authorization of Doubao supersedes the old no-remote-inference constraint.
- Credentials are server-owned `SecretStr` configuration and never persisted in jobs/results/prompts.
- Endpoint is exactly `https://ark.cn-beijing.volces.com/api/v3/responses`; no redirects.
- Remote default model identity is `doubao-seed-2-1-pro-260628`.
- Preserve strict output validation, one repair, task routing, and visual-only input.
- Bound default requests to 128 frames, 32 MiB encoded request, 1,000,000 output characters, 180 seconds; reject excess before transport.
- Use `.venv/bin/python -m pytest`; use apply_patch for edits.

### Task 1: ARK backend, routing, and worker lifecycle

**Files:**
- Create: `src/las_repro/models/ark.py`
- Create: `tests/test_ark_backend.py`
- Modify: `src/las_repro/config.py`, `src/las_repro/api.py`, `src/las_repro/cli.py`
- Modify: `tests/test_api.py`, `tests/test_cli.py`
- Modify: `.env.example`, `README.md`

**Interfaces:**
- `ArkVideoModel` implements `generate(ModelRequest) -> dict`, `request_metrics() -> dict`, and `close()`; support optional existing session/request lifecycle hooks when needed.
- Constructor accepts a secret key, server-owned alias/model mapping, timeout/frame/request/output limits, optional explicit HTTP proxy, injected httpx transport/client and frame extractor for meaningful tests.
- Settings adds `ark_api_key: SecretStr | None`, `ark_model_registry: dict[str, str]`, validated finite timeout and request limits, and optional protected proxy configuration. `allowed_model_aliases` returns the local/remote union; reject duplicate aliases.
- `ark-worker --model-name <alias> [--worker-id <id>] [--once]` creates an ARK model and the existing leased worker with device `remote:ark`, without CUDA imports. Require configured backend `ark` and nonempty credentials/allowlisted model.

- [ ] **Step 1: Add failing request/response and lifecycle tests.**

Use an httpx mock transport that parses the actual outgoing request and asserts
authorization is only in headers; exact endpoint/model/store=false; images are
inline JPEG data URLs; timestamp text equals frame timestamps in the original
video interval; no filenames/absolute paths/audio are sent. Return a Responses
envelope with reasoning followed by one output-text JSON message and assert
strict parsing and allowlisted usage metrics. Example core assertion:

```python
assert captured["store"] is False
assert captured["model"] == "doubao-seed-2-1-pro-260628"
assert model.generate(request) == {"objects": []}
assert model.request_metrics() == {"input_tokens": 12, "output_tokens": 8}
```

Test malformed JSON, duplicate keys, missing/multiple output messages, incomplete
status, foreign model identity, oversized response, invalid usage, timeout, 401,
429, 500, and redirect. Ensure failures cannot expose credentials, response
bodies, local paths, or stale metrics. Exercise cleanup after success/failure.
Frame/request bounds must fail before HTTP; reject empty/out-of-span or unordered
frames. Do not silently resample or reuse a different span/fps session.

- [ ] **Step 2: Run RED.**

```bash
.venv/bin/python -m pytest tests/test_ark_backend.py -q
```

Expected missing backend failure; retain relevant output in implementation report.

- [ ] **Step 3: Implement the bounded adapter.**

Extract frames with existing media helpers, serialize timestamp/image pairs,
and include the original stage prompt. Use `thinking={"type":"disabled"}` as
the initial documented runtime default and finite stage token budgets matching
existing Qwen stages including occlusion. Unknown stages fail explicitly.
Read HTTP bodies with a finite byte bound before JSON decoding. Accept only a
completed response for the configured identity, a single assistant message with
one output_text, and one strict JSON object. Raise `ModelOutputError` for invalid
model output so the existing worker schedules normal repair. Raise a sanitized
backend error for transport/service failures. Do not add retries beyond existing
job recovery. Use explicit proxy configuration and `trust_env=False`.

- [ ] **Step 4: Add failing configuration/API/CLI integration tests.**

Assert configured ARK aliases can be submitted, unknown aliases cannot, local
aliases still work, overlapping registries fail, caller ARK keys remain dropped,
missing secrets fail cleanly, and CLI `--once` processes exactly one routed job.
Use a real SQLite queue with injected mock transport or model factory. Verify
the role closes the worker and HTTP client even on initialization/run failure,
and neither role startup nor adapter import loads Torch/SAM/Transformers.

- [ ] **Step 5: Implement configuration and role integration.**

Use `Settings.allowed_model_aliases` at API validation; do not put remote IDs in
local `Path` registry entries. Keep Qwen and Fake default behavior unchanged.
Document opt-in env configuration, the exact model alias in Submit, explicit
proxy use, process launch, external image transmission, and local rollback.
Do not log secrets or entire settings. Do not change existing public payloads.

- [ ] **Step 6: Verify and commit.**

```bash
.venv/bin/python -m pytest tests/test_ark_backend.py tests/test_api.py tests/test_cli.py tests/test_repository_policy.py -q
.venv/bin/python -m pytest -q
git diff --check
git add src/las_repro/models/ark.py src/las_repro/config.py src/las_repro/api.py src/las_repro/cli.py tests/test_ark_backend.py tests/test_api.py tests/test_cli.py .env.example README.md
git commit -m "feat: run semantic stages through Doubao Responses"
```

Run a live frozen-video stage using environment-only credentials after unit and
integration checks; record sanitized model/hash/frame count/latency/validation
evidence. Review this complete task before resuming original SAM Task 11.

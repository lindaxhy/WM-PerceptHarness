# Design: one generic OpenAI-compatible VLM backend

Status: done (steps 1–4), 2026-09-18. Step 3 validated against Doubao ARK's
OpenAI-compatible endpoint (`doubao-seed-2-1-pro-260915`, thinking disabled via
extra_body): one synthetic video completed on both the new and the legacy
backend with the same output schema and consistent content. Step 4 removed
`ark.py`, `qwen3_vl.py`, their tests, the `[gpu]` extra, and the GPU scripts;
`reverify_semantic_stages.py` was ported to the new backend (`--base-url`).
Remaining follow-up (optional): rename the `las_repro` package.
Replaces: `models/ark.py` (Doubao/ARK Responses adapter), `models/qwen3_vl.py`
(local GPU inference).

## Problem

The harness is an event-annotation operator that should run against any VLM
given an endpoint and a key. Today it ships two vendor-specific backends
instead:

- `ark.py` hardcodes the Volcengine ARK Responses endpoint, so only Doubao
  works over the network;
- `qwen3_vl.py` loads a local Qwen3-VL checkpoint with transformers, dragging
  in the `[gpu]` extra, `PERCEPT_GPU_DEVICES`, `PERCEPT_MODEL_REGISTRY`,
  `scripts/download_model.py`, and `scripts/gpu_smoke.py`.

Every new model means a new adapter. Meanwhile Doubao ARK, Qwen (DashScope),
OpenAI, Gemini (via its OpenAI compatibility layer), and any local model
served by vLLM/SGLang all expose the same OpenAI-compatible
`/chat/completions` surface.

## Goal

One network backend, `openai`, configured entirely by environment:

```bash
PERCEPT_OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3   # any provider
PERCEPT_OPENAI_API_KEY=sk-...
PERCEPT_OPENAI_MODEL=doubao-seed-2-1-pro-260628
```

```bash
percept eval --videos ./my_videos --template embodied_action_captioning \
  --backend openai --output results/
```

Switching provider = editing three variables. Local models run behind
`vllm serve Qwen/Qwen3-VL-8B-Instruct` and use the same backend with
`PERCEPT_OPENAI_BASE_URL=http://localhost:8000/v1`.

Out of scope: the SAM3.1 CV evidence stack (`cv/`, `PERCEPT_CV_*`) is kept
unchanged, including `cv_device`. The `fake` backend stays for CI.

## New module: `models/openai_compat.py`

`OpenAICompatVideoModel`, satisfying the existing `VideoModel` protocol
(`generate(request: ModelRequest) -> dict`). It is `ark.py` with the
vendor-specific parts replaced:

| Concern | Approach |
|---|---|
| Frames | Reuse `media.extract_frames` exactly as `ark.py` does: sample JPEG frames for `request.span`, base64-encode. |
| Request | `POST {base_url}/chat/completions` with one user message: interleaved `image_url` parts (`data:` URLs, timestamp text between frames, matching the ARK payload layout) plus the prompt text. |
| Auth | `Authorization: Bearer {key}`; `PERCEPT_OPENAI_EXTRA_HEADERS` (JSON object) for providers that need more. |
| Output budget | Keep the per-stage `max_output_tokens` table (`ARK_STAGE_MAX_OUTPUT_TOKENS` moves to the new module unchanged, name generalized). |
| Structured output | `PERCEPT_OPENAI_RESPONSE_FORMAT=json_schema \| json_object \| none` (default `json_object`). `json_schema` forwards `request.response_contract`; `none` relies on prompting. In every mode the reply still goes through `parse_strict_json`, which is already the safety net for fenced/dirty JSON. |
| Provider quirks | `PERCEPT_OPENAI_EXTRA_BODY` (JSON object) merged into the request body — covers ARK's `thinking: {type: disabled}`, DashScope's `enable_thinking`, etc., without vendor code. |
| Limits | Keep `max_frames`, `max_request_bytes`, `max_output_chars`, `timeout_seconds`, `proxy` as `PERCEPT_OPENAI_*` settings with the current ARK defaults. |
| Semantic cache | `semantic_cache_identity()` includes `base_url`, resolved model id, response-format mode, extra-body hash, and a new `adapter_contract_version` (`openai-compat-chat-completions-v1`). Existing cached ARK results are therefore invalidated once — re-runs are expected and correct. |
| Errors | Same sanitization discipline as `ArkBackendError`: never echo the key or full payload; surface status code + trimmed body. |

Model naming: `PERCEPT_OPENAI_MODEL` is the provider model id. Internally it
registers as a single-entry registry `{alias: model_id}` where the alias is
the sanitized model id, so `model_alias`, storage, and the semantic cache
keep working unchanged. `--model` still selects the alias when a user
provides a multi-entry `PERCEPT_OPENAI_MODEL_REGISTRY` (JSON), for A/B runs.

## CLI and config changes

- `_BACKENDS` becomes `("openai", "fake")`. `doubao` remains for one release
  as a deprecated alias: it selects `openai` and, when `PERCEPT_OPENAI_BASE_URL`
  is unset, falls back to `PERCEPT_ARK_API_KEY` + the ARK base URL with a
  deprecation warning. Removed after migration.
- `--device` (qwen-only) is dropped; `PERCEPT_GPU_DEVICES` and the local
  `PERCEPT_MODEL_REGISTRY` path registry are removed from `Settings`.
  `cv_device` and all `cv_*` settings stay.
- `.env.example` is rewritten as provider presets (Doubao ARK, DashScope,
  OpenAI, local vLLM) — comment blocks, one uncommented.

## Deletions (after the new backend is validated)

- `src/las_repro/models/ark.py`, `src/las_repro/models/qwen3_vl.py`
  (~1,000 lines)
- `tests/test_ark_backend.py`, `tests/test_qwen_backend.py` → replaced by
  `tests/test_openai_backend.py` using the same injected-`httpx.MockTransport`
  pattern `test_ark_backend.py` already uses
- `scripts/gpu_smoke.py`; `scripts/download_model.py` loses its Qwen half
  (keep the SAM3.1 download path if it has one, else keep the script only for
  SAM assets)
- `pyproject.toml` `[gpu]` extra (torch/transformers stay only if the SAM3.1
  runtime needs them — verify against `cv/sam31.py` imports before removing)
- README "Configure a backend" table shrinks to `openai` / `fake`; the
  two-stage flow examples switch `--backend doubao` → `--backend openai`.

## Migration steps

1. **Add** `openai_compat.py` + unit tests (mock transport: payload shape,
   frame budget enforcement, response_format modes, error sanitization,
   cache identity stability). No deletions yet.
2. **Wire** into CLI as `--backend openai`; add the `doubao` alias shim.
3. **Validate**: run one fixture video through Doubao's OpenAI-compatible
   endpoint and diff against the current `ark` backend output (field-level,
   not byte-level — token sampling may differ). Then a smoke run against a
   local vLLM Qwen3-VL to confirm the local path.
4. **Delete** `ark.py`, `qwen3_vl.py`, gpu extra, related scripts/settings/
   tests; update README and `.env.example`.
5. Separate follow-up PR (optional): rename the `las_repro` package to match
   WM-PerceptHarness (the env prefix is already renamed).

Steps 1–2 and 4 are pure refactor verifiable by the test suite; step 3 is the
only part that needs a real key and a GPU host.

## Risks

- **Provider image-count/size caps differ** (some cap images per request well
  below 128). Mitigation: `PERCEPT_OPENAI_MAX_FRAMES` already exists as a knob;
  document per-provider suggestions in `.env.example`.
- **`json_schema` support is uneven** across providers. Default is the widely
  supported `json_object`; `parse_strict_json` + the existing conservative
  repair path in `output_validation.py` absorb the rest.
- **Semantic cache invalidation** on switchover re-runs previously cached
  stages once (API cost). Acceptable; note it in the PR description.
- **Thinking/reasoning defaults**: some providers enable reasoning by default,
  inflating latency/cost and sometimes wrapping JSON. `PERCEPT_OPENAI_EXTRA_BODY`
  disables it per provider; validation step 3 must check this for Doubao.

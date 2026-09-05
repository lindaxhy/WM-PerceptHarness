# Doubao semantic backend amendment

Date: 2026-09-05

The user's continuation request authorizes replacing the weak deployed Qwen
semantic model with `doubao-seed-2-1-pro-260628` through the ARK Responses API.
This amendment supersedes the earlier SAM plan's prohibition on remote semantic
inference. Its evidence, validation, cache, fallback, and acceptance requirements
remain in force. Existing Qwen output and English LAS references stay frozen.

## Decision

Add an explicit ARK semantic worker implementing the existing `VideoModel`
protocol. It consumes the same durable stage jobs and strict validators. SAM3.1
continues in its isolated local process on GPU 3. Qwen remains available for
rollback and the original comparison; ARK workers need no CUDA device. Do not
masquerade Doubao results as Qwen output or silently fall back to a different
semantic model. A cloud failure follows the existing job and branch failure
contracts and is visible in acceptance evidence.

Alternatives considered: changing only the model name cannot work because the
current Qwen worker loads local checkpoints. Calling official LAS instead would
replace the pipeline being evaluated and discard its independent SAM evidence.
The ARK backend preserves the existing pipeline and changes its semantic engine.

## Request and response boundary

Use the official HTTPS endpoint `https://ark.cn-beijing.volces.com/api/v3/responses`
and a server-owned mapping from opaque model aliases to ARK model IDs. Keep this
mapping separate from local checkpoint paths. Caller compatibility credentials
remain discarded. The worker reads its key from protected process configuration;
the key never enters SQLite, task metadata, prompts, model registries, or Git.

Send sampled JPEG images as inline data URLs with explicit original-video
timestamps, using the requested stage interval and sampling rate. This retains
the existing visual-only frame workflow and avoids temporary public media URLs,
provider-side audio processing, and ambiguous clip-relative timestamps. Reject
oversized requests explicitly; never silently lower sampling resolution.
Keep local temporary image ownership and session cleanup explicit. Bound frame
count, encoded request size, response size, and request timeout.

Use `store=false`, non-streaming requests, fixed model identity, and finite
stage token limits. Separate model output messages from reasoning items and
require a completed response containing one strict JSON object. An incomplete
response is a model-output failure eligible for the existing single schema
repair. HTTP failures expose a closed error category, never raw response bodies.
Disable automatic redirect following so authorization cannot leave the endpoint.
Record only allowlisted input/output token counts and measured inference time.

## Integration and validation

An `ark-worker` CLI role runs the existing leased worker on a logical remote
device. API submission accepts aliases from local and ARK registries, without
allowing ambiguous duplicate aliases. Existing local defaults and Fake behavior
remain compatible. Deployment explicitly selects the Doubao alias.

Keep Task 10 backend-neutral, then implement and review the ARK backend before
Task 11 orchestration. Local tests must exercise actual request assembly,
timestamp association, strict response parsing, bounded transport failures,
secret redaction, worker lifecycle, and model routing using injected HTTP
transport. A live frozen-video stage run proves the provider accepts real input.

The final evaluator/viewer must identify Qwen-only, Doubao-only, and Doubao+SAM3.1
accurately. Preserve the approved comparison against frozen Qwen-only output,
and also compare Doubao-only with Doubao+SAM3.1 to separate the effects of model
replacement and CV evidence. All original positive-occlusion review and quality
gates still apply. GPU isolation evidence must identify remote semantic workers
and local SAM honestly; do not claim ARK inference runs on GPUs 0-2.

## Observed feasibility

The supplied model completed an image request on 2026-09-05 with HTTP 200,
`status=completed`, valid JSON, and 8.843 seconds measured client latency.
This proves connectivity and image input only, not video annotation quality.
The official SDK declares inline data URLs for image and video inputs:

- https://github.com/volcengine/volcengine-python-sdk/blob/master/volcenginesdkarkruntime/types/responses/response_input_image_param.py
- https://github.com/volcengine/volcengine-python-sdk/blob/master/volcenginesdkarkruntime/types/responses/response_input_video_param.py

The local HTTP environment advertises both an HTTP proxy and a SOCKS proxy;
the installed HTTP client lacks the optional SOCKS dependency. The successful
probe explicitly used the configured HTTPS proxy. Production transport settings
must not silently depend on a developer machine's proxy configuration.

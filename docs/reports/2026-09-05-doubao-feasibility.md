# Doubao semantic replacement feasibility

Date: 2026-09-05

The user-selected `doubao-seed-2-1-pro-260628` is reachable through the ARK
Responses endpoint and accepts the harness's existing Pass A prompt with
timestamped video frames. Production deployment and complete five-demo
acceptance are not yet established by these probes.

| Probe | Evidence |
| --- | --- |
| Public documentation image | HTTP 200, completed, strict JSON output; 8.843 s client latency |
| Frozen `full_0001`, existing Pass A prompt | HTTP 200, completed; 33 JPEG frames at 3 fps; 4 actions, 7 entity candidates |
| Pass A structural validation | `CoarsePlan.model_validate` and `validate_coarse_plan` passed without repair |
| Video probe elapsed time | 43.981 s including local extraction and request |
| Video source SHA-256 | `c3243c46bad68d3b2772e82648e45b68e75a1893b0ce27edecd450226464c1e9` |
| Video request token usage | 44,748 input, 557 output, 45,305 total; zero reasoning and cached tokens |
| Generation configuration | Responses API, `store=false`, thinking disabled, 4,096 max output tokens |

Image and video calls both returned the exact requested model identity. Video
frames were encoded inline and paired with original-video timestamps. No audio,
temporary public media host, provider file upload, or LAS reference regeneration
was used. Credentials were supplied to a non-echoing prompt and held in the
probe's process environment; they are absent from this report and repository.

The first image probe stopped before networking because automatic proxy
discovery selected SOCKS while the optional `socksio` package was absent. The
successful probes explicitly selected the configured HTTPS proxy. This was a
local transport configuration problem, not a model-service rejection.

These are feasibility measurements, not a model-quality comparison: no manual
semantic verdict, full pipeline timing, occlusion score, SAM artifact, or
deployment change is claimed. The implementation plan retains the original
acceptance gates and adds a Doubao-only control to distinguish the model
replacement from SAM's contribution.

## Production adapter probe

At commit `0fd6bf8`, the actual `ArkVideoModel` implementation completed the same
frozen `full_0001` Pass A request in 29.109 seconds (including frame extraction).
Its strict response checks accepted the exact configured model identity and
completed response; `CoarsePlan` and temporal/entity validation passed without
repair. It returned four actions and eight entity candidates, with 44,616 input
and 593 output tokens. The source digest is unchanged from the table above.
This run used the normal adapter, not the earlier hand-built HTTP probe.

The adapter is opt-in and does not change deployment by itself. Complete
Doubao-only and Doubao+SAM five-demo runs remain required. A subsequent review
identified a malformed deep-JSON recursion error classification edge case;
commit `718a9ff` fixed it, with a RED/GREEN regression and clean scoped review.
Controller verification passed all 26 backend tests on that final fix.

Provider request formats were checked against the official SDK's
[image input type](https://github.com/volcengine/volcengine-python-sdk/blob/master/volcenginesdkarkruntime/types/responses/response_input_image_param.py)
and [video input type](https://github.com/volcengine/volcengine-python-sdk/blob/master/volcenginesdkarkruntime/types/responses/response_input_video_param.py).

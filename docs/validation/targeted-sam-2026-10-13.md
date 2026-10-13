# Targeted SAM validation — 2026-10-13

The v2 pipeline removes unconditional SAM passes. VLM drafting and separate visual review run first; only localized object questions trigger a text+box reference-frame probe. A visual mask check precedes tracking. Public annotations contain only `duration` and `events`; separate audits preserve execution state, draft, review, timing and evidence references.

## Automated checks

73 tests passed, including original-clock preservation, window/mode/box cache isolation, probe-before-tracking, rejected-mask retry limits, lazy GPU loading, public/audit persistence, failed/empty scoring, and manifest exports. Installation verification passed 6/6. Independent code review found and resolved audit retention, reserved filename and tracking-bound issues.

## Four real videos

Same videos as the earlier baseline-SAM pilot; hosted `doubao-seed-2-1-pro-260628`, original video frames at the existing 2 fps VLM sampling, request cache disabled, 128 MiB client request budget. Outputs were not manually corrected.

| Sample | Earlier run (s) | Targeted run (s) | Events, earlier → targeted | Targeted SAM jobs |
|---|---:|---:|---:|---:|
| full_0086, pen into cup | 459.3 | 34.9 | 3 → 3 | 0 |
| full_0076, camera movement / apple | 326.1 | 34.3 | 5 → 3 | 0 |
| full_0582, bar magnets | 498.9 | 58.1 | 8 → 6 | 0 |
| full_0568, ball and occluder | 417.2 | 39.3 | 3 → 1 | 0 |

All four completed. The reviewer requested no SAM in these runs, so this timing comparison is not an equal-quality speed benchmark or evidence that targeted SAM improved these annotations.

The camera/rolling error remains: the model called a stationary apple rolling, and its motion audit invented orientation changes. The pen was again called upright and static intervals remained. Magnet events remain fragmented. Lightweight review is implemented but does not reliably detect confidently wrong visual claims.

An initial magnet run failed because the model rounded `9.840233333333334` to `9.840233`. Validation now canonicalizes duration serialization differences of at most one microsecond to authoritative media metadata; larger discrepancies still fail. Original failed output and both successful/failed snapshots are retained.

## Separate, explicitly targeted GPU test

This diagnostic manually selected the pen as the object to examine; the candidate box itself came from the real VLM. It is separate from the four natural annotation runs, not an automatically triggered SAM call.

- Hardware/runtime: RTX 5090, pinned SAM3.1 multiplex video predictor, existing BF16 autocast, execution chunk 8, short-video sampling capped at 30 fps.
- Candidate reference: original frame 30 at approximately 0.999 s; normalized box `[0.622, 0.532, 0.669, 0.774]`.
- Single-frame probe: 4.61 s; mask visually selected the intended isolated black marker rather than adjacent markers/cup. Real VLM mask review accepted it in 8.56 s, without a correction.
- First tracking run failed. The pinned upstream batched detector adds `max_frame_num_to_track` to a nonzero start without clamping to the video length. Passing full window length from reference index 29 therefore requested frames beyond the end. The adapter now omits that limit for spatial requests, allowing the upstream default to cover the actual window. A regression test reproduces the error.
- Fixed replay: model load 36.48 s; tracking/artifact publication 22.61 s; 173 sampled frames, original index 0 at 0 s through index 173 at 5.763067 s. Reference lies inside this interval; forward and backward coverage both completed. Peak allocated GPU memory 16,061,936,128 bytes.
- **Tracking quality remains inadequate:** only two visible detections, at 1.165933 and 1.199244 s. Both masks selected the intended marker, but there was no sustained coverage through pickup or insertion. A good prompt-frame segmentation does not establish a valid full trajectory.

These experiments verify code execution, output separation and the spatial-prompt API; they do not establish annotation accuracy or readiness for large-scale unattended labeling. SAM absence is never treated as proof of object disappearance, and review concerns remain in the audit.

Natural-run source snapshot SHA-256: `4fdf52ed11d5e9ac999b06577b19f12b9fc7a74d61332c752f9a70eff8cc58e2`. Tracking-fix snapshot SHA-256: `9aabd984771ea061433c3d210ffad005b2160ef809989c05fd12207a6c0ac7bb`. The latter changes only the targeted tracking bound and its regression test relative to the source used for natural runs. Raw videos, stage responses, immutable source archives, masks, failure traces and before/after HTML are retained in the experiment artifacts.

# Semantic repair feedback alignment

Status: implementation authorized by the user's request to fix the diagnosed
issues and prior instruction to proceed by default without repeated approval.
Baseline: `8ac2538eba416e4a3397d21d9780cc76f590fdd6`.

## Scope and choice

Align semantic prompts with existing validation and provide specific closed
scene repair codes. Keep the conservative slash/path/mask guard: ordinary slash
phrasing and a relative path cannot reliably be distinguished by punctuation.
Explicitly instruct the model to use words instead. Relaxing the guard risks
artifact leakage; automatic rewriting would obscure the original model output.
Neither is selected. Prompt changes improve instructions, not proven accuracy.

Occlusion must contain at least one supported positive-duration event; if none
is supported, use unknown with empty events. Non-occlusion events remain empty.
All evidence text must obey the existing forbidden-character and mask rules.

Scene validation retains SCENE_SPATIAL_INVALID for compatibility and adds closed
categories for evidence availability, time bounds, observed timestamps, ordering,
object references, tracks, keyframes, source segments, provenance, and prohibited
text. Aggregate independently detectable errors after successful schema parsing,
including temporal, text, and provenance checks, so ordering does not hide the
observed-clock or source-segment errors. Never serialize raw exceptions, IDs,
paths, rejected text, or arbitrary codes in the repair envelope.

Keep shared provenance enforcement authoritative for both scene and hybrid
consumers. Do not introduce a second spatial validation policy. Invalid IDs may
short-circuit that row's provenance checks safely. Preserve all previous rejection
conditions and valid output identity. Preserve generic fallback for unexpected
failures. No sorting, snapping timestamps, inferred provenance, record deletion,
extra retries, or forced nonempty spatial lists.

Prompts explicitly distinguish video duration from observed frame timestamps,
require all temporally overlapping segments (not only object-relevant segments),
explain closed repair codes, and retain supported records during repair while
allowing genuine spatial abstention. Fixed code explanations contain no raw data.

## Verification and exclusions

TDD against the real sanitizer and pipeline repair envelope: combined faults
surface multiple safe codes; isolated faults retain specific categories; legal
spatial outputs remain identical; empty spatial lists remain valid; malformed
and foreign provenance remains rejected; slash/path/mask controls stay rejected.
Use synthetic fixtures only in committed tests. Verify rendered examples against
real validators; prose changes receive manual contract review, not brittle exact
wording tests. Run focused and full Python tests with configured 85% coverage,
Node viewer tests, whitespace checks, and independent code review.

No new paid calls, SAM runs, goal creation, deployment, or modification of frozen
experiment reports/private responses. Historical acceptance remains failed.
The existing unrelated untracked performance note must remain untouched.

## Design review

- [x] Reuse the completed diagnosis and inspect current consumers and tests.
- [x] Compare strict prompt alignment with guard relaxation and automatic repair.
- [x] Present bounded design; proceed under the user's default execution request.
- [x] Self-review scope, consistency, no placeholders; visual companion not needed.
- [x] Write implementation plan before code changes.

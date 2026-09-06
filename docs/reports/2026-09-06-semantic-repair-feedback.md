# Semantic repair feedback fix

Date: 2026-09-06. Product baseline: `8ac2538eba416e4a3397d21d9780cc76f590fdd6`.
Scope authorized by the user's request to fix the completed offline diagnosis.
This report is a new local implementation checkpoint, not a replacement for the
[failed bounded experiment](2026-09-06-semantic-reverification.md).

## Changes

The occlusion prompt now explicitly requires at least one supported positive
interval for an occlusion classification. If no such interval is supported,
unknown with empty events remains the conservative alternative. Both semantic
prompts explicitly prohibit slash punctuation and other existing forbidden text
content. The slash/path/mask guard itself is unchanged; no allowlist for ordinary
slash phrasing, automatic text substitution, or semantic acceptance relaxation
was introduced.

Scene validation now retains `SCENE_SPATIAL_INVALID` and adds specific closed
codes for evidence availability, time bounds, observed timestamps, ordering,
object references, segment provenance, tracks, keyframes, provenance metadata,
and prohibited text. Temporal, text and shared provenance checks aggregate
independent errors instead of an ordering error hiding all subsequent issues.
Malformed provenance identifiers may short-circuit that row's provenance checks.
The shared validator remains authoritative for scene and hybrid callers.

Repair envelopes still contain only registered codes. They never carry the
rejected response, raw exception, object IDs, or artifact paths. Unknown typed
codes are filtered and unexpected errors retain a generic closed failure.
Valid responses remain unchanged. No records are sorted, dropped, rewritten,
or assigned inferred timestamps/provenance by local postprocessing.

The scene prompt explains that video duration need not occur in observed_clock,
requires exact observed timestamps and all temporally overlapping fine segments,
and describes each closed repair category. Supported spatial records should be
rechecked rather than deleted solely to clear validation. Genuine spatial
abstention remains valid; nonempty spatial output is not forced.

## Offline evidence replay

The four archived raw responses were read and passed to the new local sanitizer
without modifying response objects or files. No model calls were made.

- Scene initial: still invalid; now reports the umbrella plus ordering,
  unobserved time, source segments, tracks, provenance and prohibited-text codes.
- Scene repair: valid and exactly equal to its original raw response; its
  existing empty spatial arrays are not reinterpreted as a new semantic success.
- Occlusion initial: still `OCCLUSION_EVENTS_EMPTY`.
- Occlusion repair: still `OCCLUSION_EVIDENCE_PROHIBITED_CONTENT`.

These unchanged acceptance outcomes are expected: the fix improves future
instructions and diagnostic feedback, not historical responses. A future real
call is needed to evaluate model compliance and retained spatial content.
Visual correctness and overall integration acceptance remain unproven/failed.

## Verification and review

Before implementation, 72 focused baseline tests passed. After adding detailed
feedback expectations, the red run produced 14 expected failures and 61 passes;
failures were missing specific codes, not import/setup errors. Initial green
verification passed 88 focused tests. The first full run passed 1,737 tests with
85.86% coverage (76.30 seconds), with two existing dependency deprecations.

Independent read-only review found no blocking issue or acceptance weakening.
The reviewer compared baseline/current provenance acceptance across 34 synthetic
cases and checked unknown-code and raw-exception filtering. Its one minor request
was to commit the latter fault-injection controls. Those two tests were added,
along with a video-end versus observed-frame regression and an available-evidence
spatial-abstention control. The resulting 91 focused tests passed, and scoped
re-review reported no remaining findings.

Final-tree verification:

```text
.venv/bin/python -m pytest -q --cov=las_repro --cov-report=term
1740 passed, 2 existing dependency warnings in 77.33s; coverage 85.87%
```

Both rendered-prompt boundary/example tests also passed. No unresolved review
findings remain. Local implementation and verification are complete; live model
compliance has not been remeasured.

Viewer tests: 20 passed. Repository-policy tests: 31 passed. Syntax checks passed
for `evaluation/viewer/js/app.js` and `evaluation/viewer/js/model.js`; whitespace
checks passed. An initial syntax invocation used a nonexistent `viewer/app.js`
path; it was corrected to the actual `viewer/js/app.js`, with no source change.

## Identity and release boundary

New prompt file SHA-256 values:

- Scene: `569b342d131efb986af8e6384f1513298175607806ac015f9fe594d465d643a6`
- Occlusion: `9528896adef0ef7ba7f88eadcda4f9b109cffdb1dcea5f6c67e2537aa0b2f1ce`

Future inference must use the new source/prompt identity; no past result is
relabeled. No paid call, SAM rerun, dependency upgrade, deployment, goal creation,
merge, or push was performed. Frozen results and viewer data have no diff from
the baseline. The existing unrelated untracked performance note is untouched.
The implementation is retained in the existing isolated feature worktree.

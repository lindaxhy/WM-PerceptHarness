# LAS capability completion implementation plan

> **For agentic workers:** Execute task by task using executing-plans; use requesting-code-review before delivery.

**Goal:** Recover useful, visually supported occlusion and scene output, then measure the unchanged five-video quality, latency and cache gates.

**Architecture:** Keep the pinned Doubao and SAM models and immutable historical evidence. Fix evidence loss at its source before changing semantic interpretation. Validate fresh outputs with the production contracts and evaluate against the frozen LAS references.

**Tech Stack:** Python 3.12, Pydantic, ARK Responses, SAM3.1, pytest, existing static viewer.

## Global constraints

- Reuse `feat/sam31-evidence-integration`; keep PR #4 draft and unmerged.
- Preserve all historical runs and the untracked user performance note.
- Preserve the 720-second latency gate, positive occlusion F1 gate and 80% human precision gate. Automated inspection is not human review.
- Never convert detector loss into occlusion or silently repair unsupported claims.
- Fresh diagnostic tranche: two existing stages, at most four ARK calls, no SAM inference. Stop repeated identical calls after that tranche; follow with a changed, evidence-backed hypothesis only.
- Follow-up experiments use separate directories and explicit call ceilings; record actual calls, tokens, timings and input/source hashes. No unbounded retries.

## Task 1: Diagnose and preserve evidence under the prompt budget

**Files:** `src/las_repro/cv/summary.py`, `tests/test_cv_summary.py`.

The current reducer sets relation and overlay caps to zero before reducing redundant observations. The retained full_0001 input has 15 tracks, zero relations and zero overlays; its prompt is 206,973 characters. Capped relations also favor early frames, potentially excluding later visibility transitions.

- [ ] Run current two-stage prompts against the authenticated original bundle and retain all responses.
- [ ] Add a regression with dense tracks and a visibility transition: under a bounded prompt, retain transition observations, overlap relations and an eligible overlay.
- [ ] Observe the regression fail with the current reducer.
- [ ] Allocate budget across evidence kinds and prioritize transition-supporting relations before redundant early-frame pairs. Preserve deterministic identity, exact timestamps and truthful truncation flags.
- [ ] Run `pytest tests/test_cv_summary.py tests/test_occlusion_semantics.py tests/test_scene_semantics.py -q`.
- [ ] Replay all five retained artifacts offline, recording before/after evidence coverage without altering originals.

## Task 2: Resolve measured semantic and runtime failures

**Files:** implicated pipeline/model/CV modules and their corresponding tests; fresh experiment records.

- [ ] Use Task 1 measurements and fresh response failures to choose the next single change; amend this plan with its concrete regression before implementation.
- [ ] Write and observe a failing behavior test for each selected fix.
- [ ] Implement and run focused tests; preserve model/input identities and output validation.
- [ ] Run a bounded diagnostic with the repaired evidence; inspect positive claims against video and report remaining uncertainty.
- [ ] Only after useful semantic output, run the five-video cohort and cache resubmission with explicit limits, preserving control and historical results.

## Task 3: Verify and deliver

**Files:** new `docs/reports/2026-09-06-las-capability-completion.md`, fresh `evaluation/results/` records, current handoff pointer.

- [ ] Run the complete Python suite with coverage, viewer tests, repository policy and `git diff --check`.
- [ ] Independently review product changes and fix findings.
- [ ] Record each gate separately; do not claim LAS parity from unit tests or five diagnostic samples.
- [ ] Preserve reviewable changes and update the handoff with evidence and any concrete remaining blocker.

## Measured Task 2 refinements (2026-09-06)

1. ARK requested image resolution is ignored. Add `image_pixel_limit` to each input image for explicit low/medium/high using (min4096,max65536/131072/262144), omit when unspecified. Keep exact frames and timestamps. Transport tests must verify each mapping and unchanged bytes before implementation. Same22-frame live probe: medium1859 input tokens versus unrestricted28919; this is not a semantic-quality comparison.
2. Joint evidence budgeting: after Task1, raw/packed candidate counts are full0004 26/0, full0021 14/0, full0001 22/5. `_summary_fits` reserves only an empty candidate envelope. Add optional validated thresholds to `summarize_cv_evidence`; when supplied, budget the actual bounded canonical candidate set alongside its summary. Production pipeline passes request thresholds. Keep the existing no-threshold API path and all serialization/validation hard bounds. Regression: dense observations plus late real visibility gaps must yield a nonempty complete candidate bundle, preserving exact PTS and truthful truncation. Do not start a new live semantic tranche until these offline checks pass.

3. Semantic replay: bind ARK semantic outputs to actual video bytes, exact prompt/context/model/effective request identity; bounded SQLite cache, current validation on hits, lease-fenced atomic publication and explicit cache provenance. Use sequential cross-task replay tests, preserving real CV keys. Detailed task brief kept in local SDD workspace.
4. Runtime concurrency is conditional on fresh cold timings after the exact mask optimization. If serial execution still fails 720 seconds, evaluate CV and Pass B after A, then Scene and Occlusion after enrichment; preserve Qwen session affinity and join children on failure. Any such change needs barrier-backed dependency/recovery tests. No concurrency change has been implemented.

5. Measured generation failure: add bounded pre-model spatial provenance choices with exact source fields; keep claim generation and strict response validation. Strengthen generic visibility/fragmentation and full-video entity inventory prompts. Separate same-track gap/uncertainty dedup before candidate identities if needed.
6. Measured mask row optimization: exact bytes.count/compress arithmetic, preserving all artifact bytes and centroid calculations; local and remote microbenchmarks show opportunity. No frame/model/threshold change; measure real cold runtime afterward.

7. Observed phase-contract defect: gap candidates currently offer only last-visible to first-revisible bounds for every event type, rejecting true enter/hidden/exit phases. Replace coarse endpoint pools with bounded exact typed interval options. Gap options derive only from trusted lifecycle PTS; uncertainty windows retain explicit visual-adjudication semantics. Validate full triples, preserve optional abstention and exact provenance, bump bundle to v2 plus prompt/validator contracts, rebuild evaluation candidates and preserve old artifacts. Keep source dedup separate. This fixes representability; partial-cover boundary precision still needs visual evidence.
8. Live scene generation now exposes a 4,096-output-token truncation: exact model response reports incomplete/length, input113344/output4096. Set ARK scene-only cap8192, preserve other stages and Qwen caps, bind actual setting in semantic cache, bump adapter contract. Add generic nonredundant spatial output guidance; do not drop required objects/events or accept truncated JSON. Verify actual transport payload and rejection behavior before another bounded scene run.
9. Mask optimization review regression: bytes subclasses can override count and bypass0/1 validation. Fix by using an exact plain raw-byte buffer, add failing subclass regression and exact metric equivalence assertions, retain archive byte checks. Review must approve before moving to the next implementation.
10. Authenticated identity audit found predictor-ID handoffs with 98.6–99.2% mask IoU and cross-label duplicate-region masks over 99.3%. Add bounded retained-geometry continuation/label-conflict cues to candidate v2 alongside typed times, preserving tracks and every classification. Runtime cues must not claim mask comparisons or physical identity; authenticate their exact scalar sources, cap search/rows/bytes, and account for them in the joint prompt budget. Explain border contact versus actual exit. A controlled semantic test must determine usefulness; if unresolved, a separately validated mask/appearance association layer is required. Dedup alone cannot solve this defect.
11. Scene copying still failed with complete JSON: two invalid relation literals, then wrong CV timestamps/segment joins/order; no raw spatial row exactly selected an offered option. Introduce an explicit SceneSemanticsChoices model-facing DTO and deterministic authenticated option projection through unchanged public validation. Persist/cache DTOs; preserve accepted nonspatial fields; support relation orientation; reject invalid selections without guessing. Add scene-only server-owned ARK flattened structured-output schema, exact offered ID enums and request/cache schema identity, with bounded schema compilation. Tiny provider probe accepted flattened schema, but full DTO support and semantic accuracy still require finite real validation. This is a declared representation change, not repair of historical output.
12. Real-artifact v2 audit found all identity cues lost at default budget (full_0002: 18 continuation + 29 label cues before packing, zero after). Reopen v2 budget fix: size threshold-aware summaries against actual bounded identity content, and allow fair whole-row optional subsets under explicit tight caps with exact source authentication and conservative completeness. Do not accept redundant summary density only because every new identity section was discarded.

13. A fresh valid occlusion diagnostic has duplicate first-ball claims. Source audit identifies consecutive stored observation ordinals crossing authenticated missing frames 45–78. Add a general pre-model continuity guard for uncertain windows, preserving all typed gap phases, observed partial-support triples within a visible run, and sparse processed sampling. Do not clip one-sided boundary evidence into invented options or deduplicate outputs after generation. Implement only after Task 2g review, with explicit validator/prompt versions and bounded regression evidence.

14. First fresh control Poll response failed public validation because generic sensitive-key redaction masks the validated `semantic_cache_key` digest. Extend only the existing schema-owned restoration after whole-result validation; retain general secret redaction and reject malformed metrics. Add actual API Poll regressions for cache metrics, mutation safety and invalid/legacy fallback before resuming cohort.

15. Fresh no-CV scene twice returned a generic enum failure; one bounded observed call located unsupported `hold` in semantic event type. Add closed field-specific scene choice enum repair codes and truthful vocabulary guidance (retain observed description, model may explicitly select unknown when no finite type fits). Keep taxonomy, strict public schema and raw output rejection; no automatic enum normalization. Bump failed-envelope validator and scene prompt contracts, test private-value exclusion and cache/repair integration.

16. Fresh pinned five-video hybrid full_0021 failed both BoundaryPlan enum attempts. The worker currently has no response-contract factory for that schema. Add authenticated, bounded server-owned BoundaryPlan structured output with exact existing coarse/fine vocabularies, closed field-specific repair feedback and prompt/validator versioning. Preserve public taxonomy, original outputs, one-repair policy and scene contract; do not normalize unknown enum values or infer the unavailable raw error words.

17. Fresh full_0001/full_0004 scenes still fail finite enums despite structured output. Offline authentication finds 106–178 KB of dense CV text in each input; a deterministic 20–45 KB scene data view preserves exact offers, targets, compact segments and lifecycle records. Introduce a 48 KiB authenticated compact data view (64 KiB complete text ceiling), retaining full private source context and unchanged public projection. Keep source-derived offers identical for this comparison; explicit full-renderer fallback for oversized sources, no silent record truncation. Update prompt/replay identity and test exact source/direction/temporal binding. Input size is a measured change, not an established cause of the failures.

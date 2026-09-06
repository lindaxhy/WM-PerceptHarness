# Bounded Semantic Re-verification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for the local operator task. Controller owns all remote execution and evidence publication.

**Goal:** Run the two repaired semantic stages once with at most one closed-code
repair each and deliver a hash-bound, truthful experiment report.

**Architecture:** A small standalone experiment operator loads a private input
bundle, rebuilds production prompts with unchanged data, reserves paid calls,
and records private raw/public sanitized artifacts. It changes no product code.

**Tech Stack:** Existing Python/Pydantic/ARK adapter, pytest, SSH, wheel tooling.

## Global Constraints

- Preserve frozen reference, mapping, results, Qwen projections and thresholds.
- No new five-video cohort, SAM analysis, model change, performance tuning or PR.
- Model identity is `doubao-seed-2-1-pro-260628`; preserve original inference settings.
- Exactly two stage/sample pairs; at most two generate calls per pair, four total.
- Only new closed validation codes permit the second call; no transport retry.
- No credentials, raw output, private paths or service IDs in public reports/stdout.
- Original database and caches remain read-only; never overwrite old experiments.
- Preserve user-owned untracked performance note and existing worktree.
- User requests default routine execution; no repeated approval prompts.

### Task 1: Implement and test the bounded standalone operator

**Files:** Create `scripts/reverify_semantic_stages.py` and
`tests/test_semantic_reverification.py` only. Root owns plans, reports, bundles,
build/release directories and all remote actions.

**Interfaces:** Use `PromptRenderer.render(stage, variables)`,
`_model_request(InferenceJob(...))`, `DEFAULT_OUTPUT_SCHEMAS.sanitize`,
`_validated_stage_result`, and `ArkVideoModel`. No production changes.

- [ ] Add tests before implementation for exact old-template/data round trip,
  current-template rendering, code-only repair and retained scene suffix.
  Public helper `rebuild_prompt(original_prompt, original_template, stage,
  repair=None)` returns current prompt plus stable data digest. Extract template
  markers by escaped literal matching with JSON values; require exact round trip
  and no ambiguous/unconsumed markers. Scene permits only the existing appended
  `[CV_EVIDENCE_SUMMARY_JSON]` JSON section, validated against schema-context
  summary by caller. Occlusion permits no suffix. Never substring-edit model text.
- [ ] Add a directly testable `run_stage` boundary with injected model for tests.
  Sanitize raw output using the real registry then pass through real stage
  validation. Persist raw JSON privately before sanitization. Initial valid
  returns without retry; invalid gets at most one code-only repair; final invalid
  remains failed. Transport/generation failure stops without retry and records
  only a closed error label. Record elapsed time, usage, request/response hashes,
  closed issue codes, semantic event/decision/classification counts.
- [ ] Add CLI `--bundle`, `--bundle-sha256`, `--wheel`, `--wheel-sha256`,
  `--source-commit`, `--output-dir`, `--api-key-file`, `--execute`.
  Bundle schema is `semantic_reverification_input_v1`, with `model_identity`,
  `settings` (only ARK timeout/max_frames/max_request_bytes/max_output_chars),
  and exactly two `stages`. Each stage has `sample_id`, `stage`, `model_name`,
  `payload`, `original_template`, `source_video_sha256`,
  `original_job_payload_sha256`. Pair order is scene/full_0001 then
  occlusion/full_0002. Payload shape is the original DB job's; it includes
  rendered prompt and schema_context. Scene context includes evidence_summary;
  occlusion context contains candidates and duration. Preserve all non-prompt
  payload fields. Controller authenticates DB, artifacts and bundle provenance.
- [ ] CLI must verify bundle hash, exact pairs/model, all video hashes, old prompt
  round trips, wheel hash and all package files against imported `las_repro`
  before constructing the network model. Validate schema names/prompt repair
  initial null and candidate data alignment. Dry run never opens credentials
  or constructs model. Default has no paid calls. `--execute` requires private
  fresh output with per-stage exclusive reservation before call, fsynced to disk;
  restart refuses existing reservation (never repeats unknown completed work).
  Raw prompt/payload/response files mode0600, output directory0700. Do not follow
  preexisting output symlinks or overwrite any path. Keep errors/stdout sanitized.
- [ ] Tests cover one successful call, one repair, final-invalid two calls,
  transport failure one call, repeat reservation zero calls, hash mismatch zero
  calls, altered inputs rejected, private permissions, no raw text in reports,
  current positive example and unchanged trusted JSON. Use existing real scene
  and occlusion fixtures where practical. Run focused test file RED then GREEN,
  full suite once, Node model/syntax and diff checks; commit only these two files.
- [ ] Write detailed RED/GREEN report to the plan's SDD `task-1-report.md`;
  return status, commit, tests and concerns. Independent task review precedes
  every remote execution. Do not access remote host or credentials.

## Controller experiment steps

- [ ] Verify linked branch, existing runtime processes, Python/dependency/model
  pins and read-only database state. Current baseline check:59focusedtests pass;
  remote host reachable, all GPUs1MiB and no compute processes.
- [ ] Prepare fresh private input bundle from the exact original DB jobs and
  old release templates. Authenticate CV artifacts/video/context, pin bundle
  hash, build/extract current wheel without modifying historical venv installs.
- [ ] Review operator, perform complete dry-run preflight, then execute once
  under durable observation. Poll the actual handle until terminal; do not
  relaunch on observation timeout. Preserve partial/reserved calls on error.
- [ ] Publish sanitized immutable reports under a new re-verification result
  directory and `docs/reports/2026-09-06-semantic-reverification.md`; include
  pass/fail rationale for starting a new five-video experiment, not execution.
- [ ] Independently review new operator/evidence only (prior product review
  remains closed), verify final hashes, unchanged frozen data and cleanup state.
  Mark this bounded goal complete only when all evidence and report requirements
  hold, regardless of stage outcome; overall integration acceptance stays separate.

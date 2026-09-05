# Poll hybrid provenance contract repair

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this bounded task.

**Goal:** Restore the already-approved public Poll compatibility contract:
valid canonical hybrid data must preserve evidence IDs, content-addressed
artifact identity and numeric model usage while credentials remain redacted.

**Architecture:** Keep generic request/legacy redaction unchanged. At the Poll
result boundary, validate canonical hybrid structure and restore only explicitly
enumerated, schema-owned harmless fields in the already-redacted copy. Do not
bypass redaction for an entire result or globally exempt key/token names.

**Tech Stack:** Existing Python, FastAPI, pytest and canonical hybrid validator.

## Global Constraints

- Existing approved SAM/Doubao architecture, model, prompts, thresholds, mapping,
  source evidence, one-repair limit and public Submit/Poll envelope remain unchanged.
- Work on feat/sam31-evidence-integration; no remote deployment or credentials.
- Never weaken generic sensitive-name redaction or return unvalidated fields
  merely because their names resemble provenance.
- Keep API CPU-only; no SAM/Torch/model loading.
- Real stored output already passed canonical validation. Do not rerun paid
  inference to repair its transport representation.

## Verified failure

The first real Doubao-only five-demo task completed. Read-only validation of
its persisted result passed, but authenticated Poll replaced
source_keyframe_ids lists with strings. Source inspection confirms generic
redaction matches any key/token substring; numeric input_tokens/output_tokens
are also affected. Available CV artifact_key will have the same collision.
The original Poll envelope and persisted result are retained privately.

### Task 1: Preserve validated, schema-owned Poll metadata

**Files:**
- Modify: src/las_repro/api.py
- Test: tests/test_api.py
- Test: tests/test_security.py (only if needed for generic-redaction regression)

**Interfaces:** Poll still returns the same PollResponse and unchanged legacy
behavior. Add a private result helper in api.py; redact remains unchanged.
Existing validate_hybrid_result validates structure and internal references
without GPU or external artifact loading.

- [ ] **Step 1: RED — actual API round-trip tests.**

Construct real canonical disabled-CV and available-CV results using existing
test helpers, complete real SQLite tasks, call the actual authenticated Poll
route and validate response.data with validate_hybrid_result. Include nonempty
keyframe IDs in action, occlusion and scene event/spatial rows, available
artifact identity, and numeric input/output usage in performance stages.
Assert exact harmless-field equality, immutable stored input, working
iter_action_captions, and unchanged envelope. Each listed path must be covered;
do not mock redaction, validator, route or store.

Add adversarial cases with unrelated credential fields, canonical-shaped but
invalid provenance/usage types, and the same field names at unapproved paths.
They must remain redacted; do not construct live-looking token literals.
Run .venv/bin/python -m pytest tests/test_api.py tests/test_security.py -q and
record the real pre-fix failure.

- [ ] **Step 2: Implement the narrow boundary helper.**

Algorithm:

```python
public = redact(original)
if not isinstance(original, dict) or "annotation_branches" not in original:
    return public
try:
    validate_hybrid_result(original)
except (ValueError, TypeError, KeyError, AttributeError):
    return public  # retain existing fail-closed redaction; restore nothing
# Copy only schema-owned paths below, never arbitrary subtrees.
return public
```

User-approved review amendment: include `AttributeError` because malformed
canonical containers (for example `cv_evidence=None`) can raise it in the
existing validator. Preserve the same fail-closed redacted result and restore
nothing on this path. Add an actual authenticated Poll regression; do not
relax the canonical validator or catch unrelated runtime failures.

Only restore:

- source_keyframe_ids on annotation_branches.action_events rows;
- source_keyframe_ids on annotation_branches.occlusion.events rows;
- source_keyframe_ids on annotation_branches.scene_facts.events rows;
- source_keyframe_ids on top-level locations/relations and their
  annotation_branches.scene_facts mirrors;
- cv_evidence.artifact_key when status is available;
- numeric input_tokens/output_tokens inside performance.stages[*].provider_metrics.

All restored lists must be copied, not shared with persisted data. Missing
optional usage fields stay absent. No restoration occurs before the complete
canonical validator succeeds. Keep other fields in the generic redacted copy.
Wire only completed Poll results through this helper. No new public API or
generic redactor policy.

- [ ] **Step 3: GREEN and covering integration tests.**

Run .venv/bin/python -m pytest tests/test_api.py tests/test_security.py
tests/test_export.py tests/test_hybrid_result.py tests/test_repository_policy.py -q.
Stage newly added files before the tracked-file policy gate. Run the full
branch-enabled suite once, using --cov=src/las_repro --cov-branch
--cov-report=term-missing; require the existing 85% gate. Run scoped Ruff and
git diff --check, without unrelated formatting/refactoring.

- [ ] **Step 4: Commit, self-review and independent task review.**

Commit only scoped files with message "fix: preserve validated hybrid Poll provenance".
Report RED/GREEN, commands/output and limitations to the task report. Root
will build/deploy the reviewed patch and re-Poll the same completed task,
compare its result with persisted canonical data, then resume the saved
five-demo driver. Original bad transport evidence must remain preserved.

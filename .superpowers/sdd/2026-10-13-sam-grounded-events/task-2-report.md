# Task 2 event scoring report

## Delivered

- Replaced `percept score fidelity` with `percept score events --reference DIR --system NAME=DIR --out DIR`, repeatable systems, configurable finite `(0,1]` tIoU threshold (default `0.75`), and OpenAI-compatible judge configuration via `PERCEPT_JUDGE_BASE_URL`, `PERCEPT_JUDGE_MODEL`, `PERCEPT_JUDGE_API_KEY` or CLI overrides.
- Reused the strict `GroundedEvents.model_validate(data)` annotation contract. Scoring uses original seconds with no duration normalization. Temporal assignment maximizes eligible pair count then summed IoU, independently from the joint assignment of edges passing the semantic decision.
- Judges every temporally eligible pair once. Requires a JSON object with exact `match: bool` and nonblank `reason`; duplicate keys, malformed responses, HTTP failures, and transport failures invalidate sample scoring. The report retains candidate decisions, descriptions, IoUs, temporal and joint matches, and unmatched indices. It does not record the API key, URL credentials, query, or path.
- Loads flat run_batch outputs and nested job directories. Identity uses explicit sample/job ID, nested job directory, then flat source video stem. Duplicate IDs fail. Ignores `.percept`, `.cv-archive`, `results.jsonl` mirrors, and root manifest `shard-NN.status.json`; malformed primary JSON is reported as an error.
- Pairs the union of IDs. Missing, failed, invalid, and judge-failed samples make the system aggregate null and CLI exit nonzero. Valid empty lists have null undefined metrics; one-sided empty F1 is zero. Micro uses counts; macro reports defined-score contributor counts. Completely empty datasets fail.
- Retained the legacy `event_matching.Event`/`match_events` API over one shared Hungarian implementation. Removed retired fidelity scorer, CLI, and embedding extra; refreshed `uv.lock` offline.

## Test-driven evidence

- RED: new scorer import failed before implementation; GREEN: initial three scoring tests passed.
- RED→GREEN: CLI module import; duplicate verdict keys and blank explicit IDs; manifest shard status indexing; URL path secret disclosure; malformed-primary error counts.
- Final: `PYTHONPATH=src /Users/lindaxu/Documents/Physis/cvpr-27/.venv/bin/python -m unittest discover -s tests` — 32 tests, OK.
- `UV_CACHE_DIR=/private/tmp/percept-uv-cache uv lock --check --offline` — resolved 137 packages.
- `git diff --check` — clean. `python -m percept_harness.cli score events --help` — CLI options render.

## Scope and limits

No hosted judge call or live GPU SAM annotation was run. HTTP behavior was tested offline with `httpx.MockTransport`, and scoring is independent of the GPU annotation runtime. Existing documentation still names fidelity; the integration task owns documentation updates. A malformed dataset file stops scoring before per-sample indexing; the CLI writes a fatal error with null aggregate and one invalid dataset error per system, rather than claiming a partial score.

# Bounded real semantic re-verification

Date: 2026-09-06. User-approved active goal; routine implementation is authorized.
Baseline: `76b8848`. Completed product fixes/reviews will not be repeated.

## Experiment

Re-evaluate exactly the original cold ordinal-0 scene job for `full_0001` and
occlusion job for `full_0002`, using the repaired production prompt assets and
validators. Preserve video, sampling, model ID `doubao-seed-2-1-pro-260628`,
schema context and all original prompt data. Start each stage once and permit
only the pipeline's existing one repair using newly observed closed codes.
At most four model requests total; transport/auth failures are not retried.
No new full-cohort run, Pass A/B, SAM analysis, model change or performance work.

Replaying old prompt bytes alone would miss the new occlusion instructions;
re-running the whole pipeline would add unrelated nondeterministic inputs and
cost. Instead reconstruct the exact original template variables, prove old
template round-trip identity, and render those same values with current packaged
templates. Preserve any authenticated scene-summary suffix verbatim. On repair,
change only validation repair data, never rejected output or candidate bounds.

## Isolation and evidence

Run a hash-verified current wheel extracted into a new private directory with
the existing dedicated semantic interpreter/dependencies. Assert every loaded
package file equals that wheel and record Python/dependency identities. This
avoids changing the installed historical environments. Verify original source,
checkpoint and BPE pins without new SAM inference. Observe owned processes,
database terminality, input hashes and GPU state before and after.

Read original SQLite and caches only. Prepare one hash-bound private input
bundle, raw requests/responses and durable call reservations in a new owner-only
directory. Do not expose credentials, raw prompts, private paths, service IDs or
model text in committed evidence. Public records include hashes, closed issues,
counts, actual request attempts, usage and timings. Reserve each stage before
its first paid call so process restart cannot repeat it.

## Completion

Deliver independently checked public JSON and a narrative report that states
whether each stage passes structural and contextual validation, whether repair
was used, and whether a new five-video experiment is justified. A failed finite
experiment completes this bounded goal when its evidence/report is verified;
it does not pass overall integration acceptance. Never overwrite historical
results or fabricate human precision, latency acceptance or cache-hit success.

Self-review: no changed acceptance gates, arbitrary retries or unavailable
human judgments are prerequisites for this bounded report. No visual design
decision is needed. Default routine execution supersedes repeated approval
prompts, not external authorization boundaries.

# Bounded real semantic re-verification

Date: 2026-09-06. Product source: `76b8848`; operator: `edbee71`.

## Outcome and decision

The bounded experiment finished, but **a new five-video experiment is not yet
recommended**. Scene validation passed after its one permitted repair, with
no location or relation records. Occlusion validation still failed after its
one permitted repair. Overall integration acceptance remains failed; the
[historical acceptance report](2026-09-04-sam31-gpu-acceptance.md) is unchanged.

Exactly four real `generate` calls returned responses, with no transport retry,
new SAM analysis, full-pipeline resubmission, model change, performance tuning,
deployment, push or PR. The supervisor ran from 02:31:30.937 to 02:35:27.497 UTC,
exited normally, and recorded 236.559 seconds. A normal operator exit means the
finite experiment was recorded, not that every stage passed.

| Sample / stage | Initial call | One code-only repair | Final result |
| --- | --- | --- | --- |
| `full_0001` / scene | 83.320s; `SCENE_SPATIAL_INVALID` | 51.904s; valid | 6 objects, 8 semantic events, **0 locations / 0 relations** |
| `full_0002` / occlusion | 57.982s; `OCCLUSION_EVENTS_EMPTY` | 41.785s; `OCCLUSION_EVIDENCE_PROHIBITED_CONTENT` | Invalid; no accepted positive occlusion event |

The [execution record](../../evaluation/results/sam31_2026-09-04/reverification-2026-09-06/execution-report.json)
contains every call's timing, closed codes, usage, and request/response hashes.
Reported usage totals 613,199 input and 7,829 output tokens. Summed generation
time is 234.991 seconds, including adapter preparation; this is not an
end-to-end video latency measurement or a monetary billing statement.

## What the results do and do not show

The scene repair consumed the specific spatial error code rather than the
former generic schema code. Passing with empty spatial lists demonstrates
contract compliance, not recovery of useful spatial evidence or correctness
of its eight semantic events.

Both raw occlusion responses pass the Pydantic shape schema but fail contextual
validation. The initial response contains 15 decisions: 2 occlusion, 4
out-of-frame and 9 unknown, with only one event. Its empty-events violation is
retained. After repair, the response contains 1 occlusion, 12 out-of-frame and
2 unknown, again with one event. These are **rejected raw model claims**, not
accepted detections or measured precision.

An [offline diagnostic](../../evaluation/results/sam31_2026-09-04/reverification-2026-09-06/offline-diagnostics.json)
located the final prohibited-content match at decision index 12's
`visual_evidence`: one slash between ASCII words, with no raw-mask filename or
control-character match. The existing guard forbids slashes as well as
serialized structures. This identifies a punctuation/guard boundary to
investigate; it does not establish that the response is safe or visually
correct. No rejected text was rewritten, no validator was loosened, and no
third call was made for either stage.

These are individual nondeterministic replays. They do not establish causal
quality improvement, human precision, full-pipeline provenance acceptance,
cache-resubmission success, the 720-second latency gate, or browser usability.

## Input and runtime controls

The experiment used the original cold ordinal-0 jobs and the unchanged model
identity `doubao-seed-2-1-pro-260628`. Original video hashes, payload hashes,
schema contexts, candidate ordering, sampling and other non-prompt payload
fields were retained. The two original prompts were exactly reconstructed;
the same parsed values were rendered with the current packaged templates.
Scene retained its authenticated evidence-summary suffix. Only closed issue
codes changed in each repair prompt.

The [environment record](../../evaluation/results/sam31_2026-09-04/reverification-2026-09-06/environment.json)
binds source archive, wheel, operator, inputs, model assets, Python and
dependencies. A wheel built from the immutable source archive was extracted
into a new private runtime; every imported package file was compared to it.
Neither historical venv was reinstalled. Both retained all 46 historical
package files before and after the experiment. SAM source revision,
checkpoint bytes/size and BPE Git-blob identity were verified without loading
SAM for inference.

The historical wheel's defaults were read using its verified installation,
an isolated interpreter with bytecode writes disabled, and no ambient `LAS_*`
settings. There were no historical overrides for the four adapter settings:
180s timeout, 128 frames, 33,554,432 request bytes, and 1,000,000 output characters.
They match the input bundle exactly.

The first preparation used a cache reader that can create locks or quarantine
bad entries. Review caught that write-capable interface before paid execution.
An audit found both locks/entry metadata dated before this experiment and no
quarantine; it found no persistent mutation from the successful reads. The
replacement preparation validated private copies and proved the original
cache's content and metadata unchanged, excluding read access times. It also
reconstructed the exact same immutable input bundle. Original preparation
artifacts were retained, not overwritten.

The [launch gate](../../evaluation/results/sam31_2026-09-04/reverification-2026-09-06/launch-gate.json)
records the complete remote dry-run and idle state. The launcher checked the
reviewed operator hash before either spawning mode. Exclusive, fsynced
supervisor/stage reservations prevented repeated execution; the same process
identity was followed to terminal state.

## Evidence and verification

The [manifest](../../evaluation/results/sam31_2026-09-04/reverification-2026-09-06/manifest.json)
hashes the six public records and records controller-helper identities. Raw
prompts, payloads, parsed model responses, reservations and validation outputs
remain private. In the execution record, `response_sha256` hashes sorted
canonical parsed JSON without a newline; it is not an HTTP-envelope hash.
`request_sha256` hashes the stage/model-alias/payload projection. The
[postflight record](../../evaluation/results/sam31_2026-09-04/reverification-2026-09-06/postflight.json)
separately lists exact private file-byte hashes, including their newlines.

Postflight rehashed all four requests/responses, reproduced all four production
validation outcomes, checked code-only repairs and unchanged non-prompt data,
and verified owner-only evidence permissions. SQLite integrity, its byte hash,
and terminal counts stayed unchanged: 5 completed / 1 failed tasks and 41
completed jobs. The [cleanup record](../../evaluation/results/sam31_2026-09-04/reverification-2026-09-06/cleanup.json)
confirms matching original-cache snapshots, no quarantine, no live experiment
supervisor or owned services, and unchanged historical package files. All four
GPUs were at 1 MiB with no compute processes in postflight.

New operator task review and two scoped helper-fix reviews closed all
pre-execution findings. The controller independently verified the final code:

```text
.venv/bin/python -m pytest -q --cov=las_repro --cov-report=term-missing
1721 passed, 2 existing dependency warnings in 76.30s; coverage 85.78%

node --test evaluation/viewer/tests/model.test.mjs
20 passed; JavaScript syntax and git diff checks clean
```

Seven private helper tests and two auditor tests also passed, including real
isolated-interpreter bytecode protection and rejection of altered raw evidence.
The existing Starlette/httpx and AnyIO deprecation warnings remain; no
dependency upgrade was made. Frozen references, historical results, mapping,
Qwen projections and thresholds are unchanged. The user-owned performance
follow-up note remains untracked and untouched.

## Next step

Do not repeat paid calls or start a new five-video cohort on this result.
First use the retained responses for offline regressions of the positive
occlusion/nonempty-event rule and the plain-text-versus-path guard boundary;
also investigate whether spatial repair can preserve supported facts instead
of returning empty lists. Any change must retain unsafe-content protection and
the original acceptance gates. Only a reviewed, explicit repair hypothesis
should justify another separately bounded real experiment. No such new fix or
experiment was implemented in this phase.

# SAM3.1 Runtime Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for this bounded fix and independent review.

**Goal:** Make the approved pinned SAM adapter execute on the actual RTX 5090 host without dropping nominal 30 fps frames through timestamp rounding.

**Architecture:** Keep the pinned Meta source and checkpoint unchanged. Select the supported PyTorch attention implementation in the adapter and handle FFprobe timestamp precision when selecting existing observed frame indices.

**Tech Stack:** Existing Python, Fraction-based sampler, PyTorch 2.10/CUDA12.8, pytest.

## Global Constraints

- Work on `feat/sam31-evidence-integration`; preserve unrelated changes and fixed upstream SAM source.
- Pinned upstream revision remains `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`.
- Checkpoint remains ModelScope `facebook/sam3.1` revision `616acbee0b9ed4177f1f389e3c13594a0a1f6398`, SHA256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`.
- Never invent or rewrite observed PTS. Select only indices from the authoritative timeline.
- Preserve short-source-rate/30fps cap and long-video 8fps plus 30fps refinement policy.
- Do not silently change sampling thresholds during OOM recovery.
- Use apply_patch and `.venv/bin/python -m pytest`; no upstream checkout edits or new attention dependency.

### Task 1: Fix host attention, propagation bounds, and timestamp precision compatibility

**Files:**
- Modify: `src/las_repro/cv/sam31.py`, `tests/test_sam31_adapter.py`
- Modify: `src/las_repro/cv/timeline.py`, `tests/test_cv_timeline.py`
- Modify: `docs/reports/2026-09-05-sam31-runtime-preflight.md`

**Evidence:**

The production adapter loads the pinned checkpoint successfully, but actual
`add_prompt` fails with `ModuleNotFoundError: flash_attn_interface`. The pinned
`build_sam3_multiplex_video_predictor` defaults `use_fa3=True`; its documented
`use_fa3=False` selects the existing supported attention implementation.
The controller is testing this single-parameter change in a private probe.

With Torch attention enabled, propagation reaches the last frame and fails with
an empty feature tensor `[5184, 0, 256]`. The adapter passes `frame_count - 1`
as `max_frame_num_to_track`. At the pinned revision, the tracker clamps an
inclusive processing end to `num_frames - 1`, while the batched detector uses
the passed count as an exclusive chunk end. Passing the actual frame count
aligns both paths for this full-video stream without changing source sampling.
The single additional diagnostic override succeeded: 92 observations from
one track, two prompts, artifact publication and verified reload, 46.010 seconds
total, peak allocation 15,700,532,224 bytes. The precision correction and a
repeat without any diagnostic overrides are still required.

The real `full_0024` source has 137 frames at nominal 30 fps. FFprobe outputs
`0.0, 0.033333, 0.066667, 0.1, ...`, and the final observed timestamp makes the
inferred rate `30.000002205882517`. The current sampler erroneously enters the
rate-cap path and its exact grid comparisons yield only 92 unique indices.
This reproduces locally without any GPU. Preserve all 137 source indices in
this case, leaving their observed timestamps unchanged.

**Interfaces:** Existing `Sam31EvidenceProvider.load`, `initial_sample_indices`,
`refinement_sample_indices`, and frame/materializer contracts stay compatible.

- [ ] **Step 1: Add failing attention-selection test.**

Use the existing injected builder test and assert that the production load call
passes `use_fa3=False` alongside its existing checkpoint/tokenizer/object-cap
arguments. Retain no-network, immutable source, cleanup, and device guarantees.

```python
assert captured_builder_kwargs["use_fa3"] is False
```

Add a faithful regression for the pinned inclusive-tracker/exclusive-detector
boundary: the stream request must pass the number of sampled frames, not the
last index. Cover a one-frame video, a non-chunk-aligned video, and an exact
chunk multiple. Assert exactly all valid frame indices are consumed once and
no out-of-range observation can be published. Keep one uninterrupted stream
per entity prompt; do not reset tracker state between execution chunks.

- [ ] **Step 2: Add failing precision tests.**

```python
timeline = FrameTimeline(frames=tuple(
    FrameTimestamp(frame_index=i, timestamp_seconds=float(f"{i / 30:.6f}"))
    for i in range(137)
))
assert initial_sample_indices(timeline, policy) == tuple(range(137))
```

Also test a genuinely higher-rate source is still capped, long-video scan and
refinement remain capped with decimal-rounded PTS, stable ordering/uniqueness,
and exact preservation of returned source timestamps. The tolerance must be
bounded by FFprobe's observed timestamp precision, not an arbitrary percent or
a broad FPS slack that changes the approved sampling policy.

- [ ] **Step 3: Record RED and implement the three bounded fixes.**

```bash
.venv/bin/python -m pytest tests/test_sam31_adapter.py tests/test_cv_timeline.py -q
```

Use a documented small timestamp-comparison tolerance in rate inference and
grid/refinement selection where required. It affects comparisons only; never
round, synthesize, or replace timeline timestamps. Retain strict rejection of
nonmonotonic/missing PTS and deterministic sampling from validated indices.

- [ ] **Step 4: Verify targeted and complete suites.**

```bash
.venv/bin/python -m pytest tests/test_sam31_adapter.py tests/test_cv_timeline.py tests/test_cv_summary.py -q
.venv/bin/python -m pytest -q
git diff --check
```

Controller reruns the real video with the corrected production adapter and
checks track observations, artifact publish/reload, original index/PTS mapping,
GPU3 isolation, and cleanup. Record a truthful failure if a later runtime issue
appears; do not treat a diagnostic override as production acceptance.

- [ ] **Step 5: Commit and report.**

```bash
git add src/las_repro/cv/sam31.py src/las_repro/cv/timeline.py tests/test_sam31_adapter.py tests/test_cv_timeline.py docs/reports/2026-09-05-sam31-runtime-preflight.md
git commit -m "fix: support SAM runtime attention and rounded frame times"
```

Report each root cause, RED/GREEN evidence, exact comparison policy, changed
files, and any unresolved runtime evidence. This remediation precedes original
SAM Task11 integration and does not mark Task14 or Task15 complete.

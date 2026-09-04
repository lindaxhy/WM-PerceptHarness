# SAM3.1-assisted LAS Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cached SAM3.1 object tracks to the existing Qwen3-VL embodied-video pipeline, produce separately validated action, occlusion, and scene outputs, and quantify alignment against the frozen English LAS references.

**Architecture:** Three single-GPU Qwen workers keep the existing structured semantic stages on GPUs 0-2, while a separate process and Python environment runs a provider-independent CV worker on GPU 3. Pass A supplies a normalized entity inventory; the CV worker publishes content-addressed evidence; enrichment, occlusion, and scene stages consume only bounded summaries and keyframes. All semantic branches degrade independently, preserve provenance, and merge through deterministic validators.

**Tech Stack:** Python 3.12, Pydantic 2, SQLite/WAL, FastAPI, FFmpeg/FFprobe, Qwen3-VL-8B-Instruct, Meta SAM3.1 Object Multiplex, PyTorch 2.10/CUDA 12.8 on the GPU host, pytest/pytest-cov, static JSON evaluation artifacts.

## Global Constraints

- Work only on `feat/sam31-evidence-integration`, whose merge base is current `origin/main` including merged PR #3. This remains a new PR: do not amend or reuse the former PR #3 head branch, and keep `origin/main..HEAD` limited to SAM-alignment work.
- Keep Qwen3-VL-8B-Instruct as the semantic model; add no training, LoRA, remote inference, Grounding DINO, DINO, DINOv3, or audio processing.
- Run Qwen workers only on physical GPUs 0-2 and the SAM3.1 worker only on physical GPU 3.
- Keep the SAM runtime in a separate Python environment and pin the official SAM repository to commit `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`.
- Production workers load code, tokenizer, and checkpoints only from configured local paths; task execution performs no download.
- The default entity prompt cap is 16. Videos up to 30 seconds use source FPS capped at 30; longer videos scan at up to 8 fps and refine one-second windows at up to 30 fps.
- Never store raw masks in SQLite, prompts, API task results, logs, or training JSONL. Store them only in digest-checked CV artifacts.
- A missing or low-confidence track is not proof of absence or occlusion. Unsupported claims resolve to `unknown` or no event.
- Retain one Qwen schema-repair attempt. SAM OOM receives one same-resolution retry with a smaller execution chunk, then degrades to Qwen-only.
- A cache hit must skip SAM inference. Cache identity covers schema, source video, checkpoint, normalized prompts, sampling, and evidence thresholds.
- Keep public Submit/Poll compatibility and the current fine-segment training export. New branch outputs are additive and exact deterministic projections must be validated.
- Every implementation step uses `.venv/bin/python -m pytest`; the pre-change baseline is 762 passing Python tests plus 8 passing viewer model tests.

---

## File Structure

New focused modules:

- `src/las_repro/cv/contracts.py` — provider-independent entity, timeline, track, artifact, status, and provenance schemas.
- `src/las_repro/cv/entities.py` — deterministic entity normalization, stable IDs, priority, deduplication, and cap warnings.
- `src/las_repro/cv/timeline.py` — exact FFprobe frame PTS extraction and two-level sampling plans.
- `src/las_repro/cv/artifacts.py` — content-addressed cache, containment/digest checks, staging, atomic publication, and quarantine.
- `src/las_repro/cv/base.py` — `CvEvidenceProvider` protocol, provider errors, and deterministic Fake provider.
- `src/las_repro/cv/worker.py` — leased CV job execution, cache lookup, OOM retry, metrics, and sanitized terminal results.
- `src/las_repro/cv/sam31.py` — lazy SAM3.1 Object Multiplex adapter; the only module that imports `sam3`, NumPy, or Torch for CV inference.
- `src/las_repro/cv/summary.py` — bounded track summaries, spatial relations, overlay selection, and visibility-change candidates.
- `src/las_repro/pipelines/occlusion.py` — occlusion decision schemas, prompt validation context, temporal checks, and positive-event projection.
- `src/las_repro/pipelines/hybrid_result.py` — canonical branch result and backward-compatible deterministic projections.
- `src/las_repro/prompts/occlusion_semantics.txt` — dedicated evidence-constrained occlusion adjudication instructions.
- `src/las_repro/evaluation/las_alignment.py` — LAS/reference parsing, deterministic one-to-one matching, F1, boundary, semantic, reliability, and review metrics.
- `src/las_repro/evaluation/viewer_projection.py` — static allowlisted hybrid projection consumed by the comparison viewer merged from PR #3.
- `scripts/evaluate_las_alignment.py` — reproducible Qwen-only versus hybrid report CLI.
- `scripts/sam31_smoke.py` — local-checkpoint, one-GPU SAM isolation and artifact smoke test.
- `docs/deployment/sam31-runtime.md` — separate-environment installation, launch, cache, and recovery instructions.

Existing modules modified only at their responsibility boundary:

- `src/las_repro/domain.py`, `store.py`, and `workers.py` for routed jobs and durable timing metadata.
- `src/las_repro/config.py`, `cli.py`, `.env.example`, and `README.md` for the 3+1 runtime.
- `src/las_repro/pipelines/validators.py`, `output_validation.py`, and `embodied.py` for entity-bearing Pass A and hybrid orchestration.
- `src/las_repro/pipelines/scene_semantics.py` and `semantic_events.py` for provenance-aware branch projections.
- `src/las_repro/models/fake.py` and `qwen3_vl.py` for new deterministic fixtures, occlusion stage routing, and request metrics.
- `src/las_repro/export.py` for additive-result validation without changing fine-segment JSONL rows.
- The merged comparison viewer, its exporter, and their Python/Node tests for a Qwen-only versus Qwen+SAM3.1 selector and evidence layers.

---

### Task 1: Per-job model routing and durable execution metrics

**Files:**
- Modify: `src/las_repro/domain.py:26-72`
- Modify: `src/las_repro/store.py:136-266,425-523,893-931,1416-1452`
- Modify: `src/las_repro/workers.py:109-255`
- Test: `tests/test_store.py`
- Test: `tests/test_workers.py`

**Interfaces:**
- Consumes: the existing `InferenceJobSpec`, `InferenceJob`, and SQLite lease state machine.
- Produces: `InferenceJobSpec.model_name: str | None`, `InferenceJob.started_at`, `InferenceJob.finished_at`, `InferenceJob.metrics`, and `SQLiteTaskStore.complete_inference_job(job_id: str, result: Mapping[str, Any], *, worker_id: str, attempt: int, now: float | None = None, metrics: Mapping[str, Any] | None = None) -> InferenceJob`.

- [ ] **Step 1: Write failing storage tests for explicit per-job routing**

```python
def test_job_spec_can_route_one_stage_to_a_different_local_model(store, task):
    [job] = store.create_inference_jobs(
        task.task_id,
        [InferenceJobSpec("cv_evidence", 0, {"request": {}}, model_name="sam3.1")],
        now=10.0,
    )
    assert job.model_name == "sam3.1"
    assert store.claim_inference_job(
        "qwen-0", model_name="qwen3-vl-8b-instruct", lease_seconds=30, now=11
    ) is None
    assert store.claim_inference_job(
        "sam-3", model_name="sam3.1", lease_seconds=30, now=11
    ).job_id == job.job_id
```

- [ ] **Step 2: Run the routing test and verify the constructor rejects the new argument**

Run: `.venv/bin/python -m pytest tests/test_store.py -k different_local_model -v`

Expected: FAIL because `InferenceJobSpec` has no `model_name` field.

- [ ] **Step 3: Add the optional alias and bind it immutably at job creation**

```python
@dataclass(frozen=True)
class InferenceJobSpec:
    stage: str
    ordinal: int
    payload: Mapping[str, Any]
    model_name: str | None = None
    affinity_worker_id: str | None = None
    affinity_fallback_at: float | None = None
    affinity_fallback_seconds: float | None = None
```

In `create_inference_jobs`, resolve `spec.model_name` with
`validate_model_alias` when present; otherwise retain `_task_model_alias`. Include
the resolved alias in `_same_job_definition` so a retry cannot change backend.

- [ ] **Step 4: Write failing tests for timing and bounded metrics persistence**

```python
def test_completed_job_retains_first_start_finish_and_metrics(store, task):
    [job] = store.create_inference_jobs(task.task_id, [InferenceJobSpec("x", 0, {})], now=10)
    running = store.claim_inference_job("gpu-0", lease_seconds=30, now=12)
    done = store.complete_inference_job(
        job.job_id,
        {"ok": True},
        worker_id="gpu-0",
        attempt=running.attempt,
        now=14,
        metrics={"inference_seconds": 1.75, "peak_allocated_bytes": 1024},
    )
    assert done.started_at == 12
    assert done.finished_at == 14
    assert done.metrics == {"inference_seconds": 1.75, "peak_allocated_bytes": 1024}
```

- [ ] **Step 5: Add a backward-compatible SQLite migration and validation**

Add `started_at REAL`, `finished_at REAL`, and `metrics TEXT` when absent. On the
first successful claim set `started_at = COALESCE(started_at, now)`; on terminal
completion set `finished_at=now` and validated canonical metrics. Permit only
these metric keys:

```python
_JOB_METRIC_KEYS = frozenset({
    "inference_seconds",
    "input_tokens",
    "output_tokens",
    "peak_allocated_bytes",
    "processed_frames",
    "entity_prompts",
    "track_count",
    "cache_hit",
    "oom_retry",
})
```

Require finite non-negative floats for `inference_seconds`; non-negative strict
integers for token, byte, frame, prompt, and track counts; and strict booleans
for `cache_hit` and `oom_retry`. Reject booleans where a number is expected,
strings, non-finite numbers, unknown keys, and canonical JSON over 4096 bytes.
Terminal failure sets `finished_at` and leaves `metrics` null. Migration tests
must open a legacy fixture without the columns and verify old jobs remain
readable.

- [ ] **Step 6: Measure Qwen worker execution without changing model output**

Wrap `model.generate(request)` with the injected monotonic clock. Merge the
elapsed value with a bounded optional `model.request_metrics()` mapping, then
pass it to `complete_inference_job`. Add Fake-model tests proving metrics never
enter the schema-validated model result.

- [ ] **Step 7: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_store.py tests/test_workers.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q`

Expected: at least 762 tests pass with no failures.

- [ ] **Step 8: Commit**

```bash
git add src/las_repro/domain.py src/las_repro/store.py src/las_repro/workers.py tests/test_store.py tests/test_workers.py
git commit -m "feat: route inference stages and record job metrics"
```

---

### Task 2: CV evidence contracts and deterministic entity normalization

**Files:**
- Create: `src/las_repro/cv/__init__.py`
- Create: `src/las_repro/cv/contracts.py`
- Create: `src/las_repro/cv/entities.py`
- Create: `tests/test_cv_contracts.py`
- Create: `tests/test_cv_entities.py`

**Interfaces:**
- Consumes: JSON-compatible task/job payloads and Pydantic 2.
- Produces: `EntityRole`, `EntityCandidate`, `EntityPrompt`, `FrameTimestamp`, `FrameTimeline`, `SamplingPolicy`, `CvEvidenceRequest`, `TrackObservation`, `CvTrack`, `ArtifactFile`, `CvEvidenceArtifact`, `EvidenceStatus`, and `normalize_entities(candidates, *, limit=16) -> NormalizedEntities`.

- [ ] **Step 1: Write strict-schema failure tests**

```python
def test_track_observation_rejects_invalid_geometry_and_time():
    with pytest.raises(ValidationError):
        TrackObservation(
            frame_index=3,
            timestamp_seconds=float("nan"),
            bbox_xyxy=(0.8, 0.1, 0.2, 0.9),
            mask_ref="masks/3.npz",
            visible=True,
            confidence=1.1,
            area_fraction=0.2,
            center_xy=(0.5, 0.5),
        )
```

Also test extra-field rejection, snake-case IDs, strictly increasing timeline
indices/timestamps, relative artifact paths, unique track IDs, closed entity
references, finite confidence, and exact schema version `cv_evidence_v1`.

- [ ] **Step 2: Run the tests and verify imports fail**

Run: `.venv/bin/python -m pytest tests/test_cv_contracts.py tests/test_cv_entities.py -v`

Expected: FAIL because `las_repro.cv` does not exist.

- [ ] **Step 3: Implement frozen strict Pydantic contracts**

```python
class EntityRole(StrEnum):
    ACTOR = "actor"
    MANIPULATED_OBJECT = "manipulated_object"
    CONTAINER = "container"
    OCCLUDER = "occluder"
    SURFACE = "surface"
    OTHER = "other"

class EntityPrompt(StrictModel):
    entity_id: ObjectId
    canonical_label: str
    aliases: tuple[str, ...]
    role: EntityRole

class CvEvidenceRequest(StrictModel):
    schema_version: Literal["cv_request_v1"]
    provider: Literal["fake", "sam31"]
    model_identity: str
    video_path: Path
    video_sha256: Sha256
    duration_seconds: Timestamp
    frame_count: Annotated[int, Field(gt=0, strict=True)]
    checkpoint_sha256: Sha256
    timeline: FrameTimeline
    entities: tuple[EntityPrompt, ...]
    sampling: SamplingPolicy
    thresholds: EvidenceThresholds
```

Use tuples and frozen models at process boundaries. Validate `bbox_xyxy` and
`center_xy` inside `[0,1]`, positive ordered intervals, relative POSIX artifact
paths without `..`, and exact digest syntax.

- [ ] **Step 4: Write entity normalization tests**

```python
def test_normalize_entities_deduplicates_and_applies_role_priority():
    normalized = normalize_entities(
        [
            EntityCandidate(name="Cup", aliases=("mug",), role=EntityRole.MANIPULATED_OBJECT),
            EntityCandidate(name=" cup ", aliases=("vessel",), role=EntityRole.OTHER),
            EntityCandidate(name="right hand", aliases=(), role=EntityRole.ACTOR),
        ],
        limit=2,
    )
    assert [item.entity_id for item in normalized.entities] == ["right_hand", "cup"]
    assert normalized.entities[1].aliases == ("mug", "vessel")
    assert normalized.omitted_count == 0
```

Cover Unicode labels, ASCII slug collision suffixes, casefold deduplication,
blank/`unknown` removal, deterministic alias ordering, all six role priorities,
the 16-item default, omitted-count warning, and input immutability.

- [ ] **Step 5: Implement normalization without NLP dependencies**

Normalize whitespace and case only; never infer nouns from free text. Sort by
the fixed role order `actor`, `manipulated_object`, `container`, `occluder`,
`surface`, `other`, then original position. Generate stable ASCII IDs from the
canonical label and append `_2`, `_3` for collisions.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_cv_contracts.py tests/test_cv_entities.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/las_repro/cv tests/test_cv_contracts.py tests/test_cv_entities.py
git commit -m "feat: define CV evidence contracts"
```

---

### Task 3: Exact frame timeline and two-level sampling

**Files:**
- Create: `src/las_repro/cv/timeline.py`
- Create: `tests/test_cv_timeline.py`

**Interfaces:**
- Consumes: `Path`, FFprobe JSON, `FrameTimeline`, and visibility-change frame indices.
- Produces: `probe_frame_timeline(path, *, run=subprocess.run) -> FrameTimeline`, `initial_sample_indices(timeline, policy) -> tuple[int, ...]`, `refinement_sample_indices(timeline, change_indices, policy) -> tuple[int, ...]`, and `materialize_sampled_frames(video_path, timeline, indices, destination, *, run=subprocess.run) -> SampledFrameSet`.

- [ ] **Step 1: Write failing FFprobe parser tests**

```python
def test_probe_frame_timeline_preserves_nonuniform_pts(tmp_path, fake_run):
    fake_run.stdout = json.dumps({"frames": [
        {"best_effort_timestamp_time": "0.000000"},
        {"best_effort_timestamp_time": "0.033000"},
        {"best_effort_timestamp_time": "0.071000"},
    ]})
    timeline = probe_frame_timeline(tmp_path / "v.mp4", run=fake_run)
    assert [point.timestamp_seconds for point in timeline.frames] == [0.0, 0.033, 0.071]
```

Reject missing timestamps, non-numeric/non-finite values, duplicates,
non-monotonic order, empty frames, malformed JSON, symlink/path changes, and a
subprocess failure. Assert the command is an argument sequence with
`-show_entries frame=best_effort_timestamp_time` and no shell.

- [ ] **Step 2: Run the parser tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_cv_timeline.py -k probe -v`

Expected: FAIL because `probe_frame_timeline` is absent.

- [ ] **Step 3: Implement exact timeline extraction**

Parse timestamps with `Decimal`, convert once to finite floats, and preserve the
decoder-order frame index. Do not synthesize timestamps from average FPS. Match
the input path inode before and after FFprobe using the same no-symlink posture
as the media layer.

- [ ] **Step 4: Write failing sampling tests**

```python
def test_long_video_scans_at_eight_fps_and_refines_one_second_windows(timeline_60s):
    policy = SamplingPolicy(short_video_seconds=30, scan_fps=8, max_fps=30, refinement_radius_seconds=1)
    scan = initial_sample_indices(timeline_60s, policy)
    refined = refinement_sample_indices(timeline_60s, (300,), policy)
    assert len(scan) <= 60 * 8 + 1
    assert all(9 <= timeline_60s.frames[i].timestamp_seconds <= 11 for i in refined)
```

Short-video tests assert all source frames are selected when source FPS is at
most 30 and deterministic nearest-PTS selection when it is higher. Long-video
tests assert first/last coverage, monotonic uniqueness, an 8 fps upper bound,
one-second refinement windows, and a 30 fps upper bound.

- [ ] **Step 5: Implement deterministic PTS-based sampling**

Choose the first frame at or after each rational sample time, retain the first
and last original frames, deduplicate indices, and never report finer temporal
resolution than the selected PTS gaps. `materialize_sampled_frames` invokes
FFmpeg with `shell=False`, selects the exact original frame indices, writes
zero-based JPEG names required by SAM, and returns an immutable mapping from
each SAM-local frame index to its original frame index and PTS. Validate the
mapping and extracted file count before returning.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_cv_timeline.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/las_repro/cv/timeline.py tests/test_cv_timeline.py
git commit -m "feat: preserve frame PTS for CV sampling"
```

---

### Task 4: Secure content-addressed CV artifact store

**Files:**
- Create: `src/las_repro/cv/artifacts.py`
- Create: `tests/test_cv_artifacts.py`

**Interfaces:**
- Consumes: `CvEvidenceRequest`, a validated staging directory, and `CvEvidenceArtifact`.
- Produces: `cv_cache_key(request) -> str`, `CvArtifactHandle`, and `CvArtifactStore.lookup(key)`, `CvArtifactStore.staging(key)`, `CvArtifactStore.publish(request, staging, artifact)`, `CvArtifactStore.load(handle)`.

- [ ] **Step 1: Write failing cache-identity tests**

```python
def test_cache_key_changes_only_for_evidence_inputs(cv_request):
    base = cv_cache_key(cv_request)
    assert cv_cache_key(cv_request.model_copy()) == base
    changed = cv_request.model_copy(update={"checkpoint_sha256": "1" * 64})
    assert cv_cache_key(changed) != base
```

Independently vary video SHA, schema version, provider/model identity,
checkpoint SHA, ordered normalized prompts, sampling, and thresholds. Confirm an
unrelated downstream Qwen prompt is not part of the request or cache key.

- [ ] **Step 2: Write failing containment, digest, and atomic-publication tests**

Create cases for `..`, absolute paths, symlink escapes, a swapped file inode,
wrong digest, missing chunk, partial manifest, oversized manifest, group/world
writable cache roots, duplicate artifact entries, interrupted `os.replace`, and
concurrent publish of the same key.

- [ ] **Step 3: Run the tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_cv_artifacts.py -v`

Expected: FAIL because `CvArtifactStore` is absent.

- [ ] **Step 4: Implement the cache using private staging and atomic publish**

```python
class CvArtifactStore:
    def lookup(self, key: str) -> CvArtifactHandle | None:
        """Return a digest-validated immutable handle, or None on a miss."""

    @contextmanager
    def staging(self, key: str) -> Iterator[Path]:
        """Yield an owner-only sibling directory and remove it on exit."""

    def publish(
        self,
        request: CvEvidenceRequest,
        staging: Path,
        artifact: CvEvidenceArtifact,
    ) -> CvArtifactHandle:
        """Validate, fsync, and atomically publish one complete entry."""

    def load(self, handle: CvArtifactHandle) -> CvEvidenceArtifact:
        """Revalidate the handle and parse its canonical manifest."""
```

Use a two-character digest prefix directory, owner-only staging, canonical JSON,
streamed SHA-256, file-count and byte limits, `fsync`, and same-directory
`os.replace`. A valid existing destination wins a publish race. Move corrupt
entries to a cache-root `quarantine/` name containing only key and timestamp;
never echo attacker-controlled names.

- [ ] **Step 5: Verify raw mask data stays outside SQLite and task JSON**

Add a repository-level test that `CvEvidenceArtifact.model_dump(mode="json")`
contains only relative `ArtifactFile` references and summaries, while `.npz`
or binary mask bytes appear only below the artifact directory.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_cv_artifacts.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/las_repro/cv/artifacts.py tests/test_cv_artifacts.py
git commit -m "feat: add content-addressed CV artifact cache"
```

---

### Task 5: Deterministic Fake provider and leased CV worker

**Files:**
- Create: `src/las_repro/cv/base.py`
- Create: `src/las_repro/cv/worker.py`
- Create: `tests/test_cv_provider.py`
- Create: `tests/test_cv_worker.py`
- Modify: `src/las_repro/cv/__init__.py`

**Interfaces:**
- Consumes: routed `InferenceJob(model_name="sam3.1")`, `CvEvidenceRequest`, and `CvArtifactStore`.
- Produces: `CvEvidenceProvider.analyze(request, staging_dir) -> CvEvidenceArtifact`, `FakeCvEvidenceProvider`, `CvProviderError`, `CvOutOfMemoryError`, and `CVEvidenceWorker.run_once() -> bool`.

- [ ] **Step 1: Write failing protocol and Fake-provider tests**

```python
def test_fake_provider_is_deterministic_and_writes_valid_artifact(cv_request, tmp_path):
    first = FakeCvEvidenceProvider().analyze(cv_request, tmp_path / "a")
    second = FakeCvEvidenceProvider().analyze(cv_request, tmp_path / "b")
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.status == EvidenceStatus.AVAILABLE
```

The fixture must create at least one actor track, one object track, mask-file
reference, visibility transition, and deterministic metrics without importing
Torch, NumPy, or SAM.

- [ ] **Step 2: Write failing worker tests for cache and routing**

```python
def test_cv_worker_claims_only_sam_jobs_and_returns_manifest_handle(store, cv_job, cache):
    worker = CVEvidenceWorker(store, FakeCvEvidenceProvider(), cache, "fake-sam", lease_seconds=30)
    assert worker.run_once(now=20)
    done = store.get_inference_job(cv_job.job_id)
    assert done.status is InferenceStatus.COMPLETED
    assert set(done.result) == {"artifact_key", "manifest_sha256", "cache_hit", "status"}
```

Test that a second identical job is a cache hit and does not call the provider,
that Qwen workers cannot claim CV jobs, and that the CV worker cannot claim Qwen
jobs.

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_cv_provider.py tests/test_cv_worker.py -v`

Expected: FAIL because the provider and worker do not exist.

- [ ] **Step 4: Implement the provider protocol and exact job payload parser**

```python
@runtime_checkable
class CvEvidenceProvider(Protocol):
    def analyze(self, request: CvEvidenceRequest, staging_dir: Path) -> CvEvidenceArtifact:
        """Write provider-owned files and return their strict manifest."""

def cv_request_from_job(job: InferenceJob) -> CvEvidenceRequest:
    if job.stage != "cv_evidence" or job.model_name != "sam3.1":
        raise ValueError("CV job routing is invalid")
    return CvEvidenceRequest.model_validate(job.payload, strict=True)
```

- [ ] **Step 5: Implement lease-safe execution and one OOM retry**

On a miss, execute the provider inside the same heartbeat pattern as
`GPUWorker`. If `CvOutOfMemoryError` occurs, discard staging, halve
`execution_chunk_frames` down to a minimum of one, and retry once without
changing request cache identity, selected frames, or thresholds. On a second
failure, persist only `"CV evidence inference failed"`. Record `cache_hit`,
`oom_retry`, processed frames, prompts, tracks, peak memory, and elapsed seconds
through Task 1's metrics column.

- [ ] **Step 6: Test lease loss, crash, malformed artifact, and cleanup**

Assert stale results are dropped, failed publication never creates a cache hit,
temporary staging is removed, BaseException expires the lease, and provider
errors cannot leak paths, prompts, or raw exception text.

- [ ] **Step 7: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_cv_provider.py tests/test_cv_worker.py tests/test_workers.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/las_repro/cv tests/test_cv_provider.py tests/test_cv_worker.py tests/test_workers.py
git commit -m "feat: execute cached CV evidence jobs"
```

---

### Task 6: 3+1 configuration and separate CV process role

**Files:**
- Modify: `src/las_repro/config.py:28-58`
- Modify: `src/las_repro/cli.py:52-100,163-283`
- Modify: `.env.example`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_repository_policy.py`

**Interfaces:**
- Consumes: `CVEvidenceWorker`, `FakeCvEvidenceProvider`, server-owned paths, and the existing CLI lifecycle.
- Produces: `las-repro cv-worker --device 3 --provider sam31`, explicit `qwen_gpu_devices`, CV settings, and Fake-mode CV execution.

- [ ] **Step 1: Write failing Settings tests with exact defaults and validation**

```python
def test_hybrid_gpu_defaults_are_three_qwen_plus_one_cv():
    settings = Settings(_env_file=None)
    assert settings.gpu_devices == (0, 1, 2)
    assert settings.cv_device == 3
    assert settings.cv_provider == "disabled"
    assert settings.cv_entity_limit == 16
    assert settings.cv_short_video_seconds == 30.0
    assert settings.cv_scan_fps == 8.0
    assert settings.cv_max_fps == 30.0
```

Add strict tests for `disabled|fake|sam31`, distinct Qwen/CV devices, existing
local repository/checkpoint/cache paths when SAM is enabled, 64-hex checkpoint
digest, positive finite timeouts/thresholds, 1-16 entity cap, and
`scan_fps <= max_fps`.

- [ ] **Step 2: Write failing CLI isolation tests**

Patch imports and assert `api`, `coordinator`, and `gpu-worker` never import
`las_repro.cv.sam31`; `cv-worker --provider sam31 --device 3` sets physical
device visibility before the lazy SAM import and constructs exactly one
provider. Assert `gpu-worker --device 3` is rejected under the default 3+1
configuration.

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_cli.py tests/test_repository_policy.py -k 'cv or hybrid_gpu' -v`

Expected: FAIL because the CV role and settings are absent.

- [ ] **Step 4: Add server-owned settings**

Add fields for provider, CV device/model alias, repository/checkpoint/BPE paths,
checkpoint digest, cache root, cache byte/file limits, entity limit, short-video
threshold, scan/max FPS, refinement radius, confidence/visibility/overlap
thresholds, execution chunk frames, timeout, and compile flag. Keep provider
disabled by default so existing deployments retain Qwen-only behavior.

- [ ] **Step 5: Implement the CV CLI lifecycle and Fake stack**

`cv-worker` supports `fake` and `sam31`; the SAM branch sets
`CUDA_VISIBLE_DEVICES` before lazy imports and addresses logical `cuda:0` inside
the provider. `run-fake` starts one `CVEvidenceWorker` thread when
`LAS_CV_PROVIDER=fake`. Shutdown stops claims, finishes or expires the current
lease, closes provider state, and checkpoints SQLite.

- [ ] **Step 6: Update `.env.example` without real paths or credentials**

Document Qwen devices `0,1,2`, CV device `3`, disabled default, local-only model
paths, digest, cache root, and all numerical defaults. Repository policy tests
must reject access-token patterns and CV GPU packages imported outside
`cv/sam31.py`.

- [ ] **Step 7: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_cli.py tests/test_repository_policy.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/las_repro/config.py src/las_repro/cli.py .env.example tests/test_cli.py tests/test_repository_policy.py
git commit -m "feat: add isolated SAM worker role"
```

---

### Task 7: Entity-bearing Pass A without an extra VLM call

**Files:**
- Modify: `src/las_repro/pipelines/validators.py:123-148`
- Modify: `src/las_repro/pipelines/output_validation.py`
- Modify: `src/las_repro/prompts/embodied_pass_a.txt`
- Modify: `src/las_repro/models/fake.py:31-104`
- Modify: `src/las_repro/pipelines/embodied.py:142-203,435-477`
- Modify: `tests/test_embodied_validators.py`
- Modify: `tests/test_embodied_pipeline.py`
- Modify: `tests/test_qwen_backend.py`

**Interfaces:**
- Consumes: `EntityCandidate`, the existing Pass A repair loop, and `normalize_entities`.
- Produces: `CoarsePlan.entity_candidates: list[EntityCandidate]` and a validated `NormalizedEntities` immediately after Pass A.

- [ ] **Step 1: Write failing CoarsePlan schema and temporal tests**

```python
def test_coarse_plan_requires_explicit_entity_candidates(valid_coarse_plan):
    valid_coarse_plan["entity_candidates"] = [{
        "name": "red container",
        "aliases": ["container"],
        "role": "manipulated_object",
    }]
    parsed = CoarsePlan.model_validate(valid_coarse_plan)
    validate_coarse_plan(parsed, parsed.actions[-1].end)
    assert parsed.entity_candidates[0].name == "red container"
```

Reject more than 64 raw candidates, duplicate aliases inside a candidate,
blank strings, unsupported roles, extra keys, and empty candidates when actions
mention a concrete target. Add explicit output-registry repair codes for each
schema family without retaining model-supplied text in errors.

- [ ] **Step 2: Run validator tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_embodied_validators.py -k entity -v`

Expected: FAIL because `CoarsePlan` has no entity inventory.

- [ ] **Step 3: Update the Pass A prompt and Fake fixture**

Require concise English canonical labels and aliases, one supported role, and
visible/action-relevant entities only. Tell the model that candidates are
evidence requests rather than claims of presence. The Fake fixture returns a
right-hand actor and a red-container manipulated object.

- [ ] **Step 4: Preserve entity candidates through Pass B context safely**

Pass B receives the validated complete `CoarsePlan`, but its output remains the
existing action/boundary schema. Parent-copy validation continues to compare
only action fields. Immediately after Pass A, call
`normalize_entities(coarse.entity_candidates, limit=settings.cv_entity_limit)`
and retain the omission warning for the final result.

- [ ] **Step 5: Update all exact fixtures and schema tests**

Change every Pass A fixture in `test_embodied_pipeline.py`,
`test_embodied_validators.py`, and `test_qwen_backend.py`. Add a prompt-injection
test proving candidate text is serialized only as trusted JSON data in later
prompts and cannot create new instruction sections.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_embodied_validators.py tests/test_embodied_pipeline.py tests/test_qwen_backend.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/las_repro/pipelines/validators.py src/las_repro/pipelines/output_validation.py src/las_repro/prompts/embodied_pass_a.txt src/las_repro/models/fake.py src/las_repro/pipelines/embodied.py tests/test_embodied_validators.py tests/test_embodied_pipeline.py tests/test_qwen_backend.py
git commit -m "feat: derive CV entities in embodied pass A"
```

---

### Task 8: SAM3.1 Object Multiplex adapter

**Files:**
- Create: `src/las_repro/cv/sam31.py`
- Create: `tests/test_sam31_adapter.py`
- Modify: `src/las_repro/cv/__init__.py`

**Interfaces:**
- Consumes: `CvEvidenceRequest`, an injected official-compatible predictor, and a private staging directory.
- Produces: `Sam31EvidenceProvider.load(repository_path, checkpoint_path, checkpoint_sha256, *, compile_model=False, predictor_factory=None) -> Sam31EvidenceProvider`, `Sam31EvidenceProvider.analyze(request, staging_dir) -> CvEvidenceArtifact`, mask chunks, normalized tracks, spatial summaries, and provider metrics.

- [ ] **Step 1: Write predictor-double tests for the official request sequence**

```python
def test_sam31_adapter_uses_local_multiplex_session_and_closes_it(cv_request, predictor, tmp_path):
    provider = Sam31EvidenceProvider(predictor=predictor, torch_module=fake_torch)
    artifact = provider.analyze(cv_request, tmp_path)
    assert predictor.requests[0]["type"] == "start_session"
    assert any(item["type"] == "add_prompt" for item in predictor.requests)
    assert any(item["type"] == "propagate_in_video" for item in predictor.stream_requests)
    assert predictor.requests[-1]["type"] == "close_session"
    assert artifact.provider == "sam31_multiplex"
```

The double returns exact arrays for `out_obj_ids`, `out_probs`,
`out_boxes_xywh`, and `out_binary_masks`. Test that `start_session.resource_path`
is a numbered sampled-frame directory, not the unsampled source video; test
multi-instance IDs, per-prompt ID namespacing, empty detections, shape mismatch,
non-finite scores, invalid boxes, wrong mask shape, out-of-range frame index,
duplicate frame output, and close in `finally`.

- [ ] **Step 2: Run adapter tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_sam31_adapter.py -v`

Expected: FAIL because `Sam31EvidenceProvider` is absent.

- [ ] **Step 3: Implement lazy local-only loading**

```python
from sam3.model_builder import build_sam3_multiplex_video_predictor

predictor = build_sam3_multiplex_video_predictor(
    checkpoint_path=str(checkpoint_path),
    load_from_HF=False,
    multiplex_count=16,
    gpus_to_use=[0],
    compile=compile_model,
)
```

Before this import, verify the configured SAM repository revision and checkpoint
SHA-256, then verify that the imported `sam3` package resolves beneath that
repository. Refuse missing/mismatched assets and network-looking model
identifiers. Do not add SAM, Torch, NumPy, OpenCV, or Hugging Face dependencies
to the core `pyproject.toml` environment.

- [ ] **Step 4: Implement sampled-frame sessions without losing source PTS**

For videos up to 30 seconds, materialize the exact initial indices once and call
`start_session(resource_path=sampled_frames.directory)`. For longer videos,
materialize the 8 fps scan, run every entity prompt, collect visibility-change
indices, map them back to original PTS, compute the one-second refinement
windows, and then materialize the union of scan and refined indices. Discard the
preliminary scan outputs and rerun every prompt on that final union so published
tracks share one stable local-index-to-original-index/PTS mapping. Never run the
predictor over the unsampled source MP4.

- [ ] **Step 5: Implement stateful prompt and propagation calls**

For each materialized directory, follow the official notebook contract:
`start_session`, then for each normalized text prompt reset prior state, call
`add_prompt` on local frame 0, stream `propagate_in_video`, and namespace returned
object IDs by entity ID. Object Multiplex handles all instances produced for
that one text prompt; prompts remain sequential so entity provenance stays
closed. Close every preliminary and final session in `finally`. Convert
normalized XYWH to validated XYXY, derive area and center from the boolean mask,
and use `out_probs` for confidence. The OOM execution-chunk retry advances over
the same final sampled-frame sequence and changes neither sampling nor PTS.

- [ ] **Step 6: Write masks and bounded overlays**

Write compressed per-prompt `.npz` mask chunks below `masks/`; calculate
relations and summary scalars before leaving the SAM environment. Write no more
than two overlay PNGs per visibility-change candidate and no more than 24
overlays per video. The core process sees only relative paths plus digests.

- [ ] **Step 7: Map CUDA OOM and collect memory metrics**

Translate only `torch.cuda.OutOfMemoryError` into `CvOutOfMemoryError`; other
errors become sanitized `CvProviderError`. Reset peak statistics before
analysis, record peak allocated bytes, clear session references, and call the
official close path before optional `empty_cache` under pressure.

- [ ] **Step 8: Run focused and full CPU tests**

Run: `.venv/bin/python -m pytest tests/test_sam31_adapter.py tests/test_repository_policy.py -q`

Expected: PASS without importing an installed SAM package.

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/las_repro/cv/sam31.py src/las_repro/cv/__init__.py tests/test_sam31_adapter.py tests/test_repository_policy.py
git commit -m "feat: adapt local SAM3.1 video tracking"
```

---

### Task 9: Bounded evidence summaries and occlusion candidates

**Files:**
- Create: `src/las_repro/cv/summary.py`
- Create: `tests/test_cv_summary.py`

**Interfaces:**
- Consumes: `CvEvidenceArtifact`, normalized entities, and `FrameTimeline`.
- Produces: `summarize_cv_evidence(artifact, *, limits) -> CvEvidenceSummary` and `build_occlusion_candidates(summary, thresholds) -> tuple[OcclusionCandidate, ...]`.

- [ ] **Step 1: Write failing summary-bound tests**

```python
def test_summary_contains_no_masks_and_is_size_bounded(cv_artifact):
    summary = summarize_cv_evidence(cv_artifact, max_tracks=64, max_observations_per_track=64)
    encoded = summary.model_dump_json()
    assert "out_binary_masks" not in encoded
    assert len(encoded) <= 200_000
```

Assert deterministic downsampling retains first, last, minimum-area,
maximum-area, lowest-confidence, and state-change observations. Test track and
relation caps, stable ordering, finite geometry, relative overlay references,
and warnings for every truncation.

- [ ] **Step 2: Write candidate classification-feature tests**

Construct tracks for visible→missing→visible behind another object's mask,
visible→edge-exit, low-confidence flicker, partial area reduction, permanent
disappearance, and missing plausible occluder. Assert only evidence-supported
cases become candidates and every counter-signal is retained.

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_cv_summary.py -v`

Expected: FAIL because summarization is absent.

- [ ] **Step 4: Implement deterministic relations and candidates**

```python
class OcclusionCandidate(StrictModel):
    candidate_id: CandidateId
    target_entity_id: ObjectId
    possible_occluder_entity_ids: tuple[ObjectId, ...]
    allowed_start_times: tuple[Timestamp, ...]
    allowed_end_times: tuple[Timestamp, ...]
    last_visible_frame: int
    first_revisible_frame: int | None
    edge_departure: bool
    low_confidence: bool
    overlay_refs: tuple[str, ...]
```

Use mask-overlap, containment, center, area ratio, edge proximity, confidence,
and lifecycle gaps. Candidate IDs are a hash-derived stable prefix plus ordinal.
Do not classify the candidate semantically in this module.

- [ ] **Step 5: Serialize summaries as trusted JSON prompt data**

Add `summary.prompt_record()` and `candidate.prompt_record()` that expose only
allowlisted scalar/list fields and relative keyframe IDs. Enforce the 200,000
character prompt-data cap before any Qwen job is created.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_cv_summary.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/las_repro/cv/summary.py tests/test_cv_summary.py
git commit -m "feat: summarize CV tracks for semantic stages"
```

---

### Task 10: Dedicated occlusion adjudication stage

**Files:**
- Create: `src/las_repro/pipelines/occlusion.py`
- Create: `src/las_repro/prompts/occlusion_semantics.txt`
- Create: `tests/test_occlusion_semantics.py`
- Modify: `src/las_repro/pipelines/output_validation.py`
- Modify: `src/las_repro/pipelines/embodied.py:51-58,357-542,1091-1177`
- Modify: `src/las_repro/models/fake.py`
- Modify: `src/las_repro/models/qwen3_vl.py:30-45`
- Modify: `tests/test_qwen_backend.py`

**Interfaces:**
- Consumes: `tuple[OcclusionCandidate, ...]`, normalized entities, video duration, frame PTS, and bounded evidence summary.
- Produces: `OcclusionDecisionSet`, `validate_occlusion_decisions(result: OcclusionDecisionSet, candidates: tuple[OcclusionCandidate, ...], *, duration: float) -> None`, `project_occlusion_events(result: OcclusionDecisionSet, candidates: tuple[OcclusionCandidate, ...], tracks: tuple[CvTrack, ...], segments: Sequence[Mapping[str, Any]], *, repair_history: tuple[str, ...]) -> list[dict[str, Any]]`, and the Qwen stage `occlusion_semantics` with one repair.

- [ ] **Step 1: Write failing strict decision-schema tests**

```python
def test_occlusion_decision_must_use_candidate_and_observed_boundaries(candidate):
    raw = {
        "decisions": [{
            "candidate_id": candidate.candidate_id,
            "classification": "occlusion",
            "target_entity_id": candidate.target_entity_id,
            "occluder_entity_id": candidate.possible_occluder_entity_ids[0],
            "events": [{"event_type": "occluded", "start": 1.0, "end": 2.0}],
            "visual_evidence": "target disappears behind the board and returns",
            "confidence": 0.9,
        }]
    }
    parsed = OcclusionDecisionSet.model_validate(raw)
    validate_occlusion_decisions(parsed, (candidate,), duration=3.0)
```

Test exact one-decision-per-candidate cardinality and order; classifications
`occlusion`, `out_of_frame`, `detector_loss`, `unknown`; zero positive events
for non-occlusion; positive ordered intervals for occlusion; target equality;
occluder must be proposed or `unknown`; start/end must be selected from the
candidate's allowed PTS values; no overlap within one candidate's same event
type; and closed video bounds.

- [ ] **Step 2: Run schema tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_occlusion_semantics.py -k decision -v`

Expected: FAIL because the schema is absent.

- [ ] **Step 3: Add a data-isolated prompt and renderer**

The prompt requires visible evidence, distinguishes occlusion from edge exit
and detector loss, permits `unknown`, forbids invented entities/times, and
returns one decision for each trusted candidate skeleton. Serialize candidates,
entities, and CV summary as canonical JSON under labeled trusted-data sections;
put only issue codes in the repair block.

- [ ] **Step 4: Register output validation and one repair attempt**

Add `OcclusionDecisionSet` to `DEFAULT_OUTPUT_SCHEMAS` with closed repair codes.
Add the prompt asset, Fake fixture, and Qwen token cap. Extend `_stage_label` and
reuse `_run_validated_stage`; a valid empty candidate tuple skips Qwen and
returns an empty decision set deterministically.

- [ ] **Step 5: Implement positive event projection with provenance**

`project_occlusion_events` emits only `occlusion_enter`, `occluded`, and
`occlusion_exit` records, ordered by start/end/candidate/event type. Each record
contains event index, target/occluder IDs, description, confidence,
`branch="occlusion"`, `model_stage="occlusion_semantics"`,
`source_candidate_id`, overlapping fine-segment indices, source track IDs,
overlay keyframe IDs, `evidence_mode="hybrid"`, the closed repair history
`("initial",)` or `("initial", "repair")`, and
`review_status="unreviewed"`. Non-positive decisions remain in the audit branch
but do not enter event scoring.

- [ ] **Step 6: Test repair, skip, and conservative failure behavior**

Assert invalid initial output creates exactly one repair job, invalid repair
degrades only the occlusion branch, zero candidates create no job, and model
output cannot inject paths, masks, unknown entities, or arbitrary timestamps.

- [ ] **Step 7: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_occlusion_semantics.py tests/test_qwen_backend.py tests/test_embodied_pipeline.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/las_repro/pipelines/occlusion.py src/las_repro/prompts/occlusion_semantics.txt src/las_repro/pipelines/output_validation.py src/las_repro/pipelines/embodied.py src/las_repro/models/fake.py src/las_repro/models/qwen3_vl.py tests/test_occlusion_semantics.py tests/test_qwen_backend.py tests/test_embodied_pipeline.py
git commit -m "feat: add evidence-constrained occlusion stage"
```

---

### Task 11: Hybrid pipeline orchestration, branch result, and fallback

**Files:**
- Create: `src/las_repro/pipelines/hybrid_result.py`
- Create: `tests/test_hybrid_result.py`
- Modify: `src/las_repro/pipelines/embodied.py:126-354`
- Modify: `src/las_repro/prompts/embodied_enrichment.txt`
- Modify: `src/las_repro/prompts/scene_semantics.txt`
- Modify: `src/las_repro/pipelines/scene_semantics.py`
- Modify: `src/las_repro/pipelines/semantic_events.py`
- Modify: `src/las_repro/pipelines/output_validation.py`
- Modify: `src/las_repro/models/fake.py`
- Modify: `src/las_repro/export.py:46-57,282-379`
- Modify: `tests/test_embodied_pipeline.py`
- Modify: `tests/test_scene_semantics.py`
- Modify: `tests/test_semantic_events.py`
- Modify: `tests/test_export.py`

**Interfaces:**
- Consumes: Pass A normalized entities, routed CV job result, `CvEvidenceSummary`, fine segments, occlusion decisions/events, and existing SceneSemantics.
- Produces: `SceneLocation`, `SceneRelation`, additive `annotation_branches`, `cv_evidence`, and `performance` fields plus exact legacy projections.

- [ ] **Step 1: Write failing canonical-result consistency tests**

```python
def test_hybrid_result_keeps_legacy_fields_as_exact_projections(hybrid_result):
    validate_hybrid_result(hybrid_result)
    assert hybrid_result["grouped_semantic_events"] == [
        legacy_action_projection(event)
        for event in hybrid_result["annotation_branches"]["action_events"]
    ]
    assert hybrid_result["semantic_events"] == hybrid_result["annotation_branches"]["legacy_scene_events"]
    assert hybrid_result["semantic_events"] == [
        legacy_scene_event_projection(event)
        for event in hybrid_result["annotation_branches"]["scene_facts"]["events"]
    ]
    assert hybrid_result["annotation_branches"]["occlusion"]["status"] == "available"
```

Define the exact additive shape:

```json
{
  "annotation_branches": {
    "action_events": [],
    "occlusion": {"status": "available", "decisions": [], "events": []},
    "scene_facts": {"status": "available", "objects": [], "initial_state": [], "final_state": [], "locations": [], "relations": [], "outcome": {"status": "unknown", "description": "No supported outcome", "confidence": 0.0}, "events": []},
    "legacy_scene_events": []
  },
  "cv_evidence": {"status": "available", "artifact_key": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "manifest_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "cache_hit": false},
  "performance": {"stages": [], "total_seconds": 0.0, "repair_count": 0, "degradation_count": 0}
}
```

Each canonical action event contains the nine existing grouped-event keys plus
`branch`, `model_stage`, `evidence_mode`, `source_track_ids`,
`source_keyframe_ids`, `repair_history`, and `review_status`.
`legacy_action_projection` removes exactly those seven additive keys, and the
result must equal `build_semantic_events(segments)` byte-for-byte after canonical
serialization. Set `branch="action"`, `model_stage="embodied_enrichment"`, and
`review_status="not_required"`. Populate source tracks deterministically from
actor/target entity matches whose observations overlap the event interval; use
`hybrid` only when at least one validated track remains, otherwise use
`vlm_only`. Repair history is the closed sequence `("initial",)` or
`("initial", "repair")` from the producing stage.

Extend `SceneSemantics` with required `locations` and `relations` lists.
`SceneLocation` contains `object_id`, `location`, `start`, `end`,
`visual_evidence`, `confidence`, and the provenance keys described below.
`SceneRelation` contains
`subject_object_id`, `relation`, `object_object_id`, `start`, `end`,
`visual_evidence`, `confidence`, `evidence_mode`, `source_track_ids`, and
`source_keyframe_ids`, plus `branch`, `model_stage`, overlapping
`source_segment_indices`, `repair_history`, and `review_status`. Its closed enum
is `left_of`, `right_of`, `above`,
`below`, `inside`, `on`, `overlapping`, `near`, `occluding`, or `unknown`.
Validate closed object references, positive in-video intervals, finite
confidence, deterministic order, and provenance against the bounded summary.
Both schemas reject blank claims and require closed object references, observed
PTS boundaries, bounded confidence, and evidence provenance. The `scene_facts`
branch projects objects, states, locations, relations, and outcome;
it also contains provenance-enriched `events` from the existing scene semantic
events. `legacy_scene_event_projection` strips the provenance keys, and both
`legacy_scene_events` and the existing top-level `semantic_events` must equal
that deterministic projection.

Status values are `available`, `unavailable`, or `disabled`. Validate warning
and status consistency, exact provenance references, deterministic event order,
and absence of mask paths other than allowlisted overlay IDs.

- [ ] **Step 2: Run result tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_hybrid_result.py -v`

Expected: FAIL because `validate_hybrid_result` is absent.

- [ ] **Step 3: Add CV job orchestration after Pass A**

When `cv_provider=disabled`, produce disabled evidence and continue unchanged.
Otherwise build the exact timeline/request, create
`InferenceJobSpec(stage="cv_evidence", ordinal=0, payload=request.model_dump(mode="json"), model_name=settings.cv_model_alias)`,
wait with `settings.cv_timeout_seconds`, load the published artifact, and build
the bounded summary. Any failed/timed-out/invalid result becomes unavailable
evidence plus `{"code":"CV_EVIDENCE_UNAVAILABLE"}`; it does not fail the task.

- [ ] **Step 4: Feed evidence into existing semantic prompts**

Enrichment and SceneSemantics receive a canonical `CV_EVIDENCE_SUMMARY_JSON`
block when available and omit it otherwise. Prompt instructions say geometry is
supporting evidence, low confidence is uncertain, and track absence is not a
semantic fact. Update the strict SceneSemantics schema, registry, prompt, Fake
fixture, and validation tests together for the required relation list. No raw
mask or absolute artifact path enters a prompt.

- [ ] **Step 5: Run and merge independent branches**

Build action events from validated enriched fine segments. Run SceneSemantics
with its existing one-repair and unavailable fallback. Run Occlusion only when
valid candidates exist. Construct `annotation_branches`, derive legacy
top-level fields, and validate the complete result before returning it. Every
canonical action, occlusion, and scene event must carry branch, model stage,
source fine-segment/track/keyframe IDs, evidence mode, confidence, repair
history, and review status; projections remove only explicitly additive keys.

- [ ] **Step 6: Build performance from immutable job records**

Emit stage names `media_decode`, `pass_a`, `sam31`, `action_enrichment`,
`occlusion`, `scene_facts`, and `merge`. For queued model stages, expose queue
seconds (`started_at-created_at`), wall seconds (`finished_at-created_at`),
inference seconds, attempt count, worker ID, cache hit, and bounded provider
metrics; for in-process stages, expose only monotonic elapsed seconds. Add exact
task-level repair and degradation counts. Exclude prompts, local paths, raw
errors, and job payloads. Compute top-level total with the coordinator monotonic
clock.

- [ ] **Step 7: Extend export validation without changing training rows**

`iter_action_captions` accepts the additive keys only when
`validate_hybrid_result` succeeds. It still yields the same fine-segment fields
and bytes for the same `segments`. Add a regression test comparing JSONL before
and after additive hybrid metadata.

- [ ] **Step 8: Test all degradation combinations**

Cover CV disabled, CV cache hit, CV job failed, corrupt manifest, zero tracks,
scene unavailable, occlusion unavailable, both unavailable, Qwen repair, and a
valid hybrid result. Assert `COMPLETED` for optional-branch degradation, explicit
warnings, `evidence_mode="vlm_only"` for action events with zero validated source
track IDs, and no fabricated occlusion events.

- [ ] **Step 9: Run focused and full tests with coverage**

Run: `.venv/bin/python -m pytest tests/test_hybrid_result.py tests/test_embodied_pipeline.py tests/test_scene_semantics.py tests/test_semantic_events.py tests/test_export.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q --cov=las_repro --cov-report=term-missing`

Expected: PASS with branch coverage at least 85%.

- [ ] **Step 10: Commit**

```bash
git add src/las_repro/pipelines/hybrid_result.py src/las_repro/pipelines/embodied.py src/las_repro/prompts/embodied_enrichment.txt src/las_repro/prompts/scene_semantics.txt src/las_repro/pipelines/scene_semantics.py src/las_repro/pipelines/semantic_events.py src/las_repro/pipelines/output_validation.py src/las_repro/models/fake.py src/las_repro/export.py tests/test_hybrid_result.py tests/test_embodied_pipeline.py tests/test_scene_semantics.py tests/test_semantic_events.py tests/test_export.py
git commit -m "feat: merge independent hybrid annotation branches"
```

---

### Task 12: LAS-alignment evaluator and acceptance gates

**Files:**
- Create: `src/las_repro/evaluation/__init__.py`
- Create: `src/las_repro/evaluation/las_alignment.py`
- Create: `tests/test_las_alignment.py`
- Create: `scripts/evaluate_las_alignment.py`
- Create: `evaluation/config/las_alignment_mapping_v1.json`

**Interfaces:**
- Consumes: frozen English LAS reference JSON, Qwen-only local result JSON, hybrid local result JSON, immutable mapping config, and optional completed human-review JSON.
- Produces: `evaluate_sample(reference, prediction, mapping, review=None) -> SampleMetrics`, `aggregate_metrics(samples) -> AggregateMetrics`, and canonical report JSON.

- [ ] **Step 1: Write failing temporal matching tests**

```python
def test_one_to_one_matching_maximizes_cardinality_then_iou():
    refs = [Event(0, 2, "move"), Event(2, 4, "move")]
    preds = [Event(0, 4, "move"), Event(0, 2, "move")]
    matches = match_events(refs, preds, temporal_iou_threshold=0.3)
    assert {(m.reference_index, m.prediction_index) for m in matches} == {(0, 1), (1, 0)}
```

Test exact half-open temporal IoU, threshold equality, one-to-one exclusivity,
empty sets, deterministic tie-breaking, event-type compatibility, and
cardinality-first/IoU-second matching. Implement a pure-Python rectangular
Hungarian assignment using edge weight `1_000_000 + IoU` for eligible pairs and
zero-weight dummies; add no SciPy or NetworkX dependency.

- [ ] **Step 2: Write failing metric tests**

Fixtures must hand-calculate Event precision/recall/F1 at 0.3 and 0.5,
occlusion interval F1/IoU, enter/exit mean and median absolute error,
actor/action/target/state/result macro-F1, completion/degradation/repair rates,
and reviewed factual precision. Assert undefined denominators serialize as
`null` with a reason, never NaN.

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_las_alignment.py -v`

Expected: FAIL because the evaluation package is absent.

- [ ] **Step 4: Implement strict reference and prediction adapters**

Validate exact LAS object/event references, finite confidence, positive bounded
intervals, and frozen manifest SHA-256 before scoring. Local scoring uses
`annotation_branches.action_events` and positive
`annotation_branches.occlusion.events`; it never scores fine segments as extra
primary events. Keep action/actor/target aliases in the versioned mapping JSON,
not hard-coded inside matching logic.

- [ ] **Step 5: Implement metrics and canonical aggregation**

Use one-to-one matches separately at 0.3 and 0.5. Compute macro-F1 by field over
comparable labels, and publish numerator/denominator beside every ratio. Include
sample count, event counts, unmatched IDs, reference/prediction digests, model
identity, artifact identity, configuration digest, and acceptance-gate booleans.

- [ ] **Step 6: Implement the CLI and input-integrity checks**

```bash
.venv/bin/python scripts/evaluate_las_alignment.py \
  --reference-manifest evaluation/references/las_official_english_2026-09-04/manifest.json \
  --qwen-results outputs/five-demo/qwen-only \
  --hybrid-results outputs/five-demo/qwen-sam31 \
  --mapping evaluation/config/las_alignment_mapping_v1.json \
  --output outputs/five-demo/las-alignment-report.json
```

The CLI refuses missing samples, hash mismatches, mixed configs/models,
non-canonical JSON, incomplete review records when factual precision is
requested, or overwriting a different existing report without an explicit
`--replace` flag.

- [ ] **Step 7: Encode the approved gates**

The report passes only when all five tasks validate, action Event F1@0.3 loses
no more than 0.05 absolute, occlusion Event F1@0.3 is positive, every positive
occlusion claim is reviewed, reviewed precision is at least 0.80, and all
provenance hashes are present.

- [ ] **Step 8: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_las_alignment.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/las_repro/evaluation scripts/evaluate_las_alignment.py evaluation/config/las_alignment_mapping_v1.json tests/test_las_alignment.py
git commit -m "feat: score one-to-one LAS alignment"
```

---

### Task 13: Hybrid projection and merged comparison-viewer binding

**Files:**
- Create: `src/las_repro/evaluation/viewer_projection.py`
- Create: `tests/test_viewer_projection.py`
- Modify: `scripts/build_comparison_viewer_data.py`
- Modify: `tests/test_comparison_viewer_export.py`
- Modify: `evaluation/viewer/index.html`
- Modify: `evaluation/viewer/styles.css`
- Modify: `evaluation/viewer/js/model.js`
- Modify: `evaluation/viewer/js/app.js`
- Modify: `evaluation/viewer/tests/model.test.mjs`
- Modify: `evaluation/viewer/README.md`
- Modify: `tests/test_comparison_viewer_static.py`

**Interfaces:**
- Consumes: one validated hybrid result, sample ID, duration, source digest, and an optional validated occlusion-review mapping; derives the result digest canonically and reads the validated artifact digest from `cv_evidence`.
- Produces: `project_hybrid_viewer_data(sample_id: str, duration: float, result: Mapping[str, Any], *, source_sha256: str, review: Mapping[str, Any] | None = None, include_fine_segments: bool = False) -> dict[str, Any]`, `normalizeHybrid(displayData, duration, expectedSampleId)`, and a right-panel selector for Qwen-only versus Qwen+SAM3.1.

- [ ] **Step 1: Write failing allowlist and provenance tests**

```python
def test_viewer_projection_exposes_branches_without_local_paths(hybrid_result):
    projected = project_hybrid_viewer_data(
        "full_0001", 10.933333333333334, hybrid_result,
        source_sha256="c3243c46bad68d3b2772e82648e45b68e75a1893b0ce27edecd450226464c1e9",
    )
    assert set(projected) == {"schema_version", "sample", "layers", "provenance", "warnings"}
    assert "/Users/" not in json.dumps(projected)
    assert "/root/" not in json.dumps(projected)
```

Assert duration bounds, event ordering, branch status, evidence mode,
confidence, source track/candidate/keyframe IDs, digest syntax, relative overlay
references, and exact rejection of raw masks, job payloads, prompts, errors,
absolute paths, and unknown fields. An optional review record may change only a
matching positive occlusion event from `unreviewed` to `supported` or
`unsupported`; missing, duplicate, or foreign review IDs fail projection.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_viewer_projection.py -v`

Expected: FAIL because the projector is absent.

- [ ] **Step 3: Implement the static projection**

Create layers `action_events`, `occlusion_events`, and `scene_facts`; retain
`fine_segments` only under an evidence-layer flag. Every event includes its
source IDs and review status. Include bounded relative overlay references but
no full mask index. Validate the projection before canonical JSON serialization.

- [ ] **Step 4: Write failing exporter and manifest-compatibility tests**

Extend fixtures with a version-2 sample containing:

```json
{
  "local_variants": [
    {"id": "qwen_only", "label": "Qwen-only", "path": "evaluation/viewer/data/local/full_0001.json", "format": "comparison_viewer_local_v1"},
    {"id": "qwen_sam31", "label": "Qwen + SAM3.1", "path": "evaluation/viewer/data/hybrid/full_0001.json", "format": "comparison_viewer_hybrid_v1"}
  ]
}
```

Require unique closed variant IDs, one Qwen-only entry, relative paths, exact
format identifiers, matching sample/duration/digests, all-or-nothing five-sample
publication, digest-checked overlay copies, and backward-compatible parsing of
the currently committed version-1 manifest until Task 15 publishes real hybrid
files.

- [ ] **Step 5: Extend the existing atomic multi-sample exporter**

Keep the existing Qwen-only arguments unchanged. Add optional
`--hybrid-input-dir`, `--hybrid-output-dir`, `--artifact-root`, and `--review`
arguments. When hybrid export is requested, verify source/result/review and
artifact hashes, project every sample, copy only referenced overlay PNGs into a
private temporary output tree, rewrite them to viewer-relative paths, validate
every JSON/image digest, and atomically replace only the named hybrid output
directory. A missing sample or overlay fails the whole export; no partial viewer
dataset is published.

- [ ] **Step 6: Write failing browser-model tests for hybrid layers**

Add Node tests proving `normalizeHybrid` accepts Action, Occlusion, Scene, and
optional Fine layers; packs overlapping events deterministically; retains
branch/evidence/provenance/review metadata; validates overlay paths; and rejects
unknown schemas, foreign sample IDs, invalid intervals, malformed provenance,
absolute paths, and raw-mask references. Add manifest tests for variant fallback
and default selection of `qwen_sam31` when present.

- [ ] **Step 7: Bind the projection to the merged viewer**

Keep LAS fixed on the left. Add an accessible right-panel result selector for
`Qwen-only` and `Qwen + SAM3.1`, plus `Action`, `Occlusion`, `Scene`, and `Fine`
layer buttons. Disable unavailable modes rather than showing stale data. Event
cards show evidence mode, track/candidate/keyframe IDs, confidence, warnings,
repair history, and review status. Clicking an event seeks the shared video;
clicking an allowlisted overlay opens a bounded evidence preview without a
backend or full mask-store access. Preserve responsive layout, keyboard focus,
safe `textContent` rendering, and synchronized LAS/local timeline behavior.

- [ ] **Step 8: Run focused viewer and full tests**

Run: `.venv/bin/python -m pytest tests/test_viewer_projection.py tests/test_comparison_viewer_export.py tests/test_comparison_viewer_static.py -q`

Expected: PASS.

Run: `node --test evaluation/viewer/tests/model.test.mjs`

Expected: PASS.

Run: `node --check evaluation/viewer/js/app.js`

Expected: no output and exit 0.

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/las_repro/evaluation/viewer_projection.py tests/test_viewer_projection.py scripts/build_comparison_viewer_data.py tests/test_comparison_viewer_export.py evaluation/viewer/index.html evaluation/viewer/styles.css evaluation/viewer/js/model.js evaluation/viewer/js/app.js evaluation/viewer/tests/model.test.mjs evaluation/viewer/README.md tests/test_comparison_viewer_static.py
git commit -m "feat: compare hybrid evidence in the demo viewer"
```

These are new commits on `feat/sam31-evidence-integration` against the viewer
already present on `main`; they do not modify the merged PR #3 branch or reopen
that PR.

---

### Task 14: SAM runtime documentation and GPU smoke tooling

**Files:**
- Create: `scripts/sam31_smoke.py`
- Create: `tests/test_sam31_smoke.py`
- Create: `docs/deployment/sam31-runtime.md`
- Modify: `README.md`
- Modify: `.env.example`

**Interfaces:**
- Consumes: pinned local SAM checkout/checkpoint, one allowed test video, physical GPU ID 3, and the production settings contract.
- Produces: a sanitized smoke JSON record and exact deployment/rollback commands.

- [ ] **Step 1: Write failing smoke-script argument and sanitization tests**

Test required repository/checkpoint/checkpoint-SHA/video/cache/device arguments,
device 3 enforcement by default, path containment, no network flags, nonzero
exit for hash mismatch/wrong device/invalid artifact, and stdout containing no
absolute paths, prompt text, tokens, or raw exceptions.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_sam31_smoke.py -v`

Expected: FAIL because the script is absent.

- [ ] **Step 3: Implement one-video smoke output**

The script builds a fixed two-entity request, probes exact PTS, runs the adapter,
publishes and reloads the artifact, checks at least one valid observation, closes
the provider, and prints canonical JSON with GPU name/index, source revision,
checkpoint SHA, elapsed seconds, peak bytes, frame/track counts, and pass/fail.

- [ ] **Step 4: Write exact separate-environment instructions**

Document Python 3.12+, PyTorch 2.7+, and CUDA 12.6+ as upstream prerequisites,
with the accepted host using PyTorch 2.10/CUDA 12.8. Pin checkout commit
`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`, install the harness and SAM into a
dedicated environment, compute the local checkpoint SHA with `sha256sum`, set
offline environment variables, initialize trusted cache/database directories,
and launch three Qwen workers plus one CV worker. Include stop, restart,
cache-quarantine, rollback, and idle-memory verification commands.

- [ ] **Step 5: Update README and example configuration**

Describe disabled/Fake/SAM modes, artifact ownership, cache invalidation, 3+1
routing, fallback semantics, performance fields, and the evaluator/viewer
projection commands. State clearly that installing core or Fake mode does not
install/import SAM packages.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_sam31_smoke.py tests/test_cli.py tests/test_repository_policy.py -q`

Expected: PASS.

Run: `.venv/bin/python -m pytest -q --cov=las_repro --cov-report=term-missing`

Expected: PASS with branch coverage at least 85%.

- [ ] **Step 7: Commit**

```bash
git add scripts/sam31_smoke.py tests/test_sam31_smoke.py docs/deployment/sam31-runtime.md README.md .env.example
git commit -m "docs: add SAM3.1 deployment and smoke workflow"
```

---

### Task 15: Four-GPU acceptance, five-demo A/B, and PR evidence

**Files:**
- Create: `evaluation/results/sam31_2026-09-04/manifest.json`
- Create: `evaluation/results/sam31_2026-09-04/las-alignment-report.json`
- Create: `evaluation/results/sam31_2026-09-04/occlusion-review.json`
- Create: `docs/reports/2026-09-04-sam31-gpu-acceptance.md`
- Modify: `evaluation/viewer/data/demo-manifest.json`
- Create: `evaluation/viewer/data/hybrid/full_0001.json`
- Create: `evaluation/viewer/data/hybrid/full_0002.json`
- Create: `evaluation/viewer/data/hybrid/full_0004.json`
- Create: `evaluation/viewer/data/hybrid/full_0021.json`
- Create: `evaluation/viewer/data/hybrid/full_0024.json`
- Create: `evaluation/viewer/data/hybrid/assets/` (only digest-named bounded overlay PNGs referenced by the five projections)

**Interfaces:**
- Consumes: the completed implementation wheel, four-GPU server, five frozen videos, frozen English LAS references, Qwen-only results, and the evaluator.
- Produces: sanitized immutable acceptance evidence and a ready-to-review independent PR branch.

- [ ] **Step 1: Build and hash the implementation wheel**

Run: `uv build --offline --wheel`

Expected: one wheel under `dist/`; record `sha256sum dist/*.whl` before transfer.

- [ ] **Step 2: Verify local quality gates before deployment**

Run: `.venv/bin/python -m pytest -q --cov=las_repro --cov-report=term-missing`

Expected: all tests pass and branch coverage is at least 85%.

Run: `git diff --check origin/main...HEAD`

Expected: no output.

- [ ] **Step 3: Install the pinned SAM runtime and validate four idle GPUs**

Follow `docs/deployment/sam31-runtime.md`. Before starting workers, capture
sanitized `nvidia-smi` evidence showing four RTX 5090 devices, no unexpected
compute processes, and adequate free memory. Verify the SAM checkout and
checkpoint digests before loading.

- [ ] **Step 4: Run isolation smoke tests**

Start Qwen workers on 0, 1, and 2, and `cv-worker --provider sam31 --device 3`.
Run the existing Qwen GPU smoke on 0-2 and `scripts/sam31_smoke.py` on 3. Assert
process/device assignment, finite latency, valid artifact reload, and no model
loaded on the wrong GPU.

- [ ] **Step 5: Run the frozen five-demo cold-cache evaluation**

Submit `full_0001`, `full_0002`, `full_0024`, `full_0021`, and `full_0004` with
the same English query and fixed Qwen settings used by the existing report.
Save canonical Poll results, SQLite terminal rows, per-job metrics, CV manifests,
artifact digests, model identities, and configuration digest. Require `5/5`
structural completion and no video exceeding 12 minutes.

- [ ] **Step 6: Run the cache-hit verification**

Resubmit the same five byte-identical inputs and configuration. Assert every CV
job returns `cache_hit=true`, SAM provider invocation count stays zero, artifact
digests equal the cold run, and all tasks remain valid. Report cold versus cached
per-stage and total latency separately.

- [ ] **Step 7: Review every positive occlusion claim**

For each projected positive event, inspect its video interval plus overlay
keyframes and record exactly one verdict (`supported` or `unsupported`), reviewer
identifier, target, occluder, event type, start/end, and a concise visual reason.
The review file validator must prove every positive event appears exactly once
and no unknown event is reviewed.

- [ ] **Step 8: Generate the A/B report and enforce gates**

Run `scripts/evaluate_las_alignment.py` with the frozen reference manifest,
Qwen-only results, hybrid results, mapping config, and completed review file.
Require all gates from Task 12. If a gate fails, preserve the truthful failure
report and diagnose before changing prompts or thresholds; do not post-hoc edit
the LAS references or mapping.

- [ ] **Step 9: Publish the five-demo hybrid viewer data**

Upgrade each sample in `evaluation/viewer/data/demo-manifest.json` to the
version-2 `local_variants` shape while retaining the frozen Qwen-only path and
adding the matching hybrid path. Then run:

```bash
.venv/bin/python scripts/build_comparison_viewer_data.py \
  --input-dir outputs/five-demo/qwen-only \
  --output-dir evaluation/viewer/data/local \
  --hybrid-input-dir outputs/five-demo/qwen-sam31 \
  --hybrid-output-dir evaluation/viewer/data/hybrid \
  --artifact-root "$LAS_CV_CACHE_ROOT" \
  --review evaluation/results/sam31_2026-09-04/occlusion-review.json \
  --manifest evaluation/viewer/data/demo-manifest.json
```

Open all five samples through the local HTTP viewer. For each sample, switch
between Qwen-only and Qwen+SAM3.1, exercise every available layer, seek from one
event per branch, and open every exported overlay. Confirm LAS stays on the
left, the selected local variant stays on the right, video/timeline/event cards
share one clock, and no browser request escapes the repository origin.

- [ ] **Step 10: Verify recovery and memory stability**

Exercise one controlled CV-worker termination during a disposable task, restart
it, and verify lease recovery without duplicate publication. After all terminal
tasks, verify SQLite integrity, no pending/running jobs, no task-local artifacts
inside the persistent cache, and stable idle GPU memory.

- [ ] **Step 11: Write sanitized acceptance evidence**

The report includes wheel/source/checkpoint/config/reference/result/artifact
digests; job counts and terminal states; per-stage cold/cache timings; GPU peak
and idle memory; metric numerators/denominators; manual review counts; warnings;
and known limitations. Exclude credentials, absolute private paths, raw prompts,
request payloads, full masks, and unsanitized exceptions.

- [ ] **Step 12: Run final repository verification**

Run: `.venv/bin/python -m pytest -q --cov=las_repro --cov-report=term-missing`

Expected: PASS with branch coverage at least 85%.

Run: `node --test evaluation/viewer/tests/model.test.mjs && node --check evaluation/viewer/js/app.js`

Expected: all Node tests pass and syntax check exits 0.

Run: `git diff --check origin/main...HEAD`

Expected: no output.

Run: `git status --short`

Expected: only intentionally tracked acceptance files are present before commit.

- [ ] **Step 13: Commit acceptance evidence and viewer data**

```bash
git add evaluation/results/sam31_2026-09-04 docs/reports/2026-09-04-sam31-gpu-acceptance.md evaluation/viewer/data/demo-manifest.json evaluation/viewer/data/hybrid
git commit -m "test: record SAM3.1 alignment acceptance"
```

- [ ] **Step 14: Request code review before push and PR creation**

Invoke `superpowers:requesting-code-review`, resolve findings through verified
fix commits, rerun the final checks, then push
`feat/sam31-evidence-integration` and create a new PR targeting `main`. Confirm
the PR diff contains no PR #3 commits and no credentials.

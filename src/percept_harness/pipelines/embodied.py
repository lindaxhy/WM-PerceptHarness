"""Natural events with independent visual review and targeted SAM verification."""
import time
from types import SimpleNamespace

from ..cv.entities import EntityCandidate
from ..domain import InferenceJobSpec
from ..media import probe_video
from ..execution import wait_for_jobs
from .base import SafePipelineError
from .event_evidence import EvidenceOrchestrator, targeted_windows
from .event_prompts import VERSION, CONFIDENCE_VERSION, prompt
from .event_schemas import EvidenceReview, SamPromptReview, SpatialTarget, validate_stage, GroundedEvents, ConfidenceDraft, SAM_CONFIDENCE_THRESHOLD
from .output_validation import DEFAULT_OUTPUT_SCHEMAS

EMBODIED_PROMPT_VERSION = VERSION


class EmbodiedActionPipeline:
    def __init__(self, *, probe=probe_video, wait_jobs=wait_for_jobs, wait_timeout=300.0):
        self._probe, self._wait_jobs, self._wait_timeout = probe, wait_jobs, wait_timeout

    def run(self, task, context):
        if context.settings.cv_provider == 'disabled':
            raise SafePipelineError('Annotation requires SAM capability; disabled CV is not supported')
        media_path = context.media_path
        if media_path is None:
            raise SafePipelineError('Annotation requires a local original video')
        duration = self._probe(media_path).duration
        confidence_gated = context.settings.event_workflow == 'confidence-gated'
        context.metadata.update({'event_workflow': context.settings.event_workflow,
                                 'annotation_version': CONFIDENCE_VERSION if confidence_gated else VERSION, 'synthetic': context.settings.cv_provider == 'fake',
                                 'phases': [], 'evidence': [], 'unresolved_concerns': [],
                                 'sam_checks': [], 'stage_timings': []})
        audit = context.metadata
        evidence = []
        user_context = task.payload.get('task_context', {}).get('prompt_context')

        def stage(name, schema, *, draft=None, review=None, target=None, ordinal=0, visuals=None):
            supplied = evidence if visuals is None else visuals
            ctx = {'duration': duration, 'event_count': len(draft['events']) if draft else 0}
            if target is not None:
                ctx.update(window_start=target['start'], window_end=target['end'])
            payload = {'video_path': str(media_path), 'span': {'start': 0.0, 'end': duration}, 'fps': 2.0,
                'schema_name': schema.__name__, 'schema_context': ctx,
                'prompt': prompt(name, duration=duration, context=user_context,
                    evidence=[e.prompt_record for e in supplied], draft=draft, review=review,
                    target=target, unresolved=audit['unresolved_concerns']),
                'visual_evidence': [image.payload() for e in supplied for image in e.images]}
            started = time.monotonic()
            try:
                [job] = context.store.create_inference_jobs(task.task_id, [InferenceJobSpec(
                    stage=name, ordinal=ordinal, payload=payload, model_name=task.payload.get('model_name'))])
                [data] = self._wait_jobs(context.store, task.task_id, [job.job_id], self._wait_timeout)
                if DEFAULT_OUTPUT_SCHEMAS.failure_codes(schema.__name__, data) is not None:
                    raise SafePipelineError(f'Invalid structured output from {name}')
                result = validate_stage(schema, data, ctx)
                audit['phases'].append(name)
                return result
            finally:
                audit['stage_timings'].append({'stage': name, 'ordinal': ordinal, 'seconds': time.monotonic() - started})

        if confidence_gated:
            draft = stage('confidence_draft', ConfidenceDraft)
            audit['draft'] = draft
            assessed = ConfidenceDraft.model_validate(draft)
            questions = [e.uncertainty for e in assessed.events
                         if e.confidence <= SAM_CONFIDENCE_THRESHOLD]
            review = SimpleNamespace(concerns=questions, unresolved=[])
            review_data = {'source': 'initial_self_assessment',
                           'concerns': [q.model_dump(mode='json') for q in questions]}
            audit['confidence_gate'] = {'threshold': SAM_CONFIDENCE_THRESHOLD,
                'low_confidence_event_indices': [i for i, e in enumerate(assessed.events)
                                                if e.confidence <= SAM_CONFIDENCE_THRESHOLD]}
        else:
            draft = stage('grounded_draft', GroundedEvents)
            audit['draft'] = draft
            review_data = stage('evidence_review', EvidenceReview, draft=draft)
            review = EvidenceReview.model_validate(review_data)
            audit['review'] = review_data
        windows, omitted = targeted_windows(review, duration)
        audit['unresolved_concerns'].extend(review.unresolved)
        audit['unresolved_concerns'].extend(omitted)
        # A revised generation is not an independent certification of resolution.
        audit['unresolved_concerns'].extend(c.reason for c in review.concerns)
        if confidence_gated:
            audit['confidence_gate']['skipped_without_target'] = [
                {'event_indices': list(q.event_indices), 'kind': q.kind, 'reason': q.reason}
                for q in review.concerns if q.target is None]
        orchestrator = None
        calls, completed = 0, 0
        for index, window in enumerate(windows):
            target = SpatialTarget.model_validate(window['target'])
            check_record = {'concern_index': window['concern_index'], 'question': window['question'],
                            'window': [window['start'], window['end']], 'attempts': [], 'tracked': False}
            audit['sam_checks'].append(check_record)
            try:
                if orchestrator is None:
                    orchestrator = EvidenceOrchestrator(task, context, media_path, duration, self._wait_jobs)
                for attempt in range(2):
                    candidates = [EntityCandidate(name=target.description, aliases=(), role='other')]
                    ordinal = index * 3 + attempt + 1
                    started = time.monotonic()
                    calls += 1
                    try:
                        probe = orchestrator.run(candidates, ordinal=ordinal, window=window, target=target, mode='probe')
                    finally:
                        audit['stage_timings'].append({'stage': 'sam_probe', 'ordinal': ordinal, 'seconds': time.monotonic() - started})
                    audit['evidence'].append(probe.audit)
                    if not probe.images:
                        decision = {'accepted': False, 'reason': 'No visible target mask; tracking skipped.', 'correction': None}
                    else:
                        decision = stage('sam_prompt_review', SamPromptReview, draft=draft,
                                         target={**window, 'target': target.model_dump(mode='json'),
                                                 'actual_reference_time': probe.audit['reference_time']},
                                         ordinal=ordinal, visuals=[probe])
                    check_record['attempts'].append({'target': target.model_dump(mode='json'),
                                                     'probe_scope': probe.audit['scope'], **decision})
                    if decision['accepted']:
                        started = time.monotonic()
                        calls += 1
                        try:
                            tracked = orchestrator.run(candidates, ordinal=index * 3 + 3,
                                                       window=window, target=target, mode='track')
                        finally:
                            audit['stage_timings'].append({'stage': 'sam_track', 'ordinal': index * 3 + 3,
                                                         'seconds': time.monotonic() - started})
                        audit['evidence'].append(tracked.audit)
                        evidence.append(tracked)
                        check_record['tracked'] = True
                        completed += 1
                        break
                    if attempt == 0 and decision['correction'] is not None:
                        target = SpatialTarget.model_validate(decision['correction'])
                    else:
                        audit['unresolved_concerns'].append(f'SAM target rejected: {decision["reason"]}')
                        break
            except Exception as error:
                check_record['error'] = type(error).__name__
                audit['unresolved_concerns'].append(f'Targeted SAM unavailable for {window["question"]}: {type(error).__name__}')
        audit['recheck_budget'] = {'selected_windows': len(windows), 'completed_windows': completed,
                                 'sam_calls': calls, 'max_prompt_attempts_per_target': 2}
        if confidence_gated and not windows:
            audit['confidence_gate']['route'] = 'draft_only'
            return assessed.public_annotation()
        if confidence_gated:
            audit['confidence_gate']['route'] = 'sam_then_revision'
        return stage('grounded_revision', GroundedEvents, draft=draft, review=review_data)

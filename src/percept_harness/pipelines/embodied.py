"""Natural event annotation with mandatory SAM grounding and separate review."""
from ..cv.entities import EntityCandidate, normalize_entities
from ..domain import InferenceJobSpec
from ..execution import wait_for_jobs
from ..media import probe_video
from .base import SafePipelineError
from .event_evidence import EvidenceOrchestrator, merged_windows
from .event_prompts import VERSION, prompt
from .event_schemas import EntityNomination, EvidenceReview, validate_stage, GroundedEvents
from .output_validation import DEFAULT_OUTPUT_SCHEMAS

EMBODIED_PROMPT_VERSION = VERSION


class EmbodiedActionPipeline:
    """Keep the historical template name, expose only duration and events."""
    def __init__(self, *, probe=probe_video, wait_jobs=wait_for_jobs, wait_timeout=300.0):
        self._probe, self._wait_jobs, self._wait_timeout = probe, wait_jobs, wait_timeout

    def run(self, task, context):
        if context.settings.cv_provider == 'disabled':
            raise SafePipelineError('Annotation requires SAM; disabled CV is not supported')
        media_path = context.media_path
        if media_path is None:
            raise SafePipelineError('Annotation requires a local original video')
        duration = self._probe(media_path).duration
        context.metadata.update({'annotation_version': VERSION, 'synthetic': context.settings.cv_provider == 'fake',
                                 'phases': [], 'evidence': [], 'unresolved_concerns': []})
        audit = context.metadata
        evidence = []
        user_context = task.payload.get('task_context', {}).get('prompt_context')

        def stage(name, schema, *, draft=None, review=None):
            ctx = {'duration': duration, 'event_count': len(draft['events']) if draft else 0}
            payload = {'video_path': str(media_path), 'span': {'start': 0.0, 'end': duration}, 'fps': 2.0,
                'schema_name': schema.__name__, 'schema_context': ctx,
                'prompt': prompt(name, duration=duration, context=user_context,
                    evidence=[e.prompt_record for e in evidence], draft=draft, review=review,
                    unresolved=audit['unresolved_concerns']),
                'visual_evidence': [image.payload() for e in evidence for image in e.images]}
            [job] = context.store.create_inference_jobs(task.task_id, [InferenceJobSpec(
                stage=name, ordinal=0, payload=payload, model_name=task.payload.get('model_name'))])
            [data] = self._wait_jobs(context.store, task.task_id, [job.job_id], self._wait_timeout)
            if DEFAULT_OUTPUT_SCHEMAS.failure_codes(schema.__name__, data) is not None:
                raise SafePipelineError(f'Invalid structured output from {name}')
            result = validate_stage(schema, data, ctx)
            audit['phases'].append(name)
            return result

        nomination = EntityNomination.model_validate(stage('entity_nomination', EntityNomination))
        candidates = list(nomination.entities)
        orchestrator = EvidenceOrchestrator(task, context, media_path, duration, self._wait_jobs)
        baseline = orchestrator.run(candidates)
        evidence.append(baseline)
        audit['evidence'].append(baseline.audit)
        audit['phases'].append('baseline_sam')
        draft = stage('grounded_draft', GroundedEvents)
        review_data = stage('evidence_review', EvidenceReview, draft=draft)
        review = EvidenceReview.model_validate(review_data)
        audit['review'] = review_data
        windows, omitted = merged_windows(review, duration)
        audit['unresolved_concerns'].extend(review.unresolved)
        audit['unresolved_concerns'].extend(omitted)
        # There is deliberately no second reviewer: these remain audit concerns,
        # not assertions that another generation has proven them resolved.
        audit['unresolved_concerns'].extend(c.reason for c in review.concerns)
        for ordinal, window in enumerate(windows, 1):
            labels = list(dict.fromkeys(window['new_entities']))
            additions = [EntityCandidate(name=label, aliases=(), role='other') for label in labels[:64]]
            limit = context.settings.cv_entity_limit
            new = normalize_entities(additions, limit=limit)
            existing = normalize_entities(candidates, limit=limit)
            new_ids = {e.canonical_label for e in new.entities}
            combined = list(new.entities) + [e for e in existing.entities if e.canonical_label not in new_ids]
            if len(labels) > 64 or new.omitted_count or len(combined) > limit:
                audit['unresolved_concerns'].append(f'Local SAM recheck {ordinal}: entity budget limited coverage; new labels prioritized')
            local_candidates = [EntityCandidate(name=e.canonical_label, aliases=e.aliases, role=e.role) for e in combined[:limit]]
            try:
                local = orchestrator.run(local_candidates, ordinal=ordinal, window=window)
                evidence.append(local)
                audit['evidence'].append(local.audit)
            except Exception as error:
                # Baseline remains mandatory; bounded local failure is explicit
                # and the final revision is instructed to be conservative.
                audit['unresolved_concerns'].append(f'Local SAM recheck {ordinal} unavailable: {type(error).__name__}')
        audit['phases'].append('local_sam_rechecks')
        final = stage('grounded_revision', GroundedEvents, draft=draft, review=review_data)
        audit['recheck_budget'] = {'rounds': 1, 'requested_windows': len(windows), 'completed_windows': len(evidence)-1}
        return final

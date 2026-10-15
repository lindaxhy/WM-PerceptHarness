"""Natural events with independent visual review and targeted SAM verification."""
import time

from ..media import probe_video
from .base import SafePipelineError
from .event_evidence import EvidenceOrchestrator, targeted_windows, MAX_RECHECKS, MAX_RECHECK_SECONDS
from .event_prompts import VERSION, CONFIDENCE_VERSION, prompt
from .event_schemas import EvidenceReview, SamPromptReview, SpatialTarget, validate_stage, GroundedEvents, ConfidenceDraft, Concern, SAM_CONFIDENCE_THRESHOLD, SamTargetPlan
from .output_validation import DEFAULT_OUTPUT_SCHEMAS


class EmbodiedActionPipeline:
    def run(self, context, *, prompt_context=None):
        media_path = context.media_path
        duration = probe_video(media_path).duration
        confidence_gated = context.settings.event_workflow == 'confidence-gated'
        context.metadata.update({'event_workflow': context.settings.event_workflow,
                                 'annotation_version': CONFIDENCE_VERSION if confidence_gated else VERSION,
                                 'phases': [], 'evidence': [], 'unresolved_concerns': [],
                                 'sam_checks': [], 'stage_timings': []})
        audit = context.metadata
        evidence = []

        def stage(name, schema, *, draft=None, review=None, target=None, ordinal=0, visuals=None):
            supplied = evidence if visuals is None else visuals
            ctx = {'duration': duration, 'event_count': len(draft['events']) if draft else 0}
            if review is not None and 'concerns' in review:
                ctx['concern_ids'] = [c['concern_id'] for c in review['concerns'] if 'concern_id' in c]
            if target is not None:
                ctx.update(window_start=target['start'], window_end=target['end'])
            span = {'start': target['start'], 'end': target['end']} if name == 'sam_target_plan' else {'start': 0.0, 'end': duration}
            payload = {'video_path': str(media_path), 'span': span, 'fps': 2.0,
                'schema_name': schema.__name__, 'schema_context': ctx,
                'prompt': prompt(name, duration=duration, context=prompt_context,
                    evidence=[e.prompt_record for e in supplied], draft=draft, review=review,
                    target=target, unresolved=audit['unresolved_concerns']),
                'visual_evidence': [image.payload() for e in supplied for image in e.images]}
            started = time.monotonic()
            try:
                data = context.executor.execute(name, payload)
                if DEFAULT_OUTPUT_SCHEMAS.failure_codes(schema.__name__, data) is not None:
                    raise SafePipelineError(f'Invalid structured output from {name}')
                result = validate_stage(schema, data, ctx, discarded_targets=(
                    audit.setdefault('discarded_sam_targets', []) if schema is ConfidenceDraft else None))
                audit['phases'].append(name)
                return result
            finally:
                audit['stage_timings'].append({'stage': name, 'ordinal': ordinal, 'seconds': time.monotonic() - started})

        if confidence_gated:
            draft = stage('confidence_draft', ConfidenceDraft)
            audit['draft'] = draft
            assessed = ConfidenceDraft.model_validate(draft)
            concerns = [Concern.model_validate({**e.uncertainty.model_dump(mode='json'), 'event_indices': [i]})
                        for i, e in enumerate(assessed.events) if e.uncertainty is not None]
            review_data = {'source': 'initial_self_assessment',
                           'concerns': [q.model_dump(mode='json') for q in concerns]}
            audit['confidence_gate'] = {'threshold': SAM_CONFIDENCE_THRESHOLD,
                'policy': 'all_explicit_uncertainties',
                'uncertain_event_indices': [i for i, e in enumerate(assessed.events) if e.uncertainty is not None],
                'requires_revision': bool(concerns),
                'low_confidence_event_indices': [i for i, e in enumerate(assessed.events)
                                                if e.confidence <= SAM_CONFIDENCE_THRESHOLD]}
        else:
            draft = stage('grounded_draft', GroundedEvents)
            audit['draft'] = draft
            review_data = stage('evidence_review', EvidenceReview, draft=draft)
            review = EvidenceReview.model_validate(review_data)
            concerns = list(review.concerns)
            known = {c.reason for c in concerns}
            # Legacy free-text issues have no time range: use the full source
            # window rather than silently excluding them from target planning.
            concerns.extend(Concern(kind='description', reason=reason, start=0.0, end=duration,
                target=None, event_indices=()) for reason in review.unresolved if reason not in known)
            audit['review'] = review_data
            audit['unresolved_concerns'].extend(review.unresolved)
        audit['unresolved_concerns'].extend(c.reason for c in concerns)
        orchestrator = None
        calls, completed, selected = 0, 0, 0
        spent = 0.0

        def collect_evidence(next_concerns, candidate, source):
            nonlocal orchestrator, calls, completed, selected, spent
            windows, omitted = targeted_windows(next_concerns, duration,
                max_windows=MAX_RECHECKS-selected, max_seconds=MAX_RECHECK_SECONDS-spent)
            audit['unresolved_concerns'].extend(omitted)
            offset = selected
            selected += len(windows)
            spent += sum(w['end'] - w['start'] for w in windows)
            for local_index, window in enumerate(windows):
                index = offset + local_index
                check_record = {'concern_index': window['concern_index'], 'question': window['question'],
                                'window': [window['start'], window['end']], 'source': source, 'kind': window['kind'],
                                'attempts': [], 'tracked': False}
                audit['sam_checks'].append(check_record)
                try:
                    plan = stage('sam_target_plan', SamTargetPlan, draft=candidate, target=window, ordinal=index, visuals=[])
                    check_record['target_plan'] = plan
                    if plan['target'] is None:
                        check_record['status'] = 'no_reliable_target'
                        audit['unresolved_concerns'].append(f"SAM target unavailable for {window['question']}: {plan['reason']}")
                        continue
                    target = SpatialTarget.model_validate(plan['target'])
                    if orchestrator is None:
                        orchestrator = EvidenceOrchestrator(context, media_path, duration)
                    for attempt in range(2):
                        ordinal = index * 3 + attempt + 1
                        started = time.monotonic()
                        calls += 1
                        try:
                            probe = orchestrator.run(ordinal=ordinal, window=window, target=target, mode='probe')
                        finally:
                            audit['stage_timings'].append({'stage': 'sam_probe', 'ordinal': ordinal, 'seconds': time.monotonic() - started})
                        audit['evidence'].append(probe.audit)
                        if not probe.images:
                            decision = {'accepted': False, 'reason': 'No visible target mask; tracking skipped.', 'correction': None}
                        else:
                            decision = stage('sam_prompt_review', SamPromptReview, draft=candidate,
                                             target={**window, 'target': target.model_dump(mode='json'),
                                                     'actual_reference_time': probe.audit['reference_time']},
                                             ordinal=ordinal, visuals=[probe])
                        check_record['attempts'].append({'target': target.model_dump(mode='json'),
                                                         'probe_scope': probe.audit['scope'], **decision})
                        if decision['accepted']:
                            started = time.monotonic()
                            calls += 1
                            try:
                                tracked = orchestrator.run(ordinal=index * 3 + 3,
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
            audit['recheck_budget'] = {'selected_windows': selected, 'completed_windows': completed,
                'sam_calls': calls, 'selected_seconds': spent, 'max_windows': MAX_RECHECKS,
                'max_seconds': MAX_RECHECK_SECONDS, 'max_prompt_attempts_per_target': 2}

        collect_evidence(concerns, draft, 'initial_review')
        if confidence_gated and not concerns:
            audit['confidence_gate']['route'] = 'draft_only'
            audit['revision_validation'] = {'status': 'not_required', 'attempts': 0}
            return assessed.public_annotation()
        if confidence_gated:
            audit['confidence_gate']['route'] = 'sam_then_revision' if calls else 'revision_only'
        from .revision import resolve_review
        return resolve_review(stage, draft, review_data, concerns, audit, collect_evidence=collect_evidence)

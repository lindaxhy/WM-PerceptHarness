"""Bounded revision and independent verification of review concerns."""
from .event_schemas import ReviewedRevision, RevisionVerification, Concern

MAX_REVISION_ATTEMPTS = 2


def accepted_assessment(response, verdict):
    """A source limitation needs independent agreement, never a tool-error waiver."""
    left, right = response.get('assessment'), verdict.get('assessment')
    left = None if left and left['category'] == 'no_source_limitation' else left
    right = None if right and right['category'] == 'no_source_limitation' else right
    if left is None and right is None:
        return True
    if not left or not right or left['category'] != right['category']:
        return False
    return (right['category'] != 'pipeline_error'
            and left['event_index'] == right['event_index']
            and max(left['start'], right['start']) < min(left['end'], right['end']))


def resolve_review(stage, draft, review, concerns, audit, *, collect_evidence):
    ledger = []
    for concern in concerns:
        ledger.append({'concern_id': f'c{len(ledger):04d}', **concern.model_dump(mode='json')})
    reasons = {entry['reason'] for entry in ledger}
    for reason in audit['unresolved_concerns']:
        if reason not in reasons:
            ledger.append({'concern_id': f'c{len(ledger):04d}', 'reason': reason})
            reasons.add(reason)
    audit['initial_unresolved_concerns'] = list(audit['unresolved_concerns'])
    audit['revision_attempts'] = []
    original = draft
    feedback = None
    for attempt in range(MAX_REVISION_ATTEMPTS):
        audit['concern_ledger'] = list(ledger)
        revision_context = {'original_review': review, 'original_draft': original,
                            'concerns': ledger, 'previous_verification': feedback}
        revision = stage('grounded_revision', ReviewedRevision, draft=draft,
                         review=revision_context, ordinal=attempt)
        candidate = {key: revision[key] for key in ('duration', 'events')}
        verification = stage('revision_verification', RevisionVerification, draft=candidate,
            review={'concerns': ledger, 'assessment_required_ids': [r['concern_id']
                for r in revision['responses'] if r.get('assessment') is not None]}, ordinal=attempt)
        self_unresolved = {r['concern_id'] for r in revision['responses'] if r['action'] == 'unresolved'}
        responses = {r['concern_id']: r for r in revision['responses']}
        failed = []
        assessments = []
        for verdict in verification['verdicts']:
            agreed = accepted_assessment(responses[verdict['concern_id']], verdict)
            if not verdict['resolved'] or verdict['concern_id'] in self_unresolved or not agreed:
                failed.append({**verdict, 'reason': verdict['reason'] if agreed else
                    'Source assessment not independently agreed or a pipeline error remains: ' + verdict['reason']})
            elif verdict.get('assessment') and verdict['assessment']['category'] != 'no_source_limitation':
                assessments.append({'concern_id': verdict['concern_id'], **verdict['assessment']})
        audit['annotation_assessments'] = assessments
        audit['remaining_review'] = failed
        audit['revision_attempts'].append({'revision': revision, 'verification': verification})
        audit['unresolved_concerns'] = [f"{v['concern_id']}: {v['reason']}" for v in failed] + [issue['reason'] for issue in verification['new_issues']]
        if not failed and not verification['new_issues']:
            categories = {a['category'] for a in assessments}
            audit['annotation_outcome'] = ('completed_with_uncertainty' if 'visual_uncertainty' in categories else
                'completed_with_observable_defects' if categories else 'completed')
            audit['revision_validation'] = {'status': 'passed', 'attempts': attempt + 1,
                                            'verifier': 'separate_vlm_call'}
            return candidate
        feedback = {**verification, 'remaining_review': failed}
        draft = candidate
        if attempt + 1 < MAX_REVISION_ATTEMPTS:
            new_concerns = []
            for issue in verification['new_issues']:
                reason = issue['reason']
                if reason not in reasons:
                    ledger.append({'concern_id': f'c{len(ledger):04d}', **issue})
                    reasons.add(reason)
                    new_concerns.append(Concern.model_validate(issue))
            collect_evidence(new_concerns, candidate, 'revision_verification')
    audit['revision_validation'] = {'status': 'needs_review', 'attempts': MAX_REVISION_ATTEMPTS,
                                    'verifier': 'separate_vlm_call'}
    audit['annotation_outcome'] = 'needs_review'
    return candidate

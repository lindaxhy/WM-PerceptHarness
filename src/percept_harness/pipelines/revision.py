"""Bounded revision and independent verification of review concerns."""
from .event_schemas import ReviewedRevision, RevisionVerification, Concern

MAX_REVISION_ATTEMPTS = 2


def accepted_assessment(response, verdict):
    """Taxonomy differences are audit data; material pipeline errors still block."""
    left, right = response.get('assessment'), verdict.get('assessment')
    return not any(a and a['category'] == 'pipeline_error' for a in (left, right))


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
        disagreements = []
        for verdict in verification['verdicts']:
            response = responses[verdict['concern_id']]
            agreed = accepted_assessment(response, verdict)
            left, right = response.get('assessment'), verdict.get('assessment')
            if ((left or {}).get('category'), (left or {}).get('event_index')) != ((right or {}).get('category'), (right or {}).get('event_index')):
                disagreements.append({'concern_id': verdict['concern_id'],
                    'revision_assessment': left, 'verifier_assessment': right})
            if not verdict['resolved'] or verdict['concern_id'] in self_unresolved or not agreed:
                failed.append({**verdict, 'reason': verdict['reason'] if agreed else
                    'A material pipeline error remains: ' + verdict['reason']})
            else:
                observed = [a for a in (left, right) if a and a['category'] in {'visual_uncertainty', 'observable_defect'}]
                if observed:
                    assessments.append({'concern_id': verdict['concern_id'], **observed[-1],
                        'categories': sorted({a['category'] for a in observed}),
                        'revision_assessment': left, 'verifier_assessment': right,
                        'description_independently_verified': True})
        audit['annotation_assessments'] = assessments
        audit['assessment_disagreements'] = disagreements
        audit['remaining_review'] = failed
        audit['revision_attempts'].append({'revision': revision, 'verification': verification})
        audit['unresolved_concerns'] = [f"{v['concern_id']}: {v['reason']}" for v in failed] + [issue['reason'] for issue in verification['new_issues']]
        if not failed and not verification['new_issues']:
            categories = {category for a in assessments for category in a['categories']}
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

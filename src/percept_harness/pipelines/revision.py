"""Bounded revision and independent verification of review concerns."""
from .event_schemas import ReviewedRevision, RevisionVerification

MAX_REVISION_ATTEMPTS = 2


def resolve_review(stage, draft, review, concerns, audit):
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
            review={'original_draft': original, 'concerns': ledger}, ordinal=attempt)
        self_unresolved = {r['concern_id'] for r in revision['responses'] if r['action'] == 'unresolved'}
        failed = [v for v in verification['verdicts'] if not v['resolved'] or v['concern_id'] in self_unresolved]
        audit['revision_attempts'].append({'revision': revision, 'verification': verification})
        audit['unresolved_concerns'] = [f"{v['concern_id']}: {v['reason']}" for v in failed] + verification['new_issues']
        if not failed and not verification['new_issues']:
            audit['revision_validation'] = {'status': 'passed', 'attempts': attempt + 1,
                                            'verifier': 'separate_vlm_call'}
            return candidate
        feedback = verification
        draft = candidate
        if attempt + 1 < MAX_REVISION_ATTEMPTS:
            for reason in verification['new_issues']:
                if reason not in reasons:
                    ledger.append({'concern_id': f'c{len(ledger):04d}', 'reason': reason})
                    reasons.add(reason)
    audit['revision_validation'] = {'status': 'needs_review', 'attempts': MAX_REVISION_ATTEMPTS,
                                    'verifier': 'separate_vlm_call'}
    return candidate

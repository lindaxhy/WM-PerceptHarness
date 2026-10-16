import unittest
from percept_harness.pipelines.event_schemas import (
    Concern, EvidenceReview, RevisionVerification, ConfidenceDraft, validate_stage,
)
from percept_harness.pipelines.event_evidence import targeted_windows


def concern(reference):
    return dict(kind='object_location', reason='Is the object still visible?',
                start=6.0, end=8.5, event_indices=[0],
                target=dict(description='green tray', reference_time=reference,
                            bbox_xyxy=[0.1, 0.2, 0.4, 0.6]))


class ReferenceWindowTests(unittest.TestCase):
    def test_earlier_and_later_references_preserve_question(self):
        for reference, expected in [(0., (0., 9.)), (11., (5.5, 11.5))]:
            data={'motion_evidence':'Not applicable.', 'concerns':[concern(reference)], 'unresolved':[]}
            result=validate_stage(EvidenceReview,data,{'duration':20.625,'event_count':1})
            c=Concern.model_validate(result['concerns'][0])
            windows,omitted=targeted_windows([c],20.625)
            self.assertFalse(omitted)
            self.assertEqual((windows[0]['start'],windows[0]['end']),expected)
            self.assertEqual(windows[0]['concern_window'],[6.,8.5])
            self.assertEqual(windows[0]['target']['reference_time'],reference)

    def test_verification_and_confidence_allow_external_reference(self):
        c=concern(11.)
        validate_stage(RevisionVerification,{'verdicts':[],'new_issues':[c]},
                       {'duration':20.625,'event_count':1,'concern_ids':[]})
        uncertainty={k:v for k,v in c.items() if k!='event_indices'}
        validate_stage(ConfidenceDraft,{'duration':20.625,'events':[
            dict(start=6.,end=8.5,description='Tray moves.',confidence=0.5,uncertainty=uncertainty)]},
            {'duration':20.625})

    def test_outside_video_still_rejected(self):
        for reference in [-1.,21.]:
            with self.assertRaises(ValueError):
                validate_stage(EvidenceReview,{'motion_evidence':'Not applicable.',
                    'concerns':[concern(reference)],'unresolved':[]},
                    {'duration':20.625,'event_count':1})

    def test_budget_charges_expanded_window_and_clips_to_video(self):
        c=Concern.model_validate(concern(20.625))
        windows,omitted=targeted_windows([c],20.625,max_seconds=5.)
        self.assertFalse(windows)
        self.assertEqual(len(omitted),1)
        windows,_=targeted_windows([c],20.625)
        self.assertEqual(windows[0]['end'],20.625)

if __name__=='__main__': unittest.main()

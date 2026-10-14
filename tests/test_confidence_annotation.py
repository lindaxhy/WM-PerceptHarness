import unittest

def concern(kind='object_identity', target=True):
    return {'kind': kind, 'reason': 'Which pen is held by the hand?', 'event_indices': [0],
            'start': 1.0, 'end': 3.0,
            'target': {'description': 'red container', 'reference_time': 2.0,
                       'bbox_xyxy': [0.1, 0.1, 0.8, 0.8]} if target else None}

def draft(confidence=0.9, kind='object_identity', target=True):
    return {'duration': 4.0, 'events': [{'start': 0.0, 'end': 4.0,
        'description': 'A hand moves the red container.', 'confidence': confidence,
        'uncertainty': {k: v for k, v in concern(kind, target).items() if k != 'event_indices'} if confidence <= 0.6 else None}]}

class ContractTests(unittest.TestCase):
    def test_invalid_confidence_or_target_cannot_enter_pipeline(self):
        from percept_harness.pipelines.event_schemas import ConfidenceDraft, validate_stage
        for value in (-0.1, 1.1, float('nan'), True, '0.5'):
            data = draft(); data['events'][0]['confidence'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_stage(ConfidenceDraft, data, {'duration': 4.0})
        data = draft(0.4); data['events'][0]['uncertainty'] = None
        with self.assertRaises(ValueError): validate_stage(ConfidenceDraft, data, {'duration': 4.0})
        data = draft(0.4); data['events'][0]['end'] = 4.1
        with self.assertRaises(ValueError): validate_stage(ConfidenceDraft, data, {'duration': 4.0})

    def test_draft_response_schema_compiles_for_real_backend(self):
        from percept_harness.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
        contract = DEFAULT_OUTPUT_SCHEMAS.model_response_contract('ConfidenceDraft', {'duration': 4.0})
        self.assertEqual(contract.format()['name'], 'ConfidenceDraft')

    def test_submicrosecond_rounding_canonicalizes_nested_target(self):
        from percept_harness.pipelines.event_schemas import ConfidenceDraft, validate_stage
        data = draft(0.4); endpoint = 4.0000005
        data['duration'] = endpoint
        event = data['events'][0]; event['end'] = endpoint
        event['uncertainty']['end'] = endpoint
        event['uncertainty']['target']['reference_time'] = endpoint
        output = validate_stage(ConfidenceDraft, data, {'duration': 4.0})
        self.assertEqual(output['events'][0]['uncertainty']['end'], 4.0)
        self.assertEqual(output['events'][0]['uncertainty']['target']['reference_time'], 4.0)
        data['events'][0]['uncertainty']['end'] = 4.001
        with self.assertRaises(ValueError): validate_stage(ConfidenceDraft, data, {'duration': 4.0})

    def test_invalid_question_is_not_hidden_by_target_fallback(self):
        from percept_harness.pipelines.event_schemas import ConfidenceDraft, validate_stage
        data = draft(0.4)
        data['events'][0]['uncertainty']['end'] = 0.5
        with self.assertRaises(ValueError):
            validate_stage(ConfidenceDraft, data, {'duration': 4.0})

import json
import unittest


class ContractTests(unittest.TestCase):
    def test_response_contract_preserves_nullable_target_and_numeric_box_items(self):
        from percept_harness.pipelines.event_schemas import EvidenceReview
        from percept_harness.models.response_contract import compile_local_schema
        schema = compile_local_schema(EvidenceReview.model_json_schema())
        target = schema['properties']['concerns']['items']['properties']['target']['anyOf']
        self.assertEqual(target[1], {'type': 'null'})
        box = target[0]['properties']['bbox_xyxy']
        self.assertEqual(box['items'], {'type': 'number', 'minimum': 0.0, 'maximum': 1.0})
        self.assertEqual((box['minItems'], box['maxItems']), (4, 4))

    def test_submicrosecond_model_rounding_preserves_authoritative_clock(self):
        from percept_harness.pipelines.event_schemas import validate_stage, GroundedEvents
        actual = 9.840233333333334
        result = validate_stage(GroundedEvents, {'duration': 9.840233, 'events': [
            {'start': 0.0, 'end': 9.840234, 'description': 'Hand moves the magnet.'}]}, {'duration': actual})
        self.assertEqual(result['duration'], actual)
        self.assertEqual(result['events'][0]['end'], actual)
        with self.assertRaises(ValueError):
            validate_stage(GroundedEvents, {'duration': 9.84, 'events': []}, {'duration': actual})

    def test_strict_event_schema_rejects_categories_and_bad_times_but_allows_overlap(self):
        from percept_harness.pipelines import output_validation
        registry = output_validation.DEFAULT_OUTPUT_SCHEMAS
        valid = {'duration': 4.0, 'events': [{'start': 0.0, 'end': 3.0, 'description': 'Hand moves.'},
                                           {'start': 1.0, 'end': 4.0, 'description': 'Cup falls.'}]}
        self.assertEqual(registry.sanitize('GroundedEvents', valid, {'duration': 4.0}), valid)
        for patch in ({'start': True}, {'end': float('inf')}, {'description': ' '}, {'skill': 'move'}, {'end': 5.0}):
            value = json.loads(json.dumps(valid))
            value['events'][0].update(patch)
            self.assertIsNotNone(registry.failure_codes('GroundedEvents', registry.sanitize('GroundedEvents', value, {'duration': 4.0})))

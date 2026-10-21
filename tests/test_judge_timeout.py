import unittest
import httpx
from percept_harness.evaluation.events import OpenAIJudge, JudgeError

class JudgeTimeoutTests(unittest.TestCase):
    def test_read_budget_and_decision_contract(self):
        def handler(request):
            self.assertEqual(request.extensions['timeout'], {'connect':30.0,'read':180.0,'write':60.0,'pool':30.0})
            return httpx.Response(200,json={'choices':[{'message':{'content':'{"match":true,"reason":"same action"}'}}]})
        judge=OpenAIJudge('https://example.invalid','test','test',transport=httpx.MockTransport(handler))
        self.assertEqual(judge('reference','prediction'),{'match':True,'reason':'same action'})

    def test_custom_timeout_and_timeout_not_a_negative_match(self):
        def handler(request):
            self.assertEqual(request.extensions['timeout']['read'],120.0)
            raise httpx.ReadTimeout('timed out',request=request)
        judge=OpenAIJudge('https://example.invalid','test','test',read_timeout_seconds=120,transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(JudgeError,'ReadTimeout'):judge('reference','prediction')

    def test_invalid_timeout(self):
        for value in [0,-1,True,float('inf'),float('nan')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                OpenAIJudge('https://example.invalid','test','test',read_timeout_seconds=value)


class JudgeModeTests(unittest.TestCase):
    def capture(self, **options):
        bodies = []
        def handler(request):
            import json
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'message': {'content': '{"match":false,"reason":"different"}'}}]})
        OpenAIJudge('https://example.invalid/v1/', 'judge-model', 'key', transport=httpx.MockTransport(handler), **options)('instruction', 'observed')
        return bodies[0]

    def test_default_request_is_unchanged(self):
        self.assertEqual(self.capture(), {
            'model': 'judge-model', 'temperature': 0, 'response_format': {'type': 'json_object'}, 'messages': [
                {'role': 'system', 'content': 'Decide whether descriptions report the same observed action, object, destination, direction, and outcome. Ignore wording and timestamps. Return only JSON with strict boolean match and brief reason. Do not include chain of thought.'},
                {'role': 'user', 'content': '{"reference": "instruction", "prediction": "observed"}'}]})

    def test_instruction_mode_and_extra_body(self):
        from percept_harness.evaluation.events import JUDGE_PROMPTS
        body = self.capture(mode='instruction', extra_body={'thinking': {'type': 'disabled'}})
        self.assertEqual(body['messages'][0]['content'], JUDGE_PROMPTS['instruction'])
        self.assertTrue(body['messages'][0]['content'].startswith('The reference is the instruction'))
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertEqual(body['messages'][1]['content'], '{"reference": "instruction", "prediction": "observed"}')

    def test_invalid_mode_and_reserved_extra_body(self):
        for options in ({'mode': 'lenient'}, {'extra_body': {'messages': []}}, {'extra_body': {'temperature': 1}}, {'extra_body': []}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                OpenAIJudge('https://example.invalid', 'test', 'test', **options)

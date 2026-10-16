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

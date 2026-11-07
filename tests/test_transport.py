import asyncio
import json
from pathlib import Path
import tempfile
import unittest

import httpx2
from quant_evidence.ai import BudgetTransport, validate_grounding
from quant_evidence.budget import BudgetError, Ledger


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/"tmp"
        base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base)
        self.root=Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def request(self,**extra):
        payload={"model":"gpt-6-luna","max_output_tokens":1000,"input":"fixture",**extra}
        return httpx2.Request("POST","https://api.openai.com/v1/responses",json=payload,headers={"Authorization":"Bearer NOT_A_REAL_SECRET"})
    async def test_mock_http_usage_saved_without_headers(self):
        async def handler(request):
            return httpx2.Response(200,json={"id":"resp_fixture","service_tier":"default","usage":{"input_tokens":100,"output_tokens":200}})
        ledger=Ledger(self.root/"ledger.sqlite3")
        transport=BudgetTransport(ledger,"mock-only",self.root/"trace",httpx2.MockTransport(handler))
        await transport.handle_async_request(self.request())
        self.assertEqual(ledger.summary()["calls"][0]["status"],"settled")
        for p in (self.root/"trace").glob("*.json"):
            self.assertNotIn("NOT_A_REAL_SECRET",p.read_text())
    async def test_budget_failure_makes_no_network_call(self):
        calls=[]
        async def handler(request):
            calls.append(1)
            raise AssertionError("should not send")
        t=BudgetTransport(Ledger(self.root/"l.sqlite3",.00001),"mock",self.root/"t",httpx2.MockTransport(handler))
        with self.assertRaises(BudgetError):
            await t.handle_async_request(self.request())
        self.assertEqual(calls,[])
    async def test_lost_connection_retains_reservation(self):
        async def handler(request):
            raise httpx2.ReadTimeout("mock timeout")
        ledger=Ledger(self.root/"l.sqlite3")
        t=BudgetTransport(ledger,"mock",self.root/"t",httpx2.MockTransport(handler))
        with self.assertRaises(httpx2.ReadTimeout):
            await t.handle_async_request(self.request())
        self.assertEqual(ledger.summary()["calls"][0]["status"],"unknown")
        with self.assertRaises(BudgetError):
            await t.handle_async_request(self.request())
    async def test_redirect_does_not_follow_or_release_unknown(self):
        async def handler(request):
            return httpx2.Response(302,headers={"location":"https://example.com"})
        ledger=Ledger(self.root/"l.sqlite3")
        t=BudgetTransport(ledger,"mock",self.root/"t",httpx2.MockTransport(handler))
        response=await t.handle_async_request(self.request())
        self.assertEqual(response.status_code,302)
        self.assertEqual(ledger.summary()["calls"][0]["status"],"unknown")


class GroundingTests(unittest.TestCase):
    def test_wrong_or_unread_evidence_cannot_pass(self):
        review={"citations":["E1"],"findings":[],"numeric_claims":[{"evidence_id":"E1","path":"values.0","value":.5}]}
        self.assertTrue(validate_grounding(review,{"E1":{"values":[.5]}},["E1"])["passed"])
        self.assertFalse(validate_grounding(review,{"E1":{"values":[.7]}},["E1"])["passed"])
        self.assertFalse(validate_grounding(review,{"E1":{"values":[.5]}},[])["passed"])
    def test_nonfinite_and_boolean_numbers_are_not_measurements(self):
        for value in (float("nan"),True):
            review={"citations":["E1"],"findings":[],"numeric_claims":[{"evidence_id":"E1","path":"value","value":value}]}
            self.assertFalse(validate_grounding(review,{"E1":{"value":value}},["E1"])["passed"])


if __name__=="__main__":
    unittest.main()

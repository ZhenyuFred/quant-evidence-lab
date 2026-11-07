import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx2
from quant_evidence.ai import review_evidence
from quant_evidence.budget import BudgetError


class SDKFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_tool_loop_over_mock_http(self):
        parent=Path(__file__).resolve().parents[1]/"tmp"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            root=Path(folder)
            calls=[]
            async def handler(request):
                payload=json.loads(await request.aread())
                calls.append(payload)
                if len(calls)==1:
                    output=[{"type":"function_call","id":"fc_fixture","call_id":"call_fixture","name":"read_evidence","arguments":"{\"evidence_id\":\"metric\"}","status":"completed"}]
                else:
                    text=json.dumps({"verdict":"supported","summary":"Fixture only","findings":[],"citations":["metric"],"numeric_claims":[{"evidence_id":"metric","path":"value","value":.5}],"limitations":[]})
                    output=[{"type":"message","id":"msg_fixture","role":"assistant","status":"completed","content":[{"type":"output_text","text":text,"annotations":[]}]}]
                return httpx2.Response(200,json={"id":f"resp_{len(calls)}","object":"response","created_at":1,"status":"completed","model":"gpt-6-luna","output":output,"parallel_tool_calls":True,"tool_choice":"auto","tools":[],"usage":{"input_tokens":100,"output_tokens":200,"total_tokens":300,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens_details":{"reasoning_tokens":0}},"service_tier":"default"})
            with patch("quant_evidence.ai.load_key",return_value="sk-fixture-not-a-key"),patch("quant_evidence.ai.httpx2.AsyncHTTPTransport",return_value=httpx2.MockTransport(handler)):
                result=await review_evidence(root,"Check metric",{"metric":{"value":.5}},model="gpt-6-luna")
                cached=await review_evidence(root,"Check metric",{"metric":{"value":.5}},model="gpt-6-luna")
                self.assertEqual(cached,result)
            self.assertEqual(result["review"]["verdict"],"supported")
            self.assertEqual(result["trace"]["read_ids"],["metric"])
            self.assertEqual(len(calls),2)
            self.assertTrue(any(i.get("type")=="function_call_output" for i in calls[1]["input"]))
            self.assertEqual(calls[0]["max_output_tokens"],5000)
            self.assertFalse(calls[0]["store"])

    async def test_failed_agent_retains_cost_and_cannot_bill_same_task_again(self):
        parent=Path(__file__).resolve().parents[1]/"tmp"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            root=Path(folder)
            calls=[]
            async def handler(request):
                calls.append(1)
                return httpx2.Response(200,json={"id":"resp_failed_fixture","object":"response","created_at":1,"status":"completed","model":"gpt-6-luna",
                    "output":[{"type":"function_call","id":"fc_fixture","call_id":"call_fixture","name":"read_evidence","arguments":"{\"evidence_id\":\"metric\"}","status":"completed"}],
                    "parallel_tool_calls":True,"tool_choice":"auto","tools":[],"usage":{"input_tokens":100,"output_tokens":200,"total_tokens":300,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens_details":{"reasoning_tokens":0}},"service_tier":"default"})
            with patch("quant_evidence.ai.load_key",return_value="sk-fixture-not-a-key"),patch("quant_evidence.ai.httpx2.AsyncHTTPTransport",return_value=httpx2.MockTransport(handler)):
                from agents import MaxTurnsExceeded
                with self.assertRaises(MaxTurnsExceeded):
                    await review_evidence(root,"One turn only",{"metric":{"value":.5}},model="gpt-6-luna",max_turns=1)
                summary=json.loads((root/"outputs"/"budget-summary.json").read_text())
                self.assertGreater(summary["conservative_usage_usd"],0)
                with self.assertRaises(BudgetError):
                    await review_evidence(root,"One turn only",{"metric":{"value":.5}},model="gpt-6-luna",max_turns=1)
            self.assertEqual(len(calls),1)


if __name__=="__main__":
    unittest.main()

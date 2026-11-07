import tempfile
import unittest
import json
import sqlite3
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path
from quant_evidence.budget import BudgetError, Ledger, conservative_cost, reservation_for


def payload():
    return {"model": "gpt-6.1-sol", "input": "test", "max_output_tokens": 1000, "service_tier": "default"}


class BudgetTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1]/"tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.path = Path(self.temp.name)/"ledger.sqlite3"
    def tearDown(self):
        self.temp.cleanup()
    def test_reservation_blocks_overspend_before_network(self):
        ledger = Ledger(self.path, .001)
        with self.assertRaises(BudgetError):
            ledger.reserve("test", payload())
        self.assertEqual(ledger.summary()["calls"], [])
    def test_inflight_and_ambiguous_requests_block_second_writer(self):
        ledger = Ledger(self.path)
        call_id = ledger.reserve("test", payload())
        with self.assertRaises(BudgetError):
            Ledger(self.path).reserve("second process", payload())
        ledger.unknown(call_id, "timeout")
        with self.assertRaises(BudgetError):
            Ledger(self.path).reserve("retry", payload())
        self.assertGreater(ledger.summary()["retained_reservations_usd"], 0)
    def test_settlement_and_reasoning_not_double_counted(self):
        ledger = Ledger(self.path)
        call_id = ledger.reserve("test", payload())
        usage = {"input_tokens":100, "output_tokens":200,"input_tokens_details":{"cached_tokens":40},"output_tokens_details":{"reasoning_tokens":150}}
        ledger.settle(call_id, {"id":"resp_test","usage":usage})
        self.assertEqual(conservative_cost("gpt-6.1-sol", usage), 2304)
        self.assertEqual(ledger.summary()["conservative_usage_usd"], .002304)
        with self.assertRaises(BudgetError):
            ledger.settle(call_id, {"usage":usage})
    def test_no_implicit_budget_raise(self):
        Ledger(self.path, 2)
        with self.assertRaises(BudgetError):
            Ledger(self.path, 40)
    def test_unknown_usage_and_hosted_tools_fail_closed(self):
        ledger = Ledger(self.path)
        call_id = ledger.reserve("test", payload())
        with self.assertRaises(BudgetError):
            ledger.settle(call_id, {"id":"missing_usage"})
        for change in ({"tools":[{"type":"web_search"}]},{"service_tier":"fast"},{"stream":True},{"previous_response_id":"prior"},{"max_output_tokens":None},{"model":"unpriced"}):
            with self.assertRaises(BudgetError):
                reservation_for(payload() | change)

    def response(self, identifier="response_fixture", **changes):
        usage={"input_tokens":1000,"output_tokens":200,"total_tokens":1200,
               "input_tokens_details":{"cached_tokens":400,"cache_write_tokens":500},
               "output_tokens_details":{"reasoning_tokens":50}}
        usage.update(changes)
        return {"id":identifier,"service_tier":"default","usage":usage}

    def test_explicit_authorization_is_audited_and_default_reopens_current_cap(self):
        ledger=Ledger(self.path)
        with self.assertRaises(BudgetError):
            Ledger(self.path,40)
        change=ledger.authorize_ceiling(40,reason="User explicitly authorized using the original $40")
        self.assertTrue(change["changed"])
        self.assertEqual(Ledger(self.path).summary()["ceiling_usd"],40)
        self.assertEqual(ledger.summary()["adjustment_history"][0]["kind"],"authorize_ceiling")
        self.assertFalse(ledger.authorize_ceiling(40,reason="Same authorization")["changed"])
        self.assertEqual(len(ledger.summary()["adjustment_history"]),1)
        for value in (41,0,-1,True,float("nan")):
            with self.assertRaises(BudgetError):
                ledger.authorize_ceiling(value,reason="invalid fixture")
        with self.assertRaises(BudgetError):
            ledger.enable_precise_pricing(reason="")

    def test_default_reopen_also_preserves_a_small_test_ceiling(self):
        Ledger(self.path,.03)
        self.assertEqual(Ledger(self.path).summary()["ceiling_usd"],.03)

    def test_precise_settlement_uses_request_default_ttl_without_forging_usage(self):
        ledger=Ledger(self.path)
        ledger.enable_precise_pricing(reason="Verified published rates and default 30m caching")
        request=payload() | {"input":"PRIVATE_PROMPT_MUST_NOT_BE_STORED_IN_METADATA"}
        identifier=ledger.reserve("fixture",request)
        response=self.response()
        original=deepcopy(response["usage"])
        ledger.settle(identifier,response)
        summary=ledger.summary()
        row=summary["calls"][0]
        self.assertEqual(row["charged"],3490)
        self.assertEqual(row["original_conservative_charged"],5040)
        self.assertEqual(row["usage"],original)
        self.assertNotIn("cache_write_tokens_details",row["usage"]["input_tokens_details"])
        self.assertEqual(row["pricing_basis"],"exact_published_rates")
        self.assertEqual(summary["guard_accounted_usage_usd"],.00349)
        self.assertTrue(summary["all_settled_pricing_exact"])
        self.assertNotIn("PRIVATE_PROMPT_MUST_NOT_BE_STORED_IN_METADATA",json.dumps(row["pricing_metadata"]))

    def test_mixed_lifetimes_without_breakdown_remain_an_upper_bound(self):
        ledger=Ledger(self.path)
        ledger.enable_precise_pricing(reason="Precision with bounded fallback")
        request=payload() | {"prompt_cache_options":{"mode":"implicit","ttl":"30m"},
            "input":[{"role":"user","content":[{"type":"input_text","text":"fixture",
                                                   "prompt_cache_breakpoint":{"mode":"explicit","ttl":"12h"}}]}]}
        identifier=ledger.reserve("fixture",request)
        ledger.settle(identifier,self.response())
        row=ledger.summary()["calls"][0]
        self.assertEqual(row["charged"],4740)
        self.assertEqual(row["pricing_basis"],"token_rate_upper_bound")
        self.assertFalse(ledger.summary()["all_settled_pricing_exact"])
        with self.assertRaises(BudgetError):
            ledger.reprice_settled({identifier:{"request":request,"response":self.response()}},reason="Cannot prove mixed TTL")

    def test_pending_or_unknown_prevents_configuration_and_migration(self):
        ledger=Ledger(self.path)
        identifier=ledger.reserve("fixture",payload())
        for status in ("pending","unknown"):
            if status=="unknown":
                ledger.unknown(identifier,"fixture unknown response")
            with self.assertRaises(BudgetError):
                ledger.authorize_ceiling(40,reason="Cannot change with unresolved cost")
            with self.assertRaises(BudgetError):
                ledger.enable_precise_pricing(reason="Cannot change with unresolved cost")
            with self.assertRaises(BudgetError):
                ledger.set_max_inflight(3,reason="Cannot change with unresolved cost")
            with self.assertRaises(BudgetError):
                ledger.reprice_settled({identifier:{}},reason="Cannot migrate unresolved")
        self.assertGreater(ledger.summary()["retained_reservations_usd"],0)

    def test_precise_inconsistent_usage_retains_reservation_and_blocks_calls(self):
        ledger=Ledger(self.path)
        ledger.enable_precise_pricing(reason="fixture")
        identifier=ledger.reserve("fixture",payload())
        with self.assertRaises(BudgetError):
            ledger.settle(identifier,self.response(total_tokens=1201))
        row=ledger.summary()["calls"][0]
        self.assertEqual(row["status"],"unknown")
        self.assertIsNone(row["charged"])
        with self.assertRaises(BudgetError):
            ledger.reserve("must not send",payload())

    def test_precise_usage_cannot_contradict_requested_ttl(self):
        ledger=Ledger(self.path)
        ledger.enable_precise_pricing(reason="fixture")
        identifier=ledger.reserve("fixture",payload())
        response=self.response()
        response["usage"]["input_tokens_details"]["cache_write_tokens_details"]={"ttl_30m_tokens":0,"ttl_12h_tokens":500}
        with self.assertRaises(BudgetError):
            ledger.settle(identifier,response)
        self.assertEqual(ledger.summary()["calls"][0]["status"],"unknown")

    def test_atomic_historical_repricing_retains_original_and_full_audit(self):
        ledger=Ledger(self.path)
        request=payload()
        response=self.response()
        identifier=ledger.reserve("fixture",request)
        ledger.settle(identifier,response)
        prior_usage=ledger.summary()["calls"][0]["usage"]
        result=ledger.reprice_settled({identifier:{"request":request,"response":response}},reason="Trusted saved request and response checked")
        self.assertEqual(result["repriced_calls"],1)
        row=ledger.summary()["calls"][0]
        self.assertEqual((row["charged"],row["original_conservative_charged"]),(3490,5040))
        self.assertEqual(row["usage"],prior_usage)
        audit=ledger.summary()["adjustment_history"][0]
        self.assertEqual(audit["before"]["charged_microusd"],5040)
        self.assertEqual(audit["after"]["delta_microusd"],-1550)
        self.assertEqual(ledger.reprice_settled({identifier:{"request":request,"response":response}},reason="Idempotent verification")["repriced_calls"],0)

    def test_repricing_rejects_modified_request_or_response_and_rolls_back_all_rows(self):
        ledger=Ledger(self.path)
        request=payload()
        identifiers=[]
        for name in ("a","b"):
            identifier=ledger.reserve(name,request)
            ledger.settle(identifier,self.response(name))
            identifiers.append(identifier)
        before=ledger.summary()
        bad=self.response("not_the_saved_id")
        with self.assertRaises(BudgetError):
            ledger.reprice_settled({identifiers[0]:{"request":request,"response":self.response("a")},
                                   identifiers[1]:{"request":request,"response":bad}},reason="Atomic fixture")
        self.assertEqual(ledger.summary(),before)
        with self.assertRaises(BudgetError):
            ledger.reprice_settled({identifiers[0]:{"request":request | {"input":"modified"},"response":self.response("a")}},reason="Hash mismatch")
        changed=self.response("a")
        changed["usage"]["output_tokens"]+=1
        with self.assertRaises(BudgetError):
            ledger.reprice_settled({identifiers[0]:{"request":request,"response":changed}},reason="Usage mismatch")
        self.assertEqual(ledger.summary(),before)

    def test_schema_upgrade_preserves_an_old_ledger_and_can_reprice_it(self):
        request=payload()
        response=self.response()
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE calls (id TEXT PRIMARY KEY,task TEXT,model TEXT,reserved INTEGER NOT NULL,charged INTEGER,status TEXT,response_id TEXT,usage TEXT,created TEXT,note TEXT)")
            db.execute("CREATE TABLE settings (name TEXT PRIMARY KEY,value INTEGER)")
            db.execute("INSERT INTO settings VALUES ('ceiling',38000000)")
            db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)",("old","fixture",request["model"],100000,5040,"settled",response["id"],json.dumps(response["usage"]),"fixture_date",None))
        db.close()
        ledger=Ledger(self.path)
        self.assertEqual(len(ledger.summary()["calls"]),1)
        ledger.reprice_settled({"old":{"request":request,"response":response}},reason="Trusted legacy trace")
        row=ledger.summary()["calls"][0]
        self.assertEqual(row["charged"],3490)
        self.assertEqual(row["original_conservative_charged"],5040)

    def test_small_luna_request_preserves_fixed_padding_and_fits_small_remaining_balance(self):
        request={"model":"gpt-6-luna","input":"x","max_output_tokens":1,"service_tier":"default"}
        reservation=reservation_for(request)
        self.assertGreaterEqual(reservation,2048)
        self.assertLess(reservation,2200)
        ledger=Ledger(self.path,.0022)
        ledger.enable_precise_pricing(reason="Small final request")
        self.assertTrue(ledger.reserve("tiny",request))

    def test_parallel_reservations_are_atomic_and_third_is_budget_blocked(self):
        request=payload()
        reservation=reservation_for(request)
        ledger=Ledger(self.path,(reservation*2+reservation//2)/1e6)
        ledger.set_max_inflight(3,reason="Explicit three-way bounded concurrency")
        barrier=Barrier(3)
        def reserve_one(index):
            barrier.wait()
            try:
                return ledger.reserve(f"parallel-{index}",request)
            except BudgetError:
                return None
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(reserve_one,range(3)))
        self.assertEqual(sum(value is not None for value in results),2)
        summary=ledger.summary()
        self.assertEqual(summary["pending_request_count"],2)
        self.assertEqual(summary["retained_reservations_usd"],2*reservation/1e6)
        self.assertGreaterEqual(summary["available_usd"],0)
        self.assertEqual(summary["max_inflight"],3)

    def test_unknown_blocks_new_parallel_calls_but_existing_pending_can_settle(self):
        ledger=Ledger(self.path)
        ledger.set_max_inflight(3,reason="fixture concurrency")
        first=ledger.reserve("one",payload())
        second=ledger.reserve("two",payload())
        ledger.unknown(first,"fixture timeout")
        with self.assertRaises(BudgetError):
            ledger.reserve("three",payload())
        ledger.settle(second,self.response())
        summary=ledger.summary()
        self.assertEqual(summary["unknown_request_count"],1)
        self.assertEqual(summary["pending_request_count"],0)
        self.assertGreater(summary["retained_reservations_usd"],0)

    def test_max_inflight_changes_are_audited_and_limit_is_enforced(self):
        ledger=Ledger(self.path)
        self.assertEqual(ledger.summary()["max_inflight"],1)
        ledger.set_max_inflight(3,reason="User-authorized bounded parallel work")
        for index in range(3):
            ledger.reserve(str(index),payload())
        with self.assertRaises(BudgetError):
            ledger.reserve("four",payload())
        self.assertEqual(ledger.summary()["adjustment_history"][0]["kind"],"set_max_inflight")


if __name__ == "__main__":
    unittest.main()

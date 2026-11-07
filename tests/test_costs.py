from copy import deepcopy
import json
import unittest

from quant_evidence.budget import PRICES
from quant_evidence.costs import CostError, SHORT_CONTEXT_LIMIT, TOKEN_RATES, compute_usage_cost, request_pricing_metadata, summarize_costs


def usage(inp=1000, out=200, cached=0, writes=0, lifetimes=None):
    details = {"cached_tokens": cached, "cache_write_tokens": writes}
    if lifetimes is not None:
        details["cache_write_tokens_details"] = {"ttl_30m_tokens": lifetimes[0], "ttl_12h_tokens": lifetimes[1]}
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
            "input_tokens_details": details, "output_tokens_details": {"reasoning_tokens": 50 if out >= 50 else 0}}


def call(identifier="one", data=None, **kwargs):
    record = {"id": identifier, "model": "gpt-6-luna", "status": "settled", "reserved": 5000,
              "charged": 350, "usage": usage() if data is None else data, "response_id": "resp_" + identifier}
    record.update(kwargs)
    return record


class UsageCostTests(unittest.TestCase):
    def metadata(self, **changes):
        return request_pricing_metadata({"model":"gpt-6.1-sol","input":"fixture","max_output_tokens":1000,**changes})

    def test_request_default_lifetime_establishes_exact_price_without_mutating_usage(self):
        data=usage(1000,200,400,500)
        before=deepcopy(data)
        result=compute_usage_cost("gpt-6.1-sol",data,pricing_metadata=self.metadata())
        self.assertEqual(result["estimated_usd"],.00349)
        self.assertTrue(result["exact_for_published_token_rates"])
        self.assertTrue(result["cache_lifetime_from_request"])
        self.assertEqual(data,before)
        self.assertNotIn("cache_write_tokens_details",data["input_tokens_details"])

    def test_explicit_12h_and_mixed_breakpoints_are_priced_honestly(self):
        metadata=self.metadata(prompt_cache_options={"mode":"implicit","ttl":"12h"})
        result=compute_usage_cost("gpt-6.1-sol",usage(1000,200,400,500),pricing_metadata=metadata)
        self.assertTrue(result["exact_for_published_token_rates"])
        self.assertEqual(result["estimated_usd"],.00474)
        metadata=self.metadata(input=[{"role":"user","content":[{"type":"input_text","text":"fixture",
            "prompt_cache_breakpoint":{"mode":"explicit","ttl":"12h"}}]}])
        self.assertIsNone(metadata["cache_write_ttl"])
        self.assertFalse(compute_usage_cost("gpt-6.1-sol",usage(1000,200,400,500),pricing_metadata=metadata)["exact_for_published_token_rates"])

    def test_explicit_no_breakpoints_disables_cache_and_rejects_contradictory_usage(self):
        metadata=self.metadata(prompt_cache_options={"mode":"explicit"})
        self.assertTrue(metadata["cache_disabled"])
        result=compute_usage_cost("gpt-6.1-sol",{"input_tokens":1000,"output_tokens":200},pricing_metadata=metadata)
        self.assertTrue(result["exact_for_published_token_rates"])
        self.assertEqual(result["estimated_usd"],.004)
        with self.assertRaises(CostError):
            compute_usage_cost("gpt-6.1-sol",usage(1000,200,1,0),pricing_metadata=metadata)

    def test_request_metadata_rejects_unknown_policy_and_does_not_store_prompt(self):
        for extra in ({"service_tier":"fast"},{"previous_response_id":"r"},
                      {"prompt_cache_options":{"ttl":"24h"}},{"prompt_cache_retention":"24h"},
                      {"tools":[{"type":"web_search"}]}):
            with self.subTest(extra=extra),self.assertRaises(CostError):
                self.metadata(**extra)
        metadata=self.metadata(input="DO_NOT_STORE_THIS_PROMPT")
        self.assertNotIn("DO_NOT_STORE_THIS_PROMPT",json.dumps(metadata))
        self.assertEqual(len(metadata["request_sha256"]),64)

    def test_opaque_or_remote_context_cannot_claim_a_text_byte_bound(self):
        for item in ({"type":"input_image","image_url":"https://example.com/image"},
                     {"type":"reasoning","encrypted_content":"opaque"},
                     {"type":"item_reference","id":"r"}):
            self.assertFalse(self.metadata(input=[item])["bounded_text_request"])

    def test_response_lifetime_breakdown_must_match_trusted_request(self):
        with self.assertRaises(CostError):
            compute_usage_cost("gpt-6.1-sol",usage(1000,200,400,500,(0,500)),pricing_metadata=self.metadata())
    def test_ordinary_input_output_and_reasoning_are_not_double_counted(self):
        result = compute_usage_cost("gpt-6.1-sol", usage())
        self.assertEqual(result["estimated_usd"], 0.004)
        self.assertTrue(result["exact_for_published_token_rates"])
        data = usage()
        data["output_tokens_details"]["reasoning_tokens"] = 200
        self.assertEqual(compute_usage_cost("gpt-6.1-sol", data)["estimated_usd"], 0.004)

    def test_mixed_ttls_are_disjoint_and_replace_input_rate(self):
        data = usage(12000, 200, 4000, 7000, (3000, 4000))
        result = compute_usage_cost("gpt-6.1-sol", data)
        # 1000*2 + 4000*.1 + 3000*2.5 + 4000*5 + 200*10.
        self.assertEqual(result["estimated_usd"], 0.0319)
        self.assertEqual(result["token_counts"]["ordinary_input"], 1000)
        self.assertTrue(result["exact_for_published_token_rates"])

    def test_missing_lifetime_split_produces_interval_not_false_precision(self):
        result = compute_usage_cost("gpt-6.1-sol", usage(1000, 200, 400, 500))
        self.assertEqual(result["lower_bound_usd"], 0.00349)
        self.assertEqual(result["upper_bound_usd"], 0.00474)
        self.assertEqual(result["estimated_usd"], result["upper_bound_usd"])
        self.assertFalse(result["exact_for_published_token_rates"])
        self.assertIn("30m/12h", result["method"])

    def test_zero_writes_do_not_need_ttl_split(self):
        result = compute_usage_cost("gpt-6-luna", usage(1000, 200, 400, 0))
        self.assertEqual(result["estimated_usd"], 0.000164)
        self.assertTrue(result["exact_for_published_token_rates"])

    def test_missing_cached_count_is_not_assumed_zero(self):
        data = usage(1000, 0, 0, 0)
        del data["input_tokens_details"]["cached_tokens"]
        result = compute_usage_cost("gpt-6-luna", data)
        self.assertEqual(result["lower_bound_usd"], 0.00001)
        self.assertEqual(result["upper_bound_usd"], 0.0001)
        self.assertFalse(result["exact_for_published_token_rates"])

    def test_missing_write_count_is_not_assumed_zero(self):
        data = usage(1000, 0, 400, 0)
        del data["input_tokens_details"]["cache_write_tokens"]
        result = compute_usage_cost("gpt-6-luna", data)
        self.assertEqual(result["lower_bound_usd"], 0.000064)
        self.assertEqual(result["upper_bound_usd"], 0.000154)

    def test_all_missing_input_details_have_honest_bounds(self):
        result = compute_usage_cost("gpt-6-luna", {"input_tokens": 1000, "output_tokens": 0})
        self.assertEqual(result["lower_bound_usd"], 0.00001)
        self.assertEqual(result["upper_bound_usd"], 0.00025)
        self.assertFalse(result["exact_for_published_token_rates"])

    def test_complete_ttls_can_determine_omitted_write_total(self):
        data = usage(1000, 0, 0, 500, (200, 300))
        del data["input_tokens_details"]["cache_write_tokens"]
        result = compute_usage_cost("gpt-6-luna", data)
        self.assertEqual(result["estimated_usd"], 0.00015)
        self.assertTrue(result["exact_for_published_token_rates"])

    def test_zero_usage_and_all_cached_boundary(self):
        zero = compute_usage_cost("gpt-6-luna", {"input_tokens": 0, "output_tokens": 0})
        self.assertEqual(zero["estimated_usd"], 0)
        self.assertTrue(zero["exact_for_published_token_rates"])
        self.assertEqual(compute_usage_cost("gpt-6-astra", usage(1000, 0, 1000, 0))["estimated_usd"], .001)

    def test_decimal_submicro_costs_are_not_rounded_up_to_ledger_units(self):
        result = compute_usage_cost("gpt-6-luna", usage(1, 0, 0, 1, (1, 0)))
        self.assertEqual(result["estimated_usd"], 0.000000125)

    def test_cached_plus_written_cannot_exceed_input(self):
        with self.assertRaises(CostError):
            compute_usage_cost("gpt-6-luna", usage(1000, 0, 600, 500))

    def test_lifetime_sums_and_partial_breakdowns_are_rejected(self):
        with self.assertRaises(CostError):
            compute_usage_cost("gpt-6-luna", usage(1000, 0, 0, 500, (200, 301)))
        data = usage(1000, 0, 0, 500, (200, 300))
        del data["input_tokens_details"]["cache_write_tokens_details"]["ttl_12h_tokens"]
        with self.assertRaises(CostError):
            compute_usage_cost("gpt-6-luna", data)

    def test_invalid_counts_totals_and_reasoning_fail(self):
        for value in (-1, 0.5, True, "100", float("nan"), 2**63):
            data = usage()
            data["input_tokens"] = value
            with self.subTest(value=value), self.assertRaises(CostError):
                compute_usage_cost("gpt-6-luna", data)
        data = usage()
        data["total_tokens"] += 1
        with self.assertRaises(CostError):
            compute_usage_cost("gpt-6-luna", data)
        data = usage()
        data["output_tokens_details"]["reasoning_tokens"] = 201
        with self.assertRaises(CostError):
            compute_usage_cost("gpt-6-luna", data)

    def test_unknown_models_and_long_context_fail_closed(self):
        for model in ("made-up", "gpt-6.1-sol-unknown-snapshot", None, []):
            with self.subTest(model=model), self.assertRaises(CostError):
                compute_usage_cost(model, usage())
        self.assertTrue(compute_usage_cost("gpt-6-luna", usage(SHORT_CONTEXT_LIMIT, 0))["exact_for_published_token_rates"])
        with self.assertRaises(CostError):
            compute_usage_cost("gpt-6-luna", usage(SHORT_CONTEXT_LIMIT + 1, 0))

    def test_rate_tables_match_budget_upper_rates(self):
        for model, rates in TOKEN_RATES.items():
            for name, budget_name in (("input", "input"), ("cached", "cached"), ("write_12h", "write"), ("output", "output")):
                self.assertEqual(float(rates[name]), PRICES[model][budget_name])


class SummaryCostTests(unittest.TestCase):
    def test_real_smoke_usage_reconciles_without_rewriting_ledger_cost(self):
        records = [call("a", usage(749, 15), charged=196), call("b", usage(794, 22), charged=210),
                   call("c", usage(901, 182), charged=317)]
        source = {"calls": records, "conservative_usage_usd": .000723, "retained_reservations_usd": 0}
        before = deepcopy(source)
        result = summarize_costs(source)
        self.assertEqual(result["estimated_usd"], .0003539)
        self.assertEqual(result["conservative_budget_usage_usd"], .000723)
        self.assertTrue(result["exact_for_published_token_rates"])
        self.assertFalse(result["account_balance_verified"])
        self.assertFalse(result["platform_invoice_verified"])
        self.assertEqual(source, before)
        json.dumps(result, allow_nan=False)

    def test_bounded_request_makes_whole_summary_bounded(self):
        result = summarize_costs({"calls": [call(data=usage(1000, 0, 0, 1000))]})
        self.assertEqual(result["lower_bound_usd"], .000125)
        self.assertEqual(result["upper_bound_usd"], .00025)
        self.assertFalse(result["exact_for_published_token_rates"])

    def test_missing_usage_never_counts_as_free_or_as_verified_reservation_cost(self):
        for status in ("pending", "unknown", "settled"):
            missing = call("unknown", status=status, charged=None, usage=None, response_id=None)
            result = summarize_costs({"calls": [call(), missing]})
            self.assertIsNone(result["estimated_usd"])
            self.assertIsNone(result["upper_bound_usd"])
            self.assertFalse(result["exact_for_published_token_rates"])
            self.assertFalse(result["complete_usage_coverage"])
            self.assertEqual(result["lower_bound_usd"], .0002)
            self.assertEqual(result["reservation_envelope_usd"], .0052)
            self.assertFalse(result["reservation_envelope_is_billing_bound"])

    def test_unknown_with_usage_is_still_unreconciled(self):
        result = summarize_costs({"calls": [call(status="unknown")]})
        self.assertTrue(result["complete_usage_coverage"])
        self.assertIsNone(result["estimated_usd"])
        self.assertEqual(result["observed_usage_cost"]["estimated_usd"], .0002)
        self.assertFalse(result["exact_for_published_token_rates"])

    def test_explicit_rejection_costs_zero_but_contradictory_rejection_fails(self):
        result = summarize_costs({"calls": [call(status="rejected", charged=0, usage=None)]})
        self.assertEqual(result["estimated_usd"], 0)
        self.assertEqual(result["rejected_request_count"], 1)
        self.assertTrue(result["exact_for_published_token_rates"])
        with self.assertRaises(CostError):
            summarize_costs({"calls": [call(status="rejected", charged=0)]})

    def test_duplicate_call_or_response_ids_fail_instead_of_double_counting(self):
        for records in ([call(), call()], [call("a"), call("b", response_id="resp_a")]):
            with self.assertRaises(CostError):
                summarize_costs({"calls": records})

    def test_empty_summary_and_invalid_rows(self):
        result = summarize_costs({"calls": []})
        self.assertEqual(result["estimated_usd"], 0)
        self.assertEqual(result["request_count"], 0)
        for bad in ({}, {"calls": [None]}, {"calls": [call(status="invented")]},
                    {"calls": [call(model="unpriced")]}):
            with self.subTest(bad=bad), self.assertRaises(CostError):
                summarize_costs(bad)


if __name__ == "__main__":
    unittest.main()

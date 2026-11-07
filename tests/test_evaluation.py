"""Offline checks of label isolation and adversarial grading failure modes."""

from copy import deepcopy
import json
import math
import unittest

from quant_evidence.evaluation import BENCHMARK_INFO, ISSUE_CODES, REVIEW_SCHEMA, build_cases, grade


def research_fixture():
    metrics = {
        "start": "2015-01", "end": "2026-09", "n_months": 141,
        "cagr": 0.08123, "ann_vol": 0.17345, "max_drawdown": -0.34987,
        "sharpe": 0.51, "total_cost": 0.071, "total_return": 1.56,
    }
    return {
        "protocol": {"splits": {"train": {"start": "1970-01", "end": "1999-12"},
                                 "validation": {"start": "2000-01", "end": "2014-12"},
                                 "holdout": {"start": "2015-01", "end": "2026-09"}}},
        "manifest": {"sources": {"industry": {
            "url": "https://mba.tuck.dartmouth.edu/example.zip", "sha256": "a" * 64,
        }}, "limitations": ["Historical source data may be revised."]},
        "selection": {"selected_name": "candidate_iv", "comparator_name": "candidate_equal",
                      "selection_end": "2014-12", "selected_using": "validation_only"},
        "results": {"candidate_iv": {"spec": {"name": "candidate_iv", "cost_bps": 10.0},
                                      "metrics": {split: deepcopy(metrics)
                                                  for split in ("train", "validation", "holdout")}}},
    }


def labelled_review(case):
    """A private test oracle. Never exposed by the project evidence tools."""
    return {
        "verdict": case["expected_verdict"],
        "summary": "The narrowly scoped claim has been checked against the supplied records.",
        "findings": [{"code": code, "detail": "The evidence contradicts the scoped claim.",
                      "evidence_ids": list(case["evidence"])} for code in case["expected_issues"]],
        "citations": list(case["evidence"]),
        "numeric_claims": [{"evidence_id": eid, "path": path,
                            "value": case["evidence"][eid]["metrics"][path.split(".")[-1]]}
                           for eid, path in case["required_numeric_paths"]],
        "limitations": ["This controlled diagnostic is not external validation."],
    }


class CaseConstructionTests(unittest.TestCase):
    def setUp(self):
        self.research = research_fixture()
        self.cases = build_cases(self.research)

    def test_balanced_24_case_controlled_benchmark(self):
        self.assertEqual(len(self.cases), 24)
        self.assertEqual(sum(not c["expected_issues"] for c in self.cases), 3)
        for issue in ISSUE_CODES:
            self.assertEqual(sum(c["expected_issues"] == [issue] for c in self.cases), 3)
        self.assertFalse(BENCHMARK_INFO["independent_external_validation"])
        self.assertEqual(len({case["case_id"] for case in self.cases}), 24)

    def test_model_visible_payload_contains_no_private_labels(self):
        for case in self.cases:
            visible = json.dumps({"question": case["question"], "evidence": case["evidence"]})
            for private in ("expected_verdict", "expected_issues", "required_numeric_paths", "mutation"):
                self.assertNotIn(private, visible)
            for expected_code in case["expected_issues"]:
                self.assertNotIn(expected_code, visible)
            self.assertRegex(case["case_id"], r"^case_\d{3}$")

    def test_real_metrics_retained_without_changing_input(self):
        self.assertEqual(self.research, research_fixture())
        for case, metric in zip(self.cases[:3], ("cagr", "ann_vol", "max_drawdown")):
            self.assertEqual(case["evidence"]["E1"]["metrics"][metric],
                             self.research["results"]["candidate_iv"]["metrics"]["holdout"][metric])
        json.dumps(self.cases, allow_nan=False)

    def test_cases_are_deeply_isolated_and_deterministic(self):
        self.assertEqual(self.cases, build_cases(self.research))
        self.cases[0]["evidence"]["E1"]["metrics"]["cagr"] = 999
        self.assertEqual(build_cases(self.research)[0]["evidence"]["E1"]["metrics"]["cagr"], 0.08123)

    def test_missing_real_metrics_fail_without_fabricating_data(self):
        for missing in ("cagr", "ann_vol", "max_drawdown"):
            bad = deepcopy(self.research)
            del bad["results"]["candidate_iv"]["metrics"]["holdout"][missing]
            with self.assertRaises(ValueError):
                build_cases(bad)

    def test_invalid_source_selection_and_fee_fail(self):
        bad = deepcopy(self.research)
        bad["selection"]["selection_end"] = "2015-01"
        with self.assertRaises(ValueError):
            build_cases(bad)
        bad = deepcopy(self.research)
        bad["results"]["candidate_iv"]["spec"]["cost_bps"] = 0
        with self.assertRaises(ValueError):
            build_cases(bad)


class GradingTests(unittest.TestCase):
    def setUp(self):
        self.cases = build_cases(research_fixture())
        self.case = self.cases[0]
        self.review = labelled_review(self.case)

    def test_all_private_oracle_outputs_pass_with_actual_reads(self):
        for case in self.cases:
            with self.subTest(case=case["case_id"]):
                result = grade(labelled_review(case), case, {"read_ids": list(case["evidence"])})
                self.assertTrue(result["passed"], result)
                self.assertTrue(result["trace_checked"])

    def test_no_citations_never_pass_even_when_labels_are_correct(self):
        for case in (self.case, self.cases[-1]):
            review = labelled_review(case)
            review["citations"] = []
            self.assertFalse(grade(review, case)["passed"])
            self.assertFalse(grade(review, case)["valid_citations"])

    def test_fabricated_numeric_value_fails(self):
        self.review["numeric_claims"][0]["value"] += 0.01
        result = grade(self.review, self.case)
        self.assertFalse(result["numeric_grounding"]["passed"])
        self.assertFalse(result["passed"])

    def test_absolute_tolerance_and_finite_numeric_type(self):
        self.review["numeric_claims"][0]["value"] += 0.0000005
        self.assertTrue(grade(self.review, self.case)["passed"])
        for value in (math.nan, math.inf, -math.inf, True, "0.08123", 10**400):
            with self.subTest(value=value):
                review = labelled_review(self.case)
                review["numeric_claims"][0]["value"] = value
                self.assertFalse(grade(review, self.case)["passed"])

    def test_nonfinite_or_boolean_evidence_is_not_numeric(self):
        for value in (math.nan, math.inf, True):
            case = deepcopy(self.case)
            case["evidence"]["E1"]["metrics"]["cagr"] = value
            self.assertFalse(grade(self.review, case)["passed"])

    def test_missing_numeric_claim_cannot_pass_numeric_question(self):
        self.review["numeric_claims"] = []
        self.assertFalse(grade(self.review, self.case)["passed"])

    def test_bad_and_unknown_numeric_paths_fail(self):
        for path in ("metrics.no_such_stat", "metrics..cagr", "", "metrics.start"):
            self.review["numeric_claims"][0]["path"] = path
            self.assertFalse(grade(self.review, self.case)["passed"])

    def test_unknown_citation_fails(self):
        self.review["citations"] = ["E999"]
        self.assertFalse(grade(self.review, self.case)["valid_citations"])

    def test_correct_claim_without_read_trace_fails_when_trace_given(self):
        result = grade(self.review, self.case, {"read_ids": []})
        self.assertFalse(result["passed"])
        self.assertFalse(result["valid_citations"])
        self.assertTrue(grade(self.review, self.case, {"read_ids": ["E1"]})["passed"])

    def test_trace_checks_finding_and_numeric_references_not_only_citations(self):
        case = self.cases[3]
        review = labelled_review(case)
        review["citations"] = ["E1"]
        self.assertFalse(grade(review, case, {"read_ids": ["E1"]})["passed"])
        case = deepcopy(self.case)
        case["evidence"]["E2"] = {"value": 1.0}
        review = labelled_review(case)
        review["citations"] = ["E1"]
        review["numeric_claims"].append({"evidence_id": "E2", "path": "value", "value": 1.0})
        self.assertFalse(grade(review, case, {"read_ids": ["E1"]})["passed"])

    def test_empty_and_malformed_outputs_fail_without_crashing(self):
        bad_outputs = [None, "", [], {}, {"verdict": "supported"}]
        for field, value in (("summary", []), ("findings", None), ("numeric_claims", 7),
                             ("limitations", "none"), ("citations", {"E1": True})):
            review = labelled_review(self.case)
            review[field] = value
            bad_outputs.append(review)
        for output in bad_outputs:
            with self.subTest(output=output):
                self.assertFalse(grade(output, self.case)["passed"])

    def test_wrong_category_and_extra_findings_fail(self):
        case = self.cases[3]
        review = labelled_review(case)
        review["findings"][0]["code"] = "cost_omitted"
        result = grade(review, case)
        self.assertFalse(result["exact_issue_match"])
        self.assertEqual(result["issue_precision"], 0)
        self.assertEqual(result["issue_recall"], 0)
        review = labelled_review(case)
        review["findings"].append({"code": "numeric_mismatch", "detail": "Spurious extra finding.",
                                   "evidence_ids": ["E1"]})
        result = grade(review, case)
        self.assertEqual(result["issue_precision"], 0.5)
        self.assertEqual(result["issue_recall"], 1)
        self.assertFalse(result["passed"])

    def test_duplicate_unknown_codes_and_uncited_findings_fail(self):
        case = self.cases[3]
        review = labelled_review(case)
        review["findings"].append(deepcopy(review["findings"][0]))
        self.assertFalse(grade(review, case)["passed"])
        review = labelled_review(case)
        review["findings"][0]["code"] = "made_up_category"
        self.assertFalse(grade(review, case)["passed"])
        review = labelled_review(case)
        review["findings"][0]["evidence_ids"] = []
        self.assertFalse(grade(review, case)["passed"])

    def test_wrong_verdict_fails_with_correct_issue(self):
        case = self.cases[3]
        review = labelled_review(case)
        review["verdict"] = "supported"
        result = grade(review, case)
        self.assertTrue(result["exact_issue_match"])
        self.assertFalse(result["verdict_correct"])
        self.assertFalse(result["passed"])

    def test_schema_extras_are_not_accepted(self):
        self.review["expected_verdict"] = "supported"
        self.assertFalse(grade(self.review, self.case)["schema_valid"])
        self.assertFalse(REVIEW_SCHEMA["additionalProperties"])

    def test_array_numeric_path_is_supported(self):
        case = deepcopy(self.case)
        case["evidence"]["E1"]["samples"] = [{"value": 3.0}]
        self.review["numeric_claims"].append({"evidence_id": "E1", "path": "samples.0.value", "value": 3})
        self.assertTrue(grade(self.review, case)["passed"])


if __name__ == "__main__":
    unittest.main()

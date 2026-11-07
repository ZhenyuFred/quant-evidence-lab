import unittest
import json
from argparse import Namespace
from pathlib import Path
import tempfile
from copy import deepcopy
from unittest.mock import AsyncMock, patch
from quant_evidence.cli import completed_review_receipt, save_cost_analysis, study_evidence, summarize_evaluation
from quant_evidence.research import fingerprint, sha


def fixture_case():
    return {"case_id":"case_001", "question":"Verify the stored value.", "evidence":{"E1":{"value":1}},
            "expected_verdict":"supported", "expected_issues":[], "required_numeric_paths":[["E1","value"]]}


def fixture_result(model="m1", mode="tools"):
    task=f"evaluation/{model}/{mode}/case_001"
    return {"model":model,"mode":mode,"task":task,"status":"completed",
            "review":{"verdict":"supported","summary":"The stored value is supported.","findings":[],
                      "citations":["E1"],"numeric_claims":[{"evidence_id":"E1","path":"value","value":1.0}],
                      "limitations":["Local test fixture only."]},
            "trace":{"read_ids":["E1"],"call_ids":["fixture-call"]},"grounding":{"passed":True,"errors":[]}}


def write_fixture_receipt(root, case=None, result=None):
    from quant_evidence.ai import INSTRUCTIONS, Review
    case = fixture_case() if case is None else case
    result = fixture_result() if result is None else result
    request_hash=fingerprint({"question":case["question"],"evidence":case["evidence"],
                              "model":result["model"],"task":result["task"],"mode":result["mode"],
                              "max_turns":6,"max_tokens":5000,"instructions":INSTRUCTIONS,
                              "schema":Review.model_json_schema()})
    path=root/"outputs"/"private"/"runs"/f"{request_hash}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    receipt={"request_hash":request_hash,"task":result["task"],"status":"completed","result":result}
    path.write_text(json.dumps(receipt),encoding="utf-8")
    return path, receipt


def write_fixture_archive(root, case, status):
    from quant_evidence.ai import INSTRUCTIONS
    study={"fixture":True}
    signature=fingerprint({"study":study,"instructions":INSTRUCTIONS,"cases":[case],
                           "runtime_sha256":sha(root/"quant_evidence"/"ai.py"),
                           "grader_sha256":sha(root/"quant_evidence"/"evaluation.py")})
    path=root/"outputs"/"evaluation"/"runs"/f"{signature}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    report={"signature":signature,"case_count":1,"records":[{
        "key":"m1/tools/case_001","case_id":"case_001","model":"m1","mode":"tools",
        "status":status,"score":None,"error_type":"FixturePostprocessingCrash"}],"status":status}
    path.write_text(json.dumps(report),encoding="utf-8")
    return path


class CompletedReceiptTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/"tmp"
        parent.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent)
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def read_fixture(self, case=None):
        case=fixture_case() if case is None else case
        return completed_review_receipt(self.root,case["question"],case["evidence"],model="m1",
                                        mode="tools",task="evaluation/m1/tools/case_001")

    def test_completed_matching_receipt_is_read_without_model_or_key_access(self):
        path, receipt=write_fixture_receipt(self.root)
        original=path.read_bytes()
        with patch("quant_evidence.ai.review_evidence",new_callable=AsyncMock) as model, patch("quant_evidence.ai.load_key") as key:
            self.assertEqual(self.read_fixture(),receipt["result"])
            model.assert_not_called()
            key.assert_not_called()
        self.assertEqual(path.read_bytes(),original)

    def test_absent_or_different_request_receipt_never_replays(self):
        self.assertIsNone(self.read_fixture())
        write_fixture_receipt(self.root)
        changed=fixture_case()
        changed["question"]="Different question"
        self.assertIsNone(self.read_fixture(changed))
        changed=fixture_case()
        changed["evidence"]["E1"]["value"]=2
        self.assertIsNone(self.read_fixture(changed))

    def test_unfinished_mismatched_and_malformed_receipts_are_refused(self):
        path, valid=write_fixture_receipt(self.root)
        variants=[]
        for status in ("running","interrupted"):
            variants.append(dict(valid,status=status))
        variants.append(dict(valid,request_hash="mismatch"))
        variants.append(dict(valid,task="different_task"))
        for field,value in (("model","different_model"),("mode","full_context"),
                            ("task","different_task"),("status","running")):
            altered=deepcopy(valid)
            altered["result"][field]=value
            variants.append(altered)
        altered=deepcopy(valid)
        del altered["result"]["review"]["summary"]
        variants.append(altered)
        altered=deepcopy(valid)
        altered["result"]["trace"]["read_ids"]="E1"
        variants.append(altered)
        altered=deepcopy(valid)
        altered["result"]["trace"]["call_ids"]=[]
        variants.append(altered)
        for variant in variants:
            path.write_text(json.dumps(variant),encoding="utf-8")
            self.assertIsNone(self.read_fixture())
        path.write_text("{incomplete JSON",encoding="utf-8")
        self.assertIsNone(self.read_fixture())

    def test_completed_output_with_grounding_errors_is_recovered_for_honest_grading(self):
        result=fixture_result()
        result["status"]="completed_with_grounding_errors"
        result["review"]["numeric_claims"][0]["value"]=99
        write_fixture_receipt(self.root,result=result)
        recovered=self.read_fixture()
        from quant_evidence.evaluation import grade
        self.assertIsNotNone(recovered)
        self.assertFalse(grade(recovered["review"],fixture_case(),recovered["trace"])["passed"])

    def test_cost_analysis_is_bound_to_entire_budget_snapshot(self):
        record={"calls":[],"conservative_usage_usd":0,"retained_reservations_usd":0}
        with patch("quant_evidence.cli.dump"):
            analysis=save_cost_analysis(record)
        self.assertEqual(analysis["budget_snapshot_sha256"],fingerprint(record))


class EvaluationRecoveryTests(unittest.TestCase):
    def record(self,key,status="completed"):
        return {"key":key,"model":"m","mode":"tools","status":status,
                "score":{"passed":True,"verdict_correct":True,"valid_citations":True} if status=="completed" else None}
    def test_extending_completed_grid_returns_to_running(self):
        report={"records":[self.record("a")],"requested_keys":["a","b"],"status":"completed_requested_grid"}
        summarize_evaluation(report)
        self.assertEqual(report["status"],"running")
        self.assertFalse(report["all_requested_cases_passed"])
        self.assertEqual(report["requested_summary"]["pending"],1)
    def test_failed_attempt_stays_in_denominator(self):
        report={"records":[self.record("a"),self.record("b","failed")],"requested_keys":["a","b"]}
        summarize_evaluation(report)
        self.assertEqual(report["groups"]["m/tools"]["n"],2)
        self.assertEqual(report["groups"]["m/tools"]["passed"],1)
        self.assertEqual(report["status"],"failed")
    def test_completion_is_for_current_requested_grid(self):
        report={"records":[self.record("a"),self.record("old","failed")],"requested_keys":["a"]}
        summarize_evaluation(report)
        self.assertEqual(report["status"],"completed_requested_grid")
        self.assertTrue(report["all_requested_cases_passed"])

    def test_all_momentum_validation_candidates_are_exposed_without_extra_holdout_data(self):
        def candidate(name, sharpe):
            return {"spec":{"name":name},"metrics":{"validation":{"sharpe":sharpe},
                                                        "holdout":{"secret_fixture":True}}}
        study={"selection":{"selected_name":"momentum_iv","comparator_name":"momentum_equal"},
               "protocol":{},"manifest":{},"comparison":{},"sensitivity":[],
               "results":{"momentum_iv":candidate("momentum_iv",0.7),
                          "momentum_equal":candidate("momentum_equal",0.8),
                          "momentum_other":candidate("momentum_other",0.4),
                          "market_buy_and_hold":candidate("market_buy_and_hold",0.9)}}
        exposed=study_evidence(study)["validation_candidates"]
        self.assertEqual(set(exposed),{"momentum_iv","momentum_equal","momentum_other"})
        self.assertEqual(exposed["momentum_other"]["validation"],{"sharpe":0.4})
        self.assertTrue(all(set(result)=={"spec","validation"} for result in exposed.values()))
        self.assertNotIn("holdout",json.dumps(exposed))


class EvaluationPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_receipt_recovers_running_or_failed_case_without_a_paid_call(self):
        from quant_evidence.cli import do_evaluate
        parent=Path(__file__).resolve().parents[1]/"tmp"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            root=Path(folder)
            (root/"quant_evidence").mkdir()
            for filename in ("ai.py","evaluation.py"):
                (root/"quant_evidence"/filename).write_text("fixture source identity")
            case=fixture_case()
            receipt_path,_=write_fixture_receipt(root,case)
            receipt_before=receipt_path.read_bytes()
            blocked=AsyncMock(side_effect=AssertionError("No model calls are permitted in recovery"))
            for prior_status in ("running","failed"):
                archive=write_fixture_archive(root,case,prior_status)
                with patch("quant_evidence.cli.ROOT",root),patch("quant_evidence.cli.run_research",return_value={"fixture":True}),patch("quant_evidence.evaluation.build_cases",return_value=[case]),patch("quant_evidence.ai.review_evidence",blocked):
                    self.assertTrue(await do_evaluate(Namespace(limit=None,models=["m1"],modes=["tools"])))
                saved=json.loads(archive.read_text(encoding="utf-8"))
                row=saved["records"][0]
                self.assertEqual(saved["status"],"completed_requested_grid")
                self.assertEqual(saved["study_sha256"],fingerprint({"fixture":True}))
                self.assertEqual(json.loads((root/saved["cases_file"]).read_text(encoding="utf-8")),[case])
                self.assertEqual(row["recovery"]["prior_status"],prior_status)
                self.assertEqual(row["recovery"]["prior_error_type"],"FixturePostprocessingCrash")
                self.assertFalse(row["recovery"]["new_api_requests"])
                self.assertEqual(saved["groups"]["m1/tools"]["n"],1)
                self.assertTrue(row["score"]["passed"])
                self.assertEqual(receipt_path.read_bytes(),receipt_before)
            blocked.assert_not_called()

            # A persisted public case pack cannot be silently replaced, even
            # when the paid model result is already cached.
            (root/saved["cases_file"]).write_text(json.dumps([{"tampered":True}]),encoding="utf-8")
            with patch("quant_evidence.cli.ROOT",root),patch("quant_evidence.cli.run_research",return_value={"fixture":True}),patch("quant_evidence.evaluation.build_cases",return_value=[case]),patch("quant_evidence.ai.review_evidence",blocked):
                with self.assertRaises(ValueError):
                    await do_evaluate(Namespace(limit=None,models=["m1"],modes=["tools"]))
            blocked.assert_not_called()

    async def test_unfinished_receipt_does_not_enable_recovery_or_new_request(self):
        from quant_evidence.cli import do_evaluate
        parent=Path(__file__).resolve().parents[1]/"tmp"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            root=Path(folder)
            (root/"quant_evidence").mkdir()
            for filename in ("ai.py","evaluation.py"):
                (root/"quant_evidence"/filename).write_text("fixture source identity")
            case=fixture_case()
            archive=write_fixture_archive(root,case,"running")
            receipt_path,receipt=write_fixture_receipt(root,case)
            receipt["status"]="interrupted"
            receipt_path.write_text(json.dumps(receipt),encoding="utf-8")
            blocked=AsyncMock(side_effect=AssertionError("No model calls are permitted in recovery"))
            with patch("quant_evidence.cli.ROOT",root),patch("quant_evidence.cli.run_research",return_value={"fixture":True}),patch("quant_evidence.evaluation.build_cases",return_value=[case]),patch("quant_evidence.ai.review_evidence",blocked):
                with self.assertRaises(RuntimeError):
                    await do_evaluate(Namespace(limit=None,models=["m1"],modes=["tools"]))
            blocked.assert_not_called()
            self.assertEqual(json.loads(archive.read_text())["records"][0]["status"],"running")

    async def test_expansion_failure_is_saved_and_not_silently_retried(self):
        from quant_evidence.cli import do_evaluate
        parent=Path(__file__).resolve().parents[1]/"tmp"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            root=Path(folder)
            (root/"quant_evidence").mkdir()
            for filename in ("ai.py","evaluation.py"):
                (root/"quant_evidence"/filename).write_text("fixture source identity")
            cases=[{"case_id":"case_001","question":"fixture","evidence":{"E1":{"value":1}}}]
            score={"passed":True,"verdict_correct":True,"valid_citations":True}
            model_result={"review":{},"trace":{"read_ids":["E1"]}}
            mocked=AsyncMock(side_effect=[model_result,model_result,RuntimeError("fixture failure")])
            with patch("quant_evidence.cli.ROOT",root),patch("quant_evidence.cli.run_research",return_value={"fixture":True}),patch("quant_evidence.evaluation.build_cases",return_value=cases),patch("quant_evidence.evaluation.grade",return_value=score),patch("quant_evidence.ai.review_evidence",mocked):
                self.assertTrue(await do_evaluate(Namespace(limit=None,models=["m1"],modes=["tools"])))
                args=Namespace(limit=None,models=["m1","m2","m3"],modes=["tools"])
                with self.assertRaises(RuntimeError):
                    await do_evaluate(args)
                saved=json.loads((root/"outputs"/"evaluation"/"results.json").read_text())
                self.assertEqual(saved["status"],"failed")
                self.assertEqual(saved["requested_summary"],{"total":3,"completed":2,"failed":1,"pending":0})
                self.assertEqual(sum(g["n"] for g in saved["groups"].values()),3)
                self.assertEqual(mocked.await_count,3)
                with self.assertRaises(RuntimeError):
                    await do_evaluate(args)
                self.assertEqual(mocked.await_count,3)


class ResearchReviewArchiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_evidence_and_question_versions_preserve_prior_reviews(self):
        from quant_evidence.cli import do_review
        parent=Path(__file__).resolve().parents[1]/"tmp"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            root=Path(folder)
            args=Namespace(question="First question",model="m1",mode="tools")
            first=fixture_result()
            first["task"]="research_review"
            second=deepcopy(first)
            second["review"]["summary"]="Different evidence was reviewed."
            third=deepcopy(first)
            third["review"]["summary"]="A different question was reviewed."
            mocked=AsyncMock(side_effect=[first,second,third])
            with patch("quant_evidence.cli.ROOT",root),patch("quant_evidence.cli.run_research",return_value={"fixture":True}),patch("quant_evidence.cli.study_evidence",side_effect=[{"E1":{"value":1}},{"E1":{"value":2}},{"E1":{"value":2}}]),patch("quant_evidence.ai.review_evidence",mocked):
                await do_review(args)
                latest=root/"outputs"/"ai"/"research_review.json"
                original=json.loads(latest.read_text(encoding="utf-8"))
                old_archive=root/original["archive_file"]
                original_bytes=old_archive.read_bytes()
                await do_review(args)
                updated=json.loads(latest.read_text(encoding="utf-8"))
                self.assertNotEqual(original["archive_file"],updated["archive_file"])
                self.assertEqual(old_archive.read_bytes(),original_bytes)
                args.question="Second question"
                await do_review(args)
                changed_question=json.loads(latest.read_text(encoding="utf-8"))
                self.assertNotEqual(updated["archive_file"],changed_question["archive_file"])
                self.assertEqual(updated["evidence_sha256"],changed_question["evidence_sha256"])
                self.assertNotEqual(updated["question_sha256"],changed_question["question_sha256"])
                self.assertEqual(old_archive.read_bytes(),original_bytes)
                self.assertEqual(changed_question["study_sha256"],fingerprint({"fixture":True}))
                self.assertEqual(mocked.await_count,3)

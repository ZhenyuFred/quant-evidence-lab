from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from .research import dump, fingerprint, run_research, sha


ROOT = Path(__file__).resolve().parents[1]


def study_evidence(study: dict) -> dict:
    chosen, comparator = study["selection"]["selected_name"], study["selection"]["comparator_name"]
    return {
        "protocol": study["protocol"],
        "source_manifest": study["manifest"],
        "selection": study["selection"],
        "selected_result": study["results"][chosen],
        "comparator_result": study["results"][comparator],
        "validation_candidates": {
            name: {"spec": result["spec"], "validation": result["metrics"]["validation"]}
            for name, result in study["results"].items() if name.startswith("momentum_")
        },
        "comparison": study["comparison"],
        "cost_sensitivity": {"scenarios":study["sensitivity"]},
    }


async def do_review(args):
    from .ai import review_evidence
    study = run_research(ROOT)
    question = args.question or ("审核本次研究问题：相同参数的行业动量中，逆波动率权重是否改善扣费后风险调整收益？"
          "按验证集选择规则、留出结果、区间估计和成本敏感性回答；负结果照实报告。请明确哪些结论可支持、哪些不能。")
    evidence = study_evidence(study)
    evidence_hash = fingerprint(evidence)
    question_hash = fingerprint(question)
    record = await review_evidence(ROOT,question,evidence,model=args.model,mode=args.mode,task="research_review")
    record["study_sha256"] = fingerprint(study)
    record["evidence_sha256"] = evidence_hash
    record["question_sha256"] = question_hash
    record["question"] = question
    archive = ROOT/"outputs"/"ai"/"research_reviews"/args.model/args.mode/evidence_hash/f"{question_hash}.json"
    record["archive_file"] = str(archive.relative_to(ROOT)).replace("\\", "/")
    if archive.exists():
        if fingerprint(json.loads(archive.read_text(encoding="utf-8"))) != fingerprint(record):
            raise ValueError("A different review already occupies this archive identity; preserve both before continuing")
    else:
        dump(archive,record)
    dump(ROOT/"outputs"/"ai"/"research_review.json",record)
    print(json.dumps(record["review"],ensure_ascii=False,indent=2))


def prepare_file_review(args):
    from .evidence_pack import build_evidence_pack
    evidence = build_evidence_pack(Path(args.csv), Path(args.metadata) if args.metadata else None)
    signature = fingerprint(evidence)
    target = ROOT/"outputs"/"imports"/signature
    dump(target/"evidence.json", evidence)
    return evidence, signature, target


async def do_file_review(args, prepared):
    from .ai import review_evidence
    evidence, signature, target = prepared
    question = args.question or ("审查这个月度回测摘要：哪些绩效结论可由数字支持，哪些来源、参数选择、"
                                "时间信息和实施成本无法验证？不要从收益序列推断不存在未来信息泄漏。")
    record = await review_evidence(ROOT,question,evidence,model=args.model,mode=args.mode,
                                   task=f"import_review/{signature}")
    dump(target/"review.json",record)
    print(json.dumps(record["review"],ensure_ascii=False,indent=2))
    print(f"Saved: {target/'review.json'}")


def save_cost_analysis(record):
    from .costs import summarize_costs
    analysis = summarize_costs(record)
    analysis["budget_snapshot_sha256"] = fingerprint(record)
    groups = {}
    for call in record["calls"]:
        parts = call.get("task", "unknown").split("/")
        name = "/".join(parts[:3]) if parts[0] == "evaluation" else parts[0]
        groups.setdefault(name, []).append(call)
    analysis["by_task_group"] = {}
    for name, calls in groups.items():
        group = summarize_costs({"calls":calls})
        analysis["by_task_group"][name] = {key:group[key] for key in
            ("request_count","estimated_usd","lower_bound_usd","upper_bound_usd","exact_for_published_token_rates")}
    dump(ROOT/"outputs"/"cost-analysis.json",analysis)
    return analysis


def summarize_evaluation(report):
    groups={}
    for record in report["records"]:
        group=groups.setdefault(f"{record['model']}/{record['mode']}",{"n":0,"completed":0,"failed":0,"running":0,"passed":0,"verdict_correct":0,"valid_citations":0})
        group["n"]+=1
        status=record.get("status","completed")
        group[status]+=1
        for field in ("passed","verdict_correct","valid_citations"):
            group[field]+=int(bool((record.get("score") or {}).get(field)))
    report["groups"]=groups
    requested=set(report["requested_keys"])
    selected=[r for r in report["records"] if r["key"] in requested]
    completed=sum(r.get("status","completed")=="completed" for r in selected)
    failed=sum(r.get("status")=="failed" for r in selected)
    report["requested_summary"]={"total":len(requested),"completed":completed,"failed":failed,"pending":len(requested)-completed-failed}
    report["status"]="failed" if failed else "completed_requested_grid" if completed==len(requested) else "running"
    report["all_requested_cases_passed"]=completed==len(requested) and all((r.get("score") or {}).get("passed",False) for r in selected)
    return report


def completed_review_receipt(root: Path, question: str, evidence: dict, *,
                             model: str, task: str, mode: str) -> dict | None:
    """Read an exact completed receipt only; never create a run or send a request.

    This mirrors the fixed evaluation settings in ai.review_evidence. A missing,
    interrupted, mismatched or malformed receipt cannot authorize paid replay.
    It repairs the crash window between saving a result and saving its case row.
    """
    from .ai import INSTRUCTIONS, Review
    request_hash = fingerprint({"question": question, "evidence": evidence, "model": model,
                                "task": task, "mode": mode, "max_turns": 6, "max_tokens": 5000,
                                "instructions": INSTRUCTIONS, "schema": Review.model_json_schema()})
    path = Path(root).resolve()/"outputs"/"private"/"runs"/f"{request_hash}.json"
    if not path.is_file() or path.is_symlink():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(saved, dict) or saved.get("status") != "completed"
                or saved.get("request_hash") != request_hash or saved.get("task") != task):
            return None
        result = saved.get("result")
        if (not isinstance(result, dict) or result.get("model") != model
                or result.get("mode") != mode or result.get("task") != task
                or result.get("status") not in ("completed", "completed_with_grounding_errors")):
            return None
        Review.model_validate(result.get("review"))
        trace = result.get("trace")
        if not isinstance(trace, dict):
            return None
        for name in ("read_ids", "call_ids"):
            values = trace.get(name)
            if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
                return None
        if not trace["call_ids"]:
            return None
        return result
    except (OSError, ValueError, TypeError):
        return None


async def do_evaluate(args):
    from .ai import INSTRUCTIONS, review_evidence
    from .evaluation import build_cases, grade
    study = run_research(ROOT)
    cases = build_cases(study)
    if args.limit:
        cases = cases[:args.limit]
    output = ROOT/"outputs"/"evaluation"/"results.json"
    signature = fingerprint({"study":study,"instructions":INSTRUCTIONS,"cases":cases,
                             "runtime_sha256":sha(ROOT/"quant_evidence"/"ai.py"),
                             "grader_sha256":sha(ROOT/"quant_evidence"/"evaluation.py")})
    archive=ROOT/"outputs"/"evaluation"/"runs"/f"{signature}.json"
    if archive.exists():
        report=json.loads(archive.read_text(encoding="utf-8"))
        if report["signature"] != signature:
            raise ValueError("Evaluation dataset or prompt changed. Preserve prior results in another version before rerunning.")
    else:
        report={"signature":signature,"benchmark_type":"controlled mutations of frozen real research evidence; not an independent external benchmark",
                "case_count":len(cases),"records":[],"status":"running"}
    study_hash = fingerprint(study)
    if report.get("study_sha256", study_hash) != study_hash:
        raise ValueError("Evaluation report belongs to a different frozen study")
    report["study_sha256"] = study_hash
    cases_path = ROOT/"outputs"/"evaluation"/"cases"/f"{signature}.json"
    if cases_path.exists():
        if fingerprint(json.loads(cases_path.read_text(encoding="utf-8"))) != fingerprint(cases):
            raise ValueError("Saved benchmark cases were modified; preserve the audit record before continuing")
    else:
        # This public audit artifact includes labels; only question/evidence are
        # ever passed to the reviewer below, exactly as in the original run.
        dump(cases_path,cases)
    report["cases_file"] = str(cases_path.relative_to(ROOT)).replace("\\", "/")
    report["requested_keys"]=[f"{model}/{mode}/{case['case_id']}" for model in args.models for mode in args.modes for case in cases]
    report["archive_file"]=str(archive.relative_to(ROOT)).replace("\\","/")

    def save():
        summarize_evaluation(report)
        dump(archive,report)
        dump(output,report)

    save()
    for model in args.models:
        for mode in args.modes:
            for case in cases:
                key=f"{model}/{mode}/{case['case_id']}"
                previous=next((r for r in report["records"] if r["key"]==key),None)
                if previous:
                    if previous.get("status","completed")!="completed":
                        recovered = completed_review_receipt(ROOT, case["question"], case["evidence"],
                                                             model=model, mode=mode, task=f"evaluation/{key}")
                        if recovered is None:
                            raise RuntimeError("A prior evaluation attempt is unfinished or failed and has no matching completed receipt. It is retained in the denominator and will not be automatically charged again.")
                        recovered_score = grade(recovered["review"], case, recovered["trace"])
                        recovery = {"source": "completed_local_receipt", "new_api_requests": False,
                                    "prior_status": previous["status"],
                                    "prior_error_type": previous.pop("error_type", None)}
                        previous.update(status="completed", output=recovered, score=recovered_score,
                                        recovery=recovery)
                        save()
                    continue
                print(f"Reviewing {key}",flush=True)
                row={"key":key,"case_id":case["case_id"],"model":model,"mode":mode,"status":"running","score":None}
                report["records"].append(row)
                save()
                try:
                    record=await review_evidence(ROOT,case["question"],case["evidence"],model=model,mode=mode,task=f"evaluation/{key}")
                    score=grade(record["review"],case,record["trace"])
                    row.update(status="completed",output=record,score=score)
                except BaseException as exc:
                    row.update(status="failed",error_type=type(exc).__name__)
                    save()
                    raise
                save()
    print(json.dumps(report["groups"],ensure_ascii=False,indent=2))
    return report["all_requested_cases_passed"]


def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description="Quant Evidence Lab — offline research and budgeted AI review")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("research",help="Reproduce or verify the frozen study; no AI calls")
    sub.add_parser("report",help="Build the self-contained offline HTML report")
    sub.add_parser("budget",help="Show local request accounting, not account credit balance")
    sub.add_parser("verify",help="Check study file hashes and artifact integrity offline")
    r=sub.add_parser("review",help="Paid: run the research reviewer through the official API")
    r.add_argument("--model",default="gpt-6.1-sol",choices=["gpt-6-luna","gpt-6.1-sol","gpt-6-astra"])
    r.add_argument("--mode",default="tools",choices=["tools","full_context"])
    r.add_argument("--question")
    f=sub.add_parser("review-file",help="Validate a selected monthly CSV and optionally send its summary to the API")
    f.add_argument("--csv",required=True,help="Explicit monthly CSV path; see examples/README.md")
    f.add_argument("--metadata",help="Optional JSON with unverified provenance declarations")
    f.add_argument("--question")
    f.add_argument("--model",default="gpt-6.1-sol",choices=["gpt-6-luna","gpt-6.1-sol","gpt-6-astra"])
    f.add_argument("--mode",default="tools",choices=["tools","full_context"])
    f.add_argument("--dry-run",action="store_true",help="Only validate/save the evidence summary; no key or API required")
    e=sub.add_parser("evaluate",help="Paid: resume a controlled reliability evaluation")
    e.add_argument("--models",nargs="+",default=["gpt-6-luna","gpt-6.1-sol"],choices=["gpt-6-luna","gpt-6.1-sol","gpt-6-astra"])
    e.add_argument("--modes",nargs="+",default=["tools"],choices=["tools","full_context"])
    e.add_argument("--limit",type=int,help="Explicit partial smoke evaluation; changes dataset signature")
    args=parser.parse_args()
    try:
        if args.command=="research":
            study=run_research(ROOT)
            print(json.dumps({"selected_pair":study["selection"],"comparison":study["comparison"]},ensure_ascii=False,indent=2))
        elif args.command=="report":
            from .report import build_report
            print(build_report(ROOT))
        elif args.command=="budget":
            from .budget import Ledger
            path=ROOT/"outputs"/"private"/"usage.sqlite3"
            record=Ledger(path).summary() if path.exists() else {"calls":[],"conservative_usage_usd":0,"ceiling_usd":38,"account_balance_verified":False,"status":"no_real_API_requests_recorded"}
            dump(ROOT/"outputs"/"budget-summary.json",record)
            analysis=save_cost_analysis(record)
            print(json.dumps({key:value for key,value in analysis.items() if key!="requests"},ensure_ascii=False,indent=2))
        elif args.command=="verify":
            from .audit import verify_study
            record=verify_study(ROOT)
            dump(ROOT/"outputs"/"verification.json",record)
            print(json.dumps(record,ensure_ascii=False,indent=2))
            if not record["passed"]:
                raise SystemExit(1)
        elif args.command=="review":
            from .ai import load_key
            load_key(ROOT)
            asyncio.run(do_review(args))
        elif args.command=="review-file":
            prepared=prepare_file_review(args)
            if args.dry_run:
                print(f"Validated locally; no API call. Evidence: {prepared[2]/'evidence.json'}")
            else:
                from .ai import load_key
                load_key(ROOT)
                asyncio.run(do_file_review(args,prepared))
        elif args.command=="evaluate":
            if args.limit is not None and not 1<=args.limit<=24:
                parser.error("--limit must be between 1 and 24")
            from .ai import load_key
            load_key(ROOT)
            if not asyncio.run(do_evaluate(args)):
                raise SystemExit(2)
    except (RuntimeError,ValueError,OSError) as exc:
        # No traceback from a credential-bearing network request is printed.
        print(f"Stopped: {type(exc).__name__}: {exc}",file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Stopped: {type(exc).__name__}. No automatic retry was performed; inspect saved safe traces and the budget ledger before resuming.",file=sys.stderr)
        raise SystemExit(1)

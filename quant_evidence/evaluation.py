"""Controlled mutation benchmark for evidence-grounded research reviews.

This is a local, labelled diagnostic, not independent external validation of a
model's ability to review arbitrary financial research. Historical numbers come
from ``research``; the audit records are deliberately controlled fixtures. Each
negative fixture changes one field or removes one item. Do not send a whole case
to a model: only ``question`` and the values returned by evidence-reading tools
are model inputs. Labels and mutation descriptions are evaluator-only metadata.

The grader checks structured claims, not every free-text sentence. A passing
score does not establish that all prose is correct or that an investment works.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


ISSUE_CODES = (
    "future_leakage", "cost_omitted", "holdout_selection",
    "unverified_source", "numeric_mismatch", "unsupported_guarantee",
    "insufficient_evidence",
)
VERDICTS = ("supported", "not_supported", "insufficient_evidence")
BENCHMARK_INFO = {
    "kind": "controlled_mutation",
    "version": 1,
    "case_count": 24,
    "clean_controls": 3,
    "mutations_per_issue": 3,
    "independent_external_validation": False,
    "scope": "Real historical metrics with locally constructed, single-defect audit records.",
    "limitations": [
        "Small authored diagnostic; results do not establish external generalization.",
        "Cases share research data and templates and are not statistically independent.",
        "Grades check structured evidence use, not arbitrary numerical claims in prose.",
    ],
}

REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "summary": {"type": "string"},
        "findings": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "code": {"type": "string", "enum": list(ISSUE_CODES)},
                    "detail": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["code", "detail", "evidence_ids"],
            },
        },
        "citations": {"type": "array", "items": {"type": "string"}},
        "numeric_claims": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "evidence_id": {"type": "string"},
                    "path": {"type": "string"},
                    "value": {"type": "number"},
                },
                "required": ["evidence_id", "path", "value"],
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "summary", "findings", "citations", "numeric_claims", "limitations"],
}

_SCOPE = (
    " Assess only the stated claim against the supplied evidence, not the whole "
    "strategy. General caveats such as historical data revisions or non-tradability "
    "belong in limitations, unless the claim explicitly contradicts them. Read the "
    "relevant evidence with the tool, cite its IDs, and put any numerical assertions "
    "in numeric_claims using exact stored values and dot-separated JSON paths. "
    "Do not infer missing measurements. This is a controlled audit exercise."
)


def _number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _previous_month(month: str) -> str:
    try:
        year, number = map(int, month.split("-"))
        if len(month) != 7 or not 1 <= number <= 12:
            raise ValueError
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("holdout start must use YYYY-MM") from exc
    return f"{year - (number == 1):04d}-{12 if number == 1 else number - 1:02d}"


def build_cases(research: dict) -> list[dict]:
    """Build 3 metric controls and 21 isolated mutation cases, without network.

    Requires the selected result, positive ``spec.cost_bps``, finite holdout
    cagr/ann_vol/max_drawdown, and the actual manifest's source URL/hash. Missing
    inputs fail rather than being replaced by invented empirical measurements.
    The caller retains the full cases privately; model tools expose evidence only.
    """
    try:
        selected_name = research["selection"]["selected_name"]
        result = research["results"][selected_name]
        holdout = result["metrics"]["holdout"]
        metrics = ("cagr", "ann_vol", "max_drawdown")
        if not all(_number(holdout.get(name)) for name in metrics):
            raise ValueError("holdout cagr, ann_vol and max_drawdown must be finite")
        cost_bps = result["spec"]["cost_bps"]
        if not _number(cost_bps) or cost_bps <= 0:
            raise ValueError("fee-omission cases require a positive source cost_bps")
        holdout_start = holdout["start"]
        prior_month = _previous_month(holdout_start)
        selection = deepcopy(research["selection"])
        selection_end = selection["selection_end"]
        if not isinstance(selection_end, str) or selection_end >= holdout_start:
            raise ValueError("source selection must end before holdout starts")
        sources = research["manifest"]["sources"]
        source_name = "industry" if "industry" in sources else sorted(sources)[0]
        source = deepcopy(sources[source_name])
        if not isinstance(source["url"], str) or not source["url"]:
            raise ValueError("source URL is absent")
        if not isinstance(source["sha256"], str) or len(source["sha256"]) != 64:
            raise ValueError("source SHA256 is absent or malformed")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("research is missing required selected-result/provenance fields") from exc

    cases: list[dict] = []

    def add(question: str, evidence: dict, issue: str | None = None,
            mutation: dict | None = None, numeric_paths: list | None = None) -> None:
        cases.append({
            "case_id": f"case_{len(cases) + 1:03d}",
            "question": question + _SCOPE,
            "evidence": deepcopy(evidence),
            "expected_verdict": ("insufficient_evidence" if issue == "insufficient_evidence"
                                 else "not_supported" if issue else "supported"),
            "expected_issues": [issue] if issue else [],
            "benchmark_kind": "controlled_mutation",
            "mutation": deepcopy(mutation),
            "required_numeric_paths": deepcopy(numeric_paths or []),
        })

    def measured(metric: str) -> dict:
        return {"selected_name": selected_name, "split": "holdout",
                "metrics": {"start": holdout_start, "end": holdout["end"],
                            metric: holdout[metric]},
                "units": "returns, volatility and drawdown are decimals, not percentages"}

    for metric in metrics:
        add(f"Claim: the canonical recorded holdout {metric} is exactly {holdout[metric]!r}. "
            "Is this a faithful report of that stored historical statistic? Include "
            "the canonical stored value explicitly in numeric_claims.",
            {"E1": measured(metric)}, numeric_paths=[["E1", f"metrics.{metric}"]])

    for field in ("latest_momentum_return_month", "latest_volatility_return_month",
                  "latest_ranking_return_month"):
        clean = {"holding_month": holdout_start, "execution": "start_of_holding_month",
                 "latest_momentum_return_month": prior_month,
                 "latest_volatility_return_month": prior_month,
                 "latest_ranking_return_month": prior_month}
        changed = deepcopy(clean)
        changed[field] = holdout_start
        add("Claim: every return used by the audited signal was observable before "
            "the holding month began. Assess signal timing only.",
            {"E1": {"rule": "A month's return is observable only after that month ends.",
                    "latest_permitted_return_month": prior_month},
             "E2": changed}, "future_leakage",
            {"evidence_id": "E2", "path": field, "before": clean[field], "after": holdout_start})

    for split in ("train", "validation", "holdout"):
        actual = deepcopy(result["metrics"][split])
        add("Claim: the audited run's returns are net of the required transaction "
            "charge. Assess whether the required fees were actually deducted, "
            "not the transcription accuracy of a performance statistic.",
            {"E1": {"required_cost_bps": cost_bps,
                    "rule": "net_return = gross_return - required_cost_bps / 10000 * turnover",
                    "historical_reference_metrics": actual},
             "E2": {"split": split, "return_label": "net_after_required_costs",
                    "applied_cost_bps": 0.0,
                    "positive_turnover": True,
                    "cost_formula": "applied_cost_bps / 10000 * turnover"}},
            "cost_omitted", {"evidence_id": "E2", "path": "applied_cost_bps",
                              "before": cost_bps, "after": 0.0})

    selection_record = {
        "selected_name": selected_name,
        "selection_metric_split": "validation",
        "performance_data_end": selection_end,
        "holdout_used_for_parameter_choice": False,
    }
    for field, replacement in (("selection_metric_split", "holdout"),
                               ("performance_data_end", holdout["end"]),
                               ("holdout_used_for_parameter_choice", True)):
        changed = deepcopy(selection_record)
        changed[field] = replacement
        add("Claim: the audited parameter choice excluded the reported holdout from "
            "selection. Assess selection contamination, not signal construction.",
            {"E1": {"reference_selection": selection,
                    "holdout_start": holdout_start, "holdout_end": holdout["end"],
                    "rule": "No holdout performance may enter parameter selection."},
             "E2": changed}, "holdout_selection",
            {"evidence_id": "E2", "path": field,
             "before": selection_record[field], "after": replacement})

    provenance = {"source_url": source["url"], "raw_sha256": source["sha256"],
                  "integrity_check": "matched_canonical_manifest"}
    changed_hash = ("0" if source["sha256"][0] != "0" else "1") + source["sha256"][1:]
    for field, replacement in (("source_url", "https://unverified.invalid/returns.csv"),
                               ("raw_sha256", changed_hash),
                               ("integrity_check", "not_performed")):
        changed = deepcopy(provenance)
        changed[field] = replacement
        add("Claim: this audit record establishes that the input came from the "
            "canonical source and its integrity was verified against the manifest. "
            "Assess the claimed provenance verification only.",
            {"E1": {"source_name": source_name, "canonical_manifest_entry": source},
             "E2": changed}, "unverified_source",
            {"evidence_id": "E2", "path": field,
             "before": provenance[field], "after": replacement})

    for metric in metrics:
        reported = float(holdout[metric]) + 0.123456
        add(f"Claim: the report's quoted holdout {metric} exactly matches the "
            "canonical measured statistic. Check transcription accuracy only. "
            "Include the canonical stored value explicitly in numeric_claims.",
            {"E1": measured(metric), "E2": {"metric": metric, "reported_value": reported,
                                            "split": "holdout", "units": "decimal"}},
            "numeric_mismatch", {"evidence_id": "E2", "path": "reported_value",
                                 "before": holdout[metric], "after": reported},
            numeric_paths=[["E1", f"metrics.{metric}"]])

    neutral = "These historical observations do not establish future performance."
    guarantees = (
        "The historical results guarantee positive returns in every future year.",
        "The historical drawdown guarantees that investors can never lose principal.",
        "The historical results prove this strategy will outperform every future market regime.",
    )
    for guarantee in guarantees:
        add("Claim: the report's forward-looking conclusion follows from the "
            "historical measurements. Assess only whether the stated certainty is supported.",
            {"E1": {"selected_name": selected_name, "historical_holdout": deepcopy(holdout)},
             "E2": {"conclusion": guarantee}}, "unsupported_guarantee",
            {"evidence_id": "E2", "path": "conclusion", "before": neutral, "after": guarantee})

    for metric in metrics:
        evidence = measured(metric)
        del evidence["metrics"][metric]
        add(f"Can the exact stored holdout {metric} be verified from this evidence "
            "packet? The underlying monthly observations are not supplied. "
            "Assess evidence sufficiency for this statistic only.",
            {"E1": evidence}, "insufficient_evidence",
            {"evidence_id": "E1", "path": f"metrics.{metric}",
             "before": holdout[metric], "operation": "remove"})
    return cases


def _resolve(document: Any, path: str) -> Any:
    if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
        raise ValueError("invalid dot-separated path")
    value = document
    for part in path.split("."):
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list) and part.isascii() and part.isdigit():
            value = value[int(part)]
        else:
            raise ValueError("path does not resolve to a JSON value")
    return value


def grade(review: Any, case: dict, trace: dict | None = None) -> dict:
    """Grade structured output against private labels and accessible evidence.

    Missing citations fail, including an abstention (cite the incomplete packet).
    If trace supplies ``read_ids``, every referenced ID must have been read.
    Numeric matching uses absolute tolerance 1e-6, no relative tolerance, and
    rejects booleans/nonfinite numbers. The metric-specific cases additionally
    require an explicit numeric claim for the canonical measurement.
    Empty/no-issue controls have precision and recall 1 only when appropriate;
    duplicate issue codes and malformed outputs cannot pass.
    """
    if (not isinstance(case, dict) or not isinstance(case.get("evidence"), dict)
            or case.get("expected_verdict") not in VERDICTS
            or not isinstance(case.get("expected_issues"), list)
            or any(code not in ISSUE_CODES for code in case["expected_issues"])):
        raise ValueError("invalid evaluator case")
    evidence = case["evidence"]
    expected = set(case["expected_issues"])
    errors: list[str] = []
    schema_valid = True
    review = review if isinstance(review, dict) else {}
    required = set(REVIEW_SCHEMA["required"])
    if set(review) != required:
        errors.append("review fields do not match the output schema")
        schema_valid = False
    if review.get("verdict") not in VERDICTS:
        errors.append("invalid verdict")
        schema_valid = False
    if not isinstance(review.get("summary"), str) or not review.get("summary", "").strip():
        errors.append("summary must be nonempty")
        schema_valid = False

    def text_list(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)

    if not text_list(review.get("limitations")):
        errors.append("limitations must be a list of strings")
        schema_valid = False
    citations = review.get("citations", [])
    citations_shape = text_list(citations)
    if not citations_shape:
        errors.append("citations must be a list of nonempty IDs")
        schema_valid = False
        citations = []
    references: list[str] = list(citations)
    findings = review.get("findings", [])
    if not isinstance(findings, list):
        findings = []
        schema_valid = False
        errors.append("findings must be a list")
    predicted: list[str] = []
    finding_citations_valid = True
    for finding in findings:
        if not isinstance(finding, dict):
            schema_valid = False
            errors.append("invalid finding")
            finding_citations_valid = False
            continue
        code = finding.get("code")
        if isinstance(code, str):
            predicted.append(code)
        if (set(finding) != {"code", "detail", "evidence_ids"}
                or code not in ISSUE_CODES or not isinstance(finding.get("detail"), str)
                or not finding.get("detail", "").strip()):
            schema_valid = False
            errors.append("invalid finding fields or issue code")
        ids = finding.get("evidence_ids")
        if not text_list(ids) or not ids:
            schema_valid = False
            finding_citations_valid = False
            errors.append("every finding needs supporting evidence IDs")
        else:
            references.extend(ids)
    predicted_set = set(predicted)
    if len(predicted_set) != len(predicted):
        schema_valid = False
        errors.append("duplicate finding codes")
    correct_issues = len(predicted_set & expected)
    precision = correct_issues / len(predicted_set) if predicted_set else (1.0 if not expected else 0.0)
    recall = correct_issues / len(expected) if expected else 1.0

    numeric = review.get("numeric_claims", [])
    numeric_errors: list[str] = []
    if not isinstance(numeric, list):
        numeric = []
        schema_valid = False
        numeric_errors.append("numeric_claims must be a list")
    matched = 0
    resolved_paths: set[tuple[str, str]] = set()
    for index, claim in enumerate(numeric):
        if (not isinstance(claim, dict) or set(claim) != {"evidence_id", "path", "value"}
                or not isinstance(claim.get("evidence_id"), str)
                or not isinstance(claim.get("path"), str) or not _number(claim.get("value"))):
            schema_valid = False
            numeric_errors.append(f"numeric claim {index} is malformed or nonfinite")
            continue
        evidence_id, path = claim["evidence_id"], claim["path"]
        references.append(evidence_id)
        try:
            observed = _resolve(evidence[evidence_id], path)
        except (KeyError, IndexError, TypeError, ValueError):
            numeric_errors.append(f"numeric claim {index} does not resolve")
            continue
        if not _number(observed) or not math.isclose(observed, claim["value"], rel_tol=0, abs_tol=1e-6):
            numeric_errors.append(f"numeric claim {index} disagrees with finite stored value")
        else:
            matched += 1
            resolved_paths.add((evidence_id, path))
    for evidence_id, path in case.get("required_numeric_paths", []):
        if (evidence_id, path) not in resolved_paths:
            numeric_errors.append(f"required canonical numeric claim absent: {evidence_id}:{path}")

    trace_checked = isinstance(trace, dict) and "read_ids" in trace
    read_ids: set[str] | None = None
    trace_valid = True
    if trace_checked:
        recorded = trace["read_ids"]
        if not isinstance(recorded, (list, tuple, set)) or not all(isinstance(item, str) for item in recorded):
            trace_valid = False
            errors.append("trace.read_ids must contain evidence IDs")
            read_ids = set()
        else:
            read_ids = set(recorded)
    valid_citations = bool(citations) and citations_shape and finding_citations_valid and trace_valid
    for evidence_id in references:
        if evidence_id not in evidence:
            valid_citations = False
            errors.append(f"unknown evidence ID: {evidence_id}")
        if read_ids is not None and evidence_id not in read_ids:
            valid_citations = False
            errors.append(f"cited evidence was not read: {evidence_id}")
    if not citations:
        errors.append("at least one top-level evidence citation is required")

    verdict_correct = review.get("verdict") == case["expected_verdict"]
    exact_issue_match = predicted_set == expected
    numeric_passed = not numeric_errors
    return {
        "case_id": case.get("case_id"),
        "verdict_correct": verdict_correct,
        "issue_precision": precision, "issue_recall": recall,
        "exact_issue_match": exact_issue_match,
        "valid_citations": valid_citations,
        "numeric_grounding": {"passed": numeric_passed, "matched": matched,
                              "total": len(numeric), "errors": numeric_errors},
        "schema_valid": schema_valid, "trace_checked": trace_checked,
        "passed": bool(schema_valid and verdict_correct and exact_issue_match
                       and valid_citations and numeric_passed),
        "errors": errors,
    }

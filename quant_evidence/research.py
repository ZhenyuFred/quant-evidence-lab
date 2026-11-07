"""Freeze the protocol, select on validation, then unlock the historical holdout."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .backtest import candidate_specs, run_backtest, equal_weight_baseline, market_baseline, metrics
from .data import fetch_dataset


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def paired_sharpe_interval(a, b, *, iterations=2000, block=12, seed=20261106):
    """Paired circular block bootstrap; descriptive, conditional on selected pair."""
    first, second = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(first) != len(second) or len(first) < block * 2:
        raise ValueError("Paired monthly series of at least two blocks required")
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(first)
    for _ in range(iterations):
        starts = rng.integers(0, n, size=(n + block - 1)//block)
        ix = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        x, y = first[ix], second[ix]
        diffs.append(float(np.sqrt(12)*(x.mean()/x.std(ddof=1) - y.mean()/y.std(ddof=1))))
    lo, hi = np.quantile(diffs, [.025, .975])
    return {"low":float(lo),"high":float(hi),"iterations":iterations,"block_months":block,"seed":seed,
            "method":"paired circular block bootstrap, conditional on validation-selected parameters",
            "limitation":"Descriptive historical interval; stationarity, revised data and selection uncertainty limit interpretation."}


def run_research(root: Path) -> dict:
    root = Path(root).resolve()
    out = root/"outputs"/"research"
    out.mkdir(parents=True, exist_ok=True)
    returns, factors, manifest = fetch_dataset(root)
    source_hashes = {k: v["sha256"] for k,v in manifest["sources"].items()}
    code_hashes = {n: sha(root/"quant_evidence"/n) for n in ("data.py","backtest.py","research.py")}
    if (out/"research.json").exists():
        saved = json.loads((out/"research.json").read_text(encoding="utf-8"))
        if saved["protocol"]["source_hashes"] != source_hashes or saved["protocol"]["code_hashes"] != code_hashes:
            raise ValueError("Saved study source/code changed. Preserve this study and create a separate version before rerunning.")
        if sha(out/"protocol.json") != saved["protocol_sha256"] or sha(out/"selection.json") != saved["selection_sha256"]:
            raise ValueError("Protocol or selection was modified after completion")
        return saved
    specs = candidate_specs()
    last_month = str(min(returns.index[-1], factors.index[-1]))
    protocol = {
        "version":1,"frozen_at":datetime.now(timezone.utc).isoformat(),
        "question":"Does inverse-volatility weighting improve the net risk-adjusted return of a validation-selected industry momentum rule relative to its equal-weight counterpart?",
        "splits":{"train":{"start":"1970-01","end":"1999-12"},"validation":{"start":"2000-01","end":"2014-12"},"holdout":{"start":"2015-01","end":last_month}},
        "candidates":[asdict(s) for s in specs],
        "selection_rule":"Choose the equal-weight candidate with greatest validation net excess-return Sharpe; break ties by name. Lock its lookback/top_k and compare the corresponding inverse_vol candidate in holdout. Do not rank holdout candidates.",
        "primary_metric":"holdout Sharpe(inverse_vol) - Sharpe(matched_equal)",
        "secondary_metrics":["CAGR","annualised volatility","max drawdown","turnover"],
        "cost_bps":10,"cost_sensitivity_bps":[0,10,25,50],
        "return_timing":"For holding month t, momentum uses exactly L months t-L-1 through t-2; volatility uses t-12 through t-1.",
        "cost_convention":"At month end deduct bps/10000 * sum(abs(target - prior drifted weights)); includes initial cash entry. Covers hypothetical outer rebalance only.",
        "holdout_accounting":"One continuous portfolio from 1970-01; holdout rebases NAV but inherits prior holdings and actual boundary-month turnover.",
        "inference":"Paired circular 12-month block bootstrap, 2000 resamples, seed 20261106. Conditional descriptive interval, not proof of future alpha.",
        "source_hashes":source_hashes,"code_hashes":code_hashes,
        "precommit_scope":"Local timestamp/hash before evaluation; not a public externally timestamped preregistration. The LLM may know historical events.",
    }
    if (out/"protocol.json").exists():
        prior = json.loads((out/"protocol.json").read_text(encoding="utf-8"))
        if {k:v for k,v in prior.items() if k != "frozen_at"} != {k:v for k,v in protocol.items() if k != "frozen_at"}:
            raise ValueError("Existing frozen protocol differs; do not overwrite it")
        protocol = prior
    else:
        dump(out/"protocol.json", protocol)
    # Deliberately pass only pre-holdout data to selection-stage computation.
    development_returns = returns.loc[:"2014-12"]
    development_factors = factors.loc[:"2014-12"]
    results = {}
    for spec in specs:
        result = run_backtest(development_returns, development_factors["RF"], spec, start="1970-01")
        results[spec.name] = {"spec":asdict(spec), "metrics":{
            "train":metrics(result["monthly"].loc["1970-01":"1999-12"]),
            "validation":metrics(result["monthly"].loc["2000-01":"2014-12"]),
        }}
    equal_candidates = [s for s in specs if s.weighting == "equal"]
    equal = sorted(equal_candidates, key=lambda s: (-results[s.name]["metrics"]["validation"]["sharpe"], s.name))[0]
    inverse = next(s for s in specs if s.lookback == equal.lookback and s.top_k == equal.top_k and s.weighting == "inverse_vol")
    selection = {"selected_name":inverse.name,"comparator_name":equal.name,"selected_using":"validation_only",
                 "selection_end":"2014-12","holdout_start":"2015-01","frozen_at":datetime.now(timezone.utc).isoformat(),
                 "protocol_sha256":sha(out/"protocol.json"),"validation_results_hash":fingerprint(results)}
    if (out/"selection.json").exists():
        existing = json.loads((out/"selection.json").read_text(encoding="utf-8"))
        if {k:v for k,v in existing.items() if k != "frozen_at"} != {k:v for k,v in selection.items() if k != "frozen_at"}:
            raise ValueError("Existing frozen selection differs")
        selection = existing
    else:
        dump(out/"selection.json", selection)
    dump(out/"validation.json", results)
    # Only the locked pair and two fixed baselines are allowed into the holdout.
    full_runs = {s.name:run_backtest(returns, factors["RF"], s, start="1970-01") for s in (equal,inverse)}
    for result in (equal_weight_baseline(returns, factors["RF"],10,start="1970-01"), market_baseline(factors,cost_bps=10,start="1970-01")):
        full_runs[result["spec"]["name"]] = result
    artifacts = {}
    for name,result in full_runs.items():
        entry = results.setdefault(name,{"spec":result["spec"],"metrics":{}})
        for split, dates in protocol["splits"].items():
            entry["metrics"][split] = metrics(result["monthly"].loc[dates["start"]:dates["end"]])
        for table in ("monthly","weights"):
            path = out/f"{name}_{table}.csv"
            result[table].to_csv(path,float_format="%.12g")
            artifacts[path.name] = sha(path)
    sensitivity = []
    for cost in protocol["cost_sensitivity_bps"]:
        for s in (equal,inverse):
            r = run_backtest(returns, factors["RF"],replace(s,cost_bps=cost),start="1970-01")
            sensitivity.append({"name":s.name,"cost_bps":cost,"metrics":metrics(r["monthly"].loc["2015-01":])})
    interval = paired_sharpe_interval(full_runs[inverse.name]["monthly"].loc["2015-01":,"excess_return"], full_runs[equal.name]["monthly"].loc["2015-01":,"excess_return"])
    iv, eq = (results[s.name]["metrics"]["holdout"] for s in (inverse,equal))
    comparison = {"delta_sharpe":iv["sharpe"]-eq["sharpe"],"delta_cagr":iv["cagr"]-eq["cagr"],
                  "delta_max_drawdown":iv["max_drawdown"]-eq["max_drawdown"],"sharpe_interval_95":interval,
                  "directional_improvement":iv["sharpe"]>eq["sharpe"],"interval_excludes_zero":interval["low"]>0 or interval["high"]<0}
    study={"protocol":protocol,"protocol_sha256":sha(out/"protocol.json"),"selection":selection,"selection_sha256":sha(out/"selection.json"),
           "manifest":manifest,"results":results,"comparison":comparison,"sensitivity":sensitivity,"artifacts":artifacts,
           "completed_at":datetime.now(timezone.utc).isoformat()}
    dump(out/"research.json",study)
    return study

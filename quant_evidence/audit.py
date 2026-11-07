"""Independent saved-artifact integrity checks, not model-quality claims."""
from __future__ import annotations

import json
from pathlib import Path
from .research import sha, fingerprint
from .data import fetch_dataset


def verify_study(root: Path) -> dict:
    root=Path(root).resolve()
    out=root/"outputs"/"research"
    study=json.loads((out/"research.json").read_text(encoding="utf-8"))
    checks={}
    for name in ("protocol","selection"):
        checks[f"{name}_hash"] = sha(out/f"{name}.json")==study[f"{name}_sha256"]
        checks[f"{name}_content"] = json.loads((out/f"{name}.json").read_text(encoding="utf-8"))==study[name]
    checks["validation_hash"] = fingerprint(json.loads((out/"validation.json").read_text(encoding="utf-8")))==study["selection"]["validation_results_hash"]
    for name,digest in study["artifacts"].items():
        path=(out/name).resolve()
        checks[f"artifact:{name}"] = path.is_relative_to(out.resolve()) and path.is_file() and sha(path)==digest
    for name,digest in study["protocol"]["code_hashes"].items():
        checks[f"code:{name}"] = sha(root/"quant_evidence"/name)==digest
    _,_,manifest=fetch_dataset(root)
    checks["data_cache_integrity"]=all(manifest["sources"][k]["sha256"]==v for k,v in study["protocol"]["source_hashes"].items())
    checks["selection_precedes_holdout"] = study["selection"]["selection_end"] < study["protocol"]["splits"]["holdout"]["start"]
    checks["only_locked_pair_and_baselines_have_holdout"] = {n for n,r in study["results"].items() if "holdout" in r["metrics"]} == {
        study["selection"]["selected_name"],study["selection"]["comparator_name"],"all_industries_equal_weight","market_buy_and_hold"}
    return {"passed":all(checks.values()),"checks":checks,"scope":"Local artifact integrity and selection boundaries. This does not verify AI capability, account balance, or historical point-in-time availability."}

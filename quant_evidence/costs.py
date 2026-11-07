"""Read-only token-cost estimates, separate from conservative budget accounting.

Rates are Standard, short-context, global-endpoint text-token rates verified on
2026-11-06. Cache reads and writes are disjoint subsets of input_tokens. A write
rate replaces ordinary input pricing; it is not an added surcharge. Output usage
already includes reasoning. These calculations are not invoices or API balances.

This module neither opens SQLite nor alters reservations. Unknown cache details
produce an interval; missing usage on an unresolved call prevents an aggregate
point estimate. Budget reservations are shown only as a planning envelope.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json

from .budget import PRICE_CHECKED, PRICE_SOURCE, PRICES


CACHE_SOURCE = "https://developers.openai.com/api/docs/guides/prompt-caching"
SHORT_CONTEXT_LIMIT = 272_000
_MILLION = Decimal(1_000_000)
TOKEN_RATES = {
    "gpt-6-luna": {"input": "0.1", "cached": "0.01", "write_30m": "0.125", "write_12h": "0.25", "output": "0.5"},
    "gpt-6.1-sol": {"input": "2", "cached": "0.1", "write_30m": "2.5", "write_12h": "5", "output": "10"},
    "gpt-6-astra": {"input": "10", "cached": "1", "write_30m": "12.5", "write_12h": "25", "output": "50"},
}
_SCOPE = ("Published Standard short-context text-token rates only; excludes taxes, "
          "credits, negotiated discounts, tools, storage and any endpoint surcharge. "
          "This is neither an invoice nor an account balance.")


class CostError(ValueError):
    """Usage cannot be safely interpreted under the published rate contract."""


def request_pricing_metadata(payload: dict) -> dict:
    """Extract non-secret rate evidence from a trusted saved request body.

    No prompt, tool argument, headers or authentication values are retained.
    Implicit/default requests use the documented 30m default unless a supplied
    breakpoint changes its lifetime. Mixed lifetimes remain indeterminate.
    """
    if not isinstance(payload, dict):
        raise CostError("Request body must be a JSON object")
    model = payload.get("model")
    _rates(model)
    if payload.get("service_tier", "default") not in (None, "default"):
        raise CostError("Only verified Standard tier can be repriced")
    if payload.get("stream") or payload.get("background") or payload.get("previous_response_id") or payload.get("conversation"):
        raise CostError("Implicit or asynchronous request context is not supported")
    tools = payload.get("tools", [])
    if not isinstance(tools, list) or any(not isinstance(tool, dict) or tool.get("type") != "function" for tool in tools):
        raise CostError("Hosted tool charges cannot be repriced as token-only usage")
    if payload.get("prompt_cache_retention") is not None:
        raise CostError("Legacy cache-retention policy is not verified for these rates")
    options = payload.get("prompt_cache_options") or {}
    if not isinstance(options, dict) or set(options) - {"mode", "ttl"}:
        raise CostError("Unknown prompt-cache options require pricing review")
    mode, default_ttl = options.get("mode", "implicit"), options.get("ttl", "30m")
    if mode not in ("implicit", "explicit") or default_ttl not in ("30m", "12h"):
        raise CostError("Unverified cache mode or lifetime")
    lifetimes = {default_ttl} if mode == "implicit" else set()
    breakpoints = 0
    bounded_text = True

    def inspect(value):
        nonlocal breakpoints, bounded_text
        if isinstance(value, dict):
            if value.get("type") in {"input_image", "input_file", "input_audio", "input_video", "item_reference"} or value.get("encrypted_content"):
                bounded_text = False
            if "cache_control" in value:
                raise CostError("Unverified cache-control field")
            if "prompt_cache_breakpoint" in value:
                point = value["prompt_cache_breakpoint"]
                if (not isinstance(point, dict) or set(point) - {"mode", "ttl"}
                        or point.get("mode") != "explicit" or point.get("ttl", default_ttl) not in ("30m", "12h")):
                    raise CostError("Unverified explicit cache breakpoint")
                breakpoints += 1
                lifetimes.add(point.get("ttl", default_ttl))
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(payload.get("input"))
    no_cache = mode == "explicit" and breakpoints == 0
    single_ttl = next(iter(lifetimes)) if len(lifetimes) == 1 else None
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return {"schema_version": 1, "model": model, "service_tier": "default",
            "cache_mode": mode, "cache_write_ttl": single_ttl, "cache_disabled": no_cache,
            "cache_ttl_basis": "explicit_no_breakpoints" if no_cache else "request_lifetimes" if options or breakpoints else "documented_30m_default",
            "bounded_text_request": bounded_text, "explicit_breakpoint_count": breakpoints,
            "request_sha256": hashlib.sha256(encoded).hexdigest(),
            "price_checked": PRICE_CHECKED, "price_source": PRICE_SOURCE, "cache_source": CACHE_SOURCE}


def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise CostError(f"{name} must be a nonnegative integer token count")
    return value


def _optional_count(obj: dict, name: str) -> int | None:
    return _count(obj[name], name) if name in obj and obj[name] is not None else None


def _details(usage: dict, name: str) -> dict:
    value = usage.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise CostError(f"{name} must be a JSON object or null")
    return value


def _rates(model: str) -> dict[str, Decimal]:
    if not isinstance(model, str) or model not in TOKEN_RATES:
        raise CostError("Unknown model: verify published rates before estimating")
    prices = {name: Decimal(value) for name, value in TOKEN_RATES[model].items()}
    # Refuse to drift silently from the independent budget guard's rate snapshot.
    for key, budget_key in (("input", "input"), ("cached", "cached"),
                            ("write_12h", "write"), ("output", "output")):
        if prices[key] != Decimal(str(PRICES[model][budget_key])):
            raise CostError("Rate snapshots disagree; verify pricing before estimating")
    return prices


def compute_usage_cost(model: str, usage: dict, *, pricing_metadata: dict | None = None) -> dict:
    """Estimate one usage object's token cost without rounding to whole cents.

    When writes are known but their retention is not, the lower bound prices
    them at 30m and the upper bound at 12h. Missing read/write counts are unknown,
    not zero. ``estimated_usd`` selects the upper bound whenever the rates are
    underdetermined. Complete zero-write usage needs no lifetime breakdown.
    """
    rates = _rates(model)
    if not isinstance(usage, dict):
        raise CostError("usage must be a JSON object")
    try:
        inp = _count(usage["input_tokens"], "input_tokens")
        out = _count(usage["output_tokens"], "output_tokens")
    except KeyError as exc:
        raise CostError("input_tokens and output_tokens are required") from exc
    if inp > SHORT_CONTEXT_LIMIT:
        raise CostError("Input exceeds the verified short-context pricing range")
    if "total_tokens" in usage and _count(usage["total_tokens"], "total_tokens") != inp + out:
        raise CostError("total_tokens must equal input_tokens plus output_tokens")
    output_details = _details(usage, "output_tokens_details")
    reasoning = _optional_count(output_details, "reasoning_tokens")
    if reasoning is not None and reasoning > out:
        raise CostError("reasoning_tokens cannot exceed output_tokens")

    details = _details(usage, "input_tokens_details")
    cached = _optional_count(details, "cached_tokens")
    writes = _optional_count(details, "cache_write_tokens")
    lifetimes = details.get("cache_write_tokens_details")
    write_30m = write_12h = None
    if lifetimes is not None:
        if not isinstance(lifetimes, dict) or not {"ttl_30m_tokens", "ttl_12h_tokens"}.issubset(lifetimes):
            raise CostError("Lifetime breakdown must contain both 30m and 12h counts")
        write_30m = _count(lifetimes["ttl_30m_tokens"], "ttl_30m_tokens")
        write_12h = _count(lifetimes["ttl_12h_tokens"], "ttl_12h_tokens")
        if writes is not None and write_30m + write_12h != writes:
            raise CostError("Lifetime counts must sum to cache_write_tokens")
        writes = write_30m + write_12h
    inferred_lifetime = False
    if pricing_metadata is not None:
        if (not isinstance(pricing_metadata, dict) or pricing_metadata.get("schema_version") != 1
                or pricing_metadata.get("model") != model or pricing_metadata.get("service_tier") != "default"
                or pricing_metadata.get("price_checked") != PRICE_CHECKED
                or pricing_metadata.get("cache_write_ttl") not in (None, "30m", "12h")
                or not isinstance(pricing_metadata.get("cache_disabled"), bool)):
            raise CostError("Invalid or stale request-pricing metadata")
        if pricing_metadata["cache_disabled"]:
            if (cached or 0) or (writes or 0):
                raise CostError("API cache usage contradicts the no-cache request")
            cached, writes, write_30m, write_12h = 0, 0, 0, 0
        elif pricing_metadata.get("cache_write_ttl") is not None:
            ttl = pricing_metadata["cache_write_ttl"]
            if write_30m is not None:
                if (ttl == "30m" and write_12h) or (ttl == "12h" and write_30m):
                    raise CostError("API lifetime counts contradict the trusted request")
            elif writes is not None:
                write_30m, write_12h = (writes, 0) if ttl == "30m" else (0, writes)
                inferred_lifetime = bool(writes)
    if (cached or 0) + (writes or 0) > inp:
        raise CostError("Disjoint cached and cache-write counts cannot exceed input_tokens")

    fixed = Decimal(cached or 0) * rates["cached"] + Decimal(out) * rates["output"]
    known_write_lower = known_write_upper = Decimal(0)
    if write_30m is not None:
        known_write_lower = known_write_upper = (Decimal(write_30m) * rates["write_30m"]
                                                  + Decimal(write_12h) * rates["write_12h"])
    elif writes is not None:
        known_write_lower = Decimal(writes) * rates["write_30m"]
        known_write_upper = Decimal(writes) * rates["write_12h"]

    # Residual tokens are ordinary input unless the corresponding breakdown is
    # absent, in which case they could instead be reads or cache writes.
    residual = inp - (cached or 0) - (writes or 0)
    residual_rates = [rates["input"]]
    if cached is None:
        residual_rates.append(rates["cached"])
    if writes is None:
        residual_rates.extend((rates["write_30m"], rates["write_12h"]))
    lower = (fixed + known_write_lower + Decimal(residual) * min(residual_rates)) / _MILLION
    upper = (fixed + known_write_upper + Decimal(residual) * max(residual_rates)) / _MILLION
    exact = lower == upper
    missing = []
    if cached is None and inp:
        missing.append("cached read count")
    if writes is None and inp:
        missing.append("cache-write count")
    if writes and write_30m is None:
        missing.append("30m/12h write split")
    method = ("Exact arithmetic from identifiable token categories at the published rates."
              if exact else "Upper-bound point estimate; missing " + ", ".join(missing)
              + "; all feasible category allocations define the reported interval.")
    if inferred_lifetime:
        method += " Cache-write lifetime is established by the trusted request, not fabricated in API usage."
    return {
        "model": model, "currency": "USD",
        "estimated_usd": float(upper), "lower_bound_usd": float(lower), "upper_bound_usd": float(upper),
        "exact_for_published_token_rates": exact, "method": method,
        "cache_lifetime_from_request": inferred_lifetime,
        "token_counts": {"input": inp, "output": out, "cached": cached, "cache_writes": writes,
                         "write_30m": write_30m, "write_12h": write_12h,
                         "ordinary_input": residual if cached is not None and writes is not None else None,
                         "reasoning_included_in_output": reasoning},
        "price_checked": PRICE_CHECKED, "price_source": PRICE_SOURCE, "cache_source": CACHE_SOURCE,
        "scope": _SCOPE,
    }


def summarize_costs(summary: dict) -> dict:
    """Analyze a ``Ledger.summary()`` snapshot without modifying its accounting.

    Settled usage and explicit zero-charge rejections are included. Pending or
    unknown calls keep the aggregate unconfirmed, even if usage is available.
    Missing usage is never treated as zero; no upper bound on actual billing is
    claimed for those calls. The separate reservation envelope is a local budget
    estimate only, not a guarantee that the platform's bill cannot exceed it.
    """
    if not isinstance(summary, dict) or not isinstance(summary.get("calls"), list):
        raise CostError("summary must contain a calls list")
    lower = upper = estimate = Decimal(0)
    envelope = Decimal(0)
    all_exact = True
    envelope_available = True
    records = []
    seen_ids, seen_response_ids = set(), set()
    unresolved, missing_usage, rejected, usage_count = [], [], 0, 0
    for call in summary["calls"]:
        if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
            raise CostError("Every ledger call needs a nonempty ID")
        call_id = call["id"]
        if call_id in seen_ids:
            raise CostError("Duplicate ledger call ID would double-count cost")
        seen_ids.add(call_id)
        response_id = call.get("response_id")
        if response_id is not None:
            if not isinstance(response_id, str) or not response_id or response_id in seen_response_ids:
                raise CostError("Invalid or duplicate response ID would double-count cost")
            seen_response_ids.add(response_id)
        status = call.get("status")
        if status not in ("settled", "rejected", "pending", "unknown"):
            raise CostError("Unknown ledger status")
        reserved = _optional_count(call, "reserved")
        charged = _optional_count(call, "charged")
        usage = call.get("usage")
        if status == "rejected":
            if usage is not None or charged not in (None, 0):
                raise CostError("Rejected call has usage or nonzero accounting; reconcile it")
            rejected += 1
            records.append({"id": call_id, "status": status, "cost": None, "estimated_usd": 0.0})
            continue
        cost = None
        if usage is not None:
            cost = compute_usage_cost(call.get("model"), usage, pricing_metadata=call.get("pricing_metadata"))
            lower += Decimal(str(cost["lower_bound_usd"]))
            upper += Decimal(str(cost["upper_bound_usd"]))
            estimate += Decimal(str(cost["estimated_usd"]))
            all_exact = all_exact and cost["exact_for_published_token_rates"]
            usage_count += 1
        else:
            missing_usage.append(call_id)
        is_unresolved = status != "settled" or usage is None
        if is_unresolved:
            unresolved.append(call_id)
            if reserved is None:
                envelope_available = False
            else:
                envelope += max(Decimal(reserved) / _MILLION, Decimal(charged or 0) / _MILLION,
                                Decimal(str(cost["upper_bound_usd"])) if cost else Decimal(0))
        else:
            envelope += Decimal(str(cost["upper_bound_usd"]))
        records.append({"id": call_id, "status": status, "cost": cost})

    complete = not unresolved
    return {
        "currency": "USD", "request_count": len(records), "usage_request_count": usage_count,
        "rejected_request_count": rejected, "unresolved_request_ids": unresolved,
        "requests_without_usage": missing_usage, "complete_usage_coverage": not missing_usage,
        "estimated_usd": float(estimate) if complete else None,
        "lower_bound_usd": float(lower), "upper_bound_usd": float(upper) if complete else None,
        "exact_for_published_token_rates": bool(complete and all_exact),
        "observed_usage_cost": {"estimated_usd": float(estimate), "lower_bound_usd": float(lower),
                                "upper_bound_usd": float(upper), "exact_for_published_token_rates": all_exact},
        "reservation_envelope_usd": float(envelope) if envelope_available else None,
        "reservation_envelope_is_billing_bound": False,
        "conservative_budget_usage_usd": summary.get("conservative_usage_usd"),
        "guard_accounted_usage_usd": summary.get("guard_accounted_usage_usd", summary.get("conservative_usage_usd")),
        "guard_accounting_mode": summary.get("guard_accounting_mode", "legacy_upper_bound"),
        "retained_budget_reservations_usd": summary.get("retained_reservations_usd"),
        "account_balance_verified": False, "platform_invoice_verified": False,
        "method": ("Sum of per-request published token-rate estimates; bounded estimates use their upper endpoints."
                   if complete else "Unresolved requests prevent an aggregate point estimate or billing upper bound; observed usage and local reservations are reported separately."),
        "price_checked": PRICE_CHECKED, "price_source": PRICE_SOURCE,
        "scope": _SCOPE, "requests": records,
    }

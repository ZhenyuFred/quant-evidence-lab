"""Persistent, conservative per-request spending reservations.

Accounting estimates are not an account balance. An ambiguous request retains its
full reservation and halts paid work until externally reconciled.
"""
from __future__ import annotations

import json
import hashlib
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path


PRICES = {
    "gpt-6-luna": {"input": .10, "cached": .01, "write": .25, "output": .50},
    "gpt-6.1-sol": {"input": 2., "cached": .10, "write": 5., "output": 10.},
    "gpt-6-astra": {"input": 10., "cached": 1., "write": 25., "output": 50.},
}
PRICE_SOURCE = "https://developers.openai.com/api/docs/pricing"
PRICE_CHECKED = "2026-11-06"


class BudgetError(RuntimeError):
    pass


def micros(dollars: float | str) -> int:
    return int((Decimal(str(dollars)) * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


def token_cost(tokens: int, usd_per_million: float) -> int:
    if tokens < 0:
        raise BudgetError("Negative token usage")
    return int((Decimal(tokens) * Decimal(str(usd_per_million))).to_integral_value(rounding=ROUND_CEILING))


def conservative_cost(model: str, usage: dict) -> int:
    """Upper estimate: all non-cached input charged at the highest cache-write rate.

This deliberately overcounts when the API doesn't expose write-duration details.
Output includes reasoning tokens; they must not be added for a second time.
"""
    p = PRICES[model]
    inp, out = int(usage["input_tokens"]), int(usage["output_tokens"])
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
    if not 0 <= cached <= inp:
        raise BudgetError("Invalid cached usage")
    return token_cost(inp - cached, p["write"]) + token_cost(cached, p["cached"]) + token_cost(out, p["output"])


def reservation_for(payload: dict) -> int:
    model = payload.get("model")
    if model not in PRICES:
        raise BudgetError("Unpriced model; verify its price before adding it")
    if payload.get("service_tier", "default") not in (None, "default"):
        raise BudgetError("Only standard service tier is allowed")
    if payload.get("stream") or payload.get("background"):
        raise BudgetError("Streaming/background requests are not supported by this ledger")
    if payload.get("previous_response_id") or payload.get("conversation"):
        raise BudgetError("Implicit remote context is not budgetable here")
    for tool in payload.get("tools", []):
        if tool.get("type") != "function":
            raise BudgetError("Only local function tools are permitted; hosted tool fees excluded")
    maximum = payload.get("max_output_tokens")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= 12000:
        raise BudgetError("Explicit output cap of 1..12000 tokens required")
    # A UTF-8 byte is a conservative token upper bound for text. Padding covers
    # role/schema framing and provider instructions, within short-context pricing.
    byte_count = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    if byte_count > 180000:
        raise BudgetError("Request exceeds this project's bounded context size")
    p = PRICES[model]
    return token_cost(byte_count + 8192, p["write"]) + token_cost(maximum, p["output"])


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _reason(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BudgetError("An explicit audit reason is required")
    return value.strip()


def _ceiling(value) -> int:
    try:
        amount = Decimal(str(value))
        if isinstance(value, bool) or not amount.is_finite() or not 0 < amount <= 40:
            raise ValueError
        return micros(str(amount))
    except (ValueError, ArithmeticError) as exc:
        raise BudgetError("Total project ceiling must be above 0 and at most $40") from exc


class Ledger:
    def __init__(self, path: Path, ceiling_usd: float | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ceiling = _ceiling(38 if ceiling_usd is None else ceiling_usd)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, task TEXT, model TEXT, reserved INTEGER NOT NULL, charged INTEGER, status TEXT, response_id TEXT, usage TEXT, created TEXT, note TEXT)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(calls)")}
            for name, kind in (("pricing_metadata", "TEXT"), ("pricing_basis", "TEXT"), ("original_conservative_charged", "INTEGER")):
                if name not in columns:
                    db.execute(f"ALTER TABLE calls ADD COLUMN {name} {kind}")
            db.execute("CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value INTEGER)")
            db.execute("CREATE TABLE IF NOT EXISTS guard_adjustments (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, created TEXT NOT NULL, reason TEXT NOT NULL, call_id TEXT, before_json TEXT NOT NULL, after_json TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO settings VALUES ('ceiling', ?)", (self.ceiling,))
            db.execute("INSERT OR IGNORE INTO settings VALUES ('precise_pricing', 0)")
            db.execute("INSERT OR IGNORE INTO settings VALUES ('max_inflight', 1)")
            existing = db.execute("SELECT value FROM settings WHERE name='ceiling'").fetchone()[0]
            if not isinstance(existing, int) or not 0 < existing <= 40_000_000:
                raise BudgetError("Stored ledger ceiling is invalid")
            if ceiling_usd is not None and existing != self.ceiling:
                raise BudgetError("Existing ledger has a different ceiling; it cannot be silently raised")
            self.ceiling = existing

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _require_resolved(db):
        if db.execute("SELECT COUNT(*) FROM calls WHERE status IN ('pending','unknown')").fetchone()[0]:
            raise BudgetError("A prior request is pending or unaccounted for; reconcile before continuing")

    @staticmethod
    def _audit(db, kind: str, reason: str, before: dict, after: dict, call_id: str | None = None):
        db.execute("INSERT INTO guard_adjustments(kind,created,reason,call_id,before_json,after_json) VALUES (?,?,?,?,?,?)",
                   (kind, datetime.now(timezone.utc).isoformat(), reason, call_id,
                    json.dumps(before, sort_keys=True), json.dumps(after, sort_keys=True)))

    def authorize_ceiling(self, ceiling_usd: float, reason: str):
        """Explicitly change the same ledger's cap, with an immutable audit row."""
        new, reason = _ceiling(ceiling_usd), _reason(reason)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_resolved(db)
            old = db.execute("SELECT value FROM settings WHERE name='ceiling'").fetchone()[0]
            used = db.execute("SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) FROM calls").fetchone()[0]
            if new < used:
                raise BudgetError("Authorized ceiling cannot be below already accounted usage")
            if new != old:
                db.execute("UPDATE settings SET value=? WHERE name='ceiling'", (new,))
                self._audit(db, "authorize_ceiling", reason, {"ceiling_microusd": old}, {"ceiling_microusd": new})
            self.ceiling = new
        return {"changed": new != old, "ceiling_usd": new / 1e6}

    def enable_precise_pricing(self, reason: str):
        """Opt into proved token-rate cost, retaining upper bounds when necessary."""
        reason = _reason(reason)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_resolved(db)
            old = db.execute("SELECT value FROM settings WHERE name='precise_pricing'").fetchone()[0]
            if not old:
                db.execute("UPDATE settings SET value=1 WHERE name='precise_pricing'")
                self._audit(db, "enable_precise_pricing", reason, {"enabled": False}, {"enabled": True})
        return {"changed": not bool(old), "enabled": True}

    def set_max_inflight(self, maximum: int, reason: str):
        """Explicitly authorize up to three concurrently reserved requests."""
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 3:
            raise BudgetError("This project permits 1..3 explicitly authorized in-flight requests")
        reason = _reason(reason)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._require_resolved(db)
            old = db.execute("SELECT value FROM settings WHERE name='max_inflight'").fetchone()[0]
            if old != maximum:
                db.execute("UPDATE settings SET value=? WHERE name='max_inflight'", (maximum,))
                self._audit(db, "set_max_inflight", reason, {"max_inflight": old}, {"max_inflight": maximum})
        return {"changed": old != maximum, "max_inflight": maximum}

    def reserve(self, task: str, payload: dict) -> str:
        reserve = reservation_for(payload)
        call_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM calls WHERE status='unknown'").fetchone()[0]:
                raise BudgetError("An unaccounted request blocks every new request until reconciliation")
            maximum = db.execute("SELECT value FROM settings WHERE name='max_inflight'").fetchone()[0]
            if db.execute("SELECT COUNT(*) FROM calls WHERE status='pending'").fetchone()[0] >= maximum:
                raise BudgetError("The authorized in-flight reservation limit has been reached")
            self.ceiling = db.execute("SELECT value FROM settings WHERE name='ceiling'").fetchone()[0]
            precise = bool(db.execute("SELECT value FROM settings WHERE name='precise_pricing'").fetchone()[0])
            metadata = None
            from .costs import CostError, request_pricing_metadata
            try:
                metadata = request_pricing_metadata(payload)
                metadata["input_token_bound"] = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 8192
                metadata["max_output_tokens"] = payload["max_output_tokens"]
                if precise and not metadata["bounded_text_request"]:
                    raise BudgetError("Precise guard requires bounded text, not remote/opaque input")
            except CostError as exc:
                if precise:
                    raise BudgetError("Request pricing policy could not be verified") from exc
            used = db.execute("SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) FROM calls").fetchone()[0]
            if used + reserve > self.ceiling:
                raise BudgetError("The next request's worst-case reservation exceeds the remaining project budget")
            db.execute("INSERT INTO calls(id,task,model,reserved,charged,status,response_id,usage,created,note,pricing_metadata,pricing_basis) VALUES (?,?,?,?,NULL,'pending',NULL,NULL,?,NULL,?,?)",
                       (call_id, task, payload["model"], reserve, datetime.now(timezone.utc).isoformat(),
                        json.dumps(metadata) if metadata is not None else None,
                        "precise_mode_pending" if precise else "legacy_upper_bound"))
        return call_id

    def settle(self, call_id: str, response: dict):
        usage = response.get("usage")
        if not usage:
            self.unknown(call_id, "Response has no usage; retain full reservation")
            raise BudgetError("Missing usage; paid work has stopped")
        error = None
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
            if not row or row["status"] != "pending":
                raise BudgetError("Only a pending request can settle")
            try:
                conservative = conservative_cost(row["model"], usage)
                charged, basis = conservative, "legacy_upper_bound"
                if row["pricing_basis"] == "precise_mode_pending":
                    from .costs import compute_usage_cost
                    metadata = json.loads(row["pricing_metadata"]) if row["pricing_metadata"] else None
                    if response.get("service_tier") not in (None, "default"):
                        raise BudgetError("Unverified response service tier")
                    cost = compute_usage_cost(row["model"], usage, pricing_metadata=metadata)
                    charged = micros(cost["upper_bound_usd"])
                    basis = "exact_published_rates" if cost["exact_for_published_token_rates"] else "token_rate_upper_bound"
                    if metadata and (usage["input_tokens"] > metadata["input_token_bound"] or usage["output_tokens"] > metadata["max_output_tokens"]):
                        raise BudgetError("Reported token counts exceeded the request's bounds")
                status = "settled" if charged <= row["reserved"] else "unknown"
                db.execute("UPDATE calls SET charged=?,status=?,response_id=?,usage=?,pricing_basis=?,original_conservative_charged=? WHERE id=?",
                           (charged, status, response.get("id"), json.dumps(usage), basis, conservative, call_id))
                if status == "unknown":
                    error = BudgetError("Reported usage exceeded the reservation; all paid work has stopped")
            except (ValueError, TypeError, KeyError, ArithmeticError, BudgetError) as exc:
                db.execute("UPDATE calls SET status='unknown',note=? WHERE id=?", (f"Pricing validation failed: {type(exc).__name__}", call_id))
                error = BudgetError("Usage pricing is uncertain; full reservation retained and paid work stopped")
        if error is not None:
            raise error

    def reprice_settled(self, evidence_by_call_id: dict, reason: str):
        """Atomically reprice settled rows from caller-supplied trusted traces.

        Each mapping value must contain ``request`` and ``response`` objects.
        Response ID and usage must exactly match the saved settlement. Any
        unresolved ledger row or unprovable price rejects the entire migration.
        API usage is preserved byte-for-byte in SQLite; inferred TTL lives only
        in separate metadata and append-only adjustment records.
        """
        from .costs import CostError, compute_usage_cost, request_pricing_metadata
        reason = _reason(reason)
        if not isinstance(evidence_by_call_id, dict) or not evidence_by_call_id:
            raise BudgetError("A nonempty trusted-evidence mapping is required")
        changes = []
        try:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                self._require_resolved(db)
                for call_id, evidence in evidence_by_call_id.items():
                    if not isinstance(call_id, str) or not isinstance(evidence, dict):
                        raise BudgetError("Malformed call-pricing evidence")
                    row = db.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
                    if not row or row["status"] != "settled" or row["charged"] is None or not row["usage"]:
                        raise BudgetError("Only settled calls with recorded usage can be repriced")
                    request, response = evidence.get("request"), evidence.get("response")
                    if not isinstance(request, dict) or not isinstance(response, dict) or request.get("model") != row["model"]:
                        raise BudgetError("Trace request model does not match the settlement")
                    usage = json.loads(row["usage"])
                    if (not row["response_id"] or response.get("id") != row["response_id"]
                            or _digest(response.get("usage")) != _digest(usage)
                            or response.get("service_tier") not in (None, "default")):
                        raise BudgetError("Trace response identity, usage or service tier does not match")
                    metadata = request_pricing_metadata(request)
                    prior_metadata = json.loads(row["pricing_metadata"]) if row["pricing_metadata"] else None
                    if prior_metadata and prior_metadata.get("request_sha256") != metadata["request_sha256"]:
                        raise BudgetError("Trace request hash differs from the reserved request")
                    if response.get("prompt_cache_options") is not None:
                        echoed = request_pricing_metadata(request | {"prompt_cache_options": response["prompt_cache_options"]})
                        if (echoed["cache_write_ttl"], echoed["cache_disabled"]) != (metadata["cache_write_ttl"], metadata["cache_disabled"]):
                            raise BudgetError("Response cache policy contradicts the request")
                    cost = compute_usage_cost(row["model"], usage, pricing_metadata=metadata)
                    if not cost["exact_for_published_token_rates"]:
                        raise BudgetError("Trace metadata does not establish an exact token-rate cost")
                    charged = micros(cost["upper_bound_usd"])
                    if charged > row["reserved"]:
                        raise BudgetError("Repriced usage exceeds the original reservation; reconcile externally")
                    before = {"charged_microusd": row["charged"], "pricing_basis": row["pricing_basis"] or "legacy_upper_bound"}
                    after = {"charged_microusd": charged, "pricing_basis": "exact_published_rates",
                             "delta_microusd": charged - row["charged"], "request_sha256": metadata["request_sha256"],
                             "response_sha256": _digest(response), "method": cost["method"]}
                    if row["charged"] == charged and row["pricing_basis"] == "exact_published_rates" and prior_metadata == metadata:
                        continue
                    original = row["original_conservative_charged"] if row["original_conservative_charged"] is not None else row["charged"]
                    db.execute("UPDATE calls SET charged=?,pricing_basis='exact_published_rates',pricing_metadata=?,original_conservative_charged=? WHERE id=?",
                               (charged, json.dumps(metadata), original, call_id))
                    self._audit(db, "reprice_settled", reason, before, after, call_id)
                    changes.append({"call_id": call_id, **after})
                used = db.execute("SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) FROM calls").fetchone()[0]
                ceiling = db.execute("SELECT value FROM settings WHERE name='ceiling'").fetchone()[0]
                if used > ceiling:
                    raise BudgetError("Migration would exceed the authorized ledger ceiling")
        except CostError as exc:
            raise BudgetError("Trusted trace could not establish safe published-rate pricing") from exc
        return {"repriced_calls": len(changes), "changes": changes}

    def reject(self, call_id: str, status: int):
        # Only explicit pre-generation rejection codes; server failures and lost
        # connections can have billed work and must retain their reservation.
        if status not in (400, 401, 403, 404, 405, 413, 422, 429):
            self.unknown(call_id, f"HTTP {status}; billing uncertain")
            return
        with self.connect() as db:
            db.execute("UPDATE calls SET charged=0,status='rejected',note=? WHERE id=? AND status='pending'", (f"HTTP {status}", call_id))

    def unknown(self, call_id: str, note: str):
        with self.connect() as db:
            db.execute("UPDATE calls SET status='unknown',note=? WHERE id=? AND status='pending'", (note, call_id))

    def summary(self) -> dict:
        with self.connect() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM calls ORDER BY created")]
            self.ceiling = db.execute("SELECT value FROM settings WHERE name='ceiling'").fetchone()[0]
            precise = bool(db.execute("SELECT value FROM settings WHERE name='precise_pricing'").fetchone()[0])
            max_inflight = db.execute("SELECT value FROM settings WHERE name='max_inflight'").fetchone()[0]
            adjustments = [dict(r) for r in db.execute("SELECT * FROM guard_adjustments ORDER BY id")]
        for row in rows:
            if row["usage"]:
                row["usage"] = json.loads(row["usage"])
            if row["pricing_metadata"]:
                row["pricing_metadata"] = json.loads(row["pricing_metadata"])
            if row["pricing_basis"] is None:
                row["pricing_basis"] = "legacy_upper_bound" if row["status"] == "settled" else row["status"]
        for event in adjustments:
            event["before"] = json.loads(event.pop("before_json"))
            event["after"] = json.loads(event.pop("after_json"))
        spent = sum(r["charged"] or 0 for r in rows)
        retained = sum(r["reserved"] for r in rows if r["charged"] is None)
        original = sum((r["original_conservative_charged"] if r["original_conservative_charged"] is not None else r["charged"]) or 0 for r in rows)
        basis_counts = {}
        for row in rows:
            if row["status"] == "settled":
                basis_counts[row["pricing_basis"]] = basis_counts.get(row["pricing_basis"], 0) + 1
        return {"ceiling_usd": self.ceiling / 1e6, "conservative_usage_usd": spent / 1e6,
                "conservative_usage_usd_note": "Compatibility alias for guard_accounted_usage_usd; precise-mode rows are not conservative legacy estimates.",
                "guard_accounted_usage_usd": spent / 1e6,
                "original_conservative_usage_usd": original / 1e6,
                "guard_accounting_mode": "precise_when_proven_else_upper_bound" if precise else "legacy_upper_bound",
                "max_inflight": max_inflight,
                "pending_request_count": sum(r["status"] == "pending" for r in rows),
                "unknown_request_count": sum(r["status"] == "unknown" for r in rows),
                "guard_pricing_basis_counts": basis_counts,
                "all_settled_pricing_exact": bool(basis_counts) and set(basis_counts) == {"exact_published_rates"},
                "guard_rounding": "Each request rounds upward to one micro-USD; published token-rate exactness is not platform invoice reconciliation.",
                "adjustment_history": adjustments,
                "retained_reservations_usd": retained / 1e6, "available_usd": (self.ceiling-spent-retained)/1e6,
                "price_checked": PRICE_CHECKED, "price_source": PRICE_SOURCE,
                "account_balance_verified": False, "calls": rows}

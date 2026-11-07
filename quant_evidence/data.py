"""Versioned, validated Kenneth French monthly research data.

No API credentials are needed. A normal second run is strictly offline; corrupt
or incomplete caches fail closed and must be refreshed explicitly.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd


INDUSTRIES = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
              "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
FACTORS = ["Mkt-RF", "SMB", "HML", "RF"]
SOURCES = {
    "industry": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_CSV.zip",
    "factors": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
}
PROCESSED = {"industry": "industry_returns.csv", "factors": "factors.csv"}
MAX_DOWNLOAD = 20 * 1024 * 1024
MAX_CSV = 30 * 1024 * 1024


class DatasetError(ValueError):
    """The data or its recorded provenance does not satisfy our contract."""


def _validate(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    columns = INDUSTRIES if kind == "industry" else FACTORS
    if list(frame.columns) != columns or frame.empty:
        raise DatasetError(f"{kind}: empty table or unexpected columns")
    if not isinstance(frame.index, pd.PeriodIndex) or frame.index.freqstr != "M":
        raise DatasetError(f"{kind}: a monthly PeriodIndex is required")
    if frame.index.hasnans or frame.index.has_duplicates:
        raise DatasetError(f"{kind}: missing or duplicate dates")
    if not frame.index.is_monotonic_increasing:
        raise DatasetError(f"{kind}: dates are not strictly ascending")
    expected = pd.period_range(frame.index[0], frame.index[-1], freq="M")
    if not frame.index.equals(expected):
        raise DatasetError(f"{kind}: missing months; no interpolation is allowed")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DatasetError(f"{kind}: missing or non-finite numeric values")
    if kind == "industry" and (values < -1).any():
        raise DatasetError("industry: total returns cannot be below -100%")
    # SMB/HML are long-short factors; imposing a long-only -100% bound on them
    # would be a mathematical error. Check cash and the implied total market.
    if kind == "factors":
        if (frame["RF"] <= -1).any() or ((frame["Mkt-RF"] + frame["RF"]) < -1).any():
            raise DatasetError("factors: invalid risk-free or implied market return")
    frame.index.name = "month"
    return frame


def _monthly_table(text: str, columns: list[str], kind: str) -> pd.DataFrame:
    rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text.lstrip("\ufeff")))]
    start = 0
    if kind == "industry":
        marker = next((i for i, row in enumerate(rows)
                       if re.search(r"Average\s+Value\s+Weighted\s+Returns.*Monthly", " ".join(row), re.I)), None)
        if marker is None:
            raise DatasetError("industry: monthly value-weighted section is absent")
        start = marker + 1
    header = next((i for i in range(start, len(rows))
                   if rows[i] == [""] + columns), None)
    if header is None:
        raise DatasetError(f"{kind}: expected monthly header is absent")
    dates, data = [], []
    for row in rows[header + 1:]:
        if not row or not any(row):
            continue
        first = row[0]
        if not re.fullmatch(r"\d{6}", first):
            # Do not accidentally import the annual or equal-weighted blocks.
            if data and re.search(r"(?:Annual|Average\s+(?:Equal|Value)\s+Weighted|Number\s+of\s+Firms|Average\s+Firm\s+Size)", " ".join(row), re.I):
                break
            raise DatasetError(f"{kind}: invalid monthly date or section: {first!r}")
        if len(row) != len(columns) + 1:
            raise DatasetError(f"{kind}: wrong number of values at {first}")
        try:
            date = pd.Period(f"{first[:4]}-{first[4:]}", freq="M")
            values = [float(value) for value in row[1:]]
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"{kind}: malformed date or numeric value at {first}") from exc
        if any(value in (-99.99, -999) for value in values):
            raise DatasetError(f"{kind}: missing-data sentinel at {first}")
        if not np.isfinite(values).all():
            raise DatasetError(f"{kind}: non-finite numeric value at {first}")
        dates.append(date)
        data.append([value / 100.0 for value in values])
    frame = pd.DataFrame(data, index=pd.PeriodIndex(dates, freq="M"), columns=columns, dtype=float)
    return _validate(frame, kind)


def parse_industry_csv(text: str) -> pd.DataFrame:
    """Read only the first value-weighted monthly table, converting % to decimals."""
    return _monthly_table(text, INDUSTRIES, "industry")


def parse_factor_csv(text: str) -> pd.DataFrame:
    """Read the monthly Mkt-RF/SMB/HML/RF table, converting % to decimals."""
    return _monthly_table(text, FACTORS, "factors")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract(raw: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = [entry for entry in archive.infolist() if entry.filename.lower().endswith(".csv")]
            if len(entries) != 1 or entries[0].file_size > MAX_CSV:
                raise DatasetError("Expected exactly one reasonably sized CSV inside the ZIP")
            return archive.read(entries[0]).decode("utf-8-sig")
    except (zipfile.BadZipFile, UnicodeError, RuntimeError) as exc:
        raise DatasetError("Invalid source ZIP or CSV encoding") from exc


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "QuantEvidence/1.0 (public academic data)"})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(MAX_DOWNLOAD + 1)
    if len(raw) > MAX_DOWNLOAD:
        raise DatasetError("Source download exceeds the size limit")
    return raw


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as file:
            temporary = Path(file.name)
            file.write(raw)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _paired(industry: pd.DataFrame, factors: pd.DataFrame) -> None:
    if not industry.index.equals(factors.index):
        raise DatasetError("Industry and factor month coverage differs; refresh both sources together")


def _cached(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    try:
        manifest = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1:
            raise DatasetError("Unsupported data manifest version")
        frames = {}
        for kind, url in SOURCES.items():
            entry = manifest["sources"][kind]
            raw_path = root / "data" / "raw" / url.rsplit("/", 1)[1]
            cache_path = root / "data" / "processed" / PROCESSED[kind]
            raw, processed = raw_path.read_bytes(), cache_path.read_bytes()
            if entry["url"] != url or entry["sha256"] != _sha256(raw):
                raise DatasetError(f"{kind}: raw source URL/hash verification failed")
            if entry["processed_sha256"] != _sha256(processed):
                raise DatasetError(f"{kind}: processed cache hash verification failed")
            frame = pd.read_csv(io.BytesIO(processed), index_col=0, float_precision="round_trip")
            frame.index = pd.PeriodIndex(frame.index.astype(str), freq="M")
            frames[kind] = _validate(frame, kind)
        _paired(frames["industry"], frames["factors"])
        return frames["industry"], frames["factors"], manifest
    except DatasetError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise DatasetError("Incomplete or invalid data cache; use refresh=True explicitly") from exc


def fetch_dataset(root: Path, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fetch the public data once, then verify and reuse the cache without network.

    ``root`` is the project root. No data is forward/back filled. Refresh downloads
    both sources and validates them before replacing any existing cache files.
    """
    root = Path(root)
    manifest_path = root / "data" / "manifest.json"
    if manifest_path.exists() and not refresh:
        return _cached(root)
    if (root / "data" / "raw").exists() and not refresh:
        # An interrupted first write should not be mistaken for an offline hit.
        if any((root / "data" / "raw").iterdir()):
            raise DatasetError("Raw files exist without a manifest; use refresh=True explicitly")
    raw = {kind: _download(url) for kind, url in SOURCES.items()}
    frames = {"industry": parse_industry_csv(_extract(raw["industry"])),
              "factors": parse_factor_csv(_extract(raw["factors"]))}
    _paired(frames["industry"], frames["factors"])
    processed = {kind: frame.to_csv(float_format="%.17g").encode("utf-8")
                 for kind, frame in frames.items()}
    retrieved_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1, "retrieved_at": retrieved_at,
        "units": "decimal monthly returns; 0.01 means 1%",
        "frequency": "monthly", "industry_weighting": "value-weighted",
        "sources": {kind: {"url": url, "sha256": _sha256(raw[kind]),
                            "processed_sha256": _sha256(processed[kind]),
                            "retrieved_at": retrieved_at,
                            "rows": len(frames[kind]), "columns": list(frames[kind].columns),
                            "first_month": str(frames[kind].index[0]),
                            "last_month": str(frames[kind].index[-1])}
                    for kind, url in SOURCES.items()},
        "limitations": [
            "Downloaded vintage is not historical point-in-time availability; source histories can be revised.",
            "French U.S. research returns changed from CRSP FIZ to CIZ starting with the January 2025 release.",
            "Industry research portfolios are not directly tradable funds; modelled outer rebalance costs do not cover underlying portfolio implementation.",
            "Historical holdouts cannot eliminate an LLM's prior knowledge of historical markets.",
        ],
        "documentation_url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
    }
    for kind, url in SOURCES.items():
        _atomic_write(root / "data" / "raw" / url.rsplit("/", 1)[1], raw[kind])
        _atomic_write(root / "data" / "processed" / PROCESSED[kind], processed[kind])
    _atomic_write(manifest_path, (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return frames["industry"], frames["factors"], manifest

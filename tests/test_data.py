"""Synthetic parser fixtures, separate from downloaded research data."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from quant_evidence.data import (
    DatasetError, FACTORS, INDUSTRIES, SOURCES, fetch_dataset,
    parse_factor_csv, parse_industry_csv,
)


def industry_text(dates=("200001", "200002", "200003")):
    lines = ["Synthetic unit-test fixture, not real market data",
             "Average Value Weighted Returns -- Monthly", "," + ",".join(INDUSTRIES)]
    for i, date in enumerate(dates):
        lines.append(date + "," + ",".join(str(i + j + 1) for j in range(12)))
    lines += ["", "Average Equal Weighted Returns -- Monthly", "," + ",".join(INDUSTRIES),
              "200001," + ",".join(["99"] * 12)]
    return "\n".join(lines)


def factor_text(dates=("200001", "200002", "200003")):
    lines = ["Synthetic unit-test fixture", "," + ",".join(FACTORS)]
    for date in dates:
        lines.append(date + ",1.5,-0.5,0.25,0.1")
    lines += ["", "Annual Factors: January-December", "," + ",".join(FACTORS), "2000,99,99,99,99"]
    return "\n".join(lines)


def zipped(text):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture.csv", text)
    return stream.getvalue()


def fake_download(request, timeout):
    text = industry_text() if request.full_url == SOURCES["industry"] else factor_text()
    return io.BytesIO(zipped(text))


class ParserTests(unittest.TestCase):
    def test_value_weighted_only_and_percent_conversion(self):
        frame = parse_industry_csv(industry_text())
        self.assertEqual(list(frame.columns), INDUSTRIES)
        self.assertEqual(len(frame), 3)
        self.assertIsInstance(frame.index, pd.PeriodIndex)
        self.assertEqual(frame.index.freqstr, "M")
        self.assertAlmostEqual(frame.iloc[0, 0], 0.01)
        self.assertAlmostEqual(frame.iloc[0, -1], 0.12)

    def test_factor_monthly_only_and_signed_values(self):
        frame = parse_factor_csv(factor_text())
        self.assertEqual(len(frame), 3)
        self.assertEqual(list(frame.columns), FACTORS)
        self.assertAlmostEqual(frame.iloc[0]["RF"], 0.001)
        self.assertAlmostEqual(frame.iloc[0]["SMB"], -0.005)

    def test_whitespace_and_bom(self):
        text = "\ufeff" + industry_text().replace(",NoDur", ", NoDur ")
        self.assertEqual(len(parse_industry_csv(text)), 3)

    def test_duplicate_unsorted_and_missing_months_rejected(self):
        for dates in [("200001", "200001"), ("200002", "200001"), ("200001", "200003")]:
            with self.subTest(dates=dates), self.assertRaises(DatasetError):
                parse_industry_csv(industry_text(dates))

    def test_invalid_numeric_values_rejected(self):
        for value in ("-99.99", "-999", "nan", "inf", "", "oops", "-100.01"):
            with self.subTest(value=value), self.assertRaises(DatasetError):
                parse_industry_csv(industry_text().replace("200001,1,", "200001," + value + ","))

    def test_invalid_dates_do_not_silently_truncate(self):
        for date in ("200013", "2000-02", "not-a-date"):
            with self.subTest(date=date), self.assertRaises(DatasetError):
                parse_factor_csv(factor_text().replace("200002", date))

    def test_wrong_header_and_weighting_rejected(self):
        for text in (industry_text().replace("NoDur", "Unknown"),
                     industry_text().replace("Average Value Weighted", "Average Equal Weighted")):
            with self.assertRaises(DatasetError):
                parse_industry_csv(text)

    def test_factor_sentinel_and_invalid_market(self):
        for value in ("-99.99", "-999", "-200", "nan"):
            with self.subTest(value=value), self.assertRaises(DatasetError):
                parse_factor_csv(factor_text().replace("1.5,-0.5", value + ",-0.5"))

    def test_long_short_factors_are_not_long_only_returns(self):
        frame = parse_factor_csv(factor_text().replace("1.5,-0.5", "1.5,-101"))
        self.assertAlmostEqual(frame.iloc[0]["SMB"], -1.01)

    def test_wrong_row_width_rejected(self):
        with self.assertRaises(DatasetError):
            parse_factor_csv(factor_text().replace("1.5,-0.5,0.25,0.1", "1.5,-0.5"))


class CacheTests(unittest.TestCase):
    def setUp(self):
        # Windows sandbox temporary roots may not permit nested writes. Keep
        # these disposable fixtures inside the explicitly writable workspace.
        self.workspace = Path(__file__).resolve().parents[1]
        self.directory = tempfile.TemporaryDirectory(prefix="test-data-", dir=self.workspace)
        self.root = Path(self.directory.name)

    def tearDown(self):
        if self.root.resolve().parent != self.workspace:
            raise RuntimeError("Refusing to clean a fixture outside the workspace")
        self.directory.cleanup()

    def populate(self):
        with patch("urllib.request.urlopen", side_effect=fake_download) as download:
            result = fetch_dataset(self.root)
        self.assertEqual(download.call_count, 2)
        return result

    def test_offline_cache_and_provenance(self):
        original, factors, manifest = self.populate()
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")):
            cached, cached_factors, cached_manifest = fetch_dataset(self.root)
        pd.testing.assert_frame_equal(original, cached)
        pd.testing.assert_frame_equal(factors, cached_factors)
        self.assertEqual(manifest, cached_manifest)
        self.assertEqual(manifest["sources"]["industry"]["url"], SOURCES["industry"])
        self.assertEqual(len(manifest["sources"]["industry"]["sha256"]), 64)
        self.assertIn("retrieved_at", manifest)

    def test_raw_hash_tampering_rejected_without_network(self):
        self.populate()
        path = self.root / "data" / "raw" / "12_Industry_Portfolios_CSV.zip"
        path.write_bytes(path.read_bytes() + b"corrupt")
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")):
            with self.assertRaisesRegex(DatasetError, "hash"):
                fetch_dataset(self.root)

    def test_processed_hash_tampering_rejected(self):
        self.populate()
        path = self.root / "data" / "processed" / "factors.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(DatasetError, "hash"):
            fetch_dataset(self.root)

    def test_missing_cache_file_rejected(self):
        self.populate()
        (self.root / "data" / "processed" / "factors.csv").unlink()
        with self.assertRaisesRegex(DatasetError, "Incomplete"):
            fetch_dataset(self.root)

    def test_refresh_repairs_corrupt_cache(self):
        self.populate()
        path = self.root / "data" / "processed" / "factors.csv"
        path.write_bytes(b"corrupt")
        with patch("urllib.request.urlopen", side_effect=fake_download):
            returns, _, _ = fetch_dataset(self.root, refresh=True)
        self.assertEqual(len(returns), 3)

    def test_coverage_mismatch_does_not_write_manifest(self):
        def mismatch(request, timeout):
            text = industry_text() if request.full_url == SOURCES["industry"] else factor_text(("200001", "200002"))
            return io.BytesIO(zipped(text))
        with patch("urllib.request.urlopen", side_effect=mismatch):
            with self.assertRaisesRegex(DatasetError, "coverage"):
                fetch_dataset(self.root)
        self.assertFalse((self.root / "data" / "manifest.json").exists())

    def test_incomplete_first_cache_requires_explicit_refresh(self):
        path = self.root / "data" / "raw"
        path.mkdir(parents=True)
        (path / "partial.zip").write_bytes(b"partial")
        with self.assertRaisesRegex(DatasetError, "without a manifest"):
            fetch_dataset(self.root)

    def test_source_url_tampering_rejected(self):
        self.populate()
        path = self.root / "data" / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["sources"]["factors"]["url"] = "https://example.com/other.zip"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "source URL"):
            fetch_dataset(self.root)


if __name__ == "__main__":
    unittest.main()

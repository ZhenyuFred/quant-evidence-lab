import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from quant_evidence.backtest import (ExperimentSpec, candidate_specs, equal_weight_baseline,
                                     market_baseline, metrics, run_backtest)


def fixture(n=40, columns=4):
    rng = np.random.default_rng(1802)
    idx = pd.period_range("2000-01", periods=n, freq="M")
    returns = pd.DataFrame(rng.normal(0.007, 0.04, (n, columns)), index=idx,
                           columns=[f"asset_{j}" for j in range(columns)])
    return returns, pd.Series(0.001, index=idx, name="RF")


class BacktestTests(unittest.TestCase):
    def test_future_data_cannot_change_past_weights_or_returns(self):
        returns, rf = fixture()
        spec = ExperimentSpec("invariance", lookback=6, top_k=2, weighting="inverse_vol")
        original = run_backtest(returns, rf, spec)
        cutoff = returns.index[27]
        changed = returns.copy()
        changed.loc[cutoff:] = 0.7 - changed.loc[cutoff:] * 3
        altered = run_backtest(changed, rf, spec)
        pd.testing.assert_frame_equal(original["weights"].loc[:cutoff], altered["weights"].loc[:cutoff])
        pd.testing.assert_frame_equal(original["monthly"].loc[:cutoff - 1],
                                      altered["monthly"].loc[:cutoff - 1])

    def test_momentum_skips_latest_month_and_has_exact_lookback(self):
        idx = pd.period_range("2020-01", periods=6, freq="M")
        # In May, score must use Feb + Mar only. Jan and Apr are excluded.
        returns = pd.DataFrame({"A": [.9, .1, .1, -.8, 0, 0],
                                "B": [-.9, 0, 0, .9, 0, 0]}, index=idx)
        rf = pd.Series(0., index=idx)
        result = run_backtest(returns, rf, ExperimentSpec("timing", 2, 1, "equal",
                                                         vol_window=2, cost_bps=0), start="2020-05")
        self.assertEqual(result["weights"].loc["2020-05", "A"], 1)
        latest_changed = returns.copy()
        latest_changed.loc["2020-04"] = [-.99, 20]
        altered = run_backtest(latest_changed, rf, ExperimentSpec("timing", 2, 1, "equal",
                                                                 vol_window=2, cost_bps=0), start="2020-05")
        pd.testing.assert_series_equal(result["weights"].iloc[0], altered["weights"].iloc[0])

    def test_drift_cost_and_entry_by_hand(self):
        idx = pd.period_range("2020-01", periods=2, freq="M")
        returns = pd.DataFrame({"A": [.1, 0], "B": [0, 0]}, index=idx)
        result = equal_weight_baseline(returns, pd.Series(0., index=idx), cost_bps=100)
        monthly = result["monthly"]
        self.assertAlmostEqual(monthly.iloc[0].gross_return, .05)
        self.assertAlmostEqual(monthly.iloc[0].turnover, 1.)
        self.assertAlmostEqual(monthly.iloc[0].net_return, .04)
        # Drifting weights are .55/1.05 and .50/1.05, so both sides trade 1/42.
        self.assertAlmostEqual(monthly.iloc[1].turnover, 1 / 21)
        self.assertAlmostEqual(monthly.iloc[1].cost, .01 / 21)
        self.assertAlmostEqual(monthly.iloc[1].wealth, 1.04 * (1 - .01 / 21))

    def test_higher_cost_never_improves_wealth_and_does_not_change_weights(self):
        returns, rf = fixture()
        spec = ExperimentSpec("cost", 6, 2)
        low = run_backtest(returns, rf, replace(spec, cost_bps=0))
        high = run_backtest(returns, rf, replace(spec, cost_bps=50))
        pd.testing.assert_frame_equal(low["weights"], high["weights"])
        self.assertTrue((high["monthly"].net_return <= low["monthly"].net_return).all())
        self.assertTrue((high["monthly"].wealth <= low["monthly"].wealth).all())

    def test_holdings_conserve_capital_and_match_gross_returns(self):
        returns, rf = fixture()
        result = run_backtest(returns, rf, ExperimentSpec("conservation", 6, 3, "inverse_vol"))
        weights, monthly = result["weights"], result["monthly"]
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
        self.assertTrue((weights >= 0).all().all())
        np.testing.assert_allclose((weights * returns.loc[weights.index]).sum(axis=1),
                                   monthly.gross_return)
        np.testing.assert_allclose(monthly.net_return, monthly.gross_return - monthly.cost)
        np.testing.assert_allclose(monthly.excess_return, monthly.net_return - rf.loc[weights.index])

    def test_zero_volatility_and_ties_are_finite_and_deterministic(self):
        returns, rf = fixture(n=20)
        returns[:] = 0.01
        result = run_backtest(returns, rf, ExperimentSpec("ties", 6, 2, "inverse_vol"))
        np.testing.assert_allclose(result["weights"].iloc[0], [.5, .5, 0, 0])

    def test_market_excess_is_converted_to_total_and_only_entry_costs(self):
        idx = pd.period_range("2020-01", periods=3, freq="M")
        factors = pd.DataFrame({"Mkt-RF": [.02, -.01, .03], "RF": [.001, .001, .001]}, index=idx)
        result = market_baseline(factors, cost_bps=10)
        np.testing.assert_allclose(result["monthly"].gross_return, [.021, -.009, .031])
        np.testing.assert_allclose(result["monthly"].turnover, [1, 0, 0], atol=1e-14)

    def test_metrics_include_initial_drawdown_and_rebase_sliced_runs(self):
        idx = pd.period_range("2020-01", periods=3, freq="M")
        data = pd.DataFrame({"net_return": [-.1, 0, .2], "excess_return": [-.1, 0, .2],
                             "turnover": [1., 0, 0], "cost": [0., 0, 0],
                             "wealth": [.9, .9, 1.08]}, index=idx)
        result = metrics(data)
        self.assertAlmostEqual(result["max_drawdown"], -.1)
        self.assertAlmostEqual(result["total_return"], .08)
        self.assertAlmostEqual(result["cagr"], 1.08 ** 4 - 1)
        self.assertAlmostEqual(metrics(data.iloc[1:])["total_return"], .2)
        self.assertIsNone(metrics(data.iloc[:1])["sharpe"])

    def test_common_start_charges_entry_for_every_strategy(self):
        returns, rf = fixture()
        start = "2002-01"
        result = run_backtest(returns, rf, ExperimentSpec("start", 6, 2), start=start)
        baseline = equal_weight_baseline(returns, rf, start=start)
        self.assertEqual(str(result["monthly"].index[0]), start)
        self.assertEqual(result["monthly"].index.tolist(), baseline["monthly"].index.tolist())
        self.assertAlmostEqual(result["monthly"].iloc[0].turnover, 1.)
        self.assertAlmostEqual(baseline["monthly"].iloc[0].turnover, 1.)

    def test_invalid_inputs_are_rejected(self):
        returns, rf = fixture()
        spec = ExperimentSpec("invalid", 6, 2)
        broken = []
        broken.append(returns.iloc[::-1])
        broken.append(pd.concat([returns.iloc[:2], returns.iloc[1:]]))
        broken.append(returns.drop(returns.index[3]))
        nan = returns.copy(); nan.iloc[1, 0] = np.nan; broken.append(nan)
        bad = returns.copy(); bad.iloc[1, 0] = -99.99; broken.append(bad)
        date_index = returns.copy(); date_index.index = date_index.index.to_timestamp(); broken.append(date_index)
        for candidate in broken:
            with self.subTest(kind=str(candidate.index[:3])):
                with self.assertRaises(ValueError):
                    run_backtest(candidate, rf, spec)
        with self.assertRaises(ValueError):
            run_backtest(returns, rf.iloc[1:], spec)
        with self.assertRaises(ValueError):
            run_backtest(returns, rf, replace(spec, top_k=8))
        with self.assertRaises(ValueError):
            run_backtest(returns.iloc[:5], rf.iloc[:5], spec)
        with self.assertRaises(ValueError):
            ExperimentSpec("bad", skip_month=0)
        with self.assertRaises(ValueError):
            ExperimentSpec("bad", weighting="future")
        with self.assertRaises(ValueError):
            ExperimentSpec("bad", cost_bps=float("nan"))

    def test_candidate_space_is_fixed_and_small(self):
        candidates = candidate_specs()
        self.assertEqual(len(candidates), 12)
        self.assertEqual(len({spec.name for spec in candidates}), 12)
        self.assertEqual({spec.lookback for spec in candidates}, {6, 9, 12})


if __name__ == "__main__":
    unittest.main()

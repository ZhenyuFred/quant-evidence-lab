"""Deterministic monthly portfolio research, with explicit timing and costs.

``lookback=L, skip_month=1`` means exactly L observations ending at t-2
for a holding month t, not the sometimes-used 11-observation "12-2" rule.
Volatility uses only the ``vol_window`` observations ending at t-1.
Research portfolios are not directly tradable securities. Cost scenarios cover
only the hypothetical outer portfolio's rebalancing, not underlying turnover.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    lookback: int = 12
    top_k: int = 3
    weighting: str = "equal"
    skip_month: int = 1
    vol_window: int = 12
    cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        for field, minimum in (("lookback", 1), ("top_k", 1),
                               ("skip_month", 1), ("vol_window", 2)):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        if self.weighting not in ("equal", "inverse_vol"):
            raise ValueError("weighting must be equal or inverse_vol")
        _validate_cost(self.cost_bps)


def _validate_cost(cost_bps: float) -> None:
    if isinstance(cost_bps, bool) or not isinstance(cost_bps, (int, float)):
        raise ValueError("cost_bps must be numeric")
    if not math.isfinite(cost_bps) or not 0 <= cost_bps <= 10000:
        raise ValueError("cost_bps must be finite and between 0 and 10000")


def _validate_month_index(index: pd.Index, label: str) -> None:
    if not isinstance(index, pd.PeriodIndex) or index.freqstr != "M":
        raise ValueError(f"{label} needs a monthly PeriodIndex")
    if index.empty or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{label} months must be non-empty, unique and sorted")
    expected = pd.period_range(index[0], index[-1], freq="M")
    if not index.equals(expected):
        raise ValueError(f"{label} has missing months")


def _validate_inputs(returns: pd.DataFrame, rf: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    _validate_month_index(returns.index, "returns")
    if returns.columns.has_duplicates or not all(isinstance(c, str) and c for c in returns.columns):
        raise ValueError("return columns must be unique, non-empty names")
    if not all(pd.api.types.is_numeric_dtype(t) for t in returns.dtypes):
        raise ValueError("returns must contain numeric decimal returns")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1).any():
        raise ValueError("returns contain missing/non-finite values or a return below -100%")
    if not isinstance(rf, pd.Series):
        raise ValueError("rf must be a Series of decimal monthly returns")
    _validate_month_index(rf.index, "rf")
    if not pd.api.types.is_numeric_dtype(rf.dtype):
        raise ValueError("rf must contain numeric decimal returns")
    aligned_rf = rf.reindex(returns.index).astype(float)
    if not np.isfinite(aligned_rf.to_numpy()).all() or (aligned_rf <= -1).any():
        raise ValueError("rf must cover every return month with finite returns above -100%")
    return returns.astype(float), aligned_rf


def _start_position(index: pd.PeriodIndex, warmup: int, start: str | pd.Period | None) -> int:
    if start is None:
        pos = warmup
    else:
        month = pd.Period(start, freq="M")
        if month not in index:
            raise ValueError("start month must occur in the input data")
        pos = int(index.get_loc(month))
        if pos < warmup:
            raise ValueError("start month does not leave enough signal history")
    if pos >= len(index):
        raise ValueError("not enough observations after signal warmup")
    return pos


def _simulate(returns: pd.DataFrame, rf: pd.Series, targets: pd.DataFrame,
              cost_bps: float, spec: dict[str, Any]) -> dict[str, Any]:
    """Debit fees at month end; gross return minus cost is the net return.

    The pre-rebalance risky weights drift with gross asset returns. Fees are
    withdrawn pro rata from end-month holdings, leaving relative weights intact.
    ``turnover`` is total buy + sell notional / beginning portfolio value.
    Initial cash-to-risky entry therefore has turnover 1, not 0 or 0.5.
    """
    _validate_cost(cost_bps)
    if targets.empty:
        raise ValueError("no portfolio months to simulate")
    target_values = targets.to_numpy(dtype=float)
    if (not np.isfinite(target_values).all() or (target_values < 0).any()
            or not np.allclose(target_values.sum(axis=1), 1.0, atol=1e-12, rtol=0)):
        raise ValueError("target weights must be finite, non-negative and sum to one")
    previous_drifted = np.zeros(len(returns.columns), dtype=float)
    wealth = 1.0
    rows: list[dict[str, float]] = []
    for month, target in targets.iterrows():
        w = target.to_numpy(dtype=float)
        r = returns.loc[month].to_numpy(dtype=float)
        gross = float(w @ r)
        turnover = float(np.abs(w - previous_drifted).sum())
        cost = float(cost_bps / 10000 * turnover)
        net = gross - cost
        if gross <= -1 or net <= -1:
            raise ValueError(f"portfolio bankrupt in {month}; cannot continue the simulation")
        wealth *= 1 + net
        if not math.isfinite(wealth):
            raise ValueError("portfolio wealth overflowed")
        rows.append({"gross_return": gross, "net_return": net,
                     "excess_return": net - float(rf.loc[month]),
                     "turnover": turnover, "cost": cost, "wealth": wealth})
        previous_drifted = w * (1 + r) / (1 + gross)
    monthly = pd.DataFrame(rows, index=targets.index.copy())
    monthly.index.name = "month"
    return {"spec": spec, "monthly": monthly, "weights": targets.copy()}


def run_backtest(returns: pd.DataFrame, rf: pd.Series, spec: ExperimentSpec,
                 *, start: str | pd.Period | None = None) -> dict[str, Any]:
    """Run a long-only momentum experiment. Optional start keeps common samples.

    Input data before start remain available for feature construction, but the
    portfolio starts from cash at start. Stable ties use input column order.
    A fixed 1e-6 monthly volatility floor prevents division by zero.
    """
    returns, rf = _validate_inputs(returns, rf)
    if not isinstance(spec, ExperimentSpec):
        raise ValueError("spec must be an ExperimentSpec")
    if spec.top_k > len(returns.columns):
        raise ValueError("top_k exceeds the available industries")
    warmup = max(spec.lookback + spec.skip_month, spec.vol_window)
    first = _start_position(returns.index, warmup, start)
    values = returns.to_numpy(dtype=float)
    targets = np.zeros((len(returns) - first, len(returns.columns)), dtype=float)
    for row, t in enumerate(range(first, len(returns))):
        end = t - spec.skip_month
        scores = np.prod(1 + values[end - spec.lookback:end], axis=0) - 1
        chosen = np.argsort(-scores, kind="stable")[:spec.top_k]
        if spec.weighting == "equal":
            weight = np.full(spec.top_k, 1 / spec.top_k)
        else:
            vol = np.std(values[t - spec.vol_window:t, chosen], axis=0, ddof=1)
            inverse = 1 / np.maximum(vol, 1e-6)
            weight = inverse / inverse.sum()
        targets[row, chosen] = weight
    weights = pd.DataFrame(targets, index=returns.index[first:], columns=returns.columns)
    return _simulate(returns, rf, weights, spec.cost_bps, asdict(spec))


def equal_weight_baseline(returns: pd.DataFrame, rf: pd.Series,
                          cost_bps: float = 10.0, *,
                          start: str | pd.Period | None = None) -> dict[str, Any]:
    returns, rf = _validate_inputs(returns, rf)
    first = _start_position(returns.index, 0, start)
    weights = pd.DataFrame(1 / len(returns.columns), index=returns.index[first:],
                           columns=returns.columns)
    return _simulate(returns, rf, weights, cost_bps,
                     {"name": "all_industries_equal_weight", "weighting": "equal",
                      "top_k": len(returns.columns), "cost_bps": float(cost_bps)})


def market_baseline(factors: pd.DataFrame, *, cost_bps: float = 10.0,
                    start: str | pd.Period | None = None) -> dict[str, Any]:
    """Market buy-and-hold reference, charging only initial entry cost.

    Mkt-RF is an excess return: the total market return is Mkt-RF + RF.
    Outer turnover is zero after entry; underlying index turnover is unmodeled.
    """
    if not isinstance(factors, pd.DataFrame) or not {"Mkt-RF", "RF"}.issubset(factors.columns):
        raise ValueError("factors must contain Mkt-RF and RF")
    returns = pd.DataFrame({"Market": factors["Mkt-RF"] + factors["RF"]})
    returns, rf = _validate_inputs(returns, factors["RF"])
    first = _start_position(returns.index, 0, start)
    weights = pd.DataFrame(1.0, index=returns.index[first:], columns=["Market"])
    return _simulate(returns, rf, weights, cost_bps,
                     {"name": "market_buy_and_hold", "cost_bps": float(cost_bps)})


def candidate_specs(cost_bps: float = 10.0) -> list[ExperimentSpec]:
    return [ExperimentSpec(name=f"momentum_{lookback}m_top{top_k}_{weighting}",
                           lookback=lookback, top_k=top_k, weighting=weighting,
                           cost_bps=cost_bps)
            for lookback in (6, 9, 12)
            for top_k in (3, 6)
            for weighting in ("equal", "inverse_vol")]


def metrics(monthly: pd.DataFrame) -> dict[str, Any]:
    """Metrics recompute wealth from returns, so slicing a run is valid.

    Fees/holdings on a slice are inherited from the continuous run. A separately
    funded experiment must instead rerun with ``start=...`` to charge entry.
    Sharpe uses monthly excess returns and sample standard deviation; annual
    volatility uses net returns. Undefined one-month/zero-vol Sharpe is None.
    """
    required = {"net_return", "excess_return", "turnover", "cost"}
    if not isinstance(monthly, pd.DataFrame) or not required.issubset(monthly.columns):
        raise ValueError("monthly is missing required return/cost columns")
    _validate_month_index(monthly.index, "monthly")
    arrays = monthly[list(sorted(required))].to_numpy(dtype=float)
    if not np.isfinite(arrays).all():
        raise ValueError("monthly metrics contain missing or non-finite values")
    net = monthly["net_return"].to_numpy(dtype=float)
    excess = monthly["excess_return"].to_numpy(dtype=float)
    if (net <= -1).any() or (monthly["turnover"] < 0).any() or (monthly["cost"] < 0).any():
        raise ValueError("monthly returns must exceed -100%; costs/turnover cannot be negative")
    n = len(net)
    wealth = np.cumprod(1 + net)
    if not np.isfinite(wealth).all():
        raise ValueError("wealth overflowed while computing metrics")
    # Include starting capital 1: the first month's loss is a real drawdown.
    path = np.r_[1.0, wealth]
    drawdown = path / np.maximum.accumulate(path) - 1
    net_std = float(np.std(net, ddof=1)) if n > 1 else None
    excess_std = float(np.std(excess, ddof=1)) if n > 1 else None
    sharpe = (float(np.mean(excess) / excess_std * np.sqrt(12))
              if excess_std is not None and excess_std > 1e-15 else None)
    return {"n_months": n, "start": str(monthly.index[0]), "end": str(monthly.index[-1]),
            "total_return": float(wealth[-1] - 1),
            "cagr": float(np.expm1(np.log1p(net).sum() * 12 / n)),
            "ann_vol": float(net_std * np.sqrt(12)) if net_std is not None else None,
            "sharpe": sharpe, "max_drawdown": float(np.min(drawdown)),
            "avg_turnover": float(monthly["turnover"].mean()),
            "total_turnover": float(monthly["turnover"].sum()),
            "total_cost": float(monthly["cost"].sum()),
            "mean_excess_return_monthly": float(np.mean(excess)),
            "ann_return_arithmetic": float(np.mean(net) * 12)}

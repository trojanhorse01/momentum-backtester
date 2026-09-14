"""Portfolio weighting schemes: equal weight, inverse volatility, risk parity
(equal risk contribution) and mean-variance (Markowitz) optimization.

The per-date weight solves (risk parity / mean-variance) are genuine numerical
optimizations via ``scipy.optimize.minimize`` -- they are not closed-form
shortcuts. Because each rebalance is an independent small optimization problem
(dimensionality = number of active assets, typically a handful to a few dozen),
the natural implementation loops over rebalance dates only (there are ~dozens
to ~hundreds of these over a multi-year backtest, not one per trading day).
The full daily P&L simulation in :mod:`momentum_backtester.engine` -- the part
that actually scales with the number of trading days -- stays fully vectorized.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

WeightingMethod = Literal["equal", "inverse_vol", "risk_parity", "mean_variance"]


# --------------------------------------------------------------------------
# Single-date weight solvers (pure numpy in, numpy out)
# --------------------------------------------------------------------------


def equal_weights(n: int) -> np.ndarray:
    """1/n weight for each of n assets."""
    if n <= 0:
        return np.array([])
    return np.full(n, 1.0 / n)


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """Weights inversely proportional to each asset's own volatility, normalized to sum to 1."""
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    vol = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    inv = 1.0 / vol
    return inv / inv.sum()


def risk_parity_weights(cov: np.ndarray, w0: np.ndarray | None = None, max_iter: int = 1000) -> np.ndarray:
    """Equal Risk Contribution (ERC) weights via numerical optimization.

    Solves for long-only weights ``w`` (summing to 1) that equalize each
    asset's contribution to total portfolio variance,
    ``RC_i = w_i * (cov @ w)_i``, by minimizing the sum of squared pairwise
    differences between risk contributions subject to ``sum(w) = 1`` and
    ``w_i >= 0``.
    """
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    if w0 is None:
        w0 = equal_weights(n)

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        if port_var <= 1e-16:
            return 0.0
        marginal = cov @ w
        risk_contrib = w * marginal
        # Work in *relative* (percent-of-total-risk) terms, not raw variance
        # units. Raw daily variances are ~1e-4 to 1e-6, which makes the raw
        # sum-of-squared-differences objective so small in absolute magnitude
        # that SLSQP's finite-difference gradient estimate is effectively
        # flat at the equal-weight starting point and the optimizer reports
        # spurious convergence without moving -- i.e. it silently returns
        # equal weights regardless of the true covariance structure.
        # Normalizing by port_var rescales the objective to O(1) so the
        # optimizer can actually resolve differences in risk contribution.
        pct_risk_contrib = risk_contrib / port_var
        target = 1.0 / n
        return float(np.sum((pct_risk_contrib - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0) for _ in range(n)]
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-16},
    )
    w = np.clip(result.x, 0.0, None)
    total = w.sum()
    if total <= 0:
        return equal_weights(n)
    return w / total


def mean_variance_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    objective: Literal["max_sharpe", "min_variance"] = "max_sharpe",
    risk_free: float = 0.0,
    long_only: bool = True,
    w0: np.ndarray | None = None,
    max_iter: int = 1000,
) -> np.ndarray:
    """Markowitz mean-variance optimal weights via ``scipy.optimize.minimize``.

    Parameters
    ----------
    mu : expected (per-period) returns for each asset.
    cov : per-period covariance matrix.
    objective : "max_sharpe" maximizes ``(w.mu - rf) / sqrt(w.cov.w)``;
        "min_variance" minimizes ``w.cov.w``.
    risk_free : per-period risk-free rate used in the Sharpe objective.
    long_only : if True, bounds each weight to [0, 1]; otherwise [-1, 1].
    w0 : optional initial guess (defaults to equal weight).

    Returns weights that satisfy ``sum(w) == 1`` (budget constraint enforced
    both by the optimizer's equality constraint and, defensively, by a final
    renormalization).
    """
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    if w0 is None:
        w0 = equal_weights(n)

    if objective == "min_variance":

        def obj(w: np.ndarray) -> float:
            return float(w @ cov @ w)

    elif objective == "max_sharpe":

        def obj(w: np.ndarray) -> float:
            port_ret = w @ mu - risk_free
            port_vol = np.sqrt(max(w @ cov @ w, 1e-16))
            return float(-port_ret / port_vol)

    else:
        raise ValueError(f"Unknown objective: {objective}")

    bound = (0.0, 1.0) if long_only else (-1.0, 1.0)
    bounds = [bound for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-14},
    )
    w = result.x
    total = w.sum()
    if abs(total) < 1e-8:
        return equal_weights(n)
    return w / total


# --------------------------------------------------------------------------
# Time-series orchestration: turn a signal + return history into a full
# daily weight panel, solving one small optimization per rebalance date.
# --------------------------------------------------------------------------


def _solve_book_weights(
    method: WeightingMethod,
    sub_returns: pd.DataFrame,
    objective: str,
    risk_free_period: float,
) -> pd.Series:
    """Solve unsigned (long-book) weights, summing to 1, for one basket of assets."""
    assets = sub_returns.columns
    n = len(assets)
    if n == 0:
        return pd.Series(dtype=float)
    cov = sub_returns.cov().values
    if method == "equal":
        w = equal_weights(n)
    elif method == "inverse_vol":
        w = inverse_vol_weights(cov)
    elif method == "risk_parity":
        w = risk_parity_weights(cov)
    elif method == "mean_variance":
        mu = sub_returns.mean().values
        w = mean_variance_weights(mu, cov, objective=objective, risk_free=risk_free_period, long_only=True)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    return pd.Series(w, index=assets)


def build_portfolio_weights(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    method: WeightingMethod = "risk_parity",
    lookback: int = 63,
    rebalance: str = "ME",
    mv_objective: Literal["max_sharpe", "min_variance"] = "max_sharpe",
    risk_free_annual: float = 0.0,
    min_history: int = 20,
) -> pd.DataFrame:
    """Turn a {-1,0,+1} signal panel into a full daily target-weight panel.

    At each rebalance date, assets flagged ``+1`` in the signal form the long
    book (weights solved to sum to +1 via ``method``) and assets flagged
    ``-1`` form the short book (weights solved to sum to -1). Between
    rebalance dates, weights are held constant (forward-filled) -- the
    backtest engine is responsible for shifting these by one period to avoid
    lookahead bias before applying them to returns.

    Parameters
    ----------
    prices : (dates x assets) price panel (used only to derive daily returns
        for covariance/mean estimation).
    signal : {-1, 0, +1} panel aligned to ``prices``, e.g. from
        :mod:`momentum_backtester.signals`.
    method : one of "equal", "inverse_vol", "risk_parity", "mean_variance".
    lookback : number of trailing daily returns used to estimate covariance
        (and mean, for mean-variance) at each rebalance date.
    rebalance : pandas offset alias for rebalance frequency (default month-end).
    mv_objective : objective used when ``method == "mean_variance"``.
    risk_free_annual : annualized risk-free rate, converted to a per-period
        rate for the Sharpe objective.
    min_history : minimum number of trailing return observations required
        before a rebalance is solved (early dates with insufficient history
        are skipped, leaving weights at 0).

    Returns
    -------
    DataFrame of daily target weights (dates x assets), NOT yet shifted for
    lookahead -- pass directly to :func:`momentum_backtester.engine.run_backtest`.
    """
    returns = prices.pct_change()
    risk_free_period = (1 + risk_free_annual) ** (1 / 252) - 1

    rebalance_dates = prices.resample(rebalance).last().index
    rebalance_dates = rebalance_dates[rebalance_dates.isin(prices.index)]
    # ensure we also solve on the very first available signal date so the
    # portfolio isn't flat for an entire extra period
    if len(rebalance_dates) == 0:
        rebalance_dates = prices.index[:1]

    weight_rows: dict[pd.Timestamp, pd.Series] = {}

    for date in rebalance_dates:
        loc = prices.index.get_loc(date)
        if loc < min_history:
            continue
        window = returns.iloc[max(0, loc - lookback + 1) : loc + 1]

        row_signal = signal.loc[date]
        longs = row_signal[row_signal > 0].index
        shorts = row_signal[row_signal < 0].index

        full_row = pd.Series(0.0, index=prices.columns)

        if len(longs) > 0:
            sub = window[longs].dropna(axis=1, how="any")
            if len(sub.columns) > 0:
                w_long = _solve_book_weights(method, sub, mv_objective, risk_free_period)
                full_row.loc[w_long.index] += w_long.values

        if len(shorts) > 0:
            sub = window[shorts].dropna(axis=1, how="any")
            if len(sub.columns) > 0:
                w_short = _solve_book_weights(method, sub, mv_objective, risk_free_period)
                full_row.loc[w_short.index] += -w_short.values

        weight_rows[date] = full_row

    if not weight_rows:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    weights_at_rebal = pd.DataFrame(weight_rows).T
    weights_at_rebal.index.name = prices.index.name
    daily_weights = weights_at_rebal.reindex(prices.index).ffill().fillna(0.0)
    return daily_weights

"""The vectorized backtest engine.

Turns a (dates x assets) target-weight panel and a (dates x assets) price
panel into a cost- and slippage-adjusted portfolio return series and equity
curve. The entire per-day simulation is pure pandas/numpy vector arithmetic
(``.shift()``, elementwise multiply, ``.cumprod()``) -- there is no Python
loop over trading days, so runtime scales to large universes and long
histories without a per-day interpreter overhead.

Timing convention (no lookahead)
---------------------------------
``weights.loc[t]`` is the target weight decided using information available
at (or before) the close of day ``t``. It is held **during** day ``t`` in the
sense that it earns the realized return from close(t-1) to close(t) -- i.e.
the P&L engine uses ``weights.shift(1)`` against ``returns`` so that the
return earned on day ``t`` only ever depends on a decision made strictly
before day ``t``'s return is known.

Transaction costs and slippage are charged as a single all-in bps rate on
*turnover* -- the L1 change in weights from one day to the next
(``sum(|w_t - w_{t-1}|)``) -- which is the standard vectorized-backtest proxy
for "how much of the portfolio had to be traded". Cost is recognized on the
date the weight change occurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Container for all outputs of :func:`run_backtest`."""

    weights: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    equity_curve: pd.Series

    @property
    def drawdown(self) -> pd.Series:
        running_max = self.equity_curve.cummax()
        return self.equity_curve / running_max - 1.0


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    membership: pd.DataFrame | None = None,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """Run a fully vectorized backtest of a target-weight panel against prices.

    Parameters
    ----------
    prices : (dates x assets) price panel.
    weights : (dates x assets) target weights aligned to ``prices``' index and
        columns (e.g. from :func:`momentum_backtester.weighting.build_portfolio_weights`).
        NOT pre-shifted -- this function applies the one-period lag internally.
    cost_bps : one-way transaction cost in basis points, charged on turnover.
    slippage_bps : additional execution slippage in basis points, charged on
        turnover exactly like ``cost_bps`` (kept as a separate parameter so
        the two frictions can be reported/attributed independently).
    membership : optional (dates x assets) boolean point-in-time universe
        mask. When provided, weights are defensively zeroed wherever
        ``membership`` is False (an asset can never be held once it has left
        the tradable universe, e.g. after a simulated delisting), and if any
        such zeroing occurs the freed-up weight is not reinvested (that
        capital simply sits in cash for that period, mirroring realistic
        delisting friction).
    initial_capital : starting equity-curve level (default 1.0).

    Returns
    -------
    BacktestResult with weights actually used, gross/net daily returns,
    turnover, costs, and the resulting equity curve.
    """
    if not prices.index.equals(weights.index):
        weights = weights.reindex(prices.index).fillna(0.0)
    weights = weights.reindex(columns=prices.columns).fillna(0.0)

    if membership is not None:
        mem = membership.reindex(index=prices.index, columns=prices.columns).fillna(False)
        weights = weights.where(mem, 0.0)

    returns = prices.pct_change()

    # position held during day t was decided at close of t-1 -> shift(1)
    shifted_weights = weights.shift(1).fillna(0.0)
    gross_returns = (shifted_weights * returns).sum(axis=1)
    gross_returns = gross_returns.fillna(0.0)
    # first row of `returns` is always NaN (no prior price) -> no gross return
    gross_returns.iloc[0] = 0.0

    # turnover(t) = L1 trade required to move from weights(t-1) to weights(t),
    # with an implicit weights(-1) = 0 (the very first allocation is a trade
    # out of cash and is charged like any other rebalance).
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()

    all_in_bps = (cost_bps + slippage_bps) / 1e4
    costs = all_in_bps * turnover

    net_returns = gross_returns - costs
    equity_curve = initial_capital * (1.0 + net_returns).cumprod()

    return BacktestResult(
        weights=weights,
        gross_returns=gross_returns,
        net_returns=net_returns,
        turnover=turnover,
        costs=costs,
        equity_curve=equity_curve,
    )

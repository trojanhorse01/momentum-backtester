"""Sanity tests for the vectorized backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from momentum_backtester.engine import run_backtest


def _make_prices(n_days=100, n_assets=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    tickers = [f"T{i}" for i in range(n_assets)]
    log_rets = rng.normal(0.0003, 0.01, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(log_rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def test_output_shapes_and_index_alignment():
    prices = _make_prices()
    weights = pd.DataFrame(1.0 / 3, index=prices.index, columns=prices.columns)
    result = run_backtest(prices, weights, cost_bps=5, slippage_bps=2)
    assert result.net_returns.shape[0] == prices.shape[0]
    assert result.equity_curve.shape[0] == prices.shape[0]
    assert result.turnover.shape[0] == prices.shape[0]
    assert list(result.weights.columns) == list(prices.columns)
    assert result.equity_curve.index.equals(prices.index)


def test_zero_signal_zero_cost_gives_flat_equity_curve():
    # no signal at all -> weights are always zero -> no trades, no costs,
    # net returns are exactly zero every period, equity curve stays at 1.0.
    prices = _make_prices()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    result = run_backtest(prices, weights, cost_bps=10, slippage_bps=10)
    assert np.allclose(result.net_returns.values, 0.0)
    assert np.allclose(result.turnover.values, 0.0)
    assert np.allclose(result.costs.values, 0.0)
    assert np.allclose(result.equity_curve.values, 1.0)


def test_zero_cost_path_matches_buy_and_hold_equal_weight():
    prices = _make_prices()
    weights = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    result = run_backtest(prices, weights, cost_bps=0.0, slippage_bps=0.0)
    expected_gross = (weights.shift(1).fillna(0.0) * prices.pct_change()).sum(axis=1)
    expected_gross.iloc[0] = 0.0
    assert np.allclose(result.net_returns.values, expected_gross.values, equal_nan=True)


def test_costs_reduce_returns_relative_to_zero_cost():
    prices = _make_prices(n_days=60, n_assets=4, seed=1)
    # a signal that flips every 5 days to generate meaningful turnover
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(0, len(prices), 5):
        active = prices.columns[(i // 5) % prices.shape[1]]
        weights.loc[weights.index[i]:, :] = 0.0
        weights.loc[weights.index[i]:, active] = 1.0

    no_cost = run_backtest(prices, weights, cost_bps=0.0, slippage_bps=0.0)
    with_cost = run_backtest(prices, weights, cost_bps=20.0, slippage_bps=10.0)
    assert with_cost.costs.sum() > 0
    assert with_cost.equity_curve.iloc[-1] < no_cost.equity_curve.iloc[-1]
    # net = gross - cost, exactly, every period
    assert np.allclose(
        with_cost.net_returns.values,
        (with_cost.gross_returns - with_cost.costs).values,
    )


def test_initial_allocation_incurs_turnover():
    prices = _make_prices(n_days=20, n_assets=2, seed=2)
    weights = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    result = run_backtest(prices, weights, cost_bps=10.0)
    # first-day turnover should reflect trading in from an all-cash start
    assert result.turnover.iloc[0] == pytest.approx(1.0)
    # after day 0 weights never change -> zero turnover afterwards
    assert np.allclose(result.turnover.iloc[1:].values, 0.0)


def test_membership_mask_zeroes_out_delisted_weight():
    prices = _make_prices(n_days=30, n_assets=2, seed=3)
    weights = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    membership = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    membership.iloc[10:, 1] = False  # asset T1 delists on day 10
    result = run_backtest(prices, weights, membership=membership)
    assert (result.weights.iloc[10:, 1] == 0.0).all()
    assert (result.weights.iloc[:10, 1] == 0.5).all()


def test_drawdown_property_matches_manual_calculation():
    prices = _make_prices(n_days=50, n_assets=2, seed=4)
    weights = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    result = run_backtest(prices, weights)
    manual_dd = result.equity_curve / result.equity_curve.cummax() - 1.0
    assert np.allclose(result.drawdown.values, manual_dd.values)

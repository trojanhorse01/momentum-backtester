"""Hand-computable known-answer tests for the metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from momentum_backtester.metrics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
)


def test_cagr_constant_return():
    # A constant 1% per period for 252 periods (exactly one year) compounds
    # to (1.01)^252 - 1 total growth, which by construction *is* the CAGR.
    returns = pd.Series([0.01] * 252)
    expected = 1.01**252 - 1.0
    assert cagr(returns, periods_per_year=252) == pytest.approx(expected, rel=1e-9)


def test_cagr_doubles_in_one_year():
    # Growth factor of exactly 2 over exactly one year of periods => CAGR = 100%.
    n = 252
    growth_factor = 2.0
    per_period = growth_factor ** (1 / n) - 1
    returns = pd.Series([per_period] * n)
    assert cagr(returns, periods_per_year=252) == pytest.approx(1.0, rel=1e-6)


def test_annualized_volatility_known_std():
    # Constant-magnitude alternating returns have a simple, hand-computable std.
    returns = pd.Series([0.01, -0.01] * 126)
    expected_daily_std = returns.std(ddof=1)
    expected = expected_daily_std * np.sqrt(252)
    assert annualized_volatility(returns) == pytest.approx(expected)


def test_sharpe_ratio_zero_vol_is_nan():
    returns = pd.Series([0.001] * 100)
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_ratio_manual_calculation():
    returns = pd.Series([0.02, 0.01, -0.01, 0.03, 0.0, -0.02, 0.015])
    rf_annual = 0.0
    periods_per_year = 252
    excess = returns - rf_annual / periods_per_year
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)
    assert sharpe_ratio(returns, rf_annual, periods_per_year) == pytest.approx(expected)


def test_sortino_ratio_only_penalizes_downside():
    # All-positive returns => zero downside deviation => Sortino is NaN (no
    # downside risk to divide by), which is the correct edge-case behavior.
    returns = pd.Series([0.01, 0.02, 0.005, 0.03])
    assert np.isnan(sortino_ratio(returns))


def test_sortino_ratio_manual_calculation():
    returns = pd.Series([0.05, -0.02, 0.01, -0.04, 0.0, 0.03])
    downside = np.minimum(returns - 0.0, 0.0)
    downside_dev = np.sqrt(np.mean(downside**2))
    expected = returns.mean() / downside_dev * np.sqrt(252)
    assert sortino_ratio(returns) == pytest.approx(expected)


def test_max_drawdown_known_path():
    # Equity path 100 -> 110 -> 90 -> 120: trough 90 after peak 110 => -18.18%
    prices = [100, 110, 90, 120]
    returns = pd.Series(prices).pct_change().dropna().reset_index(drop=True)
    expected_mdd = 90.0 / 110.0 - 1.0
    assert max_drawdown(returns) == pytest.approx(expected_mdd, rel=1e-9)


def test_max_drawdown_monotonic_up_is_zero():
    returns = pd.Series([0.01, 0.02, 0.005, 0.03])
    assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-12)


def test_calmar_ratio_matches_cagr_over_mdd():
    returns = pd.Series([0.05, -0.02, 0.01, -0.04, 0.0, 0.03, 0.02, -0.01])
    expected = cagr(returns) / abs(max_drawdown(returns))
    assert calmar_ratio(returns) == pytest.approx(expected)


def test_performance_summary_has_all_expected_keys():
    returns = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 500))
    summary = performance_summary(returns)
    expected_keys = {
        "Total Return",
        "CAGR",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "Calmar Ratio",
    }
    assert set(summary.index) == expected_keys
    assert summary.notna().all()

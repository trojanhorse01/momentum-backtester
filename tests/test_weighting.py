"""Tests for portfolio weighting schemes: inverse vol, risk parity, mean-variance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from momentum_backtester.weighting import (
    build_portfolio_weights,
    equal_weights,
    inverse_vol_weights,
    mean_variance_weights,
    risk_parity_weights,
)


def test_equal_weights_sum_to_one():
    w = equal_weights(5)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, 0.2)


def test_inverse_vol_weights_are_inverse_to_volatility():
    # Two uncorrelated assets, asset 2 has 3x the vol of asset 1.
    cov = np.array([[1.0, 0.0], [0.0, 9.0]])
    w = inverse_vol_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    # weight ratio should be inverse of vol ratio (vol ratio = 3 -> weight ratio = 1/3)
    assert w[0] / w[1] == pytest.approx(3.0, rel=1e-6)


def test_inverse_vol_weights_equal_when_equal_vol():
    cov = np.eye(4) * 2.5
    w = inverse_vol_weights(cov)
    assert np.allclose(w, 0.25)


def test_risk_parity_weights_sum_to_one_and_nonnegative():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(500, 4))
    cov = np.cov(a, rowvar=False)
    w = risk_parity_weights(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-8).all()


def test_risk_parity_equalizes_risk_contributions():
    # Construct a diagonal covariance (uncorrelated assets) with distinct
    # vols -- for uncorrelated assets, exact ERC weights are known
    # analytically: w_i proportional to 1/vol_i (same as inverse-vol).
    vols = np.array([0.1, 0.2, 0.4])
    cov = np.diag(vols**2)
    w = risk_parity_weights(cov)
    risk_contrib = w * (cov @ w)
    # all risk contributions should be equal (within optimizer tolerance)
    assert np.allclose(risk_contrib, risk_contrib.mean(), atol=1e-6)
    # and for the diagonal case this matches closed-form inverse-vol weights
    expected = inverse_vol_weights(cov)
    assert np.allclose(w, expected, atol=1e-4)


def test_risk_parity_single_asset():
    cov = np.array([[0.04]])
    w = risk_parity_weights(cov)
    assert w == pytest.approx([1.0])


def test_mean_variance_weights_sum_to_one_min_variance():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(500, 5))
    cov = np.cov(a, rowvar=False)
    mu = rng.uniform(0.0001, 0.001, size=5)
    w = mean_variance_weights(mu, cov, objective="min_variance", long_only=True)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-8).all()
    assert (w <= 1 + 1e-8).all()


def test_mean_variance_min_variance_beats_equal_weight_variance():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(1000, 6))
    cov = np.cov(a, rowvar=False)
    mu = rng.uniform(0.0001, 0.001, size=6)
    w_mv = mean_variance_weights(mu, cov, objective="min_variance")
    w_eq = equal_weights(6)
    var_mv = w_mv @ cov @ w_mv
    var_eq = w_eq @ cov @ w_eq
    assert var_mv <= var_eq + 1e-10


def test_mean_variance_max_sharpe_sums_to_one():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(500, 4))
    cov = np.cov(a, rowvar=False)
    mu = rng.uniform(0.0002, 0.002, size=4)
    w = mean_variance_weights(mu, cov, objective="max_sharpe", long_only=True)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_build_portfolio_weights_shape_and_bounds():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=300)
    tickers = [f"A{i}" for i in range(6)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=(300, 6)), axis=0)),
        index=dates,
        columns=tickers,
    )
    # simple long-only signal: always long the first 3 assets
    signal = pd.DataFrame(0, index=dates, columns=tickers)
    signal[["A0", "A1", "A2"]] = 1

    weights = build_portfolio_weights(prices, signal, method="risk_parity", lookback=40, rebalance="ME")
    assert weights.shape == prices.shape
    # weights should never exceed the long-only [0, 1] gross budget per asset
    assert (weights >= -1e-8).all().all()
    # rows after warm-up should sum close to 1 (fully invested, long-only)
    late_rows = weights.iloc[100:]
    row_sums = late_rows.sum(axis=1)
    assert (row_sums.round(6).isin([0.0]) | np.isclose(row_sums, 1.0, atol=1e-6)).all()

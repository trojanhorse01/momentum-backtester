"""Tests for the synthetic data generator and the survivorship-bias demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from momentum_backtester.data import (
    generate_gbm_universe,
    simulate_delistings,
    survivors_only_prices,
)
from momentum_backtester.engine import run_backtest
from momentum_backtester.metrics import cagr


def test_generate_gbm_universe_shape_and_reproducibility():
    prices = generate_gbm_universe(n_assets=10, n_years=2, seed=123)
    assert prices.shape[1] == 10
    assert prices.shape[0] == int(round(2 * 252))
    assert (prices > 0).all().all()

    prices2 = generate_gbm_universe(n_assets=10, n_years=2, seed=123)
    pd.testing.assert_frame_equal(prices, prices2)


def test_generate_gbm_universe_different_seed_differs():
    p1 = generate_gbm_universe(n_assets=5, n_years=1, seed=1)
    p2 = generate_gbm_universe(n_assets=5, n_years=1, seed=2)
    assert not p1.equals(p2)


def test_simulate_delistings_marks_membership_false_after_delisting():
    prices = generate_gbm_universe(n_assets=20, n_years=8, seed=42)
    pit_prices, membership = simulate_delistings(prices, frac_delisted=0.4, seed=7)

    assert membership.shape == prices.shape
    # at least some assets should have left the universe before the end
    ever_delisted = ~membership.all(axis=0)
    assert ever_delisted.sum() > 0
    # once False, an asset should stay False for the rest of the sample
    for col in membership.columns[ever_delisted]:
        col_membership = membership[col].values
        first_false = np.argmax(~col_membership)
        assert not col_membership[first_false:].any()


def test_survivors_only_excludes_delisted_names():
    prices = generate_gbm_universe(n_assets=20, n_years=8, seed=42)
    pit_prices, membership = simulate_delistings(prices, frac_delisted=0.4, seed=7)
    survivors = survivors_only_prices(pit_prices, membership)
    ever_delisted = set(membership.columns[~membership.all(axis=0)])
    assert set(survivors.columns).isdisjoint(ever_delisted)
    assert set(survivors.columns).issubset(set(prices.columns))


def test_survivorship_bias_overstates_performance():
    """The core claim of the project: a backtest that only uses assets which
    survived to the end of the sample overstates CAGR relative to the
    point-in-time universe that correctly includes (and then excludes) the
    names that were delisted along the way, since delisted names disproportionately
    were the worst performers.
    """
    prices = generate_gbm_universe(n_assets=40, n_years=10, seed=99, avg_correlation=0.2)
    pit_prices, membership = simulate_delistings(
        prices, frac_delisted=0.4, delisting_shock=(-0.9, -0.5), seed=11
    )
    survivors = survivors_only_prices(pit_prices, membership)

    # simple equal-weight long-only buy-and-hold on both universes
    pit_weights = pd.DataFrame(0.0, index=pit_prices.index, columns=pit_prices.columns)
    active_counts = membership.sum(axis=1).replace(0, np.nan)
    pit_weights = membership.div(active_counts, axis=0).fillna(0.0)

    survivor_weights = pd.DataFrame(
        1.0 / survivors.shape[1], index=survivors.index, columns=survivors.columns
    )

    pit_result = run_backtest(pit_prices, pit_weights, membership=membership)
    survivor_result = run_backtest(survivors, survivor_weights)

    pit_cagr = cagr(pit_result.net_returns)
    survivor_cagr = cagr(survivor_result.net_returns)

    assert survivor_cagr > pit_cagr

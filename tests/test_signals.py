"""Tests for time-series and cross-sectional momentum signal construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from momentum_backtester.signals import (
    cross_sectional_momentum_signal,
    time_series_momentum_signal,
    trailing_return,
)


def test_trailing_return_known_values():
    prices = pd.DataFrame({"A": [100, 110, 121, 133.1, 146.41]})
    trail = trailing_return(prices, lookback=2, skip=0)
    # trail[2] = price[2]/price[0] - 1 = 121/100 - 1 = 0.21
    assert trail["A"].iloc[2] == pytest.approx(0.21)
    # trail[4] = price[4]/price[2] - 1 = 146.41/121 - 1
    assert trail["A"].iloc[4] == pytest.approx(146.41 / 121 - 1)
    assert trail["A"].iloc[:2].isna().all()


def test_trailing_return_with_skip():
    prices = pd.DataFrame({"A": [100, 110, 121, 133.1, 146.41, 150.0]})
    trail = trailing_return(prices, lookback=2, skip=1)
    # trail[5] = price[5-1]/price[5-1-2] - 1 = price[4]/price[2]-1
    assert trail["A"].iloc[5] == pytest.approx(146.41 / 121 - 1)


def test_time_series_momentum_signal_sign():
    dates = pd.bdate_range("2021-01-01", periods=10)
    up = pd.Series(np.linspace(100, 200, 10), index=dates)  # strictly rising
    down = pd.Series(np.linspace(100, 50, 10), index=dates)  # strictly falling
    prices = pd.DataFrame({"UP": up, "DOWN": down})
    signal = time_series_momentum_signal(prices, lookback=3, skip=0)
    assert (signal["UP"].dropna() >= 0).all()
    assert (signal["UP"].iloc[4:] == 1.0).all()
    assert (signal["DOWN"].iloc[4:] == -1.0).all()


def test_time_series_momentum_respects_membership():
    dates = pd.bdate_range("2021-01-01", periods=10)
    up = pd.Series(np.linspace(100, 200, 10), index=dates)
    prices = pd.DataFrame({"UP": up})
    membership = pd.DataFrame({"UP": [True] * 5 + [False] * 5}, index=dates)
    signal = time_series_momentum_signal(prices, lookback=2, skip=0, membership=membership)
    assert (signal.loc[~membership["UP"], "UP"] == 0.0).all()


def test_cross_sectional_momentum_ranks_assets():
    dates = pd.bdate_range("2021-01-01", periods=30)
    # five assets spanning a clear performance spectrum: only the extremes
    # should be flagged when top/bottom fractions are narrow (top 1/5, bottom 1/5).
    winner = 100 * np.exp(np.cumsum(np.full(30, 0.03)))
    upper_mid = 100 * np.exp(np.cumsum(np.full(30, 0.01)))
    flat = 100 * np.exp(np.cumsum(np.full(30, 0.0001)))
    lower_mid = 100 * np.exp(np.cumsum(np.full(30, -0.01)))
    loser = 100 * np.exp(np.cumsum(np.full(30, -0.03)))
    prices = pd.DataFrame(
        {"WINNER": winner, "UPPER_MID": upper_mid, "FLAT": flat, "LOWER_MID": lower_mid, "LOSER": loser},
        index=dates,
    )

    signal = cross_sectional_momentum_signal(
        prices, lookback=10, skip=0, top_frac=0.2, bottom_frac=0.2, long_only=False
    )
    last = signal.iloc[-1]
    assert last["WINNER"] == 1.0
    assert last["LOSER"] == -1.0
    assert last["FLAT"] == 0.0
    assert last["UPPER_MID"] == 0.0
    assert last["LOWER_MID"] == 0.0


def test_cross_sectional_momentum_long_only_has_no_shorts():
    dates = pd.bdate_range("2021-01-01", periods=30)
    winner = 100 * np.exp(np.cumsum(np.full(30, 0.02)))
    loser = 100 * np.exp(np.cumsum(np.full(30, -0.02)))
    prices = pd.DataFrame({"WINNER": winner, "LOSER": loser}, index=dates)
    signal = cross_sectional_momentum_signal(prices, lookback=10, top_frac=0.5, bottom_frac=0.5, long_only=True)
    assert (signal >= 0).all().all()


def test_cross_sectional_momentum_bad_fracs_raise():
    prices = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        cross_sectional_momentum_signal(prices, top_frac=0.0)
    with pytest.raises(ValueError):
        cross_sectional_momentum_signal(prices, bottom_frac=1.0)

"""Momentum signal construction -- fully vectorized (no Python-level loops over time).

Both signal families are built from a single primitive, the *trailing return*
lookback with an optional recent-months skip (the classic "12-1" momentum
construction: use the trailing 12-month return but skip the most recent month
to avoid short-term reversal contaminating the signal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_return(prices: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Trailing total return over a lookback window, vectorized via `.shift()`.

    ``trailing_return(t) = price[t - skip] / price[t - skip - lookback] - 1``

    Using only ``.shift()`` (no rolling ``.apply``) keeps this O(1) numpy
    vectorized work regardless of lookback length, and guarantees the value at
    row ``t`` only references information available at or before ``t`` --
    i.e. no lookahead bias.

    Parameters
    ----------
    prices : (dates x assets) price panel.
    lookback : window length in periods (e.g. 252 for ~12 months of daily data).
    skip : most-recent periods to exclude (e.g. 21 for the "-1 month" in "12-1").
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if skip < 0:
        raise ValueError("skip must be non-negative")
    numerator = prices.shift(skip)
    denominator = prices.shift(skip + lookback)
    return numerator / denominator - 1.0


def time_series_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Time-series (absolute) momentum signal: +1 if trailing return > 0 else -1.

    Each asset is scored independently against its *own* history (no
    cross-sectional comparison) -- the classic trend-following construction
    of Moskowitz, Ooi & Pedersen (2012). Assets outside the point-in-time
    universe (``membership`` False, when provided) are forced flat.

    Returns a DataFrame of {-1, 0, +1} aligned to ``prices``' index/columns.
    """
    trail = trailing_return(prices, lookback=lookback, skip=skip)
    signal = np.sign(trail)
    signal = signal.fillna(0.0)
    if membership is not None:
        signal = signal.where(membership.reindex_like(signal).fillna(False), 0.0)
    return signal


def cross_sectional_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    top_frac: float = 0.3,
    bottom_frac: float = 0.3,
    long_only: bool = False,
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cross-sectional momentum signal: rank assets by trailing return each day.

    At every date, assets currently in the tradable universe are ranked by
    trailing return. The top ``top_frac`` are flagged ``+1`` (long) and, unless
    ``long_only``, the bottom ``bottom_frac`` are flagged ``-1`` (short); all
    others are ``0``. Ranking uses `.rank(pct=True)` cross-sectionally
    (``axis=1``), which is fully vectorized across both dates and assets.

    Assets are selected by exact rank *count* each row (``ceil(frac *
    n_active_assets)``), not by a fuzzy percentile-rank threshold -- this
    avoids off-by-one boundary artifacts when the number of assets is small
    or does not divide evenly by the requested fraction, and matches the
    standard "top decile / bottom decile" construction used in the momentum
    literature (Jegadeesh & Titman, 1993).

    Returns a DataFrame of {-1, 0, +1} aligned to ``prices``' index/columns.
    """
    if not 0 < top_frac <= 1 or not 0 <= bottom_frac < 1:
        raise ValueError("top_frac must be in (0, 1] and bottom_frac in [0, 1)")

    trail = trailing_return(prices, lookback=lookback, skip=skip)
    if membership is not None:
        mem = membership.reindex_like(trail).fillna(False)
        trail = trail.where(mem)

    valid = trail.notna()
    n_active = valid.sum(axis=1)
    top_n = np.ceil(top_frac * n_active)
    bottom_n = np.ceil(bottom_frac * n_active) if not long_only else pd.Series(0, index=trail.index)

    # rank_best: 1 = highest trailing return (best momentum) in the row.
    # rank_worst: 1 = lowest trailing return (worst momentum) in the row.
    # method="first" breaks ties deterministically so counts are exact.
    rank_best = trail.rank(axis=1, ascending=False, method="first")
    rank_worst = trail.rank(axis=1, ascending=True, method="first")

    signal = pd.DataFrame(0.0, index=trail.index, columns=trail.columns)
    long_mask = valid & rank_best.le(top_n, axis=0)
    signal = signal.mask(long_mask, 1.0)
    if not long_only:
        short_mask = valid & rank_worst.le(bottom_n, axis=0)
        signal = signal.mask(short_mask, -1.0)
    return signal

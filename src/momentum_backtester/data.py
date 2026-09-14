"""Synthetic multi-asset price data generation.

The core pipeline never depends on a network call: :func:`generate_gbm_universe`
produces correlated geometric-Brownian-motion price paths for a configurable
number of synthetic tickers, and :func:`simulate_delistings` overlays a
point-in-time universe (membership matrix) so the backtester can demonstrate
survivorship bias on data with a known ground truth.

A thin, clearly separated opt-in loader for real data via ``yfinance`` lives in
``examples/fetch_real_data.py`` -- it is intentionally kept out of the tested
core so the test suite never depends on network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _random_correlation_matrix(n_assets: int, rng: np.random.Generator, block_corr: float = 0.35) -> np.ndarray:
    """Build a valid (positive semi-definite) correlation matrix.

    Uses a one-factor model: each asset loads on a common market factor plus
    idiosyncratic noise, which guarantees positive semi-definiteness while
    giving realistic, uniformly positive average pairwise correlation.
    """
    loadings = rng.uniform(0.2, 0.8, size=n_assets) * np.sqrt(block_corr)
    idio_var = 1.0 - loadings**2
    cov = np.outer(loadings, loadings)
    cov[np.diag_indices(n_assets)] = loadings**2 + idio_var
    # normalize to a correlation matrix (diagonal exactly 1)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def generate_gbm_universe(
    n_assets: int = 40,
    n_years: float = 10.0,
    start: str = "2014-01-02",
    freq: str = "B",
    mu_range: tuple[float, float] = (0.02, 0.14),
    sigma_range: tuple[float, float] = (0.15, 0.45),
    avg_correlation: float = 0.35,
    seed: int | None = 42,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Generate correlated GBM daily price paths for a synthetic equity universe.

    Parameters
    ----------
    n_assets : number of synthetic tickers.
    n_years : approximate length of history in years.
    start : first business-day date of the sample.
    freq : pandas date offset alias for the trading calendar (default business days).
    mu_range, sigma_range : per-asset annualized drift / volatility are drawn
        uniformly from these ranges, giving cross-sectional dispersion in
        risk/return (needed for momentum to have anything to bite on).
    avg_correlation : target average pairwise correlation across assets.
    seed : RNG seed for full reproducibility.
    tickers : optional explicit list of ticker names (default ``SYN000, SYN001, ...``).

    Returns
    -------
    DataFrame of shape (n_days, n_assets) indexed by trading date, price levels
    starting at 100.0 for every asset.
    """
    rng = np.random.default_rng(seed)
    n_days = int(round(n_years * 252))
    dates = pd.bdate_range(start=start, periods=n_days, freq=freq)

    if tickers is None:
        tickers = [f"SYN{i:03d}" for i in range(n_assets)]
    if len(tickers) != n_assets:
        raise ValueError("len(tickers) must equal n_assets")

    corr = _random_correlation_matrix(n_assets, rng, block_corr=avg_correlation)
    chol = np.linalg.cholesky(corr)

    mu = rng.uniform(*mu_range, size=n_assets)
    sigma = rng.uniform(*sigma_range, size=n_assets)

    dt = 1.0 / 252.0
    z = rng.standard_normal(size=(n_days, n_assets))
    z_corr = z @ chol.T  # correlate the innovations

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z_corr
    log_returns = drift + diffusion

    log_prices = np.cumsum(log_returns, axis=0)
    prices = 100.0 * np.exp(log_prices)

    return pd.DataFrame(prices, index=dates, columns=tickers)


def simulate_delistings(
    prices: pd.DataFrame,
    frac_delisted: float = 0.35,
    min_history_frac: float = 0.25,
    delisting_shock: tuple[float, float] = (-0.9, -0.4),
    seed: int | None = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overlay realistic delistings on a synthetic price panel to model survivorship bias.

    Assets with the *worst* trailing cumulative performance at a randomly
    chosen "delisting eligibility" checkpoint are the ones most likely to be
    delisted (mirroring reality: weak companies go bankrupt / get acquired
    out of the index / delisted), each with a sharp negative "delisting
    shock" return. After delisting, the asset's price is held flat (NaN in
    the returned membership matrix marks it as no longer tradable) so a
    careless backtester who simply forward-fills price and keeps trading it
    would not be getting a real economic loss baked into the return.

    Parameters
    ----------
    prices : output of :func:`generate_gbm_universe`.
    frac_delisted : fraction of the universe that eventually delists.
    min_history_frac : delistings only happen after this fraction of the
        sample has elapsed (avoids delisting on day one).
    delisting_shock : (low, high) uniform range for the one-day return applied
        on the delisting date, simulating a bankruptcy/delisting price crash.
    seed : RNG seed.

    Returns
    -------
    (point_in_time_prices, membership) :
        point_in_time_prices : price panel with delisted assets' prices held
            constant (last traded price) after delisting -- the membership
            matrix is what should actually be used to exclude them from the
            tradable universe going forward.
        membership : boolean DataFrame, True where the asset is part of the
            tradable, point-in-time universe on that date.
    """
    rng = np.random.default_rng(seed)
    n_days, n_assets = prices.shape
    tickers = prices.columns

    n_delisted = int(round(frac_delisted * n_assets))
    checkpoint = int(n_days * 0.6)  # evaluate trailing performance mid-sample
    start_idx = int(n_days * min_history_frac)

    trailing_perf = prices.iloc[checkpoint] / prices.iloc[start_idx] - 1.0
    # worst performers are most likely (not certain) to be delisted: rank and
    # sample from the bottom half with higher probability, keeping some randomness
    ranked = trailing_perf.sort_values().index
    candidate_pool = list(ranked[: max(n_delisted * 2, n_delisted)])
    delisted_tickers = rng.choice(candidate_pool, size=min(n_delisted, len(candidate_pool)), replace=False)

    membership = pd.DataFrame(True, index=prices.index, columns=tickers)
    pit_prices = prices.copy()

    delist_days = rng.integers(checkpoint, n_days - 1, size=len(delisted_tickers))
    shocks = rng.uniform(*delisting_shock, size=len(delisted_tickers))

    for ticker, day_idx, shock in zip(delisted_tickers, delist_days, shocks):
        # apply the delisting shock on the delisting day, then hold price flat
        pit_prices.iloc[day_idx:, pit_prices.columns.get_loc(ticker)] = prices.iloc[day_idx][ticker] * (1 + shock)
        membership.iloc[day_idx + 1 :, membership.columns.get_loc(ticker)] = False

    return pit_prices, membership


def survivors_only_prices(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Return the (biased) panel containing only assets that survived the *entire* sample.

    This is what a naive backtest that pulls "current index constituents" and
    back-fills their full price history would use -- it silently excludes
    every company that went bankrupt/delisted, which is exactly the
    survivorship bias effect this project demonstrates.
    """
    survivors = membership.all(axis=0)
    return prices.loc[:, survivors[survivors].index]

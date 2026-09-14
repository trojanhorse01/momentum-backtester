"""Risk-adjusted performance metrics.

All functions take a ``pandas.Series`` of *periodic* (e.g. daily) simple
returns and an ``periods_per_year`` annualization factor (default 252 for
daily equity data). Formulas follow the standard definitions used throughout
quantitative finance practice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve_from_returns(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """Compound a periodic-return series into a cumulative equity curve."""
    return initial_capital * (1.0 + returns.fillna(0.0)).cumprod()


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate implied by a periodic-return series."""
    n_periods = len(returns)
    if n_periods == 0:
        return float("nan")
    # equity_curve_from_returns starts from initial_capital=1.0 and compounds
    # every period in `returns` into it, so equity.iloc[-1] IS the total
    # growth factor over all n_periods (no need to divide by equity.iloc[0],
    # which already reflects the first period's return).
    equity = equity_curve_from_returns(returns)
    total_growth = equity.iloc[-1]
    if total_growth <= 0:
        return float("nan")
    years = n_periods / periods_per_year
    if years <= 0:
        return float("nan")
    return float(total_growth ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of periodic returns."""
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio: mean excess return / volatility of excess return."""
    rf_period = risk_free_annual / periods_per_year
    excess = returns - rf_period
    vol = excess.std(ddof=1)
    if np.isnan(vol) or vol < 1e-12:
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """Annualized Sortino ratio: mean excess return / downside deviation.

    Downside deviation only penalizes returns falling below ``target_return``
    (per-period; default 0), matching the standard Sortino construction.
    """
    rf_period = risk_free_annual / periods_per_year
    excess = returns - rf_period
    downside = np.minimum(returns - target_return, 0.0)
    downside_dev = np.sqrt(np.mean(np.square(downside)))
    if np.isnan(downside_dev) or downside_dev < 1e-12:
        return float("nan")
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number, e.g. -0.23 for -23%)."""
    equity = equity_curve_from_returns(returns)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR divided by the absolute value of maximum drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(cagr(returns, periods_per_year) / abs(mdd))


def performance_summary(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute the full standard set of risk-adjusted performance metrics.

    Returns a pandas Series with: CAGR, Annualized Volatility, Sharpe Ratio,
    Sortino Ratio, Max Drawdown, Calmar Ratio, and total return.
    """
    equity = equity_curve_from_returns(returns)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else float("nan")
    return pd.Series(
        {
            "Total Return": total_return,
            "CAGR": cagr(returns, periods_per_year),
            "Annualized Volatility": annualized_volatility(returns, periods_per_year),
            "Sharpe Ratio": sharpe_ratio(returns, risk_free_annual, periods_per_year),
            "Sortino Ratio": sortino_ratio(returns, risk_free_annual, periods_per_year),
            "Max Drawdown": max_drawdown(returns),
            "Calmar Ratio": calmar_ratio(returns, periods_per_year),
        }
    )

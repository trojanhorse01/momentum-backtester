"""momentum_backtester: a vectorized momentum-strategy backtesting toolkit.

Modules
-------
data        : synthetic correlated-GBM price generation and survivorship simulation.
signals     : time-series and cross-sectional momentum signal construction.
weighting   : equal-weight, inverse-volatility, risk-parity and mean-variance
              portfolio weighting schemes.
engine      : the vectorized backtest engine (turns weights + prices into a
              cost- and slippage-adjusted equity curve).
metrics     : risk-adjusted performance metrics (Sharpe, Sortino, MDD, CAGR, ...).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("momentum-backtester")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"

__all__ = ["__version__"]

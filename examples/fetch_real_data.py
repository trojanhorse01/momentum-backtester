"""Opt-in example: run the exact same pipeline on real market data via yfinance.

This script is intentionally kept OUT of the tested core (`src/momentum_backtester`)
and out of the default example (`run_backtest.py`) so the test suite and the
main demo never depend on network access or a third-party data vendor's
availability. Install the optional extra first:

    pip install "momentum-backtester[yfinance]"

Then run:

    python examples/fetch_real_data.py --tickers AAPL MSFT GOOGL AMZN META NVDA ...

Everything downstream of the price panel (signals, weighting, the backtest
engine, metrics) is identical to `run_backtest.py` -- only the data source
changes. Note real data pulled this way is NOT survivorship-bias-free (Yahoo
Finance only serves currently-listed tickers), which is precisely why the
core pipeline's survivorship-bias demonstration in `run_backtest.py` uses the
synthetic point-in-time universe instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from momentum_backtester.engine import run_backtest  # noqa: E402
from momentum_backtester.metrics import performance_summary  # noqa: E402
from momentum_backtester.signals import cross_sectional_momentum_signal  # noqa: E402
from momentum_backtester.weighting import build_portfolio_weights  # noqa: E402

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "UNH",
    "HD", "PG", "MA", "XOM", "KO", "PEP", "COST", "ADBE", "CRM", "NFLX",
]


def main() -> None:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "yfinance is not installed. Run: pip install \"momentum-backtester[yfinance]\""
        ) from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--period", default="10y")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args()

    print(f"Downloading {len(args.tickers)} tickers over {args.period} via yfinance...")
    raw = yf.download(args.tickers, period=args.period, auto_adjust=True, progress=False)["Close"]
    prices = raw.dropna(axis=1, how="all").ffill().dropna()
    print(f"Got {prices.shape[0]} trading days x {prices.shape[1]} tickers.")

    signal = cross_sectional_momentum_signal(prices, lookback=252, skip=21, top_frac=0.3, bottom_frac=0.3)
    weights = build_portfolio_weights(prices, signal, method="risk_parity", rebalance="ME")
    result = run_backtest(prices, weights, cost_bps=args.cost_bps, slippage_bps=args.slippage_bps)
    metrics = performance_summary(result.net_returns)

    print("\nCross-sectional momentum + risk parity on real data (net of costs):")
    print(metrics.round(4).to_string())


if __name__ == "__main__":
    main()

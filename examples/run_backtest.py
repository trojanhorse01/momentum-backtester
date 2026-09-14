"""End-to-end example: synthetic data -> signals -> weighting -> vectorized
backtest with costs/slippage -> risk-adjusted performance metrics -> charts.

Run with:
    python examples/run_backtest.py

Produces (in examples/output/):
    equity_curve.png             - main strategy equity curve + drawdown
    survivorship_bias.png        - point-in-time vs survivors-only comparison
    weighting_comparison.png     - Sharpe/CAGR across weighting schemes & signal types
    metrics_summary.csv          - all metrics tables in one CSV, for the README
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from momentum_backtester.data import (  # noqa: E402
    generate_gbm_universe,
    simulate_delistings,
    survivors_only_prices,
)
from momentum_backtester.engine import run_backtest  # noqa: E402
from momentum_backtester.metrics import performance_summary  # noqa: E402
from momentum_backtester.signals import (  # noqa: E402
    cross_sectional_momentum_signal,
    time_series_momentum_signal,
)
from momentum_backtester.weighting import build_portfolio_weights  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

N_ASSETS = 45
N_YEARS = 10
LOOKBACK = 252
SKIP = 21
TOP_FRAC = 0.2
REBALANCE = "QE"  # quarterly -- keeps turnover (and cost drag) realistic for a momentum book
COST_BPS = 10.0
SLIPPAGE_BPS = 5.0
RISK_FREE_ANNUAL = 0.02
# Wider cross-sectional dispersion in drift and a lower average correlation
# than the library default give momentum something real to detect (pure i.i.d.
# GBM otherwise has no genuine serial-correlation edge beyond noisy drift
# estimation) -- see README for the full discussion of this design choice.
MU_RANGE = (-0.02, 0.20)
AVG_CORRELATION = 0.15

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    }
)


def fmt_pct(x: float) -> str:
    return f"{x:.2%}" if pd.notna(x) else "nan"


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("Quantitative Momentum Trading Strategy Backtester -- example run")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Synthetic multi-asset universe (correlated GBM, no network calls)
    # ------------------------------------------------------------------
    print(f"\n[1/6] Generating synthetic universe: {N_ASSETS} tickers, {N_YEARS} years of daily data...")
    prices_no_delist = generate_gbm_universe(
        n_assets=N_ASSETS, n_years=N_YEARS, seed=42, mu_range=MU_RANGE, avg_correlation=AVG_CORRELATION
    )
    pit_prices, membership = simulate_delistings(
        prices_no_delist, frac_delisted=0.35, delisting_shock=(-0.85, -0.45), seed=7
    )
    survivors = survivors_only_prices(pit_prices, membership)
    n_delisted = N_ASSETS - survivors.shape[1]
    print(f"      {prices_no_delist.shape[0]} trading days, {n_delisted} of {N_ASSETS} names simulated to delist.")

    # ------------------------------------------------------------------
    # 2. Signals: cross-sectional momentum (primary) + time-series momentum
    # ------------------------------------------------------------------
    print("\n[2/6] Computing momentum signals (12-1 month trailing return, point-in-time universe)...")
    xs_signal = cross_sectional_momentum_signal(
        pit_prices, lookback=LOOKBACK, skip=SKIP, top_frac=TOP_FRAC, bottom_frac=TOP_FRAC, membership=membership
    )
    ts_signal = time_series_momentum_signal(pit_prices, lookback=LOOKBACK, skip=SKIP, membership=membership)

    # ------------------------------------------------------------------
    # 3. Weighting: risk parity (primary), plus a comparison across schemes
    # ------------------------------------------------------------------
    print("\n[3/6] Solving portfolio weights (risk parity via scipy.optimize, quarterly rebalance)...")
    xs_rp_weights = build_portfolio_weights(pit_prices, xs_signal, method="risk_parity", rebalance=REBALANCE)

    # ------------------------------------------------------------------
    # 4. Vectorized backtest with transaction costs + slippage
    # ------------------------------------------------------------------
    print(f"\n[4/6] Running vectorized backtest (cost={COST_BPS}bps, slippage={SLIPPAGE_BPS}bps per unit turnover)...")
    main_result = run_backtest(
        pit_prices, xs_rp_weights, cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS, membership=membership
    )
    main_metrics = performance_summary(main_result.net_returns, risk_free_annual=RISK_FREE_ANNUAL)

    print("\n      Cross-sectional momentum + risk parity (point-in-time universe, net of costs):")
    for k, v in main_metrics.items():
        print(f"        {k:<24s}: {fmt_pct(v) if 'Ratio' not in k else f'{v:.3f}'}")

    # ------------------------------------------------------------------
    # 5. Survivorship bias demonstration: point-in-time vs survivors-only
    # ------------------------------------------------------------------
    # NOTE: this comparison deliberately uses a simple, diversified equal-
    # weight LONG-ONLY buy-and-hold-style portfolio (not the long-short
    # momentum strategy above). Survivorship bias is a universe-construction
    # problem that classically distorts any strategy that holds a broad,
    # diversified basket of "current index members" -- a long-short momentum
    # strategy actively shorts/avoids the very names that go on to delist,
    # which confounds (and can even reverse) the bias's sign. Isolating it
    # with a plain long-only benchmark, at zero cost, cleanly demonstrates
    # the effect on its own terms, matching how it is usually taught.
    print("\n[5/6] Demonstrating survivorship bias (point-in-time universe vs. survivors-only)...")

    active_counts = membership.sum(axis=1).replace(0, pd.NA)
    pit_bh_weights = membership.div(active_counts, axis=0).astype(float).fillna(0.0)
    survivors_bh_weights = pd.DataFrame(1.0 / survivors.shape[1], index=survivors.index, columns=survivors.columns)

    pit_bh_result = run_backtest(pit_prices, pit_bh_weights, cost_bps=0.0, slippage_bps=0.0, membership=membership)
    survivors_bh_result = run_backtest(survivors, survivors_bh_weights, cost_bps=0.0, slippage_bps=0.0)

    pit_bh_metrics = performance_summary(pit_bh_result.net_returns, risk_free_annual=RISK_FREE_ANNUAL)
    survivors_bh_metrics = performance_summary(survivors_bh_result.net_returns, risk_free_annual=RISK_FREE_ANNUAL)

    bias_table = pd.DataFrame(
        {
            "Point-in-Time Universe (correct)": pit_bh_metrics,
            "Survivors-Only (biased)": survivors_bh_metrics,
        }
    )
    bias_table["Bias (Survivors - PIT)"] = (
        bias_table["Survivors-Only (biased)"] - bias_table["Point-in-Time Universe (correct)"]
    )
    print("      Equal-weight, long-only, zero-cost buy-and-hold on both universes:")
    print(bias_table.round(4).to_string())

    # ------------------------------------------------------------------
    # 6. Weighting-scheme and signal-type comparison
    # ------------------------------------------------------------------
    print("\n[6/6] Comparing weighting schemes and signal types (point-in-time universe, net of costs)...")
    scheme_results: dict[str, pd.Series] = {}
    for method in ["equal", "inverse_vol", "risk_parity", "mean_variance"]:
        w = build_portfolio_weights(pit_prices, xs_signal, method=method, rebalance=REBALANCE)
        res = run_backtest(pit_prices, w, cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS, membership=membership)
        scheme_results[f"XS Momentum + {method}"] = performance_summary(res.net_returns, risk_free_annual=RISK_FREE_ANNUAL)

    ts_weights = build_portfolio_weights(pit_prices, ts_signal, method="risk_parity", rebalance=REBALANCE)
    ts_result = run_backtest(pit_prices, ts_weights, cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS, membership=membership)
    scheme_results["TS Momentum + risk_parity"] = performance_summary(ts_result.net_returns, risk_free_annual=RISK_FREE_ANNUAL)

    comparison_table = pd.DataFrame(scheme_results).T
    print(comparison_table.round(4).to_string())

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    print("\nSaving charts to examples/output/ ...")

    # Chart 1: main strategy equity curve + drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax1.plot(main_result.equity_curve.index, main_result.equity_curve.values, color="#1f77b4", lw=1.4)
    ax1.set_title("Cross-Sectional Momentum + Risk Parity -- Point-in-Time Universe, Net of Costs")
    ax1.set_ylabel("Growth of $1")
    ax1.set_yscale("log")
    dd = main_result.drawdown
    ax2.fill_between(dd.index, dd.values * 100, 0, color="#d62728", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "equity_curve.png", dpi=150)
    plt.close(fig)

    # Chart 2: survivorship bias comparison
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(pit_bh_result.equity_curve.index, pit_bh_result.equity_curve.values, label="Point-in-Time Universe (correct)", color="#1f77b4", lw=1.4)
    ax.plot(survivors_bh_result.equity_curve.index, survivors_bh_result.equity_curve.values, label="Survivors-Only (biased)", color="#ff7f0e", lw=1.4, linestyle="--")
    ax.set_title("Survivorship Bias: Equal-Weight Buy-and-Hold, Point-in-Time vs. Survivors-Only")
    ax.set_ylabel("Growth of $1")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "survivorship_bias.png", dpi=150)
    plt.close(fig)

    # Chart 3: weighting-scheme / signal-type comparison (Sharpe + CAGR bars)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = list(comparison_table.index)
    axa.barh(labels, comparison_table["Sharpe Ratio"], color="#2ca02c")
    axa.set_title("Sharpe Ratio by Strategy Configuration")
    axa.set_xlabel("Sharpe Ratio")
    axb.barh(labels, comparison_table["CAGR"] * 100, color="#9467bd")
    axb.set_title("CAGR by Strategy Configuration")
    axb.set_xlabel("CAGR (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weighting_comparison.png", dpi=150)
    plt.close(fig)

    # Save all metrics tables to one CSV for the README / further analysis
    with open(OUTPUT_DIR / "metrics_summary.csv", "w") as f:
        f.write("# Main strategy metrics\n")
        main_metrics.to_frame("Value").to_csv(f)
        f.write("\n# Survivorship bias comparison\n")
        bias_table.to_csv(f)
        f.write("\n# Weighting scheme / signal type comparison\n")
        comparison_table.to_csv(f)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Charts and metrics written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

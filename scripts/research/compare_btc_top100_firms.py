#!/usr/bin/env python3
"""
compare_btc_top100_firms.py — Comprehensive comparative analysis between Bitcoin (BTC)
and the World's Top 100 Mega-Cap Firms (represented by the S&P 100 ^OEX,
Nasdaq 100 QQQ, and the largest individual global mega-cap firms).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def run_top100_comparison():
    print("Loading Bitcoin 1-sec dataset …")
    df = pd.read_parquet(DATA_DIR / "btc_1sec_nov2026.parquet")
    df["date"] = pd.to_datetime(df["timestamp_utc"].dt.date)

    btc_daily = df.groupby("date").agg(
        btc_open=("open", "first"),
        btc_close=("close", "last")
    ).reset_index()
    btc_daily["btc_ret"] = (btc_daily["btc_close"] / btc_daily["btc_open"] - 1) * 100

    print("Fetching Global Top 100 Indices and Mega-Caps via yfinance …")
    import yfinance as yf

    # Indices: ^OEX (S&P 100), QQQ (Nasdaq 100), SPY (S&P 500)
    # Global Mega-Caps: Tech, Financials, Healthcare, Consumer, Energy
    tickers = {
        # Benchmarks
        "^OEX": "S&P 100 (Top 100)",
        "QQQ": "Nasdaq 100 (Tech 100)",
        "SPY": "S&P 500",
        # Top Global Mega-Caps
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "NVDA": "NVIDIA",
        "AMZN": "Amazon",
        "GOOGL": "Alphabet/Google",
        "META": "Meta Platforms",
        "TSLA": "Tesla",
        "JPM": "JPMorgan Chase",
        "LLY": "Eli Lilly",
        "XOM": "ExxonMobil",
        "WMT": "Walmart",
        "V": "Visa"
    }

    raw = yf.download(list(tickers.keys()), start="2026-11-01", end="2026-11-26", progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:
        closes = raw.copy()

    closes = closes.reset_index()
    closes["Date"] = pd.to_datetime(closes["Date"])

    # Calculate returns for each asset
    ret_df = pd.DataFrame()
    ret_df["Date"] = closes["Date"]
    for col in tickers.keys():
        if col in closes.columns:
            ret_df[col] = closes[col].pct_change() * 100

    # Merge with Bitcoin
    merged = btc_daily.merge(closes, left_on="date", right_on="Date", how="inner")
    merged_ret = btc_daily.merge(ret_df, left_on="date", right_on="Date", how="inner").dropna()

    # 1. Performance Overview
    print("\n" + "=" * 70)
    print("  NOVEMBER 2026: BITCOIN vs WORLD TOP 100 BENCHMARKS & FIRMS")
    print("=" * 70)

    perf_rows = []
    base_btc = merged["btc_close"].iloc[0]
    final_btc = merged["btc_close"].iloc[-1]
    btc_m_ret = (final_btc / base_btc - 1) * 100

    perf_rows.append({
        "Asset": "Bitcoin (BTC)",
        "Category": "Crypto Asset",
        "Month Return": btc_m_ret,
        "Max Drawdown": (merged["btc_close"].min() / merged["btc_close"].max() - 1) * 100,
        "Daily Volatility (Std)": merged_ret["btc_ret"].std(),
        "Correlation with BTC": 1.0,
        "Beta to Top 100": np.nan
    })

    # Benchmark beta reference: ^OEX (S&P 100)
    top100_ret = merged_ret["^OEX"]
    top100_var = np.var(top100_ret)

    for ticker, name in tickers.items():
        if ticker in merged.columns and ticker in merged_ret.columns:
            start_val = merged[ticker].iloc[0]
            end_val = merged[ticker].iloc[-1]
            tot_ret = (end_val / start_val - 1) * 100
            max_dd = (merged[ticker].min() / merged[ticker].max() - 1) * 100
            vol_daily = merged_ret[ticker].std()
            corr, _ = pearsonr(merged_ret["btc_ret"], merged_ret[ticker])
            # Beta to S&P 100
            cov_top100 = np.cov(merged_ret[ticker], top100_ret)[0, 1]
            beta_top100 = cov_top100 / top100_var if top100_var > 0 else 1.0

            # Determine category
            if ticker in ["^OEX", "QQQ", "SPY"]:
                cat = "Index / Benchmark"
            elif ticker in ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN"]:
                cat = "Mega-Cap Tech"
            elif ticker in ["JPM", "V"]:
                cat = "Financial Giant"
            elif ticker in ["LLY"]:
                cat = "Healthcare"
            elif ticker in ["XOM"]:
                cat = "Energy"
            else:
                cat = "Consumer / Auto"

            perf_rows.append({
                "Asset": f"{name} ({ticker})",
                "Category": cat,
                "Month Return": tot_ret,
                "Max Drawdown": max_dd,
                "Daily Volatility (Std)": vol_daily,
                "Correlation with BTC": corr,
                "Beta to Top 100": beta_top100
            })

    # Compute Bitcoin's Beta to Top 100 (^OEX)
    cov_btc_top100 = np.cov(merged_ret["btc_ret"], top100_ret)[0, 1]
    btc_beta = cov_btc_top100 / top100_var
    perf_rows[0]["Beta to Top 100"] = btc_beta

    perf_df = pd.DataFrame(perf_rows)
    print(f"\n{'Asset':<26} {'Category':<18} {'Return':>8} {'MaxDD':>8} {'DailyVol':>9} {'Corr(BTC)':>10} {'Beta(Top100)':>13}")
    print("─" * 98)
    for _, r in perf_df.iterrows():
        b_str = f"{r['Beta to Top 100']:>11.2f}x" if pd.notna(r['Beta to Top 100']) else "      N/A"
        print(f"{r['Asset']:<26} {r['Category']:<18} {r['Month Return']:>+7.2f}% {r['Max Drawdown']:>7.2f}% {r['Daily Volatility (Std)']:>8.2f}% {r['Correlation with BTC']:>9.2f} {b_str}")

    # ── GENERATE COMPARISON CHART ────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor="#0f0f1a")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    # Panel 1: Normalized Performance Timeline (Top left & right spanning)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0f0f1a")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333355")

    # Normalize to base 100
    dates = merged["date"]
    ax1.plot(dates, (merged["btc_close"] / merged["btc_close"].iloc[0] - 1) * 100,
             color="#00d4ff", linewidth=2.8, label="Bitcoin (BTC/USD)", zorder=10)
    ax1.plot(dates, (merged["^OEX"] / merged["^OEX"].iloc[0] - 1) * 100,
             color="#ffffff", linewidth=2.0, linestyle="--", label="S&P 100 Index (^OEX) — World Top 100", zorder=8)
    ax1.plot(dates, (merged["QQQ"] / merged["QQQ"].iloc[0] - 1) * 100,
             color="#ff7f0e", linewidth=1.8, label="Nasdaq 100 (QQQ) — Tech Top 100", zorder=7)
    ax1.plot(dates, (merged["NVDA"] / merged["NVDA"].iloc[0] - 1) * 100,
             color="#76b900", linewidth=1.5, alpha=0.85, label="NVIDIA (NVDA)", zorder=6)
    ax1.plot(dates, (merged["AAPL"] / merged["AAPL"].iloc[0] - 1) * 100,
             color="#a2aaad", linewidth=1.3, alpha=0.7, label="Apple (AAPL)", zorder=5)
    ax1.plot(dates, (merged["MSFT"] / merged["MSFT"].iloc[0] - 1) * 100,
             color="#00a4ef", linewidth=1.3, alpha=0.7, label="Microsoft (MSFT)", zorder=5)

    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax1.set_title("Normalized Return Timeline: Bitcoin vs World Top 100 Benchmarks & Mega-Caps (Nov 2026)",
                  fontsize=13, fontweight="bold", color="white", pad=10)
    ax1.set_ylabel("Cumulative Change (%)", color="white", fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", ncol=3, loc="lower left")

    # Panel 2: Correlation with Bitcoin by Asset / Sector (Bottom Left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0f0f1a")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333355")

    sub_perf = perf_df.iloc[1:].sort_values("Correlation with BTC", ascending=True)
    colors = ["#2ca02c" if c >= 0.3 else ("#ff7f0e" if c >= 0.0 else "#d62728") for c in sub_perf["Correlation with BTC"]]

    ax2.barh(sub_perf["Asset"], sub_perf["Correlation with BTC"], color=colors, edgecolor="#222240")
    ax2.axvline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.4)
    ax2.set_xlim(-0.4, 0.8)
    ax2.set_title("Correlation with Bitcoin: Tech High-Beta vs Defensive Giants", fontsize=11, fontweight="bold", color="white")
    ax2.set_xlabel("Pearson Correlation Coefficient (r)", color="white", fontsize=10)

    # Panel 3: Scatter Regression: BTC Daily Return vs S&P 100 Top 100 (Bottom Right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0f0f1a")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#333355")

    x_vals = merged_ret["^OEX"]
    y_vals = merged_ret["btc_ret"]
    ax3.scatter(x_vals, y_vals, color="#00d4ff", s=50, alpha=0.85, edgecolors="white", linewidth=0.5)

    # Fit regression line
    m_slope, b_intercept = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min() - 0.5, x_vals.max() + 0.5, 50)
    ax3.plot(x_line, m_slope * x_line + b_intercept, color="#ffd700", linestyle="--",
             linewidth=1.8, label=f"Fit: Slope (Beta) = {m_slope:.2f}x")

    ax3.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
    ax3.axvline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
    ax3.set_xlabel("S&P 100 Index Daily Return (%)", color="white", fontsize=10)
    ax3.set_ylabel("Bitcoin Daily Return (%)", color="white", fontsize=10)
    ax3.set_title(f"Market Sensitivity: Bitcoin Beta to Top 100 = {btc_beta:.2f}x", fontsize=11, fontweight="bold", color="white")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white")

    fig.tight_layout()
    out = FIG_DIR / "btc_vs_top100_world_firms.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"\n Saved comparison chart: {out}")


if __name__ == "__main__":
    run_top100_comparison()

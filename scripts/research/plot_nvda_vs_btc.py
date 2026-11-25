#!/usr/bin/env python3
"""
plot_nvda_vs_btc.py — Cross-market analysis between NVIDIA (NVDA) and Bitcoin (BTC)
in November 2026.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def plot_nvda_btc():
    # 1. Load BTC
    df = pd.read_parquet(DATA_DIR / "btc_1sec_nov2026.parquet")
    df["date"] = pd.to_datetime(df["timestamp_utc"].dt.date)
    btc_daily = df.groupby("date").agg(
        btc_open=("open", "first"),
        btc_close=("close", "last")
    ).reset_index()
    btc_daily["btc_ret"] = (btc_daily["btc_close"] / btc_daily["btc_open"] - 1) * 100

    # 2. Load NVDA
    import yfinance as yf
    nvda = yf.download("NVDA", start="2026-11-01", end="2026-11-26", progress=False)
    if isinstance(nvda.columns, pd.MultiIndex):
        nvda.columns = nvda.columns.get_level_values(0)
    nvda = nvda.reset_index()
    nvda["Date"] = pd.to_datetime(nvda["Date"])
    nvda["nvda_daily_ret"] = nvda["Close"].pct_change() * 100

    m = btc_daily.merge(nvda[["Date", "Open", "Close", "nvda_daily_ret"]], left_on="date", right_on="Date", how="inner")
    m = m.rename(columns={"Close": "nvda_close"}).dropna(subset=["nvda_daily_ret"])

    rho, p_val = spearmanr(m["btc_ret"], m["nvda_daily_ret"])

    # Figure: 2 Panels
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor="#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # Panel 1: Normalized Price Performance (% from Nov 2)
    ax1 = axes[0]
    ax1_r = ax1.twinx()
    for spine in ax1_r.spines.values():
        spine.set_edgecolor("#333355")

    ax1.plot(m["date"], m["btc_close"], color="#00d4ff", linewidth=2.2, label="Bitcoin (BTC/USD)")
    ax1_r.plot(m["date"], m["nvda_close"], color="#76b900", linewidth=2.2, linestyle="--", label="NVIDIA (NVDA)")

    ax1.set_ylabel("Bitcoin Price (USD)", color="#00d4ff", fontsize=11)
    ax1_r.set_ylabel("NVIDIA Stock Price (USD)", color="#76b900", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#00d4ff")
    ax1_r.tick_params(axis="y", labelcolor="#76b900")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.1f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # Annotate Key Drop Days
    key_dates = [
        ("2026-11-05", "NVDA -9.3%\nBTC -5.5%"),
        ("2026-11-12", "NVDA -4.3%\nBTC -3.5%"),
        ("2026-11-20", "NVDA -14.0% Flash Crash\nBTC -4.9% (Trough $55k next day)")
    ]
    for d_str, label in key_dates:
        d_val = pd.to_datetime(d_str)
        if d_val in m["date"].values:
            px = m.loc[m["date"] == d_val, "btc_close"].iloc[0]
            ax1.annotate(label, xy=(d_val, px), xytext=(d_val, px + 3500),
                         arrowprops=dict(facecolor="yellow", arrowstyle="->", lw=1.2),
                         color="yellow", fontsize=9, ha="center",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="yellow", alpha=0.9))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, facecolor="#1a1a2e", labelcolor="white", loc="lower left")
    ax1.set_title("Cross-Market Transmission: Bitcoin vs NVIDIA Price Trends (Nov 2026)", fontsize=13, fontweight="bold", color="white")

    # Panel 2: Daily Returns Comparison Bar Chart
    ax2 = axes[1]
    w = 0.35
    x = np.arange(len(m))
    ax2.bar(x - w/2, m["btc_ret"], width=w, color="#00d4ff", label="BTC Return (%)", alpha=0.85)
    ax2.bar(x + w/2, m["nvda_daily_ret"], width=w, color="#76b900", label="NVDA Return (%)", alpha=0.85)
    ax2.axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime("%b %d") for d in m["date"]], rotation=30, ha="right", color="white")
    ax2.set_ylabel("Daily Return (%)", color="white", fontsize=11)
    ax2.set_title(f"Daily Return Synchronicity (Spearman Rank Correlation ρ = +{rho:.3f}, p = {p_val:.4f})", fontsize=12, fontweight="bold", color="white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white")

    fig.tight_layout()
    out = FIG_DIR / "nvda_vs_btc_nov2026.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f" Saved comparison chart: {out}")


if __name__ == "__main__":
    plot_nvda_btc()

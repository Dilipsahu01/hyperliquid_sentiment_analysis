#!/usr/bin/env python3
"""
plot_reversal_anatomy.py — Forensic pattern analysis of the market dynamics
just before NVIDIA's massive +31.5% explosion on Nov 23, 2026.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def plot_anatomy():
    # 1. Load Bitcoin 1-sec data for the critical window (Nov 19 - Nov 24)
    df = pd.read_parquet(DATA_DIR / "btc_1sec_nov2026.parquet")
    sub_btc = df[(df["timestamp_utc"] >= "2026-11-19 00:00:00+00:00") & (df["timestamp_utc"] <= "2026-11-24 23:59:59+00:00")].copy()
    # Downsample to 1-minute for plotting
    sub_btc_min = sub_btc.iloc[::60].copy()

    # 2. Daily returns on Nov 20 (Capitulation) vs Nov 23 (The Explosion)
    tickers = ["NVDA", "TSLA", "QCOM", "MSTR", "AMD", "COIN", "GOOGL", "AAPL", "WMT", "TLT", "XOM"]
    raw = yf.download(tickers, start="2026-11-18", end="2026-11-25", progress=False)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    rets = closes.pct_change() * 100

    ret_nov20 = rets.loc["2026-11-20"]
    ret_nov23 = rets.loc["2026-11-23"]

    # 3. Figure Setup
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), facecolor="#0f0f1a", gridspec_kw={"height_ratios": [1.2, 1.0]})
    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # Panel 1: Bitcoin 1-Minute Continuous Timeline Over the Weekend
    ax1 = axes[0]
    ax1.plot(sub_btc_min["timestamp_utc"], sub_btc_min["close"], color="#00d4ff", linewidth=1.8, label="Bitcoin Continuous 1-Min Price")

    # Shade the weekend when stock markets were closed
    sat_start = pd.to_datetime("2026-11-20 21:00:00+00:00")
    mon_open = pd.to_datetime("2026-11-23 14:30:00+00:00")
    ax1.axvspan(sat_start, mon_open, color="#333366", alpha=0.35, label="Weekend: Traditional Equity Markets Closed")

    # Annotate bottom and bounce
    low_time = pd.to_datetime("2026-11-21 06:17:15+00:00")
    ax1.annotate("Saturday 06:17 UTC: BTC Capitulation Low ($55,130)\nBottom reached 56 hours before US stock open!",
                 xy=(low_time, 55130), xytext=(low_time, 59500),
                 arrowprops=dict(facecolor="#d62728", arrowstyle="->", lw=1.5),
                 color="white", fontsize=9.5, fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#d62728", alpha=0.9))

    ax1.annotate("Monday 14:30 UTC: US Cash Open\nBTC already surged +16.5% to $64,225\nNVDA gaps up +31.5%!",
                 xy=(mon_open, 64225), xytext=(mon_open, 66000),
                 arrowprops=dict(facecolor="#2ca02c", arrowstyle="->", lw=1.5),
                 color="white", fontsize=9.5, fontweight="bold", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#2ca02c", alpha=0.9))

    ax1.set_ylabel("Bitcoin Price (USD)", color="white", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax1.set_title("Leading Indicator Signature: Bitcoin Rebounded +16.5% Over the Weekend BEFORE Nvidia Opened on Monday",
                  fontsize=13, fontweight="bold", color="white", pad=10)
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", loc="lower left")

    # Panel 2: Cross-Asset Sector Divergence on Crash Friday vs Rebound Monday
    ax2 = axes[1]
    w = 0.38
    x = np.arange(len(tickers))

    bars1 = ax2.bar(x - w/2, ret_nov20, width=w, color="#d62728", label="Friday Nov 20: Capitulation / Flight to Hedges (%)", alpha=0.85)
    bars2 = ax2.bar(x + w/2, ret_nov23, width=w, color="#2ca02c", label="Monday Nov 23: Rebound / Growth Surge (%)", alpha=0.85)

    ax2.axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers, rotation=0, color="white", fontweight="bold", fontsize=10)
    ax2.set_ylabel("Daily Return (%)", color="white", fontsize=11)
    ax2.set_title("Capital Rotation Anatomy: Defensive Flight to Safety on Friday (WMT, TLT, XOM) Inverted into Monday's Tech Explosion",
                  fontsize=12, fontweight="bold", color="white", pad=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", loc="upper left")

    # Add text labels on the bars for clarity
    for bar in bars1:
        val = bar.get_height()
        if not np.isnan(val) and abs(val) > 2.0:
            offset = -1.8 if val < 0 else 0.5
            ax2.text(bar.get_x() + bar.get_width()/2, val + offset, f"{val:.1f}%", ha="center", va="top" if val < 0 else "bottom", fontsize=7.5, color="#ff9999")

    for bar in bars2:
        val = bar.get_height()
        if not np.isnan(val) and abs(val) > 2.0:
            offset = 0.5 if val > 0 else -1.8
            ax2.text(bar.get_x() + bar.get_width()/2, val + offset, f"{val:.1f}%", ha="center", va="bottom" if val > 0 else "top", fontsize=7.5, color="#99ff99")

    fig.tight_layout()
    out = FIG_DIR / "nvda_reversal_anatomy.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f" Saved Reversal Anatomy Chart: {out}")


if __name__ == "__main__":
    plot_anatomy()

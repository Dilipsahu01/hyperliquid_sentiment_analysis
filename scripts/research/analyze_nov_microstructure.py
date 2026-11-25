#!/usr/bin/env python3
"""
analyze_nov_microstructure.py — Institutional quant analysis of 1-second BTC data
for November 2026 (2,073,600 bars).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def run_quant_analysis():
    print("Loading data/btc_1sec_nov2026.parquet …")
    df = pd.read_parquet(DATA_DIR / "btc_1sec_nov2026.parquet")
    fg = pd.read_csv(DATA_DIR / "fear_greed_index.csv")
    fg["date"] = pd.to_datetime(fg["date"]).dt.date

    # 1. Macro Summary
    start_px = float(df["close"].iloc[0])
    end_px = float(df["close"].iloc[-1])
    peak_idx = df["high"].idxmax()
    peak_px = float(df.loc[peak_idx, "high"])
    peak_time = df.loc[peak_idx, "timestamp_utc"]
    low_idx = df["low"].idxmin()
    low_px = float(df.loc[low_idx, "low"])
    low_time = df.loc[low_idx, "timestamp_utc"]

    total_return = (end_px / start_px - 1) * 100
    max_drawdown = (low_px / peak_px - 1) * 100
    total_btc_vol = float(df["volume"].sum())
    total_usd_vol = float((df["volume"] * df["close"]).sum())
    total_trades = int(df["trades"].sum())

    # 2. Microstructure & Return Statistics
    r = df["log_return"]
    zero_pct = float((r == 0).mean() * 100)
    std_1s = float(r.std())
    skew_val = float(r.skew())
    kurt_val = float(r.kurtosis())

    mean_tps = float(df["trades"].mean())
    p95_tps = float(df["trades"].quantile(0.95))
    p99_tps = float(df["trades"].quantile(0.99))
    max_tps_idx = df["trades"].idxmax()
    max_tps = int(df.loc[max_tps_idx, "trades"])
    max_tps_time = df.loc[max_tps_idx, "timestamp_utc"]

    max_gain_idx = r.idxmax()
    max_gain_pct = float(r.loc[max_gain_idx] * 100)
    max_gain_time = df.loc[max_gain_idx, "timestamp_utc"]
    max_gain_usd = float(df.loc[max_gain_idx, "price_change"])

    max_drop_idx = r.idxmin()
    max_drop_pct = float(r.loc[max_drop_idx] * 100)
    max_drop_time = df.loc[max_drop_idx, "timestamp_utc"]
    max_drop_usd = float(df.loc[max_drop_idx, "price_change"])

    print("\n" + "=" * 65)
    print("  NOVEMBER 2026 — 1-SECOND BTC QUANT AUDIT")
    print("=" * 65)
    print(f"  Starting Price (Nov 1 00:00 UTC):  ${start_px:,.2f}")
    print(f"  Ending Price (Nov 24 23:59 UTC):    ${end_px:,.2f}")
    print(f"  Net Month Return:                   {total_return:+.2f}%")
    print(f"  All-Time November Peak:             ${peak_px:,.2f}  ({peak_time})")
    print(f"  November Low (Trough):              ${low_px:,.2f}  ({low_time})")
    print(f"  Max Drawdown (Peak to Trough):      {max_drawdown:.2f}%")
    print(f"  Total Traded Volume:                {total_btc_vol:,.1f} BTC (~${total_usd_vol/1e9:,.2f} Billion)")
    print(f"  Total Executed Trades:              {total_trades:,}")

    print("\n" + "─" * 65)
    print("  MICROSTRUCTURE METRICS (1-SECOND GRANULARITY)")
    print("─" * 65)
    print(f"  Zero-Return Seconds:                {zero_pct:.2f}% (price unchanged)")
    print(f"  1-sec Volatility (Std Dev):         {std_1s:.6f} ({std_1s * np.sqrt(86400*365)*100:.1f}% annualized)")
    print(f"  Skewness:                           {skew_val:+.4f}")
    print(f"  Excess Kurtosis (Fat-tail index):   {kurt_val:,.1f}  (Normal Gaussian = 0.0)")
    print(f"  Average Trade Intensity:            {mean_tps:.1f} trades/sec")
    print(f"  99th-Percentile Trade Burst:        {p99_tps:.0f} trades/sec")
    print(f"  Absolute Peak Trade Burst:          {max_tps:,} trades in a single second! ({max_tps_time})")
    print(f"  Largest 1-Second Flash Pump:        +{max_gain_pct:.2f}% (${max_gain_usd:+.2f}) at {max_gain_time}")
    print(f"  Largest 1-Second Flash Dump:        {max_drop_pct:.2f}% (${max_drop_usd:+.2f}) at {max_drop_time}")

    # 3. Daily Rollup Table
    df["date"] = df["timestamp_utc"].dt.date
    daily = df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume_btc=("volume", "sum"),
        trades=("trades", "sum"),
        daily_return=("close", lambda c: (c.iloc[-1] / c.iloc[0] - 1) * 100),
        realized_vol_ann=("log_return", lambda ret: np.sqrt(np.sum(ret**2)) * np.sqrt(365 * 86400) * 100)
    ).reset_index()

    daily = daily.merge(fg[["date", "value", "classification"]], on="date", how="left")

    print("\n" + "─" * 65)
    print("  DAILY BREAKDOWN & SENTIMENT CORRELATION")
    print("─" * 65)
    print(f"  {'Date':<10} {'Close ($)':>10} {'Chg(%)':>8} {'RealizedVol':>12} {'Trades':>10} {'F&G':>5} {'Regime':<13}")
    print("  " + "─" * 63)
    for _, r_d in daily.iterrows():
        print(f"  {str(r_d['date']):<10} {r_d['close']:>10,.1f} {r_d['daily_return']:>+7.2f}% {r_d['realized_vol_ann']:>11.1f}% {int(r_d['trades']):>10,} {int(r_d['value']) if pd.notna(r_d['value']) else 0:>5} {str(r_d['classification']):<13}")

    # 4. Hourly Seasonality (Intraday U-curve)
    df["hour_utc"] = df["timestamp_utc"].dt.hour
    hourly = df.groupby("hour_utc").agg(
        volume_btc=("volume", "mean"),
        trades=("trades", "mean"),
        std_ret=("log_return", "std")
    ).reset_index()

    # ── GENERATE CHARTS ──────────────────────────────────────────────────────────
    print("\nGenerating charts …")

    # Chart 1: 1-Second Price, Drawdown, and Realized Volatility
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), facecolor="#0f0f1a", sharex=True)
    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # Subsample 1 in every 30 seconds for plotting performance
    sub = df.iloc[::30].copy()

    # Panel 1: Price
    axes[0].plot(sub["timestamp_utc"], sub["close"], color="#00d4ff", linewidth=1.2, label="BTC/USDT 1-Sec Price")
    axes[0].set_ylabel("Price (USD)", color="white", fontsize=11)
    axes[0].set_title("Bitcoin (BTC/USDT) — November 2026 High-Frequency Price Action", fontsize=14, fontweight="bold", color="white")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white")

    # Panel 2: Trade Intensity (Burstiness)
    axes[1].plot(sub["timestamp_utc"], sub["trades"], color="#ff7f0e", linewidth=0.8, alpha=0.85, label="Trades / Second")
    axes[1].set_ylabel("Trades / Sec", color="white", fontsize=11)
    axes[1].set_title("Order Flow Arrival Intensity (Trades per Second)", fontsize=13, fontweight="bold", color="white")
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white")

    # Panel 3: Daily Realized Volatility vs Fear & Greed
    daily_dt = pd.to_datetime(daily["date"])
    axes[2].bar(daily_dt, daily["realized_vol_ann"], width=0.8, color="#d62728", alpha=0.7, label="Annualized Realized Volatility (%)")
    axes[2].set_ylabel("Realized Vol (%)", color="#d62728", fontsize=11)
    axes[2].tick_params(axis="y", labelcolor="#d62728")

    ax2_fg = axes[2].twinx()
    ax2_fg.plot(daily_dt, daily["value"], color="#ffd700", linewidth=2.0, marker="o", label="Fear & Greed Index")
    ax2_fg.set_ylabel("Fear & Greed (0-100)", color="#ffd700", fontsize=11)
    ax2_fg.tick_params(axis="y", labelcolor="#ffd700")
    ax2_fg.set_ylim(0, 100)
    for spine in ax2_fg.spines.values():
        spine.set_edgecolor("#333355")

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[2].set_title("Daily Realized Volatility (from 1s bars) vs Fear & Greed Index", fontsize=13, fontweight="bold", color="white")

    fig.tight_layout()
    chart1_path = FIG_DIR / "nov2026_btc_1sec_macro.png"
    fig.savefig(chart1_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {chart1_path}")

    # Chart 2: Intraday Hour-of-Day Seasonality (UTC & IST)
    fig, ax1 = plt.subplots(figsize=(12, 6), facecolor="#0f0f1a")
    ax1.set_facecolor("#0f0f1a")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333355")

    ax2 = ax1.twinx()
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333355")

    bars = ax1.bar(hourly["hour_utc"] - 0.2, hourly["trades"], width=0.4, color="#00d4ff", alpha=0.8, label="Mean Trades / Sec")
    line = ax2.plot(hourly["hour_utc"] + 0.2, hourly["std_ret"] * 10000, color="#ff7f0e", marker="o", linewidth=2, label="1-sec Volatility (bps)")

    ax1.set_xlabel("Hour of Day (UTC)", color="white", fontsize=11)
    ax1.set_ylabel("Mean Trades Per Second", color="#00d4ff", fontsize=11)
    ax2.set_ylabel("Volatility per Sec (Basis Points × 10⁻⁴)", color="#ff7f0e", fontsize=11)
    ax1.set_xticks(range(24))
    ax1.set_title("Intraday Microstructure U-Curve: Activity & Volatility by Hour (UTC)", fontsize=13, fontweight="bold", color="white")

    # Annotate US market open
    ax1.axvline(13.5, color="yellow", linestyle="--", alpha=0.5)
    ax1.text(13.6, ax1.get_ylim()[1] * 0.88, "US Cash Open (14:30 UTC)", color="yellow", fontsize=9)

    fig.tight_layout()
    chart2_path = FIG_DIR / "nov2026_btc_intraday_seasonality.png"
    fig.savefig(chart2_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {chart2_path}")


if __name__ == "__main__":
    run_quant_analysis()

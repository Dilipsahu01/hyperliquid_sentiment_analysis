#!/usr/bin/env python3
"""
generate_forecast_and_spike_plots.py
Generates high-resolution production visualizations:
1. outputs/figures/final_forecast_nov26_30.png: Multi-horizon daily volatility fan chart, confidence ribbons, and expected price ranges.
2. outputs/figures/next_hours_trading_spike_forecast.png: Intraday next-hours forward trading spike index, dynamic volatility price tunnel, and empirical U-curve.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches

ROOT = Path(".")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300
})


def plot_final_daily_forecast():
    """Generate high-resolution fan chart for Nov 26-30 daily volatility forecast."""
    df_fc = pd.read_csv(RESULTS_DIR / "forecast_nov26_30.csv")
    df_master = pd.read_parquet(DATA_DIR / "engineered_features_master.parquet")
    df_master["cutoff_date"] = pd.to_datetime(df_master["cutoff_date"])

    # Historical actual Nov 1 to Nov 24
    hist_nov = df_master[(df_master["cutoff_date"] >= "2026-11-01") & (df_master["cutoff_date"] <= "2026-11-24")].copy()
    hist_nov = hist_nov.sort_values("cutoff_date").reset_index(drop=True)

    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2.2, 1.2]}, sharex=False)
    fig.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")

    # --- TOP PANEL: Realized Volatility Fan Chart ---
    dates_hist = hist_nov["cutoff_date"]
    vol_hist = hist_nov["rv_ann"] * 100.0

    ax1.plot(dates_hist, vol_hist, color="#2563eb", linewidth=2.2, label="Actual Daily Realized Vol (1h RV)", zorder=4)
    ax1.scatter(dates_hist, vol_hist, color="#1e3a8a", s=30, zorder=5)

    # Transition line from last historical point (Nov 24) to Nov 26
    fc_dates = pd.to_datetime(df_fc["forecast_date"])
    point_blend = df_fc["point_vol_ann"] * 100.0
    point_base = df_fc["baseline_point_vol_pct"].str.rstrip("%").astype(float)
    point_ml = df_fc["ml_point_vol_pct"].str.rstrip("%").astype(float)

    low_80 = df_fc["interval_80_lower"] * 100.0
    high_80 = df_fc["interval_80_upper"] * 100.0
    low_95 = df_fc["interval_95_lower"] * 100.0
    high_95 = df_fc["interval_95_upper"] * 100.0

    # Shaded Prediction Intervals
    ax1.fill_between(fc_dates, low_95, high_95, color="#d97706", alpha=0.15, label="95% Empirical Prediction Interval", zorder=1)
    ax1.fill_between(fc_dates, low_80, high_80, color="#d97706", alpha=0.30, label="80% Empirical Prediction Interval", zorder=2)

    # Forecast Curves
    ax1.plot(fc_dates, point_base, color="#64748b", linestyle="--", linewidth=1.5, label="Baseline (HAR_Crypto_Log_DOW)", zorder=3)
    ax1.plot(fc_dates, point_ml, color="#7c3aed", linestyle=":", linewidth=1.8, label="ML (Lasso_Direct_ML)", zorder=3)
    ax1.plot(fc_dates, point_blend, color="#d97706", linewidth=2.8, marker="o", markersize=7, label="Selected Forecast (Equal-Weight Blend)", zorder=6)

    # Draw vertical separator
    ax1.axvline(pd.Timestamp("2026-11-25"), color="#94a3b8", linestyle="-.", linewidth=1.2, alpha=0.8)
    ax1.text(pd.Timestamp("2026-11-24 12:00"), 125, "Forecast Origin: Nov 25 11:00 UTC", color="#475569", rotation=90, verticalalignment="top", fontsize=9, fontweight="bold")

    # Annotate Key Events
    ax1.annotate("Thanksgiving\n(Thin Liquidity)\n68.4%", xy=(pd.Timestamp("2026-11-26"), 68.42), xytext=(pd.Timestamp("2026-11-26"), 105),
                 arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.2), color="#b45309", fontsize=8.5, ha="center", fontweight="bold")
    ax1.annotate("Black Friday\n68.4%", xy=(pd.Timestamp("2026-11-27"), 68.44), xytext=(pd.Timestamp("2026-11-27"), 118),
                 arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.2), color="#b45309", fontsize=8.5, ha="center")
    ax1.annotate("Weekend Vol Compression\n33.3% -> 36.6%", xy=(pd.Timestamp("2026-11-28 12:00"), 34.9), xytext=(pd.Timestamp("2026-11-28 12:00"), 18),
                 arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.2), color="#1e40af", fontsize=8.5, ha="center", fontweight="bold")
    ax1.annotate("Monday Equities Reopen\nRebound to 66.6%", xy=(pd.Timestamp("2026-11-30"), 66.63), xytext=(pd.Timestamp("2026-11-30"), 110),
                 arrowprops=dict(arrowstyle="->", color="#0d9488", lw=1.2), color="#0f766e", fontsize=8.5, ha="center", fontweight="bold")

    ax1.set_title("Bitcoin (BTC/USDT) Daily Realized Volatility: Historical Actual vs Nov 26-30 Multi-Horizon Forecast", color="#1f2937", pad=12, fontweight="bold")
    ax1.set_ylabel("Annualized Volatility (%)", color="#1f2937", fontweight="bold")
    ax1.set_ylim(10, 155)
    ax1.grid(color="#e5e7eb", linestyle="--", linewidth=0.6, alpha=0.8)
    ax1.tick_params(colors="#1f2937")
    ax1.legend(loc="upper left", facecolor="#f8fafc", edgecolor="#cbd5e1", labelcolor="#1f2937", framealpha=0.9)

    # --- BOTTOM PANEL: Expected Parkinson Daily Range % ---
    ranges = df_fc["expected_range_pct"].astype(str).str.rstrip("%").astype(float).values
    day_labels = ["Thu 11/26\nThanksgiving", "Fri 11/27\nBlack Friday", "Sat 11/28\nWeekend", "Sun 11/29\nWeekend", "Mon 11/30\nReopen"]
    bar_colors = ["#d97706", "#d97706", "#2563eb", "#2563eb", "#0d9488"]

    bars = ax2.bar(day_labels, ranges, color=bar_colors, width=0.45, edgecolor="#cbd5e1", linewidth=0.8, alpha=0.85, zorder=3)
    ax2.axhline(3.5, color="#2563eb", linestyle="--", linewidth=1.0, alpha=0.6, label="Calm Range Baseline (3.5%)")
    ax2.axhline(5.5, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.6, label="Elevated Volatility Trigger (5.5%)")

    for bar, val in zip(bars, ranges):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.18, f"{val:.2f}%", ha="center", va="bottom", color="#1f2937", fontweight="bold", fontsize=9.5)

    ax2.set_title("Expected Daily Intraday High-Low Price Range (%): E[Range%] = 8.35 x sigma_ann", color="#1f2937", pad=10, fontweight="bold")
    ax2.set_ylabel("Expected Range (%)", color="#1f2937")
    ax2.set_ylim(0, 7.5)
    ax2.grid(color="#e5e7eb", linestyle="--", linewidth=0.6, alpha=0.8)
    ax2.tick_params(colors="#1f2937")
    ax2.legend(loc="upper left", facecolor="#f8fafc", edgecolor="#cbd5e1", labelcolor="#1f2937", framealpha=0.9)

    plt.tight_layout()
    out_p = FIGURES_DIR / "final_forecast_nov26_30.png"
    plt.savefig(out_p, dpi=300, facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {out_p}")


def plot_next_hours_trading_spikes():
    """
    Generate intraday next-hours forward trading spike forecast:
    From Nov 25 11:45 UTC through Nov 26 23:00 UTC (Next 36 Hours).
    Includes:
    1. Hourly Volatility & Trading Spike Risk Index (0-100%).
    2. Dynamic Price Volatility Tunnel (1-sigma & 2-sigma cones from current price $58,643.55).
    3. Empirical 4-Year U-Curve & Trade Congestion Distribution.
    """
    df_5m = pd.read_parquet(DATA_DIR / "btc_5m_2023_2026.parquet")
    df_5m["open_time"] = pd.to_datetime(df_5m["open_time"])
    df_5m["hour"] = df_5m["open_time"].dt.hour
    df_5m["log_ret"] = np.log(df_5m["close"] / df_5m["close"].shift(1))

    # Empirical hourly profile across 410,238 bars
    hourly_stats = df_5m.groupby("hour").agg(
        vol_ann=("log_ret", lambda x: np.std(x.dropna()) * np.sqrt(12 * 24 * 365) * 100.0),
        trades_mean=("trades", "mean"),
        vol_mean=("volume", "mean")
    ).reset_index()

    current_price = 58643.55
    start_time = pd.Timestamp("2026-11-25 12:00:00+00:00")
    hours_ahead = 36
    forecast_times = [start_time + pd.Timedelta(hours=i) for i in range(hours_ahead)]

    # Build forward profile based on empirical U-curve modulated by Thanksgiving calendar and current regime
    # Current regime annualized vol is ~68.4%
    base_target_vol = 68.4
    mean_emp_vol = hourly_stats["vol_ann"].mean()
    vol_scale = base_target_vol / mean_emp_vol

    forward_vols = []
    spike_index = []
    expected_low_1s = []
    expected_high_1s = []
    expected_low_2s = []
    expected_high_2s = []

    cum_var = 0.0

    for i, t in enumerate(forecast_times):
        hr = t.hour
        emp_hr = hourly_stats.loc[hourly_stats["hour"] == hr, "vol_ann"].values[0]
        emp_trades = hourly_stats.loc[hourly_stats["hour"] == hr, "trades_mean"].values[0]

        # Thanksgiving dampening on Nov 26 (cash markets closed, but order book thins)
        is_thanksgiving = (t.date() == pd.Timestamp("2026-11-26").date())
        holiday_factor = 0.92 if is_thanksgiving else 1.0

        # Hourly annualized vol
        hr_vol = emp_hr * vol_scale * holiday_factor
        forward_vols.append(hr_vol)

        # Spike Risk Index (0 - 100%): combines trade intensity and volatility peak
        # Hours 13-16 UTC (US Cash Open) have highest spike probability
        normalized_trades = emp_trades / hourly_stats["trades_mean"].max()
        normalized_vol = emp_hr / hourly_stats["vol_ann"].max()
        risk = (0.55 * normalized_vol + 0.45 * normalized_trades) * 100.0
        if is_thanksgiving and 13 <= hr <= 17:
            # Thin book penalty creates higher tail spike risk
            risk = min(100.0, risk * 1.15)
        spike_index.append(risk)

        # 1-hour variance component
        dt_years = (i + 1) / (365.0 * 24.0)
        sigma_cum = (hr_vol / 100.0) * np.sqrt(dt_years)

        # Price corridors
        expected_high_1s.append(current_price * np.exp(1.0 * sigma_cum))
        expected_low_1s.append(current_price * np.exp(-1.0 * sigma_cum))
        expected_high_2s.append(current_price * np.exp(2.0 * sigma_cum))
        expected_low_2s.append(current_price * np.exp(-2.0 * sigma_cum))

    fig = plt.figure(figsize=(15, 12), facecolor="#ffffff")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 2.0, 1.4], hspace=0.32)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#ffffff")
        ax.grid(color="#e5e7eb", linestyle="--", linewidth=0.6, alpha=0.8)
        ax.tick_params(colors="#1f2937")
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")

    # --- PANEL 1: Hourly Realized Volatility & Spike Probability Index ---
    color_vol = "#2563eb"
    color_spike = "#dc2626"

    ax1.plot(forecast_times, forward_vols, color=color_vol, linewidth=2.4, label="Predicted Hourly Volatility (Ann %)")
    ax1.set_ylabel("Predicted Hourly Vol (%)", color=color_vol, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_vol)
    ax1.set_ylim(30, 115)

    ax1_twin = ax1.twinx()
    ax1_twin.bar(forecast_times, spike_index, width=0.03, color=color_spike, alpha=0.35, label="Trading Spike Probability Index (0-100%)")
    ax1_twin.set_ylabel("Spike Probability Index (%)", color=color_spike, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color_spike)
    ax1_twin.set_ylim(0, 110)

    # Shaded Spike Zones
    # Today's US open: 13:00 to 16:30 UTC
    ax1.axvspan(pd.Timestamp("2026-11-25 13:00:00+00:00"), pd.Timestamp("2026-11-25 16:30:00+00:00"), color="#dc2626", alpha=0.12, label="Primary Spike Zone (US Cash Open)")
    ax1.axvspan(pd.Timestamp("2026-11-25 20:00:00+00:00"), pd.Timestamp("2026-11-25 21:30:00+00:00"), color="#d97706", alpha=0.12, label="Secondary Spike Zone (21:00 UTC Close)")
    ax1.axvspan(pd.Timestamp("2026-11-26 13:00:00+00:00"), pd.Timestamp("2026-11-26 17:00:00+00:00"), color="#7c3aed", alpha=0.12, label="Thanksgiving Thin-Book Flash Zone")

    ax1.set_title("FORWARD OUTLOOK: Next 36 Hours Hourly Volatility & Trading Spike Risk (Nov 25 12:00 -> Nov 26 23:00 UTC)", color="#1f2937", pad=10, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))

    # Annotate impending US Open spike
    ax1.annotate("IMPENDING SPIKE WINDOW\nUS Cash Market Open (14:30 UTC)\nPeak Volatility ~95.8% Ann",
                 xy=(pd.Timestamp("2026-11-25 14:30:00+00:00"), 96.0),
                 xytext=(pd.Timestamp("2026-11-25 17:30:00+00:00"), 105.0),
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
                 color="#991b1b", fontweight="bold", fontsize=9)

    # --- PANEL 2: Dynamic Price Volatility Tunnel (Anchored at $58,643.55) ---
    ax2.plot([forecast_times[0]], [current_price], marker="o", markersize=8, color="#2563eb", label=f"Latest Price ($58,643.55 at 11:45 UTC)")
    ax2.plot(forecast_times, [current_price] * len(forecast_times), linestyle="--", color="#64748b", linewidth=1.2, label="Anchor Price Path")

    ax2.fill_between(forecast_times, expected_low_2s, expected_high_2s, color="#d97706", alpha=0.15, label="±2-Sigma Tunnel (95.4% Expected Range)")
    ax2.fill_between(forecast_times, expected_low_1s, expected_high_1s, color="#d97706", alpha=0.30, label="±1-Sigma Tunnel (68.3% Expected Range)")

    ax2.plot(forecast_times, expected_high_1s, color="#b45309", linestyle="-.", linewidth=1.2)
    ax2.plot(forecast_times, expected_low_1s, color="#b45309", linestyle="-.", linewidth=1.2)
    ax2.plot(forecast_times, expected_high_2s, color="#d97706", linestyle=":", linewidth=1.4)
    ax2.plot(forecast_times, expected_low_2s, color="#d97706", linestyle=":", linewidth=1.4)

    # Annotations on price bounds at key hours
    t_24h = pd.Timestamp("2026-11-26 12:00:00+00:00")
    idx_24h = forecast_times.index(t_24h)
    ax2.annotate(f"24h ±1σ: ${expected_low_1s[idx_24h]:,.0f} - ${expected_high_1s[idx_24h]:,.0f}\n24h ±2σ: ${expected_low_2s[idx_24h]:,.0f} - ${expected_high_2s[idx_24h]:,.0f}",
                 xy=(t_24h, current_price), xytext=(pd.Timestamp("2026-11-26 05:00:00+00:00"), 64500),
                 arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.2), color="#1f2937", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95))

    ax2.set_title("Forward Price Volatility Tunnel (Anchored at Nov 25 11:45 UTC Close: $58,643.55)", color="#1f2937", pad=10, fontweight="bold")
    ax2.set_ylabel("Bitcoin Price (USD)", color="#1f2937")
    ax2.yaxis.set_major_formatter("${x:,.0f}")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    ax2.legend(loc="upper left", facecolor="#f8fafc", edgecolor="#cbd5e1", labelcolor="#1f2937", framealpha=0.9)

    # --- PANEL 3: Historical Empirical Intraday Seasonality U-Curve ---
    hours = hourly_stats["hour"]
    vols = hourly_stats["vol_ann"]
    trades = hourly_stats["trades_mean"]

    ax3.plot(hours, vols, color="#2563eb", marker="s", markersize=5, linewidth=2.0, label="Empirical Volatility (% Ann)")
    ax3.set_xlabel("Hour of Day (UTC)", color="#1f2937", fontweight="bold")
    ax3.set_ylabel("Empirical Vol (%)", color="#2563eb", fontweight="bold")
    ax3.tick_params(axis="y", labelcolor="#2563eb")
    ax3.set_xlim(-0.5, 23.5)
    ax3.set_xticks(range(0, 24, 2))

    ax3_twin = ax3.twinx()
    ax3_twin.bar(hours, trades, width=0.45, color="#0d9488", alpha=0.35, label="Mean Trades / 5m Bar")
    ax3_twin.set_ylabel("Trade Intensity (Trades/5m)", color="#0d9488", fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor="#0d9488")

    # Mark current hour (Hour 11 UTC)
    ax3.axvline(11, color="#d97706", linestyle="-", linewidth=2.0)
    ax3.text(11.2, 38, "CURRENT TIME\n(11:45 UTC)", color="#b45309", fontsize=8.5, fontweight="bold")

    # Mark impending peak (Hour 14 UTC)
    ax3.axvline(14, color="#dc2626", linestyle="--", linewidth=1.5)
    ax3.text(14.2, 60, "PEAK LIQUIDITY\n(14:00 UTC)", color="#991b1b", fontsize=8.5, fontweight="bold")

    ax3.set_title("Empirical Intraday Seasonality U-Curve (410,238 5-Minute Bars, 2023-2026)", color="#1f2937", pad=10, fontweight="bold")

    plt.tight_layout()
    out_p = FIGURES_DIR / "next_hours_trading_spike_forecast.png"
    plt.savefig(out_p, dpi=300, facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {out_p}")


if __name__ == "__main__":
    plot_final_daily_forecast()
    plot_next_hours_trading_spikes()

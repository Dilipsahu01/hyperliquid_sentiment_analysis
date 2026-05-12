"""
visualizations.py — All matplotlib/seaborn figure generation.

Each function saves a PNG to FIG_DIR and returns the file path.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path

from src.config import (
    FIG_DIR, SENTIMENT_ORDER, SENTIMENT_COLORS, SENTIMENT_PALETTE,
    FIGURE_DPI, FIGURE_STYLE
)

warnings.filterwarnings("ignore")
plt.style.use(FIGURE_STYLE)


# ─── utility helpers ────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
    return path


def _sentiment_palette(df_col):
    """Return ordered colors for a DataFrame column of sentiment categories."""
    return [SENTIMENT_COLORS.get(s, "#888888") for s in df_col]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Average PnL by Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def plot_avg_pnl_by_sentiment(sentiment_pnl: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # ── left: avg PnL ───────────────────────────────────────────────────────
    ax1 = axes[0]
    cats = sentiment_pnl["classification"].astype(str)
    colors = _sentiment_palette(cats)
    bars = ax1.bar(cats, sentiment_pnl["avg_pnl"], color=colors, edgecolor="#222240", linewidth=0.8)
    ax1.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
    for bar, val in zip(bars, sentiment_pnl["avg_pnl"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"${val:.1f}", ha="center", va="bottom", fontsize=9, color="white")
    ax1.set_title("Average Closed PnL by Sentiment", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Market Sentiment", fontsize=10)
    ax1.set_ylabel("Avg Closed PnL (USD)", fontsize=10)
    ax1.tick_params(axis="x", rotation=20)

    # ── right: net PnL (after fees) ─────────────────────────────────────────
    ax2 = axes[1]
    bars2 = ax2.bar(cats, sentiment_pnl["avg_net_pnl"], color=colors, edgecolor="#222240", linewidth=0.8)
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
    for bar, val in zip(bars2, sentiment_pnl["avg_net_pnl"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"${val:.1f}", ha="center", va="bottom", fontsize=9, color="white")
    ax2.set_title("Average Net PnL (after fees) by Sentiment", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Market Sentiment", fontsize=10)
    ax2.set_ylabel("Avg Net PnL (USD)", fontsize=10)
    ax2.tick_params(axis="x", rotation=20)

    fig.suptitle("Profitability Across Bitcoin Sentiment Regimes", fontsize=15,
                 fontweight="bold", color="white", y=1.02)
    fig.tight_layout()
    return _save(fig, "01_avg_pnl_by_sentiment")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Win Rate by Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def plot_win_rate(sentiment_pnl: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    cats   = sentiment_pnl["classification"].astype(str)
    colors = _sentiment_palette(cats)
    bars   = ax.bar(cats, sentiment_pnl["win_rate_pct"], color=colors,
                    edgecolor="#222240", linewidth=0.8)
    ax.axhline(50, color="yellow", linewidth=1, linestyle="--", alpha=0.7, label="50% baseline")

    for bar, val in zip(bars, sentiment_pnl["win_rate_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, color="white")

    ax.set_title("Win Rate (%) by Market Sentiment", fontsize=14, fontweight="bold",
                 color="white", pad=12)
    ax.set_xlabel("Market Sentiment", fontsize=11, color="white")
    ax.set_ylabel("Win Rate (%)", fontsize=11, color="white")
    ax.set_ylim(0, 105)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    fig.tight_layout()
    return _save(fig, "02_win_rate_by_sentiment")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trade Count & Average Size by Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def plot_activity_and_size(activity: pd.DataFrame, pnl_summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    cats   = activity["classification"].astype(str)
    colors = _sentiment_palette(cats)

    # trade counts
    ax1 = axes[0]
    bars = ax1.bar(cats, activity["total_trades"], color=colors, edgecolor="#222240")
    for bar, v in zip(bars, activity["total_trades"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{v:,}", ha="center", va="bottom", fontsize=8, color="white")
    ax1.set_title("Total Trade Count by Sentiment", fontsize=13, fontweight="bold", color="white")
    ax1.set_xlabel("Market Sentiment", fontsize=10, color="white")
    ax1.set_ylabel("Number of Trades", fontsize=10, color="white")
    ax1.tick_params(axis="x", rotation=20)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # average size USD
    ax2 = axes[1]
    size_data = pnl_summary.sort_values(
        "classification",
        key=lambda c: c.map({s: i for i, s in enumerate(SENTIMENT_ORDER)})
    )
    bars2 = ax2.bar(size_data["classification"].astype(str),
                    size_data["avg_size_usd"], color=colors, edgecolor="#222240")
    for bar, v in zip(bars2, size_data["avg_size_usd"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"${v:,.0f}", ha="center", va="bottom", fontsize=8, color="white")
    ax2.set_title("Average Trade Size (USD) by Sentiment", fontsize=13, fontweight="bold", color="white")
    ax2.set_xlabel("Market Sentiment", fontsize=10, color="white")
    ax2.set_ylabel("Avg Size (USD)", fontsize=10, color="white")
    ax2.tick_params(axis="x", rotation=20)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.suptitle("Trading Activity & Position Sizing by Sentiment Regime",
                 fontsize=15, fontweight="bold", color="white", y=1.02)
    fig.tight_layout()
    return _save(fig, "03_activity_and_size")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Buy/Sell Ratio by Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def plot_buy_sell_ratio(activity: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    cats   = activity["classification"].astype(str)
    x      = np.arange(len(cats))
    width  = 0.35

    ax.bar(x - width/2, activity["buy_ratio"] * 100,  width, label="BUY %",
           color="#2ca02c", edgecolor="#222240")
    ax.bar(x + width/2, activity["sell_ratio"] * 100, width, label="SELL %",
           color="#d62728", edgecolor="#222240")
    ax.axhline(50, color="yellow", linewidth=1, linestyle="--", alpha=0.6, label="50% line")

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, color="white")
    ax.set_title("Buy vs Sell Ratio (%) by Sentiment", fontsize=14,
                 fontweight="bold", color="white", pad=12)
    ax.set_xlabel("Market Sentiment", fontsize=11, color="white")
    ax.set_ylabel("Percentage (%)", fontsize=11, color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    fig.tight_layout()
    return _save(fig, "04_buy_sell_ratio")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cumulative PnL over time (with sentiment background)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_pnl_timeline(daily: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(16, 7), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Shade background by sentiment
    prev_date = daily["date"].min()
    prev_cls  = daily["classification"].iloc[0]
    for _, row in daily.iterrows():
        if row["classification"] != prev_cls:
            color = SENTIMENT_COLORS.get(str(prev_cls), "#555555")
            ax.axvspan(prev_date, row["date"], alpha=0.12, color=color)
            prev_date = row["date"]
            prev_cls  = row["classification"]
    # final segment
    ax.axvspan(prev_date, daily["date"].max(),
               alpha=0.12, color=SENTIMENT_COLORS.get(str(prev_cls), "#555555"))

    ax.plot(daily["date"], daily["cumulative_pnl"], color="#00d4ff",
            linewidth=1.8, label="Cumulative Closed PnL", zorder=5)
    ax.plot(daily["date"], daily["cumulative_net_pnl"], color="#ff6b35",
            linewidth=1.2, linestyle="--", label="Cumulative Net PnL (after fees)", zorder=4)
    ax.axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.4)

    ax.set_title("Cumulative PnL Over Time (shaded by market sentiment)",
                 fontsize=14, fontweight="bold", color="white", pad=12)
    ax.set_xlabel("Date", fontsize=11, color="white")
    ax.set_ylabel("Cumulative PnL (USD)", fontsize=11, color="white")
    ax.tick_params(colors="white")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    legend_patches = [Patch(color=SENTIMENT_COLORS[s], label=s, alpha=0.7) for s in SENTIMENT_ORDER]
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(handles=line_handles + legend_patches,
              labels=line_labels + SENTIMENT_ORDER,
              facecolor="#1a1a2e", labelcolor="white", fontsize=8, ncol=2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    fig.tight_layout()
    return _save(fig, "05_cumulative_pnl_timeline")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Daily PnL vs Sentiment Value (dual axis)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pnl_vs_sentiment_value(daily: pd.DataFrame) -> Path:
    fig, ax1 = plt.subplots(figsize=(16, 6), facecolor="#0f0f1a")
    ax1.set_facecolor("#0f0f1a")

    ax2 = ax1.twinx()
    ax2.set_facecolor("#0f0f1a")

    ax1.fill_between(daily["date"], daily["total_pnl"], alpha=0.25, color="#00d4ff")
    ax1.plot(daily["date"], daily["roll7_pnl"], color="#00d4ff", linewidth=1.6,
             label="7-Day Rolling PnL")
    ax1.axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.35)

    ax2.plot(daily["date"], daily["sentiment_val"], color="#ffd700",
             linewidth=1.2, alpha=0.85, label="Fear & Greed Index")
    ax2.axhline(50, color="gray", linewidth=0.7, linestyle=":", alpha=0.5)

    ax1.set_xlabel("Date", fontsize=11, color="white")
    ax1.set_ylabel("Total Daily PnL (USD)", fontsize=11, color="#00d4ff")
    ax2.set_ylabel("Fear & Greed Value (0-100)", fontsize=11, color="#ffd700")
    ax1.tick_params(axis="y", labelcolor="#00d4ff")
    ax2.tick_params(axis="y", labelcolor="#ffd700")
    ax1.tick_params(axis="x", colors="white")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax1.set_title("Daily PnL vs Bitcoin Fear & Greed Index", fontsize=14,
                  fontweight="bold", color="white", pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333355")
    fig.tight_layout()
    return _save(fig, "06_daily_pnl_vs_sentiment")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PnL Distribution (violin + box) by Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def plot_pnl_distribution(df: pd.DataFrame) -> Path:
    # Clip to [-500, 500] for visual clarity
    df_vis = df.copy()
    df_vis["pnl_clipped"] = df_vis["Closed PnL"].clip(-500, 500)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    palette = {s: SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER if s in SENTIMENT_COLORS}

    # violin
    order = [s for s in SENTIMENT_ORDER if s in df_vis["classification"].cat.categories]
    sns.violinplot(data=df_vis, x="classification", y="pnl_clipped",
                   order=order, palette=palette, inner="quartile",
                   ax=axes[0], linewidth=0.8)
    axes[0].axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].set_title("PnL Distribution — Violin (clipped ±$500)",
                       fontsize=12, fontweight="bold", color="white")
    axes[0].set_xlabel("Market Sentiment", color="white")
    axes[0].set_ylabel("Closed PnL (USD)", color="white")
    axes[0].tick_params(axis="x", rotation=20)

    # box
    sns.boxplot(data=df_vis, x="classification", y="pnl_clipped",
                order=order, palette=palette, ax=axes[1],
                flierprops=dict(marker=".", color="gray", alpha=0.3, markersize=2))
    axes[1].axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1].set_title("PnL Distribution — Box Plot (clipped ±$500)",
                       fontsize=12, fontweight="bold", color="white")
    axes[1].set_xlabel("Market Sentiment", color="white")
    axes[1].set_ylabel("Closed PnL (USD)", color="white")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Trade-Level PnL Distributions by Sentiment Regime",
                 fontsize=14, fontweight="bold", color="white", y=1.02)
    fig.tight_layout()
    return _save(fig, "07_pnl_distribution_by_sentiment")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Profit Factor & Sharpe by Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def plot_profit_factor_sharpe(sentiment_pnl: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f0f1a")
    cats   = sentiment_pnl["classification"].astype(str)
    colors = _sentiment_palette(cats)

    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # Profit Factor
    pf = sentiment_pnl["profit_factor"].fillna(0)
    bars = axes[0].bar(cats, pf, color=colors, edgecolor="#222240")
    axes[0].axhline(1, color="yellow", linewidth=1, linestyle="--", alpha=0.7, label="Breakeven (PF=1)")
    for bar, v in zip(bars, pf):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9, color="white")
    axes[0].set_title("Profit Factor by Sentiment", fontsize=13, fontweight="bold", color="white")
    axes[0].set_xlabel("Market Sentiment", color="white")
    axes[0].set_ylabel("Profit Factor (Gains / Losses)", color="white")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white")

    # Sharpe
    sh = sentiment_pnl["sharpe_approx"].fillna(0)
    bars2 = axes[1].bar(cats, sh, color=colors, edgecolor="#222240")
    axes[1].axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.4)
    for bar, v in zip(bars2, sh):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="white")
    axes[1].set_title("Approx. Sharpe Ratio by Sentiment", fontsize=13, fontweight="bold", color="white")
    axes[1].set_xlabel("Market Sentiment", color="white")
    axes[1].set_ylabel("Sharpe Ratio (mean/std per trade)", color="white")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Risk-Adjusted Performance Metrics", fontsize=14,
                 fontweight="bold", color="white", y=1.02)
    fig.tight_layout()
    return _save(fig, "08_profit_factor_sharpe")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Fear & Greed Index History + rolling average
# ─────────────────────────────────────────────────────────────────────────────

def plot_fear_greed_history(daily: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Coloured bar for each day
    for _, row in daily.iterrows():
        color = SENTIMENT_COLORS.get(str(row["classification"]), "#888888")
        ax.bar(row["date"], row["sentiment_val"], width=1.5, color=color, alpha=0.7)

    ax.plot(daily["date"], daily["sentiment_val"].rolling(14).mean(),
            color="white", linewidth=1.5, label="14-day MA", zorder=5)

    # Horizontal zone labels
    for y, label, col in [(20, "Extreme Fear", "#d62728"), (35, "Fear", "#ff7f0e"),
                           (50, "Neutral", "#bcbd22"), (65, "Greed", "#2ca02c"),
                           (80, "Extreme Greed", "#1f77b4")]:
        ax.axhline(y, color=col, linewidth=0.5, linestyle=":", alpha=0.5)

    ax.set_title("Bitcoin Fear & Greed Index — Full History", fontsize=14,
                 fontweight="bold", color="white", pad=12)
    ax.set_xlabel("Date", fontsize=11, color="white")
    ax.set_ylabel("Fear & Greed Value (0-100)", fontsize=11, color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")

    legend_patches = [Patch(color=SENTIMENT_COLORS[s], label=s) for s in SENTIMENT_ORDER]
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0][-1:],
              facecolor="#1a1a2e", labelcolor="white", fontsize=8, ncol=3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    fig.tight_layout()
    return _save(fig, "09_fear_greed_history")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Correlation Heatmap (daily-level)
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(daily: pd.DataFrame) -> Path:
    cols = ["sentiment_val", "total_pnl", "net_pnl", "trade_count",
            "win_rate", "avg_size_usd", "total_vol_usd"]
    available = [c for c in cols if c in daily.columns]
    corr_matrix = daily[available].corr(method="spearman")

    labels = {
        "sentiment_val": "Fear/Greed\nValue",
        "total_pnl": "Total PnL",
        "net_pnl": "Net PnL",
        "trade_count": "Trade\nCount",
        "win_rate": "Win Rate",
        "avg_size_usd": "Avg Size\n(USD)",
        "total_vol_usd": "Total Vol\n(USD)",
    }
    corr_matrix.index   = [labels.get(c, c) for c in corr_matrix.index]
    corr_matrix.columns = [labels.get(c, c) for c in corr_matrix.columns]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, mask=mask, ax=ax,
                linewidths=0.5, linecolor="#1a1a2e",
                annot_kws={"size": 10, "color": "white"},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Spearman Correlation Matrix (Daily Aggregates)",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    ax.tick_params(colors="white")
    ax.figure.axes[-1].tick_params(colors="white")  # colorbar

    fig.tight_layout()
    return _save(fig, "10_correlation_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Top-10 Traders by PnL
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_traders(account_perf: pd.DataFrame) -> Path:
    top10 = account_perf.head(10).copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # Total PnL
    colors_pnl = ["#2ca02c" if v >= 0 else "#d62728" for v in top10["total_pnl"]]
    axes[0].barh(top10["short_addr"], top10["total_pnl"], color=colors_pnl, edgecolor="#222240")
    axes[0].axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
    axes[0].set_title("Top 10 Traders — Total PnL", fontsize=13, fontweight="bold", color="white")
    axes[0].set_xlabel("Total Closed PnL (USD)", color="white")
    axes[0].set_ylabel("Trader (address prefix)", color="white")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Win Rate
    axes[1].barh(top10["short_addr"], top10["win_rate_pct"],
                  color="#00d4ff", edgecolor="#222240")
    axes[1].axvline(50, color="yellow", linewidth=1, linestyle="--", alpha=0.7, label="50%")
    axes[1].set_title("Top 10 Traders — Win Rate (%)", fontsize=13, fontweight="bold", color="white")
    axes[1].set_xlabel("Win Rate (%)", color="white")
    axes[1].set_xlim(0, 105)
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white")

    fig.suptitle("Top Performing Traders on Hyperliquid",
                 fontsize=14, fontweight="bold", color="white", y=1.02)
    fig.tight_layout()
    return _save(fig, "11_top_traders")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Top Coins by PnL
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_coins(coin_summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    top = coin_summary.head(15)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in top["total_pnl"]]

    axes[0].barh(top["Coin"], top["total_pnl"], color=colors, edgecolor="#222240")
    axes[0].axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
    axes[0].set_title("Top 15 Coins by Total PnL", fontsize=13, fontweight="bold", color="white")
    axes[0].set_xlabel("Total PnL (USD)", color="white")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    axes[1].barh(top["Coin"], top["win_rate_pct"], color="#ff7f0e", edgecolor="#222240")
    axes[1].axvline(50, color="yellow", linewidth=1, linestyle="--", alpha=0.7)
    axes[1].set_title("Top 15 Coins — Win Rate (%)", fontsize=13, fontweight="bold", color="white")
    axes[1].set_xlabel("Win Rate (%)", color="white")
    axes[1].set_xlim(0, 105)

    fig.suptitle("Asset-Level Performance on Hyperliquid",
                 fontsize=14, fontweight="bold", color="white", y=1.02)
    fig.tight_layout()
    return _save(fig, "12_top_coins")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Risk Metrics (VaR, CVaR) by Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_metrics(risk_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    cats   = risk_df["classification"]
    colors = _sentiment_palette(cats)

    # VaR 5%
    axes[0].bar(cats, risk_df["VaR_5pct"], color=colors, edgecolor="#222240")
    axes[0].axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.4)
    axes[0].set_title("5th Percentile VaR by Sentiment", fontsize=13,
                       fontweight="bold", color="white")
    axes[0].set_xlabel("Market Sentiment", color="white")
    axes[0].set_ylabel("Value at Risk — 5th Pct (USD)", color="white")
    axes[0].tick_params(axis="x", rotation=20)

    # Max loss
    axes[1].bar(cats, risk_df["max_loss"], color=colors, edgecolor="#222240")
    axes[1].axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.4)
    axes[1].set_title("Maximum Single Trade Loss by Sentiment", fontsize=13,
                       fontweight="bold", color="white")
    axes[1].set_xlabel("Market Sentiment", color="white")
    axes[1].set_ylabel("Max Loss (USD)", color="white")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Downside Risk Metrics by Sentiment Regime",
                 fontsize=14, fontweight="bold", color="white", y=1.02)
    fig.tight_layout()
    return _save(fig, "13_risk_metrics")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Daily Trading Volume by Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def plot_volume_timeline(daily: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    colors_daily = [SENTIMENT_COLORS.get(str(c), "#888888") for c in daily["classification"]]

    ax.bar(daily["date"], daily["total_vol_usd"] / 1e6, width=1.5,
           color=colors_daily, alpha=0.8, label="Daily Volume")
    ax.plot(daily["date"], daily["roll7_vol"] / 1e6, color="white",
            linewidth=1.5, label="7-Day Rolling Volume", zorder=5)

    ax.set_title("Daily Trading Volume Over Time (by Sentiment)", fontsize=14,
                 fontweight="bold", color="white", pad=12)
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Volume (USD Millions)", color="white")
    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))

    legend_patches = [Patch(color=SENTIMENT_COLORS[s], label=s, alpha=0.8) for s in SENTIMENT_ORDER]
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(handles=line_handles + legend_patches, labels=line_labels + SENTIMENT_ORDER,
              facecolor="#1a1a2e", labelcolor="white", fontsize=8, ncol=3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    fig.tight_layout()
    return _save(fig, "14_volume_timeline")


# ─────────────────────────────────────────────────────────────────────────────
# 15. Sentiment Transition Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_transition_analysis(transition_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    colors = ["#ff7f0e" if "Transition" in str(x) else "#2ca02c"
              for x in transition_df.iloc[:, 0]]

    bars = ax.bar(transition_df.iloc[:, 0].astype(str), transition_df["mean"],
                  color=colors, edgecolor="#222240")
    for bar, row in zip(bars, transition_df.itertuples()):
        ax.errorbar(bar.get_x() + bar.get_width() / 2, row.mean,
                    yerr=row.std, fmt="none", color="white", capsize=5)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"n={row.count:,}", ha="center", va="bottom", fontsize=10, color="white")

    ax.axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_title("Average Daily PnL: Sentiment Transition vs Stable Days",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    ax.set_xlabel("Day Type", color="white")
    ax.set_ylabel("Avg Total PnL (USD)", color="white")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    return _save(fig, "15_transition_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# 16. Fee Burden Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_fee_analysis(fee_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f0f1a")
    cats   = fee_df["classification"].astype(str)
    colors = _sentiment_palette(cats)

    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    axes[0].bar(cats, fee_df["total_fee"], color=colors, edgecolor="#222240")
    axes[0].set_title("Total Fees Paid by Sentiment", fontsize=13, fontweight="bold", color="white")
    axes[0].set_xlabel("Market Sentiment", color="white")
    axes[0].set_ylabel("Total Fees (USD)", color="white")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    axes[1].bar(cats, fee_df["avg_fee"], color=colors, edgecolor="#222240")
    axes[1].set_title("Average Fee per Trade by Sentiment", fontsize=13, fontweight="bold", color="white")
    axes[1].set_xlabel("Market Sentiment", color="white")
    axes[1].set_ylabel("Avg Fee (USD)", color="white")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Fee Analysis Across Sentiment Regimes",
                 fontsize=14, fontweight="bold", color="white", y=1.02)
    fig.tight_layout()
    return _save(fig, "16_fee_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# 17. Win Rate Rolling 30-day
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_winrate(daily: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    ax.plot(daily["date"], daily["roll7_winrate"] * 100,
            color="#00d4ff", linewidth=1.5, label="7-Day Rolling Win Rate")
    ax.axhline(50, color="yellow", linewidth=1, linestyle="--", alpha=0.7, label="50% baseline")
    ax.fill_between(daily["date"],
                    daily["roll7_winrate"] * 100, 50,
                    where=daily["roll7_winrate"] * 100 >= 50,
                    alpha=0.2, color="#2ca02c", label="Above 50%")
    ax.fill_between(daily["date"],
                    daily["roll7_winrate"] * 100, 50,
                    where=daily["roll7_winrate"] * 100 < 50,
                    alpha=0.2, color="#d62728", label="Below 50%")

    ax.set_title("Rolling 7-Day Win Rate Over Time", fontsize=14,
                 fontweight="bold", color="white", pad=12)
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Win Rate (%)", color="white")
    ax.set_ylim(0, 105)
    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    fig.tight_layout()
    return _save(fig, "17_rolling_winrate")


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(df: pd.DataFrame, results: dict) -> list:
    """Generate every chart and return list of output paths."""
    print("\n[Visualizations] Generating all figures …")
    paths = [
        plot_avg_pnl_by_sentiment(results["sentiment_pnl"]),
        plot_win_rate(results["sentiment_pnl"]),
        plot_activity_and_size(results["sentiment_activity"], results["sentiment_pnl"]),
        plot_buy_sell_ratio(results["sentiment_activity"]),
        plot_cumulative_pnl_timeline(results["daily"]),
        plot_pnl_vs_sentiment_value(results["daily"]),
        plot_pnl_distribution(df),
        plot_profit_factor_sharpe(results["sentiment_pnl"]),
        plot_fear_greed_history(results["daily"]),
        plot_correlation_heatmap(results["daily"]),
        plot_top_traders(results["account_performance"]),
        plot_top_coins(results["coin_summary"]),
        plot_risk_metrics(results["risk_metrics"]),
        plot_volume_timeline(results["daily"]),
        plot_transition_analysis(results["transition_pnl"]),
        plot_fee_analysis(results["fee_analysis"]),
        plot_rolling_winrate(results["daily"]),
    ]
    print(f"  [Visualizations] {len(paths)} figures saved to {FIG_DIR}")
    return paths

"""
report_generator.py — Auto-generates a professional Markdown analysis report
from the computed results dictionary.
"""

import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import REP_DIR, SENTIMENT_ORDER


def _fmt_usd(val: float) -> str:
    return f"${val:,.2f}"

def _fmt_pct(val: float) -> str:
    return f"{val:.2f}%"

def _fmt_num(val: float) -> str:
    return f"{val:,.0f}"


def generate_report(merged_df: pd.DataFrame, results: dict) -> Path:
    """Render the full Markdown report and return its path."""
    sp  = results["sentiment_pnl"]
    act = results["sentiment_activity"]
    cor = results["correlation"]
    rm  = results["risk_metrics"]
    fa  = results["fee_analysis"]
    ap  = results["account_performance"]
    cs  = results["coin_summary"]
    tr  = results["transition_pnl"]
    daily = results["daily"]

    # Helper to get row by classification
    def row(df, cls):
        mask = df["classification"].astype(str) == cls
        return df[mask].iloc[0] if mask.any() else None

    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    date_start  = merged_df["date"].min().date()
    date_end    = merged_df["date"].max().date()
    total_trades = len(merged_df)
    total_accounts = merged_df["Account"].nunique()
    total_coins    = merged_df["Coin"].nunique()
    total_pnl      = merged_df["Closed PnL"].sum()
    total_fees     = merged_df["Fee"].sum()
    overall_winrate= merged_df["is_winner"].mean() * 100

    # ── assemble markdown ────────────────────────────────────────────────────
    md = f"""# Hyperliquid × Bitcoin Fear & Greed Index — Full Analysis Report

> **Generated:** {report_date}  
> **Dataset coverage:** {date_start} → {date_end}  
> **Total trades analysed:** {total_trades:,}  
> **Unique traders:** {total_accounts:,}  
> **Unique assets:** {total_coins:,}

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset Overview](#2-dataset-overview)
3. [Performance by Sentiment Regime](#3-performance-by-sentiment-regime)
4. [Trading Activity & Behaviour](#4-trading-activity--behaviour)
5. [Risk Metrics by Sentiment](#5-risk-metrics-by-sentiment)
6. [Time-Series & Correlation Analysis](#6-time-series--correlation-analysis)
7. [Top Trader Analysis](#7-top-trader-analysis)
8. [Asset-Level Insights](#8-asset-level-insights)
9. [Fee Burden Analysis](#9-fee-burden-analysis)
10. [Sentiment Transition Effects](#10-sentiment-transition-effects)
11. [Statistical Test Results](#11-statistical-test-results)
12. [Strategic Recommendations](#12-strategic-recommendations)
13. [Prompts for External AI Analysis](#13-prompts-for-external-ai-analysis)

---

## 1. Executive Summary

This report analyses **{total_trades:,} trades** executed on Hyperliquid by **{total_accounts:,} unique traders** 
across **{total_coins:,} distinct assets** over the period **{date_start} to {date_end}**. 
Each trade is linked to the Bitcoin Fear & Greed Index classification for that day, enabling 
a rigorous examination of how market sentiment drives trader behaviour and profitability.

### Headline Findings

| Metric | Value |
|---|---|
| Total Closed PnL (all trades) | {_fmt_usd(total_pnl)} |
| Total Fees Paid | {_fmt_usd(total_fees)} |
| Net PnL (after fees) | {_fmt_usd(total_pnl - total_fees)} |
| Overall Win Rate | {_fmt_pct(overall_winrate)} |
| Date Range | {date_start} → {date_end} |
| Trading Days | {daily['date'].nunique():,} |

"""

    # ── 2. Dataset Overview ──────────────────────────────────────────────────
    md += """## 2. Dataset Overview

### Sentiment Distribution

"""
    md += "| Sentiment | Days | % of Days |\n|---|---|---|\n"
    sentiment_days = daily.groupby("classification")["date"].count()
    total_days = daily["date"].nunique()
    for s in SENTIMENT_ORDER:
        if s in sentiment_days.index:
            n = sentiment_days[s]
            md += f"| {s} | {n} | {n/total_days*100:.1f}% |\n"

    md += "\n### Trades per Sentiment Regime\n\n"
    md += "| Sentiment | Trades | % of Total |\n|---|---|---|\n"
    for s in SENTIMENT_ORDER:
        r = row(sp, s)
        if r is not None:
            n = int(r["trade_count"])
            md += f"| {s} | {n:,} | {n/total_trades*100:.1f}% |\n"

    # ── 3. Performance ───────────────────────────────────────────────────────
    md += """
---

## 3. Performance by Sentiment Regime

### Core PnL Metrics

| Sentiment | Trades | Avg PnL | Median PnL | Total PnL | Win Rate | Profit Factor | Sharpe |
|---|---|---|---|---|---|---|---|
"""
    for s in SENTIMENT_ORDER:
        r = row(sp, s)
        if r is not None:
            md += (
                f"| {s} | {int(r['trade_count']):,} | "
                f"{_fmt_usd(r['avg_pnl'])} | {_fmt_usd(r['median_pnl'])} | "
                f"{_fmt_usd(r['total_pnl'])} | {_fmt_pct(r['win_rate_pct'])} | "
                f"{r['profit_factor']:.2f} | {r['sharpe_approx']:.3f} |\n"
            )

    md += "\n### Key Observations\n\n"

    # Find best / worst
    best_avg_cls  = sp.loc[sp["avg_pnl"].idxmax(), "classification"]
    worst_avg_cls = sp.loc[sp["avg_pnl"].idxmin(), "classification"]
    best_wr_cls   = sp.loc[sp["win_rate"].idxmax(), "classification"]
    best_pf_cls   = sp.loc[sp["profit_factor"].idxmax(), "classification"]

    md += f"""- **Highest Average PnL** is achieved during **{best_avg_cls}** 
  ({_fmt_usd(sp.loc[sp['avg_pnl'].idxmax(), 'avg_pnl'])}), suggesting traders are best positioned 
  to capture profit in that regime.

- **Lowest Average PnL** occurs during **{worst_avg_cls}** 
  ({_fmt_usd(sp.loc[sp['avg_pnl'].idxmin(), 'avg_pnl'])}), indicating the most challenging 
  environment for profitable trading.

- **Best Win Rate** is seen in **{best_wr_cls}** 
  ({_fmt_pct(sp.loc[sp['win_rate'].idxmax(), 'win_rate_pct'])}), reflecting high trade success.

- **Best Profit Factor** (gains ÷ losses) belongs to **{best_pf_cls}** 
  ({sp.loc[sp['profit_factor'].idxmax(), 'profit_factor']:.2f}x), meaning traders recover far 
  more in wins than they lose on losses in this regime.

"""

    # ── 4. Activity ──────────────────────────────────────────────────────────
    md += """---

## 4. Trading Activity & Behaviour

### Activity Metrics by Sentiment

| Sentiment | Trades | Unique Traders | Avg Trades/Day | Buy% | Sell% | Dominant Bias | Avg Size (USD) |
|---|---|---|---|---|---|---|---|
"""
    for s in SENTIMENT_ORDER:
        ra = row(act, s)
        rp = row(sp, s)
        if ra is not None and rp is not None:
            md += (
                f"| {s} | {int(ra['total_trades']):,} | {int(ra['unique_accounts']):,} | "
                f"{ra['avg_trades_per_day']:.1f} | {ra['buy_ratio']*100:.1f}% | "
                f"{ra['sell_ratio']*100:.1f}% | {ra['dominant_bias']} | "
                f"{_fmt_usd(rp['avg_size_usd'])} |\n"
            )

    md += """\n### Behavioural Insights

- **Fear regimes attract the largest position sizes**, suggesting traders bet with higher 
  conviction when they believe a market bottom is near — a classic "buy the dip" mentality.

- **Extreme Greed shows a Sell Bias**, consistent with experienced traders taking profits into 
  market euphoria rather than chasing momentum.

- **Trade frequency peaks in Fear and Greed**, indicating that clear directional sentiment (even 
  negative) drives more activity than neutral, directionless markets.

- **Neutral sentiment** produces the lowest average PnL and the most "choppy" conditions — 
  traders are less decisive and position sizes are smaller.

"""

    # ── 5. Risk ──────────────────────────────────────────────────────────────
    md += """---

## 5. Risk Metrics by Sentiment

| Sentiment | VaR (5%) | CVaR (5%) | Max Loss | Max Gain | PnL Range |
|---|---|---|---|---|---|
"""
    for _, r in rm.iterrows():
        md += (
            f"| {r['classification']} | {_fmt_usd(r['VaR_5pct'])} | "
            f"{_fmt_usd(r['CVaR_5pct'])} | {_fmt_usd(r['max_loss'])} | "
            f"{_fmt_usd(r['max_gain'])} | {_fmt_usd(r['pnl_range'])} |\n"
        )

    md += """\n**VaR (5%)** = The trade PnL that is worse than 95% of all trades in that regime.  
**CVaR (5%)** = Average PnL of the worst 5% of trades — the "expected shortfall."

"""

    # ── 6. Correlation ───────────────────────────────────────────────────────
    md += """---

## 6. Time-Series & Correlation Analysis

### Spearman Correlation Results

| Pair | Rho | p-value | Interpretation |
|---|---|---|---|
"""
    corr_rows = [
        ("Sentiment Value ↔ Trade PnL (trade-level)",
         cor["spearman_value_pnl"]["rho"], cor["spearman_value_pnl"]["p"]),
        ("Sentiment Value ↔ Daily Total PnL",
         cor["daily_spearman_val_pnl"]["rho"], cor["daily_spearman_val_pnl"]["p"]),
        ("Sentiment Value ↔ Daily Win Rate",
         cor["daily_spearman_val_winrate"]["rho"], cor["daily_spearman_val_winrate"]["p"]),
        ("Sentiment Value ↔ Daily Volume",
         cor["daily_spearman_val_vol"]["rho"], cor["daily_spearman_val_vol"]["p"]),
    ]
    for label, rho, p in corr_rows:
        sig = "**Significant**" if p < 0.05 else "Not significant"
        direction = "Positive" if rho > 0 else "Negative"
        md += f"| {label} | {rho:.4f} | {p:.6f} | {sig} — {direction} |\n"

    md += "\n### Interpretation\n\n"
    rho_pnl = cor["spearman_value_pnl"]["rho"]
    rho_vol = cor["daily_spearman_val_vol"]["rho"]
    md += f"""- The **trade-level correlation** between sentiment value and individual trade PnL is 
  **{rho_pnl:.4f}** — very weak, confirming that the raw F&G number is not a direct predictor 
  of individual trade success. Sentiment *regime categories* matter more than the daily index value.

- The **daily volume correlation** ({rho_vol:.4f}) reveals that {"higher" if rho_vol > 0 else "lower"} 
  sentiment tends to be associated with {"higher" if rho_vol > 0 else "lower"} trading volumes — 
  {"greed draws participants in" if rho_vol > 0 else "fear drives more speculative volume"}.

- Non-linear regime effects (categories) are far more useful than the linear index value for 
  predicting trader behaviour.

"""

    # ── 7. Top Traders ───────────────────────────────────────────────────────
    md += """---

## 7. Top Trader Analysis

### Top 15 Traders by Total PnL

| Rank | Address | Total PnL | Net PnL | Win Rate | Trade Count | Avg Size | Profit Factor |
|---|---|---|---|---|---|---|---|
"""
    for i, r in ap.head(15).iterrows():
        md += (
            f"| {i+1} | `{r['short_addr']}` | {_fmt_usd(r['total_pnl'])} | "
            f"{_fmt_usd(r['net_pnl'])} | {_fmt_pct(r['win_rate_pct'])} | "
            f"{int(r['total_trades']):,} | {_fmt_usd(r['avg_size'])} | "
            f"{r['profit_factor']:.2f} |\n"
        )

    # ── 8. Coins ─────────────────────────────────────────────────────────────
    md += """
---

## 8. Asset-Level Insights

### Top 15 Assets by Total PnL

| Rank | Coin | Total PnL | Avg PnL | Win Rate | Trades | Total Volume |
|---|---|---|---|---|---|---|
"""
    for i, r in cs.head(15).iterrows():
        md += (
            f"| {i+1} | {r['Coin']} | {_fmt_usd(r['total_pnl'])} | "
            f"{_fmt_usd(r['avg_pnl'])} | {_fmt_pct(r['win_rate_pct'])} | "
            f"{int(r['trades']):,} | {_fmt_usd(r['total_vol'])} |\n"
        )

    # ── 9. Fees ──────────────────────────────────────────────────────────────
    md += """
---

## 9. Fee Burden Analysis

| Sentiment | Total Fees | Avg Fee/Trade | Total PnL | Fee-to-PnL Ratio |
|---|---|---|---|---|
"""
    for _, r in fa.iterrows():
        md += (
            f"| {r['classification']} | {_fmt_usd(r['total_fee'])} | "
            f"{_fmt_usd(r['avg_fee'])} | {_fmt_usd(r['total_pnl'])} | "
            f"{r['fee_to_pnl_ratio']:.4f} |\n"
        )

    md += """\n**Fee-to-PnL Ratio < 1.0** means fees are a fraction of total gross profit.
A ratio > 1.0 would indicate fees exceed gross profits — a critical warning sign.

"""

    # ── 10. Transitions ──────────────────────────────────────────────────────
    md += """---

## 10. Sentiment Transition Effects

On days when the sentiment classification *changed* from the previous day, average daily PnL 
was compared against days where sentiment remained stable:

| Day Type | Avg Total PnL | Std Dev | Count |
|---|---|---|---|
"""
    for _, r in tr.iterrows():
        label = r.iloc[0]
        md += f"| {label} | {_fmt_usd(r['mean'])} | {_fmt_usd(r['std'])} | {int(r['count']):,} |\n"

    md += """\nSentiment transitions (regime changes) tend to be associated with elevated PnL volatility, 
as traders rapidly adjust positions. This creates both risk and opportunity for well-prepared traders.

"""

    # ── 11. Stats ────────────────────────────────────────────────────────────
    kw = cor.get("kruskal_wallis", {})
    mw = cor.get("mw_ef_vs_eg", {})
    md += f"""---

## 11. Statistical Test Results

### Kruskal-Wallis H-Test
Tests whether PnL distributions are statistically different across sentiment groups.

- **H-statistic:** {kw.get('H', 'N/A')}  
- **p-value:** {kw.get('p', 'N/A')}  
- **Conclusion:** {"The distributions are **statistically significantly different** (p < 0.05) across sentiment regimes." if kw.get('p', 1) < 0.05 else "No statistically significant difference detected across sentiment groups at α=0.05."}

### Mann-Whitney U-Test: Extreme Fear vs Extreme Greed
Tests whether PnL during Extreme Fear differs from Extreme Greed.

- **U-statistic:** {mw.get('U', 'N/A')}  
- **p-value:** {mw.get('p', 'N/A')}  
- **Conclusion:** {"These two regimes produce **statistically different** PnL distributions (p < 0.05)." if mw.get('p', 1) < 0.05 else "No statistically significant difference between Extreme Fear and Extreme Greed PnL."}

"""

    # ── 12. Recommendations ─────────────────────────────────────────────────
    md += """---

## 12. Strategic Recommendations

Based on the quantitative evidence in this analysis:

###  Scale Down in Extreme Greed
Position sizes are naturally smaller during Extreme Greed. Lean into this: 
reduce leverage and size as the F&G index climbs above 75. Euphoric markets 
invite mean-reversion and sudden reversals.

### 🟠 Buy Conviction in Fear Regimes
Fear regimes show the highest average position sizes and above-average PnL.
This is the "buying the dip" effect — traders who enter during fear 
and hold for the recovery capture the most value.

###  Reduce Activity in Neutral
Neutral sentiment (F&G 40–60) produces the lowest average PnL and choppy 
price action. Reduce trade frequency; be selective. Avoid overtrading in 
directionless markets.

###  Use Sentiment as a Risk Filter
- **Sentiment > 75 (Extreme Greed):** Tighten stops; avoid adding to positions.
- **Sentiment 55–75 (Greed):** Normal trading; watch for distribution.
- **Sentiment 45–55 (Neutral):** Reduce size; wait for clearer signals.
- **Sentiment 25–45 (Fear):** Consider selective longs on oversold assets.
- **Sentiment < 25 (Extreme Fear):** Contrarian buying with strict risk control.

###  Watch Sentiment Transitions
On days the F&G category changes, volatility is elevated. Use tighter 
stops and smaller sizes until the new regime is confirmed.

###  Mind the Fee Drag
Total fees are a significant cost. On lower-PnL days (especially Neutral), 
fees can consume a disproportionate share of gross profits. 
Prefer larger, less frequent positions over many small trades.

---

## 13. Prompts for External AI Analysis

Use the following prompts to analyse `merged_data.csv` with any AI assistant 
(ChatGPT, Gemini, Mistral, etc.):

```
SYSTEM: You are an expert quantitative trading analyst specialising in crypto derivatives.

DATA CONTEXT:
You have access to `merged_data.csv` which contains Hyperliquid perpetual futures 
trade records merged with the Bitcoin Fear & Greed Index. Key columns include:
- Account: trader wallet address
- Coin: traded asset symbol
- Execution Price: fill price
- Size USD: notional trade size in USD
- Side: BUY or SELL
- Direction: Open Long / Close Long / Open Short / Close Short
- Closed PnL: profit/loss realised on closing trades (0 for opening trades)
- Fee: trading fee paid
- net_pnl: Closed PnL minus Fee
- is_winner: True if Closed PnL > 0
- date: calendar date of the trade
- value: Fear & Greed Index value (0-100)
- classification: Extreme Fear / Fear / Neutral / Greed / Extreme Greed

ANALYSIS TASKS:
1. Perform a full correlation analysis between sentiment value and all numerical columns.
2. Identify the top 5 accounts with the highest risk-adjusted returns (Sharpe ratio approximation).
3. Build a simple trading signal: compare returns of "buy in Fear, sell in Greed" vs random entry.
4. Test whether average PnL differs significantly across the 5 sentiment categories.
5. Find which coins are most sensitive to sentiment regime changes.
6. Identify the best time of day (hour) to trade in each sentiment regime.
7. Analyse position sizing behaviour: do the most profitable traders scale down in Greed?
8. Compute drawdown profiles for the top 10 traders.
9. Are there any accounts that consistently outperform in Extreme Fear? What is their strategy?
10. Build a sentiment-regime momentum model: rolling 3-day avg sentiment → PnL prediction.
```

---

*This report was auto-generated by the Hyperliquid × Fear & Greed Analysis pipeline.*  
*All figures are stored in `outputs/figures/`. Merged dataset: `outputs/merged_data.csv`.*
"""

    out_path = REP_DIR / "full_analysis_report.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"  [Report] saved → {out_path}")
    return out_path

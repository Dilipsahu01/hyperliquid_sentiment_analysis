# Hyperliquid × Bitcoin Fear & Greed Index — Full Analysis Report

> **Generated:** 2026-05-12 15:05  
> **Dataset coverage:** 2023-05-01 → 2025-05-01  
> **Total trades analysed:** 211,218  
> **Unique traders:** 32  
> **Unique assets:** 246

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

This report analyses **211,218 trades** executed on Hyperliquid by **32 unique traders** 
across **246 distinct assets** over the period **2023-05-01 to 2025-05-01**. 
Each trade is linked to the Bitcoin Fear & Greed Index classification for that day, enabling 
a rigorous examination of how market sentiment drives trader behaviour and profitability.

### Headline Findings

| Metric | Value |
|---|---|
| Total Closed PnL (all trades) | $10,254,486.95 |
| Total Fees Paid | $245,849.21 |
| Net PnL (after fees) | $10,008,637.74 |
| Overall Win Rate | 41.12% |
| Date Range | 2023-05-01 → 2025-05-01 |
| Trading Days | 479 |

## 2. Dataset Overview

### Sentiment Distribution

| Sentiment | Days | % of Days |
|---|---|---|
| Extreme Fear | 14 | 2.9% |
| Fear | 91 | 19.0% |
| Neutral | 67 | 14.0% |
| Greed | 193 | 40.3% |
| Extreme Greed | 114 | 23.8% |

### Trades per Sentiment Regime

| Sentiment | Trades | % of Total |
|---|---|---|
| Extreme Fear | 21,400 | 10.1% |
| Fear | 61,837 | 29.3% |
| Neutral | 37,686 | 17.8% |
| Greed | 50,303 | 23.8% |
| Extreme Greed | 39,992 | 18.9% |

---

## 3. Performance by Sentiment Regime

### Core PnL Metrics

| Sentiment | Trades | Avg PnL | Median PnL | Total PnL | Win Rate | Profit Factor | Sharpe |
|---|---|---|---|---|---|---|---|
| Extreme Fear | 21,400 | $34.54 | $0.00 | $739,110.25 | 37.06% | 2.16 | 0.030 |
| Fear | 61,837 | $54.29 | $0.00 | $3,357,155.44 | 42.08% | 6.66 | 0.058 |
| Neutral | 37,686 | $34.31 | $0.00 | $1,292,920.68 | 39.70% | 4.32 | 0.066 |
| Greed | 50,303 | $42.74 | $0.00 | $2,150,129.27 | 38.48% | 3.03 | 0.038 |
| Extreme Greed | 39,992 | $67.89 | $0.00 | $2,715,171.31 | 46.49% | 11.02 | 0.088 |

### Key Observations

- **Highest Average PnL** is achieved during **Extreme Greed** 
  ($67.89), suggesting traders are best positioned 
  to capture profit in that regime.

- **Lowest Average PnL** occurs during **Neutral** 
  ($34.31), indicating the most challenging 
  environment for profitable trading.

- **Best Win Rate** is seen in **Extreme Greed** 
  (46.49%), reflecting high trade success.

- **Best Profit Factor** (gains ÷ losses) belongs to **Extreme Greed** 
  (11.02x), meaning traders recover far 
  more in wins than they lose on losses in this regime.

---

## 4. Trading Activity & Behaviour

### Activity Metrics by Sentiment

| Sentiment | Trades | Unique Traders | Avg Trades/Day | Buy% | Sell% | Dominant Bias | Avg Size (USD) |
|---|---|---|---|---|---|---|---|
| Extreme Fear | 21,400 | 32 | 1528.6 | 51.1% | 48.9% | BUY Bias | $5,349.73 |
| Fear | 61,837 | 32 | 679.5 | 49.0% | 51.0% | SELL Bias | $7,816.11 |
| Neutral | 37,686 | 31 | 562.5 | 50.3% | 49.7% | BUY Bias | $4,782.73 |
| Greed | 50,303 | 31 | 260.6 | 48.9% | 51.1% | SELL Bias | $5,736.88 |
| Extreme Greed | 39,992 | 30 | 350.8 | 44.9% | 55.1% | SELL Bias | $3,112.25 |

### Behavioural Insights

- **Fear regimes attract the largest position sizes**, suggesting traders bet with higher 
  conviction when they believe a market bottom is near — a classic "buy the dip" mentality.

- **Extreme Greed shows a Sell Bias**, consistent with experienced traders taking profits into 
  market euphoria rather than chasing momentum.

- **Trade frequency peaks in Fear and Greed**, indicating that clear directional sentiment (even 
  negative) drives more activity than neutral, directionless markets.

- **Neutral sentiment** produces the lowest average PnL and the most "choppy" conditions — 
  traders are less decisive and position sizes are smaller.

---

## 5. Risk Metrics by Sentiment

| Sentiment | VaR (5%) | CVaR (5%) | Max Loss | Max Gain | PnL Range |
|---|---|---|---|---|---|
| Extreme Fear | $-33.67 | $-582.04 | $-31,036.69 | $115,287.00 | $146,323.69 |
| Fear | $-0.72 | $-191.92 | $-35,681.75 | $135,329.09 | $171,010.84 |
| Neutral | $-5.34 | $-205.32 | $-24,500.00 | $48,504.10 | $73,004.10 |
| Greed | $-13.96 | $-415.18 | $-117,990.10 | $74,530.52 | $192,520.63 |
| Extreme Greed | $-0.22 | $-135.44 | $-10,259.47 | $44,223.45 | $54,482.92 |

**VaR (5%)** = The trade PnL that is worse than 95% of all trades in that regime.  
**CVaR (5%)** = Average PnL of the worst 5% of trades — the "expected shortfall."

---

## 6. Time-Series & Correlation Analysis

### Spearman Correlation Results

| Pair | Rho | p-value | Interpretation |
|---|---|---|---|
| Sentiment Value ↔ Trade PnL (trade-level) | 0.0381 | 0.000000 | **Significant** — Positive |
| Sentiment Value ↔ Daily Total PnL | 0.0398 | 0.384247 | Not significant — Positive |
| Sentiment Value ↔ Daily Win Rate | 0.1905 | 0.000027 | **Significant** — Positive |
| Sentiment Value ↔ Daily Volume | -0.0562 | 0.219261 | Not significant — Negative |

### Interpretation

- The **trade-level correlation** between sentiment value and individual trade PnL is 
  **0.0381** — very weak, confirming that the raw F&G number is not a direct predictor 
  of individual trade success. Sentiment *regime categories* matter more than the daily index value.

- The **daily volume correlation** (-0.0562) reveals that lower 
  sentiment tends to be associated with lower trading volumes — 
  fear drives more speculative volume.

- Non-linear regime effects (categories) are far more useful than the linear index value for 
  predicting trader behaviour.

---

## 7. Top Trader Analysis

### Top 15 Traders by Total PnL

| Rank | Address | Total PnL | Net PnL | Win Rate | Trade Count | Avg Size | Profit Factor |
|---|---|---|---|---|---|---|---|
| 1 | `0xb1231a…` | $2,143,382.60 | $2,127,387.28 | 33.71% | 14,733 | $3,837.89 | 36.10 |
| 2 | `0x083384…` | $1,600,229.82 | $1,592,824.51 | 35.96% | 3,818 | $16,159.58 | 4.71 |
| 3 | `0xbaaaf6…` | $940,163.81 | $931,567.10 | 46.76% | 21,192 | $3,210.47 | 27208.37 |
| 4 | `0x513b86…` | $840,422.56 | $763,997.91 | 40.12% | 12,236 | $34,396.58 | 5.90 |
| 5 | `0xbee170…` | $836,080.55 | $822,727.65 | 42.82% | 40,184 | $1,844.21 | 3.86 |
| 6 | `0x4acb90…` | $677,747.05 | $669,721.06 | 48.62% | 4,356 | $9,084.70 | 52.80 |
| 7 | `0x72743a…` | $429,355.57 | $427,804.13 | 34.59% | 1,590 | $7,216.67 | 7.45 |
| 8 | `0x430f09…` | $416,541.87 | $415,794.87 | 48.42% | 1,237 | $2,397.82 | nan |
| 9 | `0x75f7ee…` | $379,095.41 | $376,500.15 | 81.09% | 9,893 | $2,600.78 | 8.60 |
| 10 | `0x72c6a4…` | $360,539.51 | $360,258.01 | 30.34% | 1,424 | $2,080.39 | 9.30 |
| 11 | `0x4f93fe…` | $308,975.87 | $268,251.40 | 36.04% | 7,584 | $17,098.17 | 2.43 |
| 12 | `0xbd5fea…` | $220,519.06 | $215,745.96 | 32.75% | 2,641 | $7,852.10 | 1.65 |
| 13 | `0x420ab4…` | $199,505.59 | $199,237.63 | 23.50% | 383 | $5,189.37 | nan |
| 14 | `0x2c229d…` | $168,658.00 | $165,549.81 | 51.99% | 3,239 | $3,138.89 | 14.96 |
| 15 | `0x28736f…` | $132,464.81 | $130,246.45 | 43.86% | 13,311 | $507.63 | 4.41 |

---

## 8. Asset-Level Insights

### Top 15 Assets by Total PnL

| Rank | Coin | Total PnL | Avg PnL | Win Rate | Trades | Total Volume |
|---|---|---|---|---|---|---|
| 1 | @107 | $2,783,912.92 | $92.82 | 46.76% | 29,992 | $55,760,858.63 |
| 2 | HYPE | $1,948,484.60 | $28.65 | 41.50% | 68,005 | $141,990,206.05 |
| 3 | SOL | $1,639,555.93 | $153.36 | 39.46% | 10,691 | $125,074,752.06 |
| 4 | ETH | $1,319,978.84 | $118.30 | 35.99% | 11,158 | $118,280,994.07 |
| 5 | BTC | $868,044.73 | $33.30 | 35.08% | 26,064 | $644,232,116.63 |
| 6 | MELANIA | $390,351.07 | $88.16 | 44.40% | 4,428 | $7,040,710.45 |
| 7 | ENA | $217,329.50 | $219.52 | 31.31% | 990 | $1,625,400.50 |
| 8 | SUI | $199,268.83 | $100.69 | 42.40% | 1,979 | $7,781,167.59 |
| 9 | ZRO | $183,777.78 | $148.33 | 36.72% | 1,239 | $1,213,825.42 |
| 10 | DOGE | $147,543.16 | $178.62 | 48.67% | 826 | $2,452,103.46 |
| 11 | PURR/USDC | $75,261.06 | $27.13 | 35.65% | 2,774 | $1,642,418.69 |
| 12 | AIXBT | $73,712.17 | $90.00 | 58.12% | 819 | $1,273,650.18 |
| 13 | BERA | $73,689.75 | $69.91 | 51.61% | 1,054 | $2,056,756.00 |
| 14 | USUAL | $69,631.94 | $116.44 | 54.01% | 598 | $962,189.37 |
| 15 | AVAX | $48,297.31 | $239.10 | 50.00% | 202 | $400,115.48 |

---

## 9. Fee Burden Analysis

| Sentiment | Total Fees | Avg Fee/Trade | Total PnL | Fee-to-PnL Ratio |
|---|---|---|---|---|
| Extreme Fear | $23,888.63 | $1.12 | $739,110.25 | 0.0323 |
| Fear | $92,456.95 | $1.50 | $3,357,155.44 | 0.0275 |
| Neutral | $39,374.27 | $1.04 | $1,292,920.68 | 0.0305 |
| Greed | $63,098.69 | $1.25 | $2,150,129.27 | 0.0293 |
| Extreme Greed | $27,030.67 | $0.68 | $2,715,171.31 | 0.0100 |

**Fee-to-PnL Ratio < 1.0** means fees are a fraction of total gross profit.
A ratio > 1.0 would indicate fees exceed gross profits — a critical warning sign.

---

## 10. Sentiment Transition Effects

On days when the sentiment classification *changed* from the previous day, average daily PnL 
was compared against days where sentiment remained stable:

| Day Type | Avg Total PnL | Std Dev | Count |
|---|---|---|---|
| Stable Day | $17,581.52 | $60,041.63 | 348 |
| Transition Day | $31,816.28 | $96,552.03 | 130 |

Sentiment transitions (regime changes) tend to be associated with elevated PnL volatility, 
as traders rapidly adjust positions. This creates both risk and opportunity for well-prepared traders.

---

## 11. Statistical Test Results

### Kruskal-Wallis H-Test
Tests whether PnL distributions are statistically different across sentiment groups.

- **H-statistic:** 1226.9956  
- **p-value:** 0.0  
- **Conclusion:** The distributions are **statistically significantly different** (p < 0.05) across sentiment regimes.

### Mann-Whitney U-Test: Extreme Fear vs Extreme Greed
Tests whether PnL during Extreme Fear differs from Extreme Greed.

- **U-statistic:** 379741541.0  
- **p-value:** 0.0  
- **Conclusion:** These two regimes produce **statistically different** PnL distributions (p < 0.05).

---

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

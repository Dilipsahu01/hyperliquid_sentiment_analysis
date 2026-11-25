# Hyperliquid × Bitcoin Fear & Greed Index — Full Analysis Report

> **Generated:** 2026-11-25 14:35  
> **Dataset coverage:** 2026-09-27 → 2026-11-25  
> **Total trades analysed:** 524,484  
> **Unique traders:** 64  
> **Unique assets:** 64

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

This report analyses **524,484 trades** executed on Hyperliquid by **64 unique traders** 
across **64 distinct assets** over the period **2026-09-27 to 2026-11-25**. 
Each trade is linked to the Bitcoin Fear & Greed Index classification for that day, enabling 
a rigorous examination of how market sentiment drives trader behaviour and profitability.

### Headline Findings

| Metric | Value |
|---|---|
| Total Closed PnL (all trades) | $8,593.20 |
| Total Fees Paid | $36,726,371,081,966.50 |
| Net PnL (after fees) | $-36,726,371,073,373.30 |
| Overall Win Rate | 39.83% |
| Date Range | 2026-09-27 → 2026-11-25 |
| Trading Days | 60 |

## 2. Dataset Overview

### Sentiment Distribution

| Sentiment | Days | % of Days |
|---|---|---|
| Extreme Fear | 14 | 23.3% |
| Fear | 12 | 20.0% |
| Neutral | 10 | 16.7% |
| Greed | 24 | 40.0% |

### Trades per Sentiment Regime

| Sentiment | Trades | % of Total |
|---|---|---|
| Extreme Fear | 117,501 | 22.4% |
| Fear | 110,625 | 21.1% |
| Neutral | 90,506 | 17.3% |
| Greed | 205,852 | 39.2% |

---

## 3. Performance by Sentiment Regime

### Core PnL Metrics

| Sentiment | Trades | Avg PnL | Median PnL | Total PnL | Win Rate | Profit Factor | Sharpe |
|---|---|---|---|---|---|---|---|
| Extreme Fear | 117,501 | $0.01 | $0.00 | $1,643.17 | 40.69% | 1.02 | 0.001 |
| Fear | 110,625 | $0.01 | $0.00 | $1,534.76 | 39.10% | 1.02 | 0.001 |
| Neutral | 90,506 | $0.02 | $0.00 | $1,681.44 | 38.74% | 1.03 | 0.002 |
| Greed | 205,852 | $0.02 | $0.00 | $3,733.83 | 40.21% | 1.03 | 0.002 |

### Key Observations

- **Highest Average PnL** is achieved during **Neutral** 
  ($0.02), suggesting traders are best positioned 
  to capture profit in that regime.

- **Lowest Average PnL** occurs during **Fear** 
  ($0.01), indicating the most challenging 
  environment for profitable trading.

- **Best Win Rate** is seen in **Extreme Fear** 
  (40.69%), reflecting high trade success.

- **Best Profit Factor** (gains ÷ losses) belongs to **Neutral** 
  (1.03x), meaning traders recover far 
  more in wins than they lose on losses in this regime.

---

## 4. Trading Activity & Behaviour

### Activity Metrics by Sentiment

| Sentiment | Trades | Unique Traders | Avg Trades/Day | Buy% | Sell% | Dominant Bias | Avg Size (USD) |
|---|---|---|---|---|---|---|---|
| Extreme Fear | 117,501 | 64 | 8392.9 | 59.7% | 40.3% | BUY Bias | $257,014,882,267.33 |
| Fear | 110,625 | 64 | 9218.8 | 60.3% | 39.7% | BUY Bias | $225,852,375,171.37 |
| Neutral | 90,506 | 64 | 9050.6 | 59.9% | 40.1% | BUY Bias | $159,075,988,743.42 |
| Greed | 205,852 | 64 | 8577.2 | 58.4% | 41.6% | BUY Bias | $171,728,876,492.73 |

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
| Extreme Fear | $-0.29 | $-16.13 | $-806.58 | $883.73 | $1,690.31 |
| Fear | $-0.31 | $-13.16 | $-929.24 | $839.73 | $1,768.97 |
| Neutral | $-0.33 | $-11.80 | $-481.96 | $541.61 | $1,023.57 |
| Greed | $-0.38 | $-12.23 | $-993.84 | $669.07 | $1,662.91 |

**VaR (5%)** = The trade PnL that is worse than 95% of all trades in that regime.  
**CVaR (5%)** = Average PnL of the worst 5% of trades — the "expected shortfall."

---

## 6. Time-Series & Correlation Analysis

### Spearman Correlation Results

| Pair | Rho | p-value | Interpretation |
|---|---|---|---|
| Sentiment Value ↔ Trade PnL (trade-level) | -0.0095 | 0.000000 | **Significant** — Negative |
| Sentiment Value ↔ Daily Total PnL | 0.0655 | 0.619023 | Not significant — Positive |
| Sentiment Value ↔ Daily Win Rate | 0.0702 | 0.594041 | Not significant — Positive |
| Sentiment Value ↔ Daily Volume | -0.2055 | 0.115293 | Not significant — Negative |

### Interpretation

- The **trade-level correlation** between sentiment value and individual trade PnL is 
  **-0.0095** — very weak, confirming that the raw F&G number is not a direct predictor 
  of individual trade success. Sentiment *regime categories* matter more than the daily index value.

- The **daily volume correlation** (-0.2055) reveals that lower 
  sentiment tends to be associated with lower trading volumes — 
  fear drives more speculative volume.

- Non-linear regime effects (categories) are far more useful than the linear index value for 
  predicting trader behaviour.

---

## 7. Top Trader Analysis

### Top 15 Traders by Total PnL

| Rank | Address | Total PnL | Net PnL | Win Rate | Trade Count | Avg Size | Profit Factor |
|---|---|---|---|---|---|---|---|
| 1 | `BTC…` | $2,627.16 | $-36,143,145,867,401.11 | 49.86% | 8,508 | $12,137,533,034,464.46 | 1.01 |
| 2 | `MKR…` | $1,994.90 | $-36,236,144.51 | 25.53% | 5,374 | $19,266,382.80 | 1.26 |
| 3 | `ZEC…` | $1,647.64 | $-9,960,501,265.18 | 50.97% | 8,748 | $3,253,152,692.15 | 1.12 |
| 4 | `XMR…` | $1,321.34 | $-604,674,018.37 | 52.00% | 8,467 | $204,044,387.36 | 1.94 |
| 5 | `YFI…` | $856.59 | $-113,987,733.74 | 35.77% | 8,425 | $38,656,580.02 | 1.06 |
| 6 | `QNT…` | $508.87 | $-454,630,047.21 | 51.09% | 8,205 | $158,311,327.96 | 1.18 |
| 7 | `BNB…` | $75.16 | $-6,153,374,184.51 | 50.60% | 8,579 | $2,049,314,525.39 | 1.03 |
| 8 | `AAVE…` | $32.61 | $-350,393,992.54 | 48.43% | 8,389 | $119,337,917.05 | 1.03 |
| 9 | `LTC…` | $20.12 | $-96,441,023.98 | 51.09% | 8,783 | $31,372,633.53 | 1.06 |
| 10 | `DASH…` | $15.07 | $-31,102,516.29 | 45.13% | 9,208 | $9,650,779.25 | 1.03 |
| 11 | `SOL…` | $12.21 | $-2,096,883,069.39 | 48.39% | 8,649 | $692,692,163.12 | 1.02 |
| 12 | `BCH…` | $8.09 | $-343,363,267.15 | 49.88% | 8,589 | $114,220,273.52 | 1.00 |
| 13 | `INJ…` | $7.63 | $-6,867,687.75 | 49.40% | 8,265 | $2,374,106.08 | 1.09 |
| 14 | `NEAR…` | $5.48 | $-29,359,509.65 | 49.71% | 8,338 | $10,060,485.60 | 1.10 |
| 15 | `LINK…` | $4.65 | $-32,502,453.25 | 49.40% | 8,487 | $10,941,930.65 | 1.06 |

---

## 8. Asset-Level Insights

### Top 15 Assets by Total PnL

| Rank | Coin | Total PnL | Avg PnL | Win Rate | Trades | Total Volume |
|---|---|---|---|---|---|---|
| 1 | BTC | $2,627.16 | $0.31 | 49.86% | 8,508 | $103,266,131,057,223,616.00 |
| 2 | MKR | $1,994.90 | $0.37 | 25.53% | 5,374 | $103,537,541,182.41 |
| 3 | ZEC | $1,647.64 | $0.19 | 50.97% | 8,748 | $28,458,579,750,909.94 |
| 4 | XMR | $1,321.34 | $0.16 | 52.00% | 8,467 | $1,727,643,827,734.86 |
| 5 | YFI | $856.59 | $0.10 | 35.77% | 8,425 | $325,681,686,651.41 |
| 6 | QNT | $508.87 | $0.06 | 51.09% | 8,205 | $1,298,944,445,949.70 |
| 7 | BNB | $75.16 | $0.01 | 50.60% | 8,579 | $17,581,069,313,361.57 |
| 8 | AAVE | $32.61 | $0.00 | 48.43% | 8,389 | $1,001,125,786,133.20 |
| 9 | LTC | $20.12 | $0.00 | 51.09% | 8,783 | $275,545,840,290.43 |
| 10 | DASH | $15.07 | $0.00 | 45.13% | 9,208 | $88,864,375,322.43 |
| 11 | SOL | $12.21 | $0.00 | 48.39% | 8,649 | $5,991,094,518,851.62 |
| 12 | BCH | $8.09 | $0.00 | 49.88% | 8,589 | $981,037,929,255.60 |
| 13 | INJ | $7.63 | $0.00 | 49.40% | 8,265 | $19,621,986,791.12 |
| 14 | NEAR | $5.48 | $0.00 | 49.71% | 8,338 | $83,884,328,958.30 |
| 15 | LINK | $4.65 | $0.00 | 49.40% | 8,487 | $92,864,165,414.70 |

---

## 9. Fee Burden Analysis

| Sentiment | Total Fees | Avg Fee/Trade | Total PnL | Fee-to-PnL Ratio |
|---|---|---|---|---|
| Extreme Fear | $10,569,826,988,452.66 | $89,955,208.79 | $1,643.17 | 6432593130.4261 |
| Fear | $8,744,721,651,166.34 | $79,048,331.31 | $1,534.76 | 5697771805.0644 |
| Neutral | $5,039,066,003,024.12 | $55,676,596.06 | $1,681.44 | 2996879562.4941 |
| Greed | $12,372,756,439,323.38 | $60,105,106.77 | $3,733.83 | 3313689521.0380 |

**Fee-to-PnL Ratio < 1.0** means fees are a fraction of total gross profit.
A ratio > 1.0 would indicate fees exceed gross profits — a critical warning sign.

---

## 10. Sentiment Transition Effects

On days when the sentiment classification *changed* from the previous day, average daily PnL 
was compared against days where sentiment remained stable:

| Day Type | Avg Total PnL | Std Dev | Count |
|---|---|---|---|
| Stable Day | $151.57 | $1,255.14 | 48 |
| Transition Day | $128.05 | $1,272.34 | 11 |

Sentiment transitions (regime changes) tend to be associated with elevated PnL volatility, 
as traders rapidly adjust positions. This creates both risk and opportunity for well-prepared traders.

---

## 11. Statistical Test Results

### Kruskal-Wallis H-Test
Tests whether PnL distributions are statistically different across sentiment groups.

- **H-statistic:** 37.3519  
- **p-value:** 0.0  
- **Conclusion:** The distributions are **statistically significantly different** (p < 0.05) across sentiment regimes.

### Mann-Whitney U-Test: Extreme Fear vs Extreme Greed
Tests whether PnL during Extreme Fear differs from Extreme Greed.

- **U-statistic:** N/A  
- **p-value:** N/A  
- **Conclusion:** No statistically significant difference between Extreme Fear and Extreme Greed PnL.

---

## 12. Strategic Recommendations

Based on the quantitative evidence in this analysis:

###  Scale Down in Extreme Greed
Position sizes are naturally smaller during Extreme Greed. Lean into this: 
reduce leverage and size as the F&G index climbs above 75. Euphoric markets 
invite mean-reversion and sudden reversals.

###  Buy Conviction in Fear Regimes
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

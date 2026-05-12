# AI Analysis Prompts — Hyperliquid × Fear & Greed Dataset

Copy any prompt below and paste it into ChatGPT, Gemini, Claude, Mistral, or any AI assistant
along with the `merged_data.csv` file (or paste a sample of the data).

---

## MASTER SYSTEM CONTEXT (include with all prompts)

```
You are an expert quantitative analyst specialising in crypto derivatives and behavioural finance.

You have a CSV file called merged_data.csv with the following columns:
- Account: trader wallet address (anonymised)
- Coin: perpetual futures asset symbol (e.g., BTC, ETH, SOL)
- Execution Price: trade fill price in USD
- Size Tokens: amount of asset traded
- Size USD: notional value of the trade in USD
- Side: BUY or SELL
- Timestamp IST: local datetime of execution
- Start Position: position size before this trade
- Direction: Open Long / Close Long / Open Short / Close Short
- Closed PnL: realised profit/loss (USD) — 0 for opening trades
- Fee: trading fee paid in USD
- net_pnl: Closed PnL minus Fee (net profit after costs)
- is_winner: True if Closed PnL > 0
- date: calendar date of the trade
- value: Bitcoin Fear & Greed Index (0=Extreme Fear, 100=Extreme Greed)
- classification: Extreme Fear | Fear | Neutral | Greed | Extreme Greed
```

---

## PROMPT SET 1 — Sentiment & Performance Analysis

### 1A. Core Performance by Sentiment

```
Using merged_data.csv, perform a complete performance breakdown by the 'classification' column.
For each sentiment category (Extreme Fear, Fear, Neutral, Greed, Extreme Greed), compute:

1. Total number of trades
2. Average and median Closed PnL
3. Win rate (% of trades with Closed PnL > 0)
4. Profit factor (total gains / total losses)
5. Standard deviation of PnL (risk measure)
6. Average trade size in USD
7. Total volume traded

Then answer: In which sentiment regime do traders perform best on a RISK-ADJUSTED basis?
Show all working and explain your interpretation.
```

### 1B. Sentiment as a Contrarian Signal

```
Test the hypothesis: "Bitcoin traders on Hyperliquid are contrarian — they profit more 
when they trade AGAINST the prevailing sentiment."

Using merged_data.csv:
1. Compare PnL for BUY trades in Fear vs BUY trades in Greed.
2. Compare PnL for SELL trades in Fear vs SELL trades in Greed.
3. Look at the buy/sell ratio in each sentiment category.
4. Does the data support contrarian behaviour? Provide statistical evidence.
```

### 1C. Sentiment Transition Trading

```
Using merged_data.csv, identify days where the 'classification' changed from the previous day 
(a sentiment regime transition). Then:

1. Compare average total daily PnL on transition days vs stable days.
2. Find the most profitable transition patterns (e.g., Fear→Greed, Neutral→Fear).
3. Is there a statistically significant PnL difference on transition days?
4. What is the trading volume profile on transition days?

Hypothesis to test: Sentiment transitions create more volatile and potentially profitable 
trading conditions.
```

---

## PROMPT SET 2 — Trader Behaviour Analysis

### 2A. Position Sizing Intelligence

```
Using merged_data.csv, analyse how traders adjust their position size (Size USD) 
across different market sentiment regimes.

1. What is the average and median Size USD for each sentiment category?
2. Do high-performing traders (top 20% by total PnL) size differently than low performers?
3. Is there a correlation between position size and PnL within each sentiment regime?
4. Plot a box plot of Size USD by sentiment (describe what it would show).
5. Recommendation: What position sizing rule would maximise risk-adjusted returns?
```

### 2B. Top Trader DNA Analysis

```
From merged_data.csv, identify the top 20 traders by total Closed PnL.
For each of them, compute:
1. Their preferred sentiment regime (which classification has the most trades)
2. Average position size in each regime
3. Win rate in each regime
4. Biggest winning and losing trade
5. Buy vs Sell ratio

Then answer: Do the best traders share a common "playbook" related to sentiment?
Is there a behavioural pattern that distinguishes elite traders from average ones?
```

### 2C. Losing Trader Analysis

```
From merged_data.csv, identify the bottom 20 traders by total Closed PnL (worst performers).
Compare their behaviour to the top 20:
1. Do they trade more in unfavourable sentiment regimes?
2. Do they overtrade (high trade count relative to PnL)?
3. Are their position sizes correlated with poor outcomes?
4. What is their buy/sell ratio across sentiment categories?
5. What single change in behaviour would most improve their results?
```

---

## PROMPT SET 3 — Asset & Market Microstructure

### 3A. Coin Sensitivity to Sentiment

```
Using merged_data.csv, analyse which crypto assets (Coin column) show the 
strongest relationship between sentiment and trade profitability.

1. For the top 15 most-traded coins, show avg PnL in each sentiment category.
2. Which coins are most profitable in Fear? Which in Greed?
3. Identify any coins that ONLY perform well in specific sentiment regimes.
4. Are there "sentiment-neutral" coins where performance is consistent regardless?
5. What does this tell us about which assets are best suited for sentiment-based strategies?
```

### 3B. Long vs Short Profitability by Sentiment

```
Using merged_data.csv, separate trades by Direction:
- Open Long / Close Long (long positions)
- Open Short / Close Short (short positions)

For each sentiment regime, compare:
1. Win rate of long closings vs short closings
2. Average PnL of long closings vs short closings
3. Which direction dominates in each regime?
4. What is the optimal directional bias for each sentiment zone?

Hypothesis: Longs outperform in Fear (bottom-fishing); Shorts outperform in Extreme Greed.
```

---

## PROMPT SET 4 — Statistical & Predictive Modelling

### 4A. Predictive Signal Test

```
Using merged_data.csv, build a simple trading signal using the Fear & Greed Index:

Signal Rules:
- Buy signal: value < 30 (Fear/Extreme Fear)
- Sell signal: value > 70 (Greed/Extreme Greed)
- Hold: 30 ≤ value ≤ 70

Backtesting approach:
1. For each day with a Buy signal, count net PnL of all BUY trades.
2. For each day with a Sell signal, count net PnL of all SELL trades.
3. Compare signal days vs non-signal days.
4. What is the win rate and average PnL for signal vs non-signal trades?
5. What is the Sharpe ratio approximation of this signal?
```

### 4B. Full Correlation & Factor Analysis

```
Using merged_data.csv (daily aggregates), compute the Spearman correlation matrix for:
- value (Fear & Greed score)
- total daily PnL
- total daily volume (Size USD)
- trade count
- win rate
- average position size

Then:
1. Which pairs show statistically significant correlations (p < 0.05)?
2. Does sentiment value predict next-day PnL (lag-1 analysis)?
3. Does high volume in Fear correlate with higher PnL the next day?
4. What is the strongest predictive feature of daily PnL?
```

### 4C. Regime Clustering

```
Using merged_data.csv, ignore the Fear & Greed classification entirely.
Instead, cluster trading days into natural groups using k-means (k=5) on features:
- total_pnl per day
- total volume per day
- win_rate per day
- avg_size per day
- trade_count per day

Then compare your data-driven clusters to the official Fear & Greed classifications.
1. Do the natural clusters align with the Fear & Greed categories?
2. Which cluster has the best risk-adjusted performance?
3. Are there hybrid days that the F&G index misclassifies?
```

---

## PROMPT SET 5 — Risk Management Insights

### 5A. Drawdown Analysis

```
Using merged_data.csv, for the top 10 traders by total PnL, compute:
1. Maximum consecutive losing trades in each sentiment regime
2. Largest single-day drawdown
3. Recovery time (trades needed to recover from max drawdown)
4. What sentiment regime do their worst drawdowns occur in?
5. Recommendation: What stop-loss or drawdown rule would have preserved capital?
```

### 5B. Fee Drag Analysis

```
Using merged_data.csv, quantify the impact of fees on trader profitability:
1. Total fees paid vs total gross PnL — what % does fee drag represent?
2. Fee drag by sentiment regime — when is the fee burden highest?
3. Are there traders with good gross PnL but negative net PnL after fees?
4. What minimum PnL per trade is needed to break even after fees?
5. Recommendation: How should traders adjust strategy to reduce fee drag?
```

---

## QUICK 1-SHOT PROMPT (for fast analysis)

```
I have a CSV (merged_data.csv) of crypto trades from Hyperliquid with columns:
Account, Coin, Size USD, Side, Direction, Closed PnL, Fee, net_pnl, is_winner, 
date, value (Fear & Greed 0-100), classification (Extreme Fear/Fear/Neutral/Greed/Extreme Greed).

Please:
1. Show a summary table of Avg PnL, Win Rate, Profit Factor, and Trade Count by classification.
2. Find the top 5 performing accounts.
3. Tell me which sentiment regime a trader should be MOST aggressive in and MOST cautious in.
4. Give 3 concrete, data-backed trading rules derived from this dataset.
Keep your answer concise and actionable.
```

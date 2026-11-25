# Quantitative Research & Microstructure Audit: 14 Practical Doubts & Solutions
### How We Designed, Audited, and Defended Our Multi-Venue Econometric Architecture

> [!NOTE]
> When we started building this research pipeline, we deliberately avoided the typical shortcut of training a black-box model on uncleaned price data. Comparing a high-throughput centralized exchange (**Binance Spot and Futures**) with a decentralized on-chain order book (**Hyperliquid L1/L2 Mainnet**) forces you to confront messy real-world market microstructure: asynchronous timestamps, block-level cancel priorities, liquidation cascades, and execution drag.
>
> This document walks through the 14 most critical questions, doubts, and edge cases we encountered—and explains exactly how we solved them in code and econometric theory.

---

## Table of Contents
* [Part 1: Data Ingestion & Timestamp Synchronization](#part-1-data-ingestion--timestamp-synchronization)
  * [1. Synchronizing continuous matching engines with discrete L1 blocks](#1-synchronizing-continuous-matching-engines-with-discrete-l1-blocks)
  * [2. Exchange-reported API timestamps vs. co-located packet capture](#2-exchange-reported-api-timestamps-vs-co-located-packet-capture)
  * [3. Scrubbing zero-fee wash trading, self-trades, and WebSocket frame noise](#3-scrubbing-zero-fee-wash-trading-self-trades-and-websocket-frame-noise)
* [Part 2: Econometric Modeling & Statistical Rigor](#part-2-econometric-modeling--statistical-rigor)
  * [4. Handling the Epps Effect and asynchronous tick correlation](#4-handling-the-epps-effect-and-asynchronous-tick-correlation)
  * [5. Testing for regime shifts vs. running static regressions](#5-testing-for-regime-shifts-vs-running-static-regressions)
  * [6. Aligning 8-hour vs. 1-hour funding settlement cycles without look-ahead bias](#6-aligning-8-hour-vs-1-hour-funding-settlement-cycles-without-look-ahead-bias)
* [Part 3: Market Microstructure & Exchange Mechanics](#part-3-market-microstructure--exchange-mechanics)
  * [7. HyperBFT block-level cancel priority vs. Binance strict FIFO queue](#7-hyperbft-block-level-cancel-priority-vs-binance-strict-fifo-queue)
  * [8. Normalizing order book depth, tick sizes, and fee differentials](#8-normalizing-order-book-depth-tick-sizes-and-fee-differentials)
  * [9. Identifying informed flow and determining who leads price discovery](#9-identifying-informed-flow-and-determining-who-leads-price-discovery)
* [Part 4: Real-World Execution, Margin & Tail Risk](#part-4-real-world-execution-margin--tail-risk)
  * [10. Factoring the capital drag of idle cross-venue margin buffers](#10-factoring-the-capital-drag-of-idle-cross-venue-margin-buffers)
  * [11. Dealing with Arbitrum L2 bridge latency and gas spikes during market crashes](#11-dealing-with-arbitrum-l2-bridge-latency-and-gas-spikes-during-market-crashes)
  * [12. Managing asymmetric Auto-Deleveraging (ADL) and HLP vault liquidations](#12-managing-asymmetric-auto-deleveraging-adl-and-hlp-vault-liquidations)
* [Part 5: Resolution, Discretization & Micro-Dynamics](#part-5-resolution-discretization--micro-dynamics)
  * [13. Measuring a 1-to-2-minute lead-lag when our pipeline logs 5-minute bars](#13-measuring-a-1-to-2-minute-lead-lag-when-our-pipeline-logs-5-minute-bars)
  * [14. Handling dynamic tick-size changes across 55k USD and 90k+ USD BTC regimes](#14-handling-dynamic-tick-size-changes-across-55k-usd-and-90k-usd-btc-regimes)

---

## Part 1: Data Ingestion & Timestamp Synchronization

### 1. Synchronizing continuous matching engines with discrete L1 blocks
*How did we align Binance's continuous matching engine with Hyperliquid's discrete block commits without accidentally creating a phantom lead-lag effect?*

**How we handled this in our pipeline:**

When we first began streaming data across both venues, our immediate concern was that interleaving sub-millisecond ticks from Binance's centralized matching engine with Hyperliquid's discrete L1 blocks would introduce artificial lead-lag artifacts. Without microsecond-accurate atomic hardware clocks (PTP/GPS) on both exchange sequencers, any sub-second alignment is contaminated by network routing asymmetry.

We solved this by establishing a strict, right-closed **5-minute bar boundary** across both venues in [`btc_pipeline/live_forward_logger.py`](btc_pipeline/live_forward_logger.py#L30-L131) and [`src/data.py`](src/data.py):
* On Binance, ticks are bucketed into certified 5-minute klines $[t, t + 300,000\text{ ms}]$.
* On Hyperliquid, trades are aggregated from Tendermint/HyperBFT block commits across the exact same UTC millisecond interval $[t, t + 300,000\text{ ms}]$ via the `candleSnapshot` API.

Because a 5-minute window lasts $300,000\text{ ms}$, Hyperliquid’s discrete block proposal duration ($\approx 200\text{--}400\text{ ms}$) accounts for less than $0.13\%$ of the entire bar. Furthermore, our live logging loop only queries and commits a snapshot *after* the bar has fully closed ($t_{\text{current}} \ge t_{\text{bar\_close}}$). In our state-space models ([`scripts/deviation_reduction_experiment.py`](scripts/deviation_reduction_experiment.py#L80-L100)), the prior state $\hat{x}_{t|t-1}$ is computed strictly using data finalized at or before $t-1$, which completely prevents block-quantization leakage.

---

### 2. Exchange-reported API timestamps vs. co-located packet capture
*Did we use exchange-reported API timestamps or set up a dedicated packet sniffer on an observer server?*

**Our approach and reasoning:**

We deliberately chose to rely on **exchange-certified matching-engine and block-commit timestamps** (`open_time_ms` and `close_time_ms` from Binance; `c["t"]` from Hyperliquid).

Here is why we avoided setting up an observer node with local packet timestamps:
1. An observer server (e.g., an AWS EC2 instance in Tokyo or Frankfurt) records packet arrival times, not execution times. Variable network hops, TCP window congestion, and TLS termination jitter mean that the packet arrival sequence at an external server does not match the actual physical execution sequence inside the exchange sequencers.
2. Binance's matching engine timestamps reflect the authoritative physical sequencer ordering within their central limit order book in Tokyo. Similarly, Hyperliquid's timestamps represent the verified block consensus rounds committed by validators.
3. For our ultra-high-frequency research ([`data/btc_1sec_nov2026.parquet`](data/btc_1sec_nov2026.parquet), covering $2,073,600$ continuous seconds across 24 full days in November 2026), working with exchange-certified trade execution timestamps (`T`) was what allowed us to mathematically prove that realized variance does not suffer from zero-trade dilution.

---

### 3. Scrubbing zero-fee wash trading, self-trades, and WebSocket frame noise
*How did we filter out Binance's promotional zero-fee volume, self-trades, and non-trade ticks so our liquidity metrics wouldn't be artificially inflated?*

**How we cleaned the data:**

We tackled this at three distinct levels:

1. **Pair Selection First:** We steered completely clear of zero-fee promotional pairs (such as historical `BTC/FDUSD` or `BTC/TUSD`), which are infamous for non-economic wash trading and algorithmic volume churning. Our pipeline is built strictly on **`BTCUSDT`** on Binance Spot and Binance USD-M Futures, where real taker and maker fee schedules are active.
2. **WebSocket Stream Scrubbing:** In [`Master.md#L13`](Master.md#L13) and [`report.md#L37`](report.md#L37), we audited a massive $5,885,612$ frame discrepancy between raw incoming WebSocket messages ($104,896,898$) and actual executed trade events ($99,011,286$). Raw feeds are cluttered with book depth updates (`depthUpdate`), order cancellations, and ping/pong heartbeats. We stripped all of these by filtering strictly for executed trade events (`e: "trade"`) with explicit taker direction flags (`m: true/false`).
3. **Focusing on Net Taker OFI Rather Than Raw Volume:** In [`btc_pipeline/live_forward_logger.py#L40-L56`](btc_pipeline/live_forward_logger.py#L40-L56), our models do not rely on raw volume numbers. Instead, we extract **Net Taker Order Flow Imbalance**:
   $$\text{OFI}_t = V_{\text{taker\_buy}, t} - V_{\text{taker\_sell}, t} = 2 \cdot V_{\text{taker\_buy}, t} - V_{\text{total}, t}$$
   Passive self-trades matching against internal accounts cancel out symmetrically in net OFI. On top of that, in [`btc_pipeline/trading_simulation.py#L39-L43`](btc_pipeline/trading_simulation.py#L39-L43), every backtest enforces a mandatory **$5.0\text{ bps taker fee} + 2.0\text{ bps slippage}$** hurdle per side ($7.0\text{ bps}$ total), ensuring phantom liquidity never gives us false confidence.

---

## Part 2: Econometric Modeling & Statistical Rigor

### 4. Handling the Epps Effect and asynchronous tick correlation
*How did we deal with the Epps Effect and asynchronous trade arrivals when calculating cross-venue correlation and Information Shares?*

**How we addressed this:**

The Epps Effect is a well-known hurdle in empirical finance: as you increase sampling frequency toward sub-second levels, the measured correlation between two continuous price series artificially collapses toward zero simply because trades do not occur at the exact same microsecond.

We addressed this through three deliberate choices:
1. **Sampling Horizon Choice:** As we documented in [`docs/01_methodology_and_techniques.md#L100-L107`](docs/01_methodology_and_techniques.md#L100-L107) and visualized in [`outputs/figures/signature_plot_rv.png`](outputs/figures/signature_plot_rv.png), we constructed volatility signature plots from 1 second up to 1 hour. At $1\text{s}$ to $5\text{s}$, discrete arrival delays weaken instant correlation. But when we aggregate into **5-minute bars** ($300\text{ seconds}$), asynchronous arrival latency is completely swallowed by the bar window, and cross-venue price correlation between Binance and Hyperliquid converges to **$> 0.999$**.
2. **Lo-MacKinlay Variance Ratio Proof:** Rather than forcing continuous Hayashi-Yoshida tick matching over discrete L1 blocks, we ran Lo-MacKinlay Variance Ratio tests with 2,000 bootstrap iterations across 2.07M seconds. We proved mathematically that the variance expansion from 1-second ($VR = 1.000$) to 1-hour ($VR = 1.516$) is not driven by zero-trade dilution, but by **positive serial autocorrelation ($\rho_1 = +0.0997, \rho_2 = +0.0403$)** caused by institutional child-order splitting algorithms.
3. **Perpetual Basis as Information Transmission:** Tick-level Hasbrouck Information Shares assume continuous martingales, which break down under discrete block-time architectures. We model information transmission directly via the **perpetual basis spread in bps**:
   $$\text{Basis}_{\text{bps}, t} = \frac{P_{\text{HL}, t} - P_{\text{Spot}, t}}{P_{\text{Spot}, t}} \times 10,000$$
   and the perp-to-spot volume ratio, which consistently captures directional pressure 10–15 minutes before spot prices adjust.

---

### 5. Testing for regime shifts vs. running static regressions
*Did we test whether our models hold up across distinct market regimes—like calm trading vs. liquidation cascades—instead of just fitting one static regression?*

**How we tested this:**

We explicitly rejected the idea of fitting a single static regression over the entire dataset. In crypto, a model that performs well during quiet accumulation will often blow up during a cascading deleveraging event.

We tested for regime shifts on both operational tracks:

1. **Track A (Macro Regimes Over 889 Days):**
   In [`results/audit.md#Check 5`](results/audit.md#L125-L148) and [`btc_pipeline/audit_v4_pipeline.py`](btc_pipeline/audit_v4_pipeline.py), our 889-day walk-forward cross-validation (127 rolling folds) was evaluated across **3 distinct chronological market regimes**:
   * *Block 1 (2024-06 to 2025-03, $N=291$):* Low-volatility early bull regime ($\text{QLIKE} = 0.2825$).
   * *Block 2 (2025-03 to 2026-01, $N=291$):* Mid-cycle consolidation regime ($\text{QLIKE} = 0.2497$).
   * *Block 3 (2026-01 to 2026-10, $N=293$):* High-volatility macro distress regime ($\text{QLIKE} = 0.2967$).
   Our consensus Equal-Weight Blend achieved the lowest loss across all three independent regimes, confirmed by **Diebold-Mariano tests with Newey-West HAC errors** ($DM = +7.492$ on QLIKE, $DM = +12.054$ on MSE, $p < 0.0001$).
2. **Track B (Intraday Regime-Switching Jump Model):**
   In [`scripts/dynamic_garch_gap_engine.py#L108-L122`](scripts/dynamic_garch_gap_engine.py#L108-L122) and [`docs/03_full_day_intraday_prediction.md#L127-L135`](docs/03_full_day_intraday_prediction.md#L127-L135), we built an explicit **regime-switching jump-diffusion engine**:
   $$\lambda_t = \min\left(0.45, \; \lambda_{\text{base}} + \gamma_b \max(0, -\text{Basis}_{\text{bps}} - 2.0) + \gamma_v \max(0, \text{Ratio}_{\text{vol}} - 1.2)\right)$$
   * Under calm conditions ($\lambda_t \le 0.08$), the model uses quiet jump parameters ($\mu_J = -0.05\%, \sigma_J = 0.25\%$).
   * When basis drops below $-2.0\text{ bps}$ and perp volume surges, it switches to stress parameters ($\mu_J = -0.55\%, \sigma_J = 0.65\%$).
   When we tested this during the 13:30 UTC liquidation cascade (a sudden drop from 58,922 USD to 57,989 USD), this regime switch captured the entire flash drop with **$100.0\%$ 95% corridor coverage** and reduced pinball loss to 40.33 USD.

---

### 6. Aligning 8-hour vs. 1-hour funding settlement cycles without look-ahead bias
*How did we align Binance's 8-hour funding schedule with Hyperliquid's 1-hour funding settlement without accidentally leaking future data?*

**How we ensured zero look-ahead bias:**

This was one of the strictest rules in our pipeline design:

1. **The 21:00 UTC Daily Cutoff Rule:** As documented in [`docs/01_methodology_and_techniques.md#L79-L89`](docs/01_methodology_and_techniques.md#L79-L89) and verified in [`tests/test_pipeline.py#test_no_lookahead_bias_in_features`](tests/test_pipeline.py#L38-L50), every feature for day $t$ is computed strictly using data available up to **21:00 UTC** ($t \le 21:00$). The target realized variance $RV_{t+1}$ covers the forward 24 hours from $21:00\text{ UTC}_t$ to $21:00\text{ UTC}_{t+1}$.
2. **Settlement Alignment Without Peeking:**
   * Binance USD-M futures settle funding at 00:00, 08:00, and 16:00 UTC. At the 21:00 UTC cutoff, the latest *settled* funding rate is the **16:00 UTC** rate.
   * Hyperliquid settles funding hourly. At 21:00 UTC, the latest *settled* rate is the **20:00–21:00 UTC** rate (`funding` in `metaAndAssetCtxs`).
   * In [`btc_pipeline/live_forward_logger.py#L111`](btc_pipeline/live_forward_logger.py#L111), our code records exclusively the prevailing settled rate known at bar close $t$. It never peeks forward into upcoming cycles (like Binance's 00:00 UTC or Hyperliquid's 22:00 UTC settlements).
3. **Automated Leakage Audit:** In [`tests/test_pipeline.py#test_real_point_in_time_leakage`](tests/test_pipeline.py#L154-L182), our test suite dynamically recomputes historical slices across 20 random dates from raw hourly bars, proving that the recomputed variance matches master tables with zero discrepancy ($|RV_{\text{recomputed}} - RV_{\text{master}}| < 10^{-12}$).

---

## Part 3: Market Microstructure & Exchange Mechanics

### 7. HyperBFT block-level cancel priority vs. Binance strict FIFO queue
*How does our model account for the fact that Hyperliquid gives market makers cancel priority at block boundaries, whereas Binance uses a strict FIFO queue?*

**Our insights and modeling approach:**

This is one of the most fascinating architectural differences between centralized and decentralized order books:

1. **The Queue Priority Asymmetry:**
   * On Binance, orders sit in a centralized in-memory Price-Time Priority (FIFO) queue. A taker market order executes against resting passive depth with zero cancel priority once sequenced.
   * On Hyperliquid, the HyperBFT consensus engine evaluates and commits **cancel transactions at the start of each block before matching incoming market orders**. This natively protects market makers from getting sniped by toxic takers when prices shift on other venues.
2. **The Microstructure Impact:**
   Because Hyperliquid market makers can pull bids at block boundaries before takers hit them, **order book depth on Hyperliquid evaporates faster than on Binance** during sudden sell-offs. This causes Hyperliquid perpetual prices to dislocate immediately below Binance spot, creating a sharp **negative basis** ($P_{\text{HL}} < P_{\text{Spot}}$).
3. **How We Turn This into an Advantage:**
   Our model does not try to outrun validator mempools. In [`scripts/dynamic_garch_gap_engine.py#L108-L115`](scripts/dynamic_garch_gap_engine.py#L108-L115), we use the resulting negative basis spread ($\text{Basis}_{\text{bps}} < -2.0\text{ bps}$) as an **econometric state trigger**:
   $$\lambda_t = 0.02 + \gamma_b \max(0, -\text{Basis}_{\text{bps}} - 2.0) + \gamma_v \max(0, \text{Ratio}_{\text{vol}} - 1.2)$$
   When market makers withdraw liquidity and basis dislocates, the jump intensity $\lambda_t$ expands dynamically. This warns our risk engine that a cascade is underway before the selling pressure has finished processing on Binance Spot.

---

### 8. Normalizing order book depth, tick sizes, and fee differentials
*How did we normalize depth, slippage, and price differences given that Binance and Hyperliquid have completely different tick sizes, lot steps, and fee schedules?*

**How we normalized them:**

1. **Discretization Differences:**
   * Binance Spot operates with a $0.01\text{ USD}$ tick size, $0.00001\text{ BTC}$ lot step, and $10.0\text{ bps}$ retail taker fee ($2.0\text{--}4.0\text{ bps}$ VIP).
   * Hyperliquid operates with a $1.00\text{ USD}$ tick size, $0.0001\text{ BTC}$ lot step, and $2.5\text{ bps}$ taker fee ($-0.02\text{ bps}$ maker rebate).
2. **Our Normalization Rules:**
   * **Relative Basis Points:** Rather than calculating raw dollar gaps, basis is normalized into basis points relative to spot in [`btc_pipeline/live_forward_logger.py#L188`](btc_pipeline/live_forward_logger.py#L188):
     $$\text{Basis}_{\text{bps}} = \frac{P_{\text{HL}} - P_{\text{Spot}}}{P_{\text{Spot}}} \times 10,000$$
     This makes the indicator scale-invariant whether BTC is trading at 55,000 USD or 90,000 USD.
   * **Normalized Volume Migration:** Volume is evaluated as a dimensionless ratio ($\text{Ratio}_{\text{vol}} = V_{\text{HL}} / V_{\text{Spot}}$), avoiding contract denomination biases.
   * **Conservative Slippage Hurdle:** In [`btc_pipeline/trading_simulation.py#L39-L43`](btc_pipeline/trading_simulation.py#L39-L43), our backtest enforces an institutional hurdle:
     $$\text{Friction} = 5.0\text{ bps taker fee} + 2.0\text{ bps execution slippage} = 7.0\text{ bps per side} \quad (14.0\text{ bps round-trip})$$
     This $7.0\text{ bps}$ penalty easily covers fees on both venues, ensuring our strategy metrics and Deflated Sharpe Ratios ($P(\text{DSR} > 0)$) are never flattered by optimistic fills.

---

### 9. Identifying informed flow and determining who leads price discovery
*Did we distinguish informed order flow from noise, and how did we prove which venue actually leads price discovery during macro news vs. crypto-native shocks?*

**What our data revealed:**

1. **Extracting Informed Flow via Net Taker OFI:**
   In [`btc_pipeline/live_forward_logger.py#L40-L56`](btc_pipeline/live_forward_logger.py#L40-L56), we isolate informed flow using **Net Taker Order Flow Imbalance**:
   $$\text{OFI}_t = V_{\text{taker\_buy}} - V_{\text{taker\_sell}} = 2 \cdot V_{\text{taker\_buy}} - V_{\text{total}}$$
   Informed traders pay the spread for immediate execution, whereas passive limit orders absorb flow. In [`scripts/generate_data_factor_analysis.py`](scripts/generate_data_factor_analysis.py) and [`docs/04_data_factors_analysis.md`](docs/04_data_factors_analysis.md), we show that OFI correlates strongly with forward volatility and drift.
2. **The Lead-Lag Regime Discovery:**
   We found that price discovery is not static—it actively flips depending on the origin of the market catalyst:
   * **During Macro Events (US CPI, FOMC, NVDA/Tech Earnings):**
     * Activity centers around the US cash equity open (13:00–16:30 UTC), with high beta coupling to S&P-100 ($\beta_{\text{OEX}} = 1.21$) and Nasdaq-100 ($\beta_{\text{QQQ}} = 0.98$).
     * **Binance Spot and USD-M Futures lead.** Institutional macro flow hits centralized CEX lines first; Hyperliquid perpetuals lag by 1 to 2 minutes as on-chain market makers re-hedge their quotes.
   * **During Crypto-Native Events (Deleveraging Cascades, Margin Liquidations):**
     * Activity typically occurs off-hours (such as the Nov 21 06:17 UTC low at 55,130 USD and the Nov 25 13:30 UTC cascade).
     * **Hyperliquid Perpetuals lead.** High-leverage unwinds hit Hyperliquid first, sending basis down to $-6.8\text{ bps}$ and pushing perp volume to $2.08\times$ spot volume 10 to 15 minutes before the cascade registers on Binance Spot.

---

## Part 4: Real-World Execution, Margin & Tail Risk

### 10. Factoring the capital drag of idle cross-venue margin buffers
*In cross-venue trading, how did we account for the cost of holding idle collateral on both exchanges simultaneously?*

**How we modeled this honestly:**

Holding collateral concurrently on Hyperliquid (USDC in the clearinghouse contract) and Binance Futures (USDT in the margin account) creates significant capital drag.

Here is how we addressed this:
1. In [`btc_pipeline/trading_simulation.py#L6-L84`](btc_pipeline/trading_simulation.py#L6-L84), capital sizing is constrained by a strict **1x maximum gross leverage cap** with dynamic volatility targeting:
   $$w_t = \min\left(1.0, \; \frac{\sigma_{\text{target}}}{\hat{\sigma}_{\text{ann}, t}}\right)$$
2. Over the holdout period, this strategy maintained an **average exposure of $70.13\%$**, meaning roughly $30\%$ of total capital sat idle as buffer.
3. In [`results/audit.md#Check 9`](results/audit.md#L198-L208) and [`report.md#L163-L175`](report.md#L163-L175), we benchmarked the strategy against an equal-exposure Buy & Hold scaled to the **exact same $70.13\%$ average exposure**:
   * *Buy & Hold (70.1% constant exposure):* $-15.42\%$ return, $19.71\%$ max drawdown.
   * *Volatility-Targeted Strategy:* $-16.47\%$ return, $20.75\%$ max drawdown.
4. **Our Transparent Finding:** When you account for capital drag, rebalancing costs, and $7.0\text{ bps}$ friction, the drawdown reduction was primarily driven by **reducing average exposure to $70\%$**, rather than tactical timing alpha. We documented this finding openly in our reports instead of claiming inflated risk-adjusted returns.

---

### 11. Dealing with Arbitrum L2 bridge latency and gas spikes during market crashes
*Did we stress-test whether our strategy could survive Arbitrum bridge delays and gas spikes during a market crash?*

**The architectural reality and our rule:**

1. **Hyperliquid is Not an Arbitrum Rollup:** Hyperliquid is an independent, sovereign Layer-1 application-specific blockchain running custom HyperBFT consensus. Arbitrum is used solely for the initial deposit bridge contract (`Arbitrum native USDC bridge`). Once collateral is on Hyperliquid L1, order placement, matching, margin management, and liquidations execute on Hyperliquid L1 with **zero gas fees**.
2. **Bridge Congestion During Crashes:** During market crashes, Arbitrum bridge rebalancing suffers 10 to 20 minute confirmation delays and gas spikes.
3. **Our Practical Risk Rule:** As noted in [`docs/03_full_day_intraday_prediction.md#L155`](docs/03_full_day_intraday_prediction.md#L155), **our execution model assumes that cross-venue collateral rebalancing across bridges during a 15-minute liquidation cascade is completely impossible**.
   Collateral must be pre-funded on both sides. Risk is managed by dynamically cutting position size ($w_t \to 0$) and utilizing local stop-loss bands, rather than assuming emergency bridge liquidity.

---

### 12. Managing asymmetric Auto-Deleveraging (ADL) and HLP vault liquidations
*How does our risk model protect against asymmetric ADL or HLP vault liquidations on Hyperliquid while a Binance hedge stays wide open?*

**Our risk mechanisms:**

1. **The Asymmetric Deleveraging Danger:**
   On Hyperliquid, liquidations are absorbed first by the **HLP (Hyperliquid Liquidity Provider) vault backstop**. If market dislocations overwhelm the vault, Auto-Deleveraging (ADL) force-closes positions of profitable, high-leverage counterparties. If you hold a hedged spread (e.g., Short Binance vs. Long Hyperliquid) and Hyperliquid triggers ADL on your long leg, you are suddenly left with an unhedged naked short on Binance.
2. **How We Mitigate This:**
   * **Pre-Emptive Quantile Stops Before ADL:** In [`docs/03_full_day_intraday_prediction.md#L42-L44`](docs/03_full_day_intraday_prediction.md#L42-L44) and [`scripts/dynamic_garch_gap_engine.py#L123-L130`](scripts/dynamic_garch_gap_engine.py#L123-L130), Strategy Track B2 calculates empirical $95\%$ and $97.5\%$ downside quantiles ($q_{0.025}$). Algorithms use this boundary as an active stop-loss to unwind positions *before* exchange liquidation or ADL thresholds are reached.
   * **Basis Dislocation & Vault Inventory Trigger:** When perpetual basis dislocates to $-6.8\text{ bps}$ and perp volume surges ($>1.2\times$ spot), signaling acute vault strain and ADL tail risk, the model widens uncertainty bands to $\pm 180\text{ USD}$ and triggers automated exposure de-leveraging to zero, eliminating asymmetric single-leg exposure before ADL can trigger.

---

## Part 5: Resolution, Discretization & Micro-Dynamics

### 13. Measuring a 1-to-2-minute lead-lag when our pipeline logs 5-minute bars
*How did we measure a 1-to-2-minute lead-lag if our production synchronized pipeline operates on 5-minute bar boundaries?*

**How the two research layers interact:**

This measurement and the production pipeline operate at two distinct architectural levels:

1. **The Research Layer (1-Second Continuous Cross-Correlation):**
   The 1-to-2-minute lead-lag was derived using our high-frequency **1-second dataset** ([`data/btc_1sec_nov2026.parquet`](data/btc_1sec_nov2026.parquet), containing $2,073,600$ continuous observations across November 2026) in [`scripts/research/analyze_nov_microstructure.py`](scripts/research/analyze_nov_microstructure.py):
   * We evaluated the cross-correlation of 1-second log returns between Binance Spot/Futures and Hyperliquid mark prices across continuous second-lag offsets $\tau \in [-300\text{s}, +300\text{s}]$:
     $$CCF(\tau) = \text{Corr}\left(r_{t}^{\text{Binance}}, \; r_{t+\tau}^{\text{Hyperliquid}}\right)$$
   * During scheduled US macro releases (14:30 UTC cash open / CPI / FOMC), the peak cross-correlation occurred at **$\tau^* \in [+60\text{s}, +120\text{s}]$**. Institutional flow hits Binance's central order book first from US connectivity lines. Decentralized market makers on Hyperliquid take 60 to 120 seconds to adjust quotes across L1 block rounds, revealing a measurable **1-to-2-minute Binance-leads-Hyperliquid lag**.
2. **The Production Pipeline (5-Minute Basis Dislocation):**
   In our production forward logger ([`btc_pipeline/live_forward_logger.py`](btc_pipeline/live_forward_logger.py)), we do not stream sub-second order book updates to avoid race conditions. Instead, this sub-bar lead-lag is captured econometrically at the 5-minute boundary via the **cross-venue perpetual basis spread**:
   $$\text{Basis}_{\text{bps}, t} = \frac{P_{\text{HL}, t} - P_{\text{Spot}, t}}{P_{\text{Spot}, t}} \times 10,000$$
   When Hyperliquid leads during crypto liquidations (e.g., the 13:30 UTC cascade), the L1 book marks down before spot completes its executions. This registers at bar close $t$ as a negative basis dislocation ($-6.8\text{ bps}$) with a surge in volume ratio ($\text{Ratio}_{\text{vol}} = 2.08\times$), which in [`scripts/dynamic_garch_gap_engine.py`](scripts/dynamic_garch_gap_engine.py#L108-L122) serves as the ex-ante trigger for the jump intensity $\lambda_{t+1}$ in the subsequent 5-minute interval.

---

### 14. Handling dynamic tick-size changes across 55k USD and 90k+ USD BTC regimes
*Did our historical backtest account for dynamic tick-size changes on Hyperliquid as Bitcoin moved between 55k USD and 90k+ USD?*

**Our approach and mathematical bounds:**

We accounted for this through **scale-invariant return modeling** and a **conservative institutional friction hurdle**:

1. **Scale-Invariant Basis Normalization:**
   Hyperliquid perpetual contracts enforce price discretization governed by `metaAndAssetCtxs` ($1.00\text{ USD}$ per tick for BTC perp, compared to $0.01\text{ USD}$ on Binance Spot). As Bitcoin moved between 55,000 USD and 90,000 USD, the physical tick size changed in relative basis points:
   * At **$55,000\text{ USD}$**, a $1.00\text{ USD}$ tick represents:
     $$\frac{1.00}{55,000} \approx \mathbf{0.182\text{ bps}}$$
   * At **$90,000\text{ USD}$**, a $1.00\text{ USD}$ tick represents:
     $$\frac{1.00}{90,000} \approx \mathbf{0.111\text{ bps}}$$
   Because all feature engineering in [`btc_pipeline/feature_engineering.py`](btc_pipeline/feature_engineering.py) and [`src/data.py`](src/data.py) models volatility through continuous log returns:
   $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
   and normalizes perpetual basis into **basis points relative to spot**, our signals are dimensionless and scale-invariant across both the $55\text{k}$ and $90\text{k}\text{ USD}$ price regimes.
2. **Conservative Slippage Hurdle Subsumes Tick Discretization:**
   In [`btc_pipeline/trading_simulation.py#L39-L43`](btc_pipeline/trading_simulation.py#L39-L43), execution is evaluated under fixed institutional execution drag:
   $$\text{Friction} = 5.0\text{ bps taker fee} + 2.0\text{ bps execution slippage} = \mathbf{7.0\text{ bps per side}} \quad (14.0\text{ bps round-trip})$$
   * At $55,000\text{ USD}$, $2.0\text{ bps}$ represents **$11.00\text{ USD}$** of price slippage ($11\times$ the $1.00\text{ USD}$ tick).
   * At $90,000\text{ USD}$, $2.0\text{ bps}$ represents **$18.00\text{ USD}$** of price slippage ($18\times$ the $1.00\text{ USD}$ tick).
   Because our applied slippage penalty ($11.00\text{--}18.00\text{ USD}$) is an **order of magnitude larger than the physical tick size** ($1.00\text{ USD}$), any dynamic tick discretization adjustment is completely absorbed by the conservative friction buffer.
3. **What Was Explicitly Not Modeled:**
   Our backtest **did not simulate passive Level-3 queue priority** or maker rebate capture across changing tick increments. The trading simulation assumes takers crossing the full spread under a $7.0\text{ bps}$ execution penalty. For macro volatility forecasting and 5-minute execution risk corridors, physical tick size variations are second-order effects relative to the $7.0\text{ bps}$ transaction friction and $6.8\text{ bps}$ liquidation basis dislocations.

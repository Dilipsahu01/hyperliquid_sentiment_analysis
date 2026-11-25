# Quantitative Research & Microstructure Audit: Questions and Answers
### Technical Defense of the Multi-Venue Econometric and Execution Architecture

> [!IMPORTANT]
> **Audit Context:** This document compiles the technical defense and rigorous econometric explanations for the Bitcoin (BTC) multi-venue research pipeline, spanning centralized order books (**Binance Spot and USD-M Futures**) and decentralized perpetual contracts (**Hyperliquid L1/L2 Mainnet**). Every answer is strictly grounded in the verified code, empirical findings, and mathematical proofs implemented in this repository.

---

## Table of Contents
* [Part 1: Data & Synchronization (Q1–Q3)](#part-1-data--synchronization)
* [Part 2: Econometrics & Statistical Rigor (Q4–Q6)](#part-2-econometrics--statistical-rigor)
* [Part 3: Market Microstructure & Mechanics (Q7–Q9)](#part-3-market-microstructure--mechanics)
* [Part 4: Real-World Execution & Risk (Q10–Q12)](#part-4-real-world-execution--risk)
* [Part 5: Resolution, Discretization & Micro-Dynamics (Q13–Q14)](#part-5-resolution-discretization--micro-dynamics)

---

## Part 1: Data & Synchronization

### Question 1
**"How did you synchronize timestamps between Binance’s continuous matching engine and Hyperliquid’s discrete L1 block times without creating artificial lead-lag bias?"**

#### Technical Answer
Sub-millisecond tick interleaving across two asynchronous consensus architectures without dedicated atomic clock hardware (PTP/GPS) produces phantom lead-lag artifacts driven by network route asymmetry rather than actual economic price discovery.

To eliminate this bias, synchronization in [`btc_pipeline/live_forward_logger.py`](btc_pipeline/live_forward_logger.py#L30-L131) and [`src/data.py`](src/data.py) is enforced at discrete, right-closed **5-minute bar boundaries** (`startTime` and `close_time_ms` in UTC milliseconds):

1. **Bucket Alignment:**
   * **Binance Spot:** Continuous matching-engine events are aggregated into exchange-certified klines covering $[t, t + 300,000\text{ ms}]$.
   * **Hyperliquid Mainnet:** Aggregated from discrete HyperBFT L1 consensus block-height commits bucketed across the identical interval $[t, t + 300,000\text{ ms}]$ via the `candleSnapshot` API payload.
2. **Causal Polling Rule:**
   As implemented in [`btc_pipeline/live_forward_logger.py#L149-L213`](btc_pipeline/live_forward_logger.py#L149-L213), multi-venue snapshots are merged strictly on matching UTC timestamps. Bars are requested and recorded only *after* the 5-minute interval has fully elapsed ($t_{\text{current}} \ge t_{\text{bar\_close}}$).
3. **Block-Time Invariance:**
   At a 5-minute sampling resolution ($300,000\text{ ms}$), Hyperliquid’s discrete L1 block proposal time ($\approx 200\text{--}400\text{ ms}$) represents $< 0.13\%$ of the bar duration. This eliminates block-quantization artifacts and guarantees zero artificial lead-lag leakage across bar boundaries.
4. **State-Space Temporal Separation:**
   In intraday Kalman tracking ([`scripts/deviation_reduction_experiment.py#L80-L100`](scripts/deviation_reduction_experiment.py#L80-L100)), state predictions $\hat{x}_{t|t-1}$ for bar $t$ use exclusively observations and realized variance finalized at or before $t-1$.

---

### Question 2
**"Did you use exchange-reported API timestamps, or did you record packet arrival times from a single co-located observer node?"**

#### Technical Answer
We strictly utilized **exchange-certified matching-engine and block-commit API timestamps** (`k[0]` open time and `k[6]` close time from Binance REST; `c["t"]` from Hyperliquid Mainnet `candleSnapshot`).

#### Econometric Justification
1. **Elimination of Observer-Location Asymmetry:**
   Recording packet arrival times from a single external observer node (e.g., an AWS Tokyo or Frankfurt EC2 instance) introduces variable network path asymmetry, TCP retransmissions, and cloud virtualization latency. These network hops distort the physical execution order inside the exchange sequencers.
2. **Authoritative Sequencer Time:**
   Binance’s internal timestamps reflect the official physical matching sequence within its central limit order book engine sequencer (Tokyo/AWS). Hyperliquid’s timestamps represent consensus validator block commit rounds (Tendermint/HyperBFT consensus).
3. **High-Frequency Variance Proof:**
   For high-frequency 1-second variance research ([`data/btc_1sec_nov2026.parquet`](data/btc_1sec_nov2026.parquet), covering $2,073,600$ continuous seconds across 24 full days in November 2026), we analyzed trade events using Binance's certified trade execution timestamps (`T`). This enabled exact mathematical proof of realized variance non-dilution without observer NIC packet noise.

---

### Question 3
**"How did you clean the data for Binance’s zero-fee pairs, self-trades, and wash volume so your liquidity metrics aren't inflated?"**

#### Technical Answer
Data cleanliness and liquidity integrity were safeguarded at three structural levels:

1. **Pair Selection:**
   We explicitly excluded zero-fee promotional pairs (e.g., historical `BTC/FDUSD` or `BTC/TUSD`), which are prone to wash trading and zero-margin algorithmic churn. Our pipeline relies strictly on **`BTCUSDT`** on Binance Spot and Binance USD-M Futures (`BTCUSDT` perpetual), where standard taker and maker fee schedules are active.
2. **WebSocket and Tick Stream Cleaning:**
   In [`Master.md#L13`](Master.md#L13) and [`report.md#L37`](report.md#L37), our audit detected and resolved a $5,885,612$ non-trade frame discrepancy between raw WebSocket events ($104,896,898$) and executed trade records ($99,011,286$):
   * Raw feeds contain order book depth delta updates (`depthUpdate`), cancellations, and heartbeat ping/pongs.
   * We filtered strictly on executed trade events (`e: "trade"`) with valid maker/taker flags (`m: true/false`), eliminating non-trade quote chatter.
3. **Taker-Based Order Flow Imbalance (OFI):**
   In [`btc_pipeline/live_forward_logger.py#L40-L56`](btc_pipeline/live_forward_logger.py#L40-L56), liquidity metrics do not rely on raw volume. We extract **Net Taker OFI**:
   $$\text{OFI}_t = V_{\text{taker\_buy}, t} - V_{\text{taker\_sell}, t} = 2 \cdot V_{\text{taker\_buy}, t} - V_{\text{total}, t}$$
   Passive wash trades matching against self-controlled accounts cancel out symmetrically in net OFI.
4. **Execution Drag Hurdle:**
   In [`btc_pipeline/trading_simulation.py#L39-L43`](btc_pipeline/trading_simulation.py#L39-L43), all backtests enforce a mandatory **$5.0\text{ bps taker fee} + 2.0\text{ bps slippage}$** friction per side ($7.0\text{ bps}$ per side, $14.0\text{ bps}$ round-trip), strictly penalizing any phantom liquidity.

---

## Part 2: Econometrics & Statistical Rigor

### Question 4
**"How did you adjust for the Epps Effect and asynchronous tick arrival when calculating cross-venue correlation and Information Shares (e.g., did you use Hayashi-Yoshida estimators or refresh-time sampling)?"**

#### Technical Answer
The Epps Effect dictates that empirical correlation between two continuous assets artificially decays toward zero as sampling frequency approaches sub-second intervals due to asynchronous tick arrivals.

#### Our Econometric Approach
1. **Sampling Scale Invariance:**
   As demonstrated in [`docs/01_methodology_and_techniques.md#L100-L107`](docs/01_methodology_and_techniques.md#L100-L107) and [`outputs/figures/signature_plot_rv.png`](outputs/figures/signature_plot_rv.png), we evaluated empirical volatility signature plots across horizons: $1\text{s}, 5\text{s}, 1\text{m}, 5\text{m}, 15\text{m}, 1\text{h}$.
   * At $1\text{s}$ to $5\text{s}$, discrete arrival delays reduce instantaneous correlation.
   * At the **5-minute bar frequency** ($300\text{ seconds}$), asynchronous arrivals are completely absorbed within the bar interval, causing cross-venue price correlation between Binance and Hyperliquid to converge to **$> 0.999$**.
2. **Lo-MacKinlay Variance Ratio Decomposition:**
   Rather than imposing Hayashi-Yoshida tick matching on discrete L1 blocks, we conducted Lo-MacKinlay Variance Ratio tests with 2,000 bootstrap iterations across 2.07M seconds. We mathematically proved that realized variance expansion from 1-second ($VR = 1.000$) to 1-hour ($VR = 1.516$) is driven by **positive serial autocorrelation ($\rho_1 = +0.0997, \rho_2 = +0.0403$)** from institutional order-splitting algorithms, completely rejecting the zero-trade dilution hypothesis.
3. **Information Transmission via Basis Dislocation:**
   Tick-level Hasbrouck Information Shares assume continuous Brownian martingales, which fail under block-time consensus. We model cross-venue lead-lag directly through the **perpetual basis spread**:
   $$\text{Basis}_{\text{bps}, t} = \frac{P_{\text{HL}, t} - P_{\text{Spot}, t}}{P_{\text{Spot}, t}} \times 10,000$$
   and perp-to-spot volume migration ratio $\text{Ratio}_{\text{vol}}$, capturing directional transmission 10–15 minutes ahead of spot price adjustments.

---

### Question 5
**"Did you test for regime shifts—separating normal market hours from high-volatility liquidation cascades—rather than running a single static regression across the whole dataset?"**

#### Technical Answer
We explicitly tested and rejected static regressions across the full sample on both Strategy Track A and Strategy Track B:

1. **Track A (Macro / Multi-Day Regime Testing):**
   * In [`results/audit.md#Check 5`](results/audit.md#L125-L148) and [`btc_pipeline/audit_v4_pipeline.py`](btc_pipeline/audit_v4_pipeline.py), the 889-day walk-forward cross-validation (127 rolling folds) was segmented into **3 distinct chronological market regimes**:
     * *Block 1 (2024-06 to 2025-03, $N=291$):* Low-volatility regime ($\text{QLIKE} = 0.2825$).
     * *Block 2 (2025-03 to 2026-01, $N=291$):* Mid-cycle consolidation ($\text{QLIKE} = 0.2497$).
     * *Block 3 (2026-01 to 2026-10, $N=293$):* High-volatility macro distress ($\text{QLIKE} = 0.2967$).
   * Performance stability was verified via **Diebold-Mariano tests with Newey-West HAC errors** ($DM = +7.492$ on QLIKE, $DM = +12.054$ on MSE, $p < 0.0001$).
2. **Track B (Intraday Regime-Switching Jump Model):**
   * In [`scripts/dynamic_garch_gap_engine.py#L108-L122`](scripts/dynamic_garch_gap_engine.py#L108-L122) and [`docs/03_full_day_intraday_prediction.md#L127-L135`](docs/03_full_day_intraday_prediction.md#L127-L135), we implemented a **regime-switching jump-diffusion engine**:
     $$\lambda_t = \min\left(0.45, \; \lambda_{\text{base}} + \gamma_b \max(0, -\text{Basis}_{\text{bps}} - 2.0) + \gamma_v \max(0, \text{Ratio}_{\text{vol}} - 1.2)\right)$$
     * *Calm Regime ($\lambda_t \le 0.08$):* $\mu_J = -0.05\%$, $\sigma_J = 0.25\%$.
     * *Liquidation Stress Regime ($\lambda_t > 0.08$):* Triggered when basis dislocates below $-2.0\text{ bps}$ and perp volume surges, shifting jump parameters to $\mu_J = -0.55\%$, $\sigma_J = 0.65\%$.
   * During the 13:30 UTC liquidation cascade (drop from 58,922 USD to 57,989 USD), this regime switch delivered **$100.0\%$ coverage** on the 95% corridor and reduced pinball loss to 40.33 USD.

---

### Question 6
**"How did you prevent look-ahead bias when aligning Binance’s 8-hour funding cycle against Hyperliquid’s 1-hour funding settlement?"**

#### Technical Answer
1. **The 21:00 UTC Daily Cutoff Rule:**
   Detailed in [`docs/01_methodology_and_techniques.md#L79-L89`](docs/01_methodology_and_techniques.md#L79-L89) and verified in [`tests/test_pipeline.py#test_no_lookahead_bias_in_features`](tests/test_pipeline.py#L38-L50):
   * Features for day $t$ strictly incorporate data up to **21:00 UTC** ($t \le 21:00$).
   * Target realized variance $RV_{t+1}$ covers the subsequent 24-hour period from $21:00\text{ UTC}_t$ to $21:00\text{ UTC}_{t+1}$.
2. **Settlement Alignment Without Peeking:**
   * **Binance USD-M Futures:** Settle funding every 8 hours (00:00, 08:00, 16:00 UTC). At 21:00 UTC, the latest *settled* rate is the **16:00 UTC** cycle.
   * **Hyperliquid Mainnet:** Settles funding hourly. At 21:00 UTC, the latest *settled* rate is the **20:00–21:00 UTC** rate (`funding` in `metaAndAssetCtxs`).
   * In [`btc_pipeline/live_forward_logger.py#L111`](btc_pipeline/live_forward_logger.py#L111), the engine records exclusively the prevailing settled rate known at bar close $t$, never peeking forward into future cycles (e.g., Binance's 00:00 UTC or Hyperliquid's 22:00 UTC rate).
3. **Leakage Audit Verification:**
   In [`tests/test_pipeline.py#test_real_point_in_time_leakage`](tests/test_pipeline.py#L154-L182), raw hourly bars across 20 random historical dates were re-aggregated from scratch, proving zero look-ahead discrepancy ($|RV_{\text{recomputed}} - RV_{\text{master}}| < 10^{-12}$).

---

## Part 3: Market Microstructure & Mechanics

### Question 7
**"How does your model account for Hyperliquid’s block-level cancel priority (where market makers can cancel stale quotes before takers hit them) compared to Binance’s strict FIFO queue?"**

#### Technical Answer
1. **Queue Priority Discrepancy:**
   * **Binance:** Uses a centralized in-memory matching engine with strict Price-Time Priority (FIFO). Taker market orders execute against resting liquidity with zero cancel priority.
   * **Hyperliquid (HyperBFT):** Consensus processes **cancel transactions at the beginning of each block before evaluating new market orders**. Market makers can cancel stale quotes at block boundaries before takers can hit them.
2. **Microstructure Consequence:**
   * Because market makers on Hyperliquid can pull bids before takers execute, **depth thins on Hyperliquid faster than on Binance** during sudden sell-offs.
   * This causes Hyperliquid perpetual prices to drop immediately below Binance spot, creating an acute **negative basis dislocation** ($P_{\text{HL}} < P_{\text{Spot}}$).
3. **Model Integration:**
   Our model does not attempt to race the L1 validator queue. In [`scripts/dynamic_garch_gap_engine.py#L108-L115`](scripts/dynamic_garch_gap_engine.py#L108-L115), it treats the resulting negative basis spread ($\text{Basis}_{\text{bps}} < -2.0\text{ bps}$) as an **econometric state trigger**:
   $$\lambda_t = 0.02 + \gamma_b \max(0, -\text{Basis}_{\text{bps}} - 2.0) + \gamma_v \max(0, \text{Ratio}_{\text{vol}} - 1.2)$$
   When market makers cancel bids and basis dislocates, jump intensity expands dynamically, warning our risk engine that a liquidation cascade is beginning before Binance Spot completes its executions.

---

### Question 8
**"How did you normalize order book depth and slippage between the two venues given their different tick sizes, lot sizes, and fee tiers?"**

#### Technical Answer
1. **Discretization Differences:**
   * Binance Spot: Tick size $0.01\text{ USD}$, lot step $0.00001\text{ BTC}$, taker fee $10.0\text{ bps}$ (retail) to $2.0\text{--}4.0\text{ bps}$ (VIP).
   * Hyperliquid Perp: Tick size $1.00\text{ USD}$, lot step $0.0001\text{ BTC}$, taker fee $2.5\text{ bps}$ (maker rebate $-0.02\text{ bps}$).
2. **Applied Normalizations:**
   * **Dimensionless Basis Points:** Cross-venue price divergence is normalized into basis points relative to spot in [`btc_pipeline/live_forward_logger.py#L188`](btc_pipeline/live_forward_logger.py#L188):
     $$\text{Basis}_{\text{bps}} = \frac{P_{\text{HL}} - P_{\text{Spot}}}{P_{\text{Spot}}} \times 10,000$$
     making the indicator scale-invariant across BTC price regimes ($55,000\text{ to }90,000\text{ USD}$).
   * **Dimensionless Volume Migration Ratio:** Volume is expressed as $\text{Ratio}_{\text{vol}} = V_{\text{HL}} / V_{\text{Spot}}$, eliminating contract step-size denomination bias.
   * **Conservative Friction Hurdle:** In [`btc_pipeline/trading_simulation.py#L39-L43`](btc_pipeline/trading_simulation.py#L39-L43), backtesting applies an institutional hurdle:
     $$\text{Friction} = 5.0\text{ bps taker fee} + 2.0\text{ bps execution slippage} = 7.0\text{ bps per side} \quad (14.0\text{ bps round-trip})$$
     This $7.0\text{ bps}$ penalty exceeds fees on both exchanges, ensuring reported returns and Deflated Sharpe Ratios ($P(\text{DSR} > 0)$) are free from micro-spread distortion.

---

### Question 9
**"Did you separate informed order flow from retail/noise flow, and how did you identify which venue actually leads price discovery during macro news versus crypto-native events?"**

#### Technical Answer
1. **Informed vs. Noise Flow Isolation:**
   * Extracted via **Net Taker Order Flow Imbalance (OFI)**:
     $$\text{OFI}_t = V_{\text{taker\_buy}} - V_{\text{taker\_sell}} = 2 \cdot V_{\text{taker\_buy}} - V_{\text{total}}$$
     Aggressive taker orders represent informed flow paying the spread for immediate execution, whereas passive limit orders absorb noise.
   * In [`scripts/generate_data_factor_analysis.py`](scripts/generate_data_factor_analysis.py) and [`docs/04_data_factors_analysis.md`](docs/04_data_factors_analysis.md), OFI is correlated against forward returns and volatility.
2. **Venue Lead-Lag Identification:**
   * **Macro News Events (US CPI, FOMC, Big Tech Earnings):**
     * Concentrated around the US cash equity open (13:00–16:30 UTC), with beta coupling to S&P-100 ($\beta_{\text{OEX}} = 1.21$) and Nasdaq-100 ($\beta_{\text{QQQ}} = 0.98$).
     * **Binance Spot and USD-M Futures lead price discovery.** Macro flow enters centralized venues first; Hyperliquid perpetuals lag by 1 to 2 minutes as decentralized market makers update quotes.
   * **Crypto-Native Events (Cascading Liquidations, Off-Hours Squeezes):**
     * Concentrated outside US trading hours (e.g., Nov 21 06:17 UTC low at 55,130 USD and Nov 25 13:30 UTC cascade).
     * **Hyperliquid Perpetuals lead price discovery.** High-leverage unwinds hit Hyperliquid first, driving basis deeply negative ($-6.8\text{ bps}$) and elevating perp volume to $2.08\times$ spot volume 10 to 15 minutes before the cascade hits Binance Spot.

---

## Part 4: Real-World Execution & Risk

### Question 10
**"In your cross-venue strategy or arbitrage tests, how did you model the capital cost of holding idle margin buffers on both exchanges simultaneously?"**

#### Technical Answer
1. **The Dual-Collateral Drag:**
   Cross-venue execution requires holding collateral concurrently on both venues: native USDC deposited into Hyperliquid's clearinghouse contract, and USDT deposited into Binance Futures margin accounts.
2. **Capital Modeling in Backtest:**
   * In [`btc_pipeline/trading_simulation.py#L6-L84`](btc_pipeline/trading_simulation.py#L6-L84), capital sizing is constrained by a **1x maximum gross leverage cap** with dynamic volatility targeting ($w_t = \min(1.0, \sigma_{\text{target}} / \hat{\sigma}_t)$).
   * Across the holdout period, average exposure was **$70.13\%$**, requiring $\sim 30\%$ idle collateral buffering.
3. **Empirical Benchmark Against Equal-Exposure Buy & Hold:**
   In [`results/audit.md#Check 9`](results/audit.md#L198-L208) and [`report.md#L163-L175`](report.md#L163-L175), we benchmarked the strategy against an equal-exposure Buy & Hold ($70.13\%$ constant exposure):
   * *Buy & Hold (70.1% exposure):* $-15.42\%$ return, $19.71\%$ max drawdown.
   * *Volatility-Targeted Strategy:* $-16.47\%$ return, $20.75\%$ max drawdown.
4. **Empirical Finding:**
   When factoring in capital drag, margin buffers, and $7.0\text{ bps}$ friction, drawdown reduction was driven by **lowering average exposure to $70\%$**, rather than tactical timing alpha. We documented this finding directly in the audit rather than claiming inflated returns.

---

### Question 11
**"Did you stress-test your model for Arbitrum L2 bridge congestion and gas spikes when rebalancing collateral during a market crash?"**

#### Technical Answer
1. **Architecture Clarification:**
   * Hyperliquid is **not an Arbitrum L2 execution rollup**; it is an independent, sovereign Layer-1 application-specific blockchain powered by custom HyperBFT consensus.
   * Arbitrum is used solely for the initial deposit bridge contract (`Arbitrum native USDC bridge`). Once collateral is deposited on Hyperliquid L1, order placement, execution, margin management, and liquidations execute on Hyperliquid L1 with **zero gas fees**.
2. **Bridge Latency Realities:**
   * During acute market crashes, Arbitrum bridge rebalancing incurs a 10 to 20 minute confirmation delay and elevated L1 gas costs.
3. **Our Risk Rule:**
   As noted in [`docs/03_full_day_intraday_prediction.md#L155`](docs/03_full_day_intraday_prediction.md#L155), **intraday cross-venue bridge rebalancing during a 15-minute liquidation cascade is treated as impossible**.
   Collateral must be pre-funded on both sides. The risk engine manages margin risk by dynamically de-risking positions ($w_t \to 0$) and utilizing local stop-loss bands, rather than assuming just-in-time bridge transfers.

---

### Question 12
**"How does your risk model handle asymmetric Auto-Deleveraging (ADL) or HLP vault backstop liquidations on Hyperliquid while your Binance hedge remains open?"**

#### Technical Answer
1. **The Asymmetric Deleveraging Risk:**
   On Hyperliquid, liquidations are absorbed first by the **HLP (Hyperliquid Liquidity Provider) vault backstop**. If market dislocations exceed vault capacity, Auto-Deleveraging (ADL) force-closes positions of profitable, high-leverage counterparties. If an arbitrageur holds a hedged spread (e.g., Short Binance vs. Long Hyperliquid) and Hyperliquid triggers ADL on the long leg, the trader is left with an unhedged naked short on Binance.
2. **Our Risk Management Safeguards:**
   * **Pre-Emptive Quantile Stops Before ADL:** In [`docs/03_full_day_intraday_prediction.md#L42-L44`](docs/03_full_day_intraday_prediction.md#L42-L44) and [`scripts/dynamic_garch_gap_engine.py#L123-L130`](scripts/dynamic_garch_gap_engine.py#L123-L130), Strategy Track B2 calculates calibrated empirical $95\%$ and $97.5\%$ downside quantiles ($q_{0.025}$). Execution algorithms use this boundary as an active stop-loss to unwind positions *prior* to exchange liquidation or ADL thresholds.
   * **Basis Dislocation & HLP Inventory Trigger:** When perpetual basis dislocates to $-6.8\text{ bps}$ and perp volume surges ($>1.2\times$ spot), signaling vault strain and ADL tail risk, the model widens uncertainty bands to $\pm 180\text{ USD}$ and triggers automated exposure de-leveraging to zero, eliminating asymmetric single-leg exposure before ADL can trigger.

---

## Part 5: Resolution, Discretization & Micro-Dynamics

### Question 13
**"How did you measure a 1-to-2-minute lead-lag if your primary synchronized pipeline operates on 5-minute bar boundaries?"**

#### Technical Answer
The lead-lag measurement and the production forecasting pipeline operate at two distinct architectural resolutions within this repository:

1. **The Measurement Layer (1-Second Continuous Cross-Correlation):**
   The 1-to-2-minute lead-lag was derived using the high-frequency **1-second dataset** ([`data/btc_1sec_nov2026.parquet`](data/btc_1sec_nov2026.parquet), containing $2,073,600$ continuous observations across November 2026) in [`scripts/research/analyze_nov_microstructure.py`](scripts/research/analyze_nov_microstructure.py):
   * **Cross-Correlation Function ($CCF$):** We evaluated cross-correlations of 1-second log returns between Binance Spot/Futures and Hyperliquid Perpetual mark prices across continuous second-lag offsets $\tau \in [-300\text{s}, +300\text{s}]$:
     $$CCF(\tau) = \text{Corr}\left(r_{t}^{\text{Binance}}, \; r_{t+\tau}^{\text{Hyperliquid}}\right)$$
   * **Macro News Releases (14:30 UTC US Cash Open / CPI / FOMC):** Peak cross-correlation occurred at **$\tau^* \in [+60\text{s}, +120\text{s}]$**. Institutional algorithmic flow hits Binance's central limit order book first from US connectivity lines. Decentralized market makers on Hyperliquid take 60 to 120 seconds to adjust quotes across L1 block proposal rounds, revealing a measurable **1-to-2-minute Binance-leads-Hyperliquid lag** during macro news events.
2. **The Production Pipeline (Transmission into 5-Minute Basis Dislocation):**
   In the production forecasting engine ([`btc_pipeline/live_forward_logger.py`](btc_pipeline/live_forward_logger.py)), we do not stream sub-second order book updates to avoid race conditions. Instead, this sub-bar lead-lag is captured econometrically at the 5-minute boundary via the **cross-venue perpetual basis spread**:
   $$\text{Basis}_{\text{bps}, t} = \frac{P_{\text{HL}, t} - P_{\text{Spot}, t}}{P_{\text{Spot}, t}} \times 10,000$$
   When Hyperliquid leads during crypto-native liquidations (e.g., the 13:30 UTC cascade), the L1 book thins out and marks down before the spot matching engine finishes executing taker orders. This appears at bar close $t$ as an acute negative basis dislocation ($-6.8\text{ bps}$) accompanied by an elevated volume migration ratio ($\text{Ratio}_{\text{vol}} = 2.08\times$). In [`scripts/dynamic_garch_gap_engine.py`](scripts/dynamic_garch_gap_engine.py#L108-L122), this state transition at bar $t$ serves as the ex-ante trigger for the jump intensity $\lambda_{t+1}$ in the subsequent 5-minute interval.

---

### Question 14
**"Did your historical backtest account for dynamic tick-size changes on Hyperliquid as BTC moved between 55k USD and 90k+ USD?"**

#### Technical Answer
Our historical backtesting framework accounted for this through **scale-invariant return modeling** and a **conservative institutional friction hurdle**, rather than an order-queue tick simulator:

1. **Scale-Invariant Relative Basis Normalization:**
   Hyperliquid perpetual contracts enforce price discretization governed by `metaAndAssetCtxs` ($1.00\text{ USD}$ per tick for BTC perp, compared to $0.01\text{ USD}$ on Binance Spot). As Bitcoin moved between 55,000 USD and 90,000 USD, the physical tick size changed in relative basis points:
   * At **$55,000\text{ USD}$**, a $1.00\text{ USD}$ tick represents:
     $$\frac{1.00}{55,000} \approx \mathbf{0.182\text{ bps}}$$
   * At **$90,000\text{ USD}$**, a $1.00\text{ USD}$ tick represents:
     $$\frac{1.00}{90,000} \approx \mathbf{0.111\text{ bps}}$$
   Because all feature engineering in [`btc_pipeline/feature_engineering.py`](btc_pipeline/feature_engineering.py) and [`src/data.py`](src/data.py) models volatility through continuous log returns:
   $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
   and normalizes perpetual basis into **basis points relative to spot**, our signals are mathematically dimensionless and scale-invariant across both the $55\text{k}$ and $90\text{k}\text{ USD}$ price regimes.
2. **Conservative Slippage Hurdle Subsumes Tick Discretization:**
   In [`btc_pipeline/trading_simulation.py#L39-L43`](btc_pipeline/trading_simulation.py#L39-L43), execution is evaluated under fixed institutional execution drag:
   $$\text{Friction} = 5.0\text{ bps taker fee} + 2.0\text{ bps execution slippage} = \mathbf{7.0\text{ bps per side}} \quad (14.0\text{ bps round-trip})$$
   * At $55,000\text{ USD}$, $2.0\text{ bps}$ represents **$11.00\text{ USD}$** of price slippage ($11\times$ the $1.00\text{ USD}$ tick).
   * At $90,000\text{ USD}$, $2.0\text{ bps}$ represents **$18.00\text{ USD}$** of price slippage ($18\times$ the $1.00\text{ USD}$ tick).
   Because our applied execution slippage penalty ($11.00\text{--}18.00\text{ USD}$) is an **order of magnitude larger than the physical tick size** ($1.00\text{ USD}$), any dynamic tick discretization adjustment is completely absorbed by the conservative friction buffer.
3. **What Was Explicitly Not Modeled:**
   Our backtest **did not simulate passive Level-3 queue priority** or maker rebate capture across changing tick increments. The trading simulation assumes takers crossing the full spread under a $7.0\text{ bps}$ execution penalty. For macro volatility forecasting and 5-minute execution risk corridors, physical tick size variations are second-order effects relative to the $7.0\text{ bps}$ transaction friction and $6.8\text{ bps}$ liquidation basis dislocations.

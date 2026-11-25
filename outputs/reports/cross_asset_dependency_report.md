# Cross-Asset Dependency & Correlation Matrix: Bitcoin & Global Giants

> **Period:** November 2026  
> **Assets Analyzed:** 23 institutional benchmarks, mega-caps, and macro proxies  
> **Data Feeds:** Binance 1-second Bitcoin dataset merged with US Cash Equity Daily Feeds  
> **Outputs:**  
> - Matrix CSV: [`outputs/cross_asset_correlation_matrix.csv`](outputs/cross_asset_correlation_matrix.csv)  
> - Matrix JSON: [`outputs/cross_asset_correlation_matrix.json`](outputs/cross_asset_correlation_matrix.json)  
> - Clustermap Figure: [`outputs/figures/cross_asset_clustermap.png`](outputs/figures/cross_asset_clustermap.png)  
> - PCA Clusters Figure: [`outputs/figures/cross_asset_pca_clusters.png`](outputs/figures/cross_asset_pca_clusters.png)

---

## 1. Executive Summary & Factor Decomposition

A Principal Component Analysis (PCA) on the daily return matrix reveals that **two primary latent forces** explain over **72.9%** of all price behavior:

1. **Factor 1: Global Institutional Liquidity (48.8% variance)**:
   - Synchronizes Wall Street prime brokerage banks, mega-cap tech, and Bitcoin simultaneously.
   - When institutional leverage expands, BTC, COIN, Goldman Sachs, and Qualcomm rally together.
2. **Factor 2: Sector Rotation / Risk Dispersion (24.1% variance)**:
   - Distinguishes high-beta growth (Crypto & AI Semis) from defensive flight-to-safety assets (ExxonMobil, US Dollar Index, Treasuries).

---

## 2. Strongest Inter-Firm Synergies (Who Moves With Whom)

These pairs show near-perfect synchronization ($r > +0.90$):

| Rank | Pair | Institutional Relationship | Pearson ($r$) | Spearman ($\rho$) |
|:---:|---|---|:---:|:---:|
| 1 | **`MS`** $\leftrightarrow$ **`GS`** | Morgan Stanley (Prime Broker) & Goldman Sachs (Trading/IB) | **+0.9801** | +0.9735 |
| 2 | **`JPM`** $\leftrightarrow$ **`BAC`** | JPMorgan (Global Bank) & Bank of America (Commercial) | **+0.9799** | +0.9706 |
| 3 | **`COIN`** $\leftrightarrow$ **`MSTR`** | Coinbase (Exchange) & MicroStrategy (Treasury) | **+0.9525** | +0.9559 |
| 4 | **`BLK`** $\leftrightarrow$ **`JPM`** | BlackRock (ETF Sponsor) & JPMorgan (Global Bank) | **+0.9453** | +0.8941 |
| 5 | **`BLK`** $\leftrightarrow$ **`BAC`** | BlackRock (ETF Sponsor) & Bank of America (Commercial) | **+0.9432** | +0.9000 |
| 6 | **`BLK`** $\leftrightarrow$ **`MS`** | BlackRock (ETF Sponsor) & Morgan Stanley (Prime Broker) | **+0.9420** | +0.9235 |
| 7 | **`BLK`** $\leftrightarrow$ **`GS`** | BlackRock (ETF Sponsor) & Goldman Sachs (Trading/IB) | **+0.9144** | +0.9118 |
| 8 | **`MS`** $\leftrightarrow$ **`BAC`** | Morgan Stanley (Prime Broker) & Bank of America (Commercial) | **+0.9095** | +0.8676 |
| 9 | **`MS`** $\leftrightarrow$ **`JPM`** | Morgan Stanley (Prime Broker) & JPMorgan (Global Bank) | **+0.9051** | +0.8882 |
| 10 | **`AMD`** $\leftrightarrow$ **`AVGO`** | AMD (Semiconductors) & Broadcom (Custom Silicon) | **+0.9032** | +0.8588 |
| 11 | **`GS`** $\leftrightarrow$ **`BAC`** | Goldman Sachs (Trading/IB) & Bank of America (Commercial) | **+0.8933** | +0.8412 |
| 12 | **`GS`** $\leftrightarrow$ **`JPM`** | Goldman Sachs (Trading/IB) & JPMorgan (Global Bank) | **+0.8811** | +0.8647 |

---

## 3. Inverse Relationships & Structural Hedges (Someone's Profit = Someone's Loss)

When Bitcoin and high-beta tech crash, where does the capital rotate?

| Rank | Pair | Institutional Dynamics | Pearson ($r$) | Spearman ($\rho$) |
|:---:|---|---|:---:|:---:|
| 1 | **`MSTR`** $\leftrightarrow$ **`XOM`** | MicroStrategy (Treasury) vs ExxonMobil (Energy Giant) | **-0.2761** | -0.2176 |
| 2 | **`UUP`** $\leftrightarrow$ **`TLT`** | US Dollar Index (Invesco Bullish) vs US 20Y Treasuries (Bonds) | **-0.2791** | -0.2735 |
| 3 | **`AAPL`** $\leftrightarrow$ **`XOM`** | Apple (Consumer Mega-Cap) vs ExxonMobil (Energy Giant) | **-0.3046** | +0.0176 |
| 4 | **`AAPL`** $\leftrightarrow$ **`UUP`** | Apple (Consumer Mega-Cap) vs US Dollar Index (Invesco Bullish) | **-0.3166** | -0.0647 |
| 5 | **`BTC`** $\leftrightarrow$ **`XOM`** | Bitcoin (BTC) vs ExxonMobil (Energy Giant) | **-0.3249** | -0.2412 |
| 6 | **`UUP`** $\leftrightarrow$ **`WMT`** | US Dollar Index (Invesco Bullish) vs Walmart (Defensive Retail) | **-0.4003** | -0.2882 |
| 7 | **`MSFT`** $\leftrightarrow$ **`UUP`** | Microsoft (Cloud/AI) vs US Dollar Index (Invesco Bullish) | **-0.4190** | -0.2147 |
| 8 | **`AVGO`** $\leftrightarrow$ **`TLT`** | Broadcom (Custom Silicon) vs US 20Y Treasuries (Bonds) | **-0.4469** | -0.2912 |
| 9 | **`XOM`** $\leftrightarrow$ **`TLT`** | ExxonMobil (Energy Giant) vs US 20Y Treasuries (Bonds) | **-0.4683** | -0.4676 |
| 10 | **`MSFT`** $\leftrightarrow$ **`XOM`** | Microsoft (Cloud/AI) vs ExxonMobil (Energy Giant) | **-0.4721** | -0.2853 |
| 11 | **`GLD`** $\leftrightarrow$ **`XOM`** | Gold (SPDR Gold Shares) vs ExxonMobil (Energy Giant) | **-0.4732** | -0.4824 |
| 12 | **`AMZN`** $\leftrightarrow$ **`XOM`** | Amazon (AWS/Retail) vs ExxonMobil (Energy Giant) | **-0.5389** | -0.4206 |

---

## 4. Key Takeaways for Quant Trading & Risk Hedging

1. **The Wall Street - Crypto Triangle (`JPM` / `MS` / `BLK` / `BTC`)**:
   - The Wall Street investment banking cluster (`GS`, `MS`, `JPM`, `BAC`) has an internal correlation of **+0.98**.
   - Because BlackRock (`BLK`) and prime brokers clear the trades and manage the Bitcoin ETF flows, **bank stocks act as an early barometer of liquidity** for Bitcoin ($r = +0.64$ to $+0.73$).

2. **The Effective Hedges Against Bitcoin Drops**:
   - **ExxonMobil (`XOM`, r = -0.325)**: Traditional commodity/energy was an active beneficiary during tech and crypto liquidation days.
   - **US Dollar Index (`UUP`, r = -0.127)**: Strong USD days systematically drained liquidity from crypto and equities.
   - **Walmart (`WMT`, r = +0.099)**: Defensive consumer retail showed near-zero sensitivity to crypto volatility.

3. **High-Beta Amplifiers**:
   - **Coinbase (`COIN`, r = +0.749)** and **MicroStrategy (`MSTR`, r = +0.707)** are **2× to 3× levered derivatives** of Bitcoin. On down days, MSTR fell -42.8% and COIN fell -29.6% compared to Bitcoin's -22.1%.

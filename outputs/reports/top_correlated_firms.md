# Top Global Firms with Strongest Correlation to Bitcoin (BTC)

> **Analysis Window:** November 2026  
> **Target Asset:** Bitcoin (BTC/USD)  
> **Source:** Daily institutional price action cross-referenced with Binance 1-second OHLCV  
> **Saved Outputs:**  
> - CSV: [`outputs/top_correlated_firms_with_btc.csv`](file://outputs/top_correlated_firms_with_btc.csv)  
> - JSON: [`outputs/top_correlated_firms_with_btc.json`](file://outputs/top_correlated_firms_with_btc.json)

---

##  Top 16 Highly Correlated Mega-Cap Firms ($r \ge 0.50$)

| Rank | Ticker | Company Name | Sector / Economic Role | Pearson ($r$) | Spearman ($\rho$) | $p$-value | Significant? |
|:---:|:---:|---|---|:---:|:---:|:---:|:---:|
| 1 | **`COIN`** | Coinbase Global Inc. | Crypto Exchange & Custody | **+0.7489** | +0.7647 | 0.00084 |  Yes (p < 0.05) |
| 2 | **`QCOM`** | Qualcomm Inc. | Mobile, Edge AI & Wireless Semis | **+0.7472** | +0.7735 | 0.00088 |  Yes (p < 0.05) |
| 3 | **`MS`** | Morgan Stanley | Institutional Prime Brokerage & Wealth | **+0.7270** | +0.6588 | 0.00142 |  Yes (p < 0.05) |
| 4 | **`GS`** | Goldman Sachs Group Inc. | Institutional Trading & Investment Banking | **+0.7165** | +0.6382 | 0.00179 |  Yes (p < 0.05) |
| 5 | **`MSTR`** | MicroStrategy Inc. | Bitcoin Treasury & Enterprise Analytics | **+0.7071** | +0.7382 | 0.00219 |  Yes (p < 0.05) |
| 6 | **`BLK`** | BlackRock Inc. | Global Asset Manager (iShares Bitcoin Trust IBIT) | **+0.7064** | +0.5735 | 0.00222 |  Yes (p < 0.05) |
| 7 | **`BAC`** | Bank of America Corp. | Diversified Global Banking | **+0.6797** | +0.5029 | 0.00377 |  Yes (p < 0.05) |
| 8 | **`MSFT`** | Microsoft Corp. | Enterprise Cloud, Software & AI | **+0.6583** | +0.5176 | 0.00557 |  Yes (p < 0.05) |
| 9 | **`JPM`** | JPMorgan Chase & Co. | Global Liquidity & Commercial Banking | **+0.6352** | +0.5029 | 0.00820 |  Yes (p < 0.05) |
| 10 | **`AAPL`** | Apple Inc. | Consumer Hardware & Ecosystem | **+0.6310** | +0.4265 | 0.00877 |  Yes (p < 0.05) |
| 11 | **`NFLX`** | Netflix Inc. | Digital Media & Streaming | **+0.6215** | +0.5382 | 0.01018 |  Yes (p < 0.05) |
| 12 | **`AMD`** | Advanced Micro Devices Inc. | AI Accelerators, GPUs & CPUs | **+0.6148** | +0.5882 | 0.01127 |  Yes (p < 0.05) |
| 13 | **`PYPL`** | PayPal Holdings Inc. | Digital Payments & Crypto On-Ramp | **+0.5844** | +0.4441 | 0.01744 |  Yes (p < 0.05) |
| 14 | **`AMZN`** | Amazon.com Inc. | Cloud Infrastructure (AWS) & E-Commerce | **+0.5594** | +0.4706 | 0.02426 |  Yes (p < 0.05) |
| 15 | **`AVGO`** | Broadcom Inc. | Custom AI Silicon & Data Center Networking | **+0.5220** | +0.4882 | 0.03808 |  Yes (p < 0.05) |
| 16 | **`ORCL`** | Oracle Corp. | Cloud Infrastructure & Database Systems | **+0.5079** | +0.5294 | 0.04461 |  Yes (p < 0.05) |

---

##  Key Macro Drivers Behind the High Correlation

1. **Crypto Treasury & Exchange Proxies (`COIN`, `MSTR`, `BLK`)**:
   - **Coinbase (`COIN`, r = +0.749)** and **MicroStrategy (`MSTR`, r = +0.707)** have direct operational and balance-sheet exposure to Bitcoin spot price volatility.
   - **BlackRock (`BLK`, r = +0.706)** is the sponsor of the iShares Bitcoin Trust (IBIT); fee revenue and multi-asset institutional flow align directly with crypto risk-on days.

2. **Wall Street Prime Brokerage Banks (`MS`, `GS`, `BAC`, `JPM`)**:
   - **Morgan Stanley (r = +0.727)**, **Goldman Sachs (r = +0.717)**, and **JPMorgan (r = +0.635)** exhibit over +0.63 correlation with Bitcoin.
   - Modern prime brokers clear crypto derivatives, finance institutional leverage, and facilitate ETF authorized participant (AP) arbitrage. When bank liquidity contracts, crypto liquidations immediately follow.

3. **Mega-Cap Big Tech & Hardware Compute (`QCOM`, `MSFT`, `AAPL`, `AMD`, `AMZN`)**:
   - **Qualcomm (r = +0.747)**, **Microsoft (r = +0.658)**, and **Apple (r = +0.631)** move with Bitcoin due to shared institutional ownership (passive index funds, risk-parity algorithms, and high-beta tech mandates).

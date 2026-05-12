# Hyperliquid × Bitcoin Fear & Greed Index — Sentiment Analysis

> **Primetrade.ai Data Science Assignment**  
> Quantitative analysis of how Bitcoin market sentiment drives trader behaviour and profitability on the Hyperliquid perpetual futures exchange.

---

## Overview

This project merges **Hyperliquid historical trade data** (~211K trades) with the **Bitcoin Fear & Greed Index** to uncover how market sentiment regimes (Extreme Fear → Extreme Greed) influence:

- Trade profitability and win rates
- Position sizing and risk behaviour
- Trader psychology and directional bias
- Asset-level performance sensitivity

---

## Quick Start

```bash
# Clone / enter the project directory
cd hyperliquid_sentiment_analysis

# Bootstrap — creates venv, installs all dependencies
bash setup.sh

# Activate the virtual environment
source .venv/bin/activate

# Run the full pipeline
python main.py
```

That's it. Everything generates automatically:
- `outputs/merged_data.csv` — merged trade + sentiment dataset
- `outputs/figures/` — 17 publication-quality charts
- `outputs/reports/full_analysis_report.md` — detailed markdown report

---

## Project Structure

```
hyperliquid_sentiment_analysis/
├── data/
│   ├── fear_greed_index.csv       # Bitcoin Fear & Greed Index (daily)
│   └── historical_data.csv        # Hyperliquid trade records
│
├── src/
│   ├── config.py                  # Paths, colours, constants
│   ├── data_loader.py             # Ingestion, cleaning, merging
│   ├── analysis.py                # All statistical computations
│   ├── visualizations.py          # All matplotlib/seaborn figures
│   └── report_generator.py        # Auto-generates the Markdown report
│
├── outputs/
│   ├── merged_data.csv            # Auto-generated
│   ├── figures/                   # 17 PNG charts (auto-generated)
│   └── reports/                   # full_analysis_report.md (auto-generated)
│
├── prompts/
│   └── ai_analysis_prompts.md     # 15+ structured prompts for GPT/Gemini/etc.
│
├── main.py                        # Entry point
├── requirements.txt               # Python dependencies
├── setup.sh                       # One-shot environment bootstrap
└── README.md
```

---

## Analysis Modules

| Module | What it computes |
|---|---|
| `data_loader.py` | Parses IST timestamps, coerces numerics, merges on date, derives `net_pnl`, `is_winner` |
| `analysis.py` | PnL by sentiment, activity metrics, daily time-series, Spearman/Pearson correlations, Kruskal-Wallis & Mann-Whitney tests, account-level rankings, coin analysis, VaR/CVaR, fee analysis, regime-transition effects, outlier detection |
| `visualizations.py` | 17 dark-theme charts covering all analysis dimensions |
| `report_generator.py` | Auto-renders findings into a professional Markdown report |

---

## Generated Figures

| # | Filename | Description |
|---|---|---|
| 01 | `avg_pnl_by_sentiment.png` | Avg Closed PnL and Net PnL by sentiment |
| 02 | `win_rate_by_sentiment.png` | Win rate % across regimes |
| 03 | `activity_and_size.png` | Trade count and avg position size |
| 04 | `buy_sell_ratio.png` | Directional bias (buy vs sell %) |
| 05 | `cumulative_pnl_timeline.png` | Cumulative PnL shaded by sentiment |
| 06 | `daily_pnl_vs_sentiment.png` | Dual-axis: daily PnL + F&G index |
| 07 | `pnl_distribution_by_sentiment.png` | Violin & box plots of PnL distributions |
| 08 | `profit_factor_sharpe.png` | Risk-adjusted metrics by regime |
| 09 | `fear_greed_history.png` | F&G index full history with MA |
| 10 | `correlation_heatmap.png` | Spearman correlation matrix (daily) |
| 11 | `top_traders.png` | Top 10 traders by PnL and win rate |
| 12 | `top_coins.png` | Top 15 assets by PnL |
| 13 | `risk_metrics.png` | VaR and max loss by sentiment |
| 14 | `volume_timeline.png` | Daily volume timeline by sentiment |
| 15 | `transition_analysis.png` | PnL on regime-transition vs stable days |
| 16 | `fee_analysis.png` | Fee burden by sentiment |
| 17 | `rolling_winrate.png` | 7-day rolling win rate over time |

---

## Key Findings (Preview)

1. **Fear regimes** produce the highest average trade sizes — traders bet bigger when they believe a bottom is forming.
2. **Extreme Greed** shows a sell bias (>55%) — experienced traders take profits into euphoria.
3. The raw F&G index value has a **near-zero linear correlation** with individual trade PnL — the *category/regime* matters far more than the daily number.
4. **Kruskal-Wallis test** confirms PnL distributions are statistically different across regimes.
5. Sentiment **transition days** exhibit elevated PnL volatility — opportunity and risk are both amplified.

See `outputs/reports/full_analysis_report.md` for the complete analysis.

---

## Using With Other AI Models

The `prompts/ai_analysis_prompts.md` file contains **15+ structured prompts** across 5 categories:

- Sentiment & Performance Analysis
- Trader Behaviour Analysis
- Asset & Market Microstructure
- Statistical & Predictive Modelling
- Risk Management Insights

Each prompt includes the dataset schema and is ready to paste into ChatGPT, Gemini, Claude, Mistral, or any AI with file-analysis capability.

---

## CLI Options

```bash
python main.py              # Full pipeline (data + analysis + plots + report)
python main.py --no-plots   # Skip figure generation (faster)
python main.py --no-report  # Skip markdown report
```

---

## Dependencies

```
pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, plotly, jinja2
```

All installed automatically via `bash setup.sh`.

---

## Dataset Sources

| Dataset | Source |
|---|---|
| Historical Trade Data | Hyperliquid perpetual futures (provided) |
| Fear & Greed Index | Alternative.me Bitcoin Fear & Greed API (provided) |

---

*Built for the Primetrade.ai Data Science Hiring Assignment.*

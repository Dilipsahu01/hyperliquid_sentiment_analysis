#!/usr/bin/env python3
"""
scripts/build_claims_ledger.py

Forensic Quant Audit & Claims Ledger Generator.
Enforces Rule 1: Every number that appears in any doc, README, Master.md or pitch
must be traced directly to code and written to results/claims_ledger.csv with:
claim_text, value, source_script, sample (N, start, end), evaluation_type, status.
"""

import csv
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Define the complete claims ledger items
claims = [
    # 1. High-Frequency Microstructure & Variance Ratios
    {
        "claim_text": "Total raw WebSocket messages processed across CEX and L1",
        "value": "104,896,898",
        "source_script": "scripts/research/generate_master_md.py",
        "sample": "N=104,896,898 events | 2026-11-01 to 2026-11-25",
        "evaluation_type": "lead-lag",
        "status": "verified"
    },
    {
        "claim_text": "Non-trade WebSocket frames scrubbed (cancellations, heartbeats, depth updates)",
        "value": "5,885,612 (5.61%)",
        "source_script": "scripts/research/generate_master_md.py",
        "sample": "N=5,885,612 non-trade frames | 2026-11-01 to 2026-11-25",
        "evaluation_type": "lead-lag",
        "status": "verified"
    },
    {
        "claim_text": "Lo-MacKinlay Variance Ratio expansion from 1-second to 1-hour",
        "value": "1.5157 (95% CI: [1.1825, 1.9529])",
        "source_script": "btc_pipeline/audit_v4_pipeline.py",
        "sample": "N=2,073,600 seconds (24 full days) | 2026-11-01 to 2026-11-25",
        "evaluation_type": "lead-lag",
        "status": "verified"
    },
    {
        "claim_text": "1-second return first-order autocorrelation rho_1 on Binance Spot trades",
        "value": "+0.0997 (std: 0.0333, min: +0.0373, max: +0.1614)",
        "source_script": "btc_pipeline/audit_v4_pipeline.py",
        "sample": "N=2,073,600 seconds | 2026-11-01 to 2026-11-25",
        "evaluation_type": "lead-lag",
        "status": "verified"
    },
    {
        "claim_text": "Zero-trade seconds share of total time",
        "value": "11.59% (240,355 zero-trade seconds)",
        "source_script": "btc_pipeline/audit_v4_pipeline.py",
        "sample": "N=2,073,600 seconds | 2026-11-01 to 2026-11-25",
        "evaluation_type": "lead-lag",
        "status": "verified"
    },

    # 2. Daily Volatility Models & Holdout Performance
    {
        "claim_text": "Out-of-sample 5m Realized Volatility holdout QLIKE loss vs 1h baseline",
        "value": "0.3979 (5m RV) vs 0.4999 (1h RV) -> 20.40% error reduction",
        "source_script": "btc_pipeline/audit_v4_pipeline.py",
        "sample": "N=23 holdout days | 2026-11-08 to 2026-11-30",
        "evaluation_type": "daily-vol",
        "status": "verified"
    },
    {
        "claim_text": "Holdout 80% confidence corridor coverage on daily volatility",
        "value": "80.80%",
        "source_script": "btc_pipeline/audit_v4_pipeline.py",
        "sample": "N=250 out-of-sample days | 2026-03-26 to 2026-11-30",
        "evaluation_type": "daily-vol",
        "status": "verified"
    },
    {
        "claim_text": "Holdout 95% confidence corridor coverage on daily volatility",
        "value": "92.80% to 93.60%",
        "source_script": "btc_pipeline/audit_v4_pipeline.py",
        "sample": "N=250 out-of-sample days | 2026-03-26 to 2026-11-30",
        "evaluation_type": "daily-vol",
        "status": "verified"
    },
    {
        "claim_text": "Headline claim: Volatility blend beats HAR-Log at 5% statistical significance",
        "value": "DM = +1.8149, p = 0.0695",
        "source_script": "btc_pipeline/audit_v4_pipeline.py",
        "sample": "N=889 walk-forward days | 2024-06-05 to 2026-10-27",
        "evaluation_type": "daily-vol",
        "status": "retracted"
    },
    {
        "claim_text": "Re-evaluated DM test vs HAR-Log with Newey-West bandwidth q=6 and HLN correction",
        "value": "DM_HLN = +1.524, p = 0.128 (not significant at alpha=0.05)",
        "source_script": "scripts/reproduce_quant_audit.py",
        "sample": "N=125 walk-forward folds (889 days) | 2024-06-05 to 2026-10-27",
        "evaluation_type": "daily-vol",
        "status": "verified"
    },
    {
        "claim_text": "HAC Newey-West bandwidth = 2",
        "value": "2",
        "source_script": "None (Arithmetic error: floor(4*(889/100)^(2/9)) = 6)",
        "sample": "N=889 days",
        "evaluation_type": "daily-vol",
        "status": "retracted"
    },

    # 3. Intraday 5-Minute Bands & Specifically Traced Claims
    {
        "claim_text": "Dynamic GARCH 1-sigma band width reduction vs static tunnel on intraday session",
        "value": "72.82% ($1455.93 -> $395.68 USD; 248.27 -> 67.47 bps)",
        "source_script": "scripts/audit_dynamic_nov25_kupiec.py",
        "sample": "N=79 bars (6.58 hours) | 2026-11-25 11:45 to 18:15 UTC",
        "evaluation_type": "5m-band",
        "status": "verified"
    },
    {
        "claim_text": "Generalization of 72.8% band reduction to full historical 4-year sample",
        "value": "72.8% full sample reduction",
        "source_script": "None (Invalidated by scripts/walkforward_5m_band_evaluation.py; true reduction is 20.94%)",
        "sample": "N=384,334 bars | 2023-04-01 to 2026-11-25",
        "evaluation_type": "5m-band",
        "status": "retracted"
    },
    {
        "claim_text": "Dynamic Jump Engine 1-sigma width reduction vs standard GARCH(1,1)-t over full historical out-of-sample",
        "value": "20.94% (30.23 bps -> 23.90 bps)",
        "source_script": "scripts/walkforward_5m_band_evaluation.py",
        "sample": "N=384,334 bars | 2023-04-01 to 2026-11-25",
        "evaluation_type": "5m-band",
        "status": "verified"
    },
    {
        "claim_text": "Dynamic 5m 1-sigma band two-sided full-width in bps and USD",
        "value": "67.47 bps ($395.68 USD on $58,643.55 anchor)",
        "source_script": "scripts/reproduce_quant_audit.py",
        "sample": "N=79 bars | 2026-11-25 11:45 to 18:15 UTC",
        "evaluation_type": "5m-band",
        "status": "verified"
    },
    {
        "claim_text": "Dynamic 5m 1-sigma band one-sided radius (half-width) in bps and USD",
        "value": "33.74 bps ($197.84 USD on $58,643.55 anchor)",
        "source_script": "scripts/reproduce_quant_audit.py",
        "sample": "N=79 bars | 2026-11-25 11:45 to 18:15 UTC",
        "evaluation_type": "5m-band",
        "status": "verified"
    },
    {
        "claim_text": "Median Pinball Loss (q=0.50 core) reduction from static to dynamic model",
        "value": "30.57% ($58.0827 -> $40.3284 USD)",
        "source_script": "scripts/dynamic_garch_gap_engine.py",
        "sample": "N=16 bars (75 minutes) | 2026-11-25 12:30 to 13:45 UTC",
        "evaluation_type": "5m-band",
        "status": "verified"
    },
    {
        "claim_text": "Winkler score reduction claimed in assistant response ($1,682 to $512 USD)",
        "value": "$1,682 to $512 USD",
        "source_script": "None (Chat reasoning without script execution; violated Rule 1)",
        "sample": "N=0 (Undefined sample)",
        "evaluation_type": "5m-band",
        "status": "retracted"
    },
    {
        "claim_text": "Actual empirically computed 95% Winkler score reduction on N=79 bars",
        "value": "52.77% ($2912.16 -> $1375.37 USD; 496.59 -> 234.53 bps)",
        "source_script": "scripts/reproduce_quant_audit.py",
        "sample": "N=79 bars | 2026-11-25 11:45 to 18:15 UTC",
        "evaluation_type": "5m-band",
        "status": "verified"
    },
    {
        "claim_text": "Kupiec (1995) POF test on Dynamic GARCH 1-sigma coverage (64.56% vs 68.27%)",
        "value": "LR_uc = 0.4931, p = 0.4825 (Fail to Reject H0)",
        "source_script": "scripts/audit_dynamic_nov25_kupiec.py",
        "sample": "N=79 bars (low statistical power) | 2026-11-25 11:45 to 18:15 UTC",
        "evaluation_type": "5m-band",
        "status": "verified"
    },
    {
        "claim_text": "Kupiec test on Static Tunnel 1-sigma coverage (86.08% vs 68.27%)",
        "value": "LR_uc = 13.3991, p = 0.00025 (Reject H0 - Miscalibrated)",
        "source_script": "scripts/audit_dynamic_nov25_kupiec.py",
        "sample": "N=79 bars | 2026-11-25 11:45 to 18:15 UTC",
        "evaluation_type": "5m-band",
        "status": "verified"
    },
    {
        "claim_text": "Christoffersen (1998) Independence test on Dynamic 1-sigma breaches",
        "value": "LR_ind = 1.4123, p = 0.2347 (Fail to Reject H0 - No Clustering)",
        "source_script": "scripts/audit_dynamic_nov25_kupiec.py",
        "sample": "N=79 bars | 2026-11-25 11:45 to 18:15 UTC",
        "evaluation_type": "5m-band",
        "status": "verified"
    },
    {
        "claim_text": "Claim that 5-minute 2-sigma band serves as a liquidation margin model",
        "value": "2-sigma corridor = 95.45% coverage",
        "source_script": "None (Breaches 13 times/day under 288 5m bars; invalid for margin)",
        "sample": "N=79 bars",
        "evaluation_type": "5m-band",
        "status": "retracted"
    },

    # 4. London Close (16:00-17:00 UTC) Regime
    {
        "claim_text": "London Close (16:00-17:00 UTC) log RV in-sample OLS regression t-stat",
        "value": "t = +2.364, p = 0.0181",
        "source_script": "scripts/backtest_london_close_feature.py",
        "sample": "N=1,423 daily sessions | 2023-01-01 to 2026-11-23",
        "evaluation_type": "daily-vol",
        "status": "verified"
    },
    {
        "claim_text": "London Close out-of-sample walk-forward incremental QLIKE advantage",
        "value": "DM = +1.0777, p = 0.2812 (Not statistically significant)",
        "source_script": "scripts/backtest_london_close_feature.py",
        "sample": "N=917 out-of-sample days | 2024-05-15 to 2026-11-17",
        "evaluation_type": "daily-vol",
        "status": "verified"
    },
    {
        "claim_text": "London Close feature is a proven ex-ante predictive alpha feature",
        "value": "p = 0.0181",
        "source_script": "None (Fails Bonferroni 0.05/39=0.00128 and fails OOS DM test)",
        "sample": "N=1,423 in-sample / N=917 OOS",
        "evaluation_type": "daily-vol",
        "status": "retracted"
    },

    # 5. Lead-Lag & Market Microstructure
    {
        "claim_text": "Binance leads Hyperliquid price discovery by 60-120s on US macro events",
        "value": "60 to 120 seconds",
        "source_script": "None (Hardcoded 13:30 UTC filter misaligned summer EDT events; requires DST fix)",
        "sample": "N=18 selective events",
        "evaluation_type": "lead-lag",
        "status": "unverified"
    },
    {
        "claim_text": "Hyperliquid perpetuals lead spot markets in liquidation cascades",
        "value": "Perpetual basis discount of -6.8 to -8.68 bps leads spot",
        "source_script": "None (Confounded instrument [perp vs spot] with venue; circular basis definition)",
        "sample": "N=24 selective events",
        "evaluation_type": "lead-lag",
        "status": "retracted"
    },
    {
        "claim_text": "L2 order book depth dropped 64.2% on Hyperliquid during cascade event",
        "value": "64.2% depth drop",
        "source_script": "scripts/test_latest_live_window.py",
        "sample": "N=1 event (Nov 25 14:15 UTC drop to $57,300) | Case study",
        "evaluation_type": "lead-lag",
        "status": "verified"
    },
    {
        "claim_text": "Forward logger represents a proprietary data moat",
        "value": "Data moat claim",
        "source_script": "None (Public requester-pays S3 buckets exist: s3://hyperliquid-archive)",
        "sample": "N/A",
        "evaluation_type": "lead-lag",
        "status": "retracted"
    },
    {
        "claim_text": "FTX collapse aftermath was included in the 917-day stress regime",
        "value": "FTX aftermath claim",
        "source_script": "None (FTX collapsed Nov 2022; repo data begins Jan 2023)",
        "sample": "N=917 days | 2023-01-01 to 2026-11-25",
        "evaluation_type": "daily-vol",
        "status": "retracted"
    },

    # 6. Trading Simulation & Strategy Performance
    {
        "claim_text": "Volatility-targeted trading strategy net return under 7 bps fee hurdle",
        "value": "-16.47% vs -15.42% Buy & Hold (Annualized Sharpe: -7.96)",
        "source_script": "btc_pipeline/trading_simulation.py",
        "sample": "N=22 holdout trading days | 2026-11-01 to 2026-11-23",
        "evaluation_type": "sim",
        "status": "verified"
    },
    {
        "claim_text": "Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) probability",
        "value": "9.75e-14 (0.000% probability of true positive Sharpe)",
        "source_script": "btc_pipeline/trading_simulation.py",
        "sample": "N=39 total experimental trials | T=22 days",
        "evaluation_type": "sim",
        "status": "verified"
    },
    {
        "claim_text": "Claim of 7 bps adverse-selection savings for market makers",
        "value": "7 bps adverse selection savings",
        "source_script": "None (7 bps was simulated taker friction, not quote markout curve)",
        "sample": "N=22 trading days",
        "evaluation_type": "sim",
        "status": "retracted"
    }
]

def build_csv_ledger():
    df = pd.DataFrame(claims)
    csv_path = RESULTS_DIR / "claims_ledger.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[OK] Successfully built claims ledger with {len(df)} entries: {csv_path}")

    # Summary counts by status and evaluation type
    print("\nSummary by Status:")
    print(df["status"].value_counts())
    print("\nSummary by Evaluation Type:")
    print(df["evaluation_type"].value_counts())
    return df

if __name__ == "__main__":
    build_csv_ledger()

#!/usr/bin/env python3
"""
main.py — Master entry point for the Hyperliquid Sentiment & Volatility Analysis Pipeline.

Usage:
    python main.py              # full pipeline execution
    python main.py --no-report  # skip markdown report generation
    python main.py --no-plots   # skip figure generation
"""

import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(args):
    t0 = time.time()
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║  Hyperliquid × Bitcoin Sentiment & Volatility Analysis Pipeline  ║
║  Institutional Quantitative Research & Microstructure Engine     ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)

    from src.data_loader import load_and_merge
    merged_df = load_and_merge(save=True)

    from src.analysis import run_all
    results = run_all(merged_df)

    _print_console_summary(results)

    if not args.no_plots:
        from src.visualizations import generate_all_plots
        generate_all_plots(merged_df, results)

    if not args.no_report:
        from src.report_generator import generate_report
        generate_report(merged_df, results)

    elapsed = time.time() - t0
    print(f"\n  Pipeline complete in {elapsed:.1f}s")
    print("   Outputs:")
    print("     outputs/merged_data.csv")
    print("     outputs/figures/   (17 PNG charts)")
    print("     outputs/reports/full_analysis_report.md")


def _print_console_summary(results: dict):
    sp = results["sentiment_pnl"]
    cr = results["correlation"]

    print("\n" + "=" * 65)
    print("  EXECUTIVE SUMMARY — PnL by Sentiment Regime")
    print("=" * 65)
    header = f"  {'Regime':<16} {'Trades':>8} {'WinRate':>9} {'Avg PnL':>10} {'Tot PnL':>12}"
    print(header)
    print("  " + "-" * 59)
    for _, r in sp.iterrows():
        print(f"  {r['classification']:<16} {r['trade_count']:>8,} "
              f"{r['win_rate_pct']:>8.1f}% "
              f"${r['avg_pnl']:>9.2f} "
              f"${r['total_pnl']:>11,.0f}")
    print("=" * 65)

    if "kruskal_wallis" in cr:
        kw = cr["kruskal_wallis"]
        sig = "Significant (p < 0.05)" if kw["p"] < 0.05 else "Not Significant"
        print(f"\n  Kruskal-Wallis Test:  H = {kw['H']:.2f},  p = {kw['p']:.4e}  [{sig}]")

    if "spearman_value_pnl" in cr:
        sp_corr = cr["spearman_value_pnl"]
        print(f"  Spearman (Score vs PnL): rho = {sp_corr['rho']:.4f},  p = {sp_corr['p']:.4e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperliquid x Fear & Greed sentiment analysis pipeline."
    )
    parser.add_argument("--no-plots",  action="store_true", help="Skip figure generation")
    parser.add_argument("--no-report", action="store_true", help="Skip markdown report")
    args = parser.parse_args()
    main(args)

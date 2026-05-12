#!/usr/bin/env python3
"""
main.py — Entry point for the Hyperliquid × Fear & Greed Index Analysis.

Usage:
    python main.py              # full pipeline
    python main.py --no-report  # skip markdown report
    python main.py --no-plots   # skip figure generation
"""

import sys
import time
import argparse
from pathlib import Path

# ── ensure project root is on sys.path ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(args):
    t0 = time.time()
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║  Hyperliquid × Bitcoin Fear & Greed Index — Analysis Pipeline   ║
║  Primetrade.ai Data Science Assignment                          ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)

    # 1. Load & merge data
    from src.data_loader import load_and_merge
    merged_df = load_and_merge(save=True)

    # 2. Run all statistical analyses
    from src.analysis import run_all
    results = run_all(merged_df)

    # 3. Print console summary
    _print_console_summary(results)

    # 4. Generate plots
    if not args.no_plots:
        from src.visualizations import generate_all_plots
        generate_all_plots(merged_df, results)

    # 5. Generate markdown report
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
    sp  = results["sentiment_pnl"]
    cor = results["correlation"]

    print("\n" + "═" * 65)
    print("  SENTIMENT vs PERFORMANCE — QUICK SUMMARY")
    print("═" * 65)
    print(f"  {'Sentiment':<18} {'Trades':>8} {'AvgPnL':>10} {'WinRate':>9} {'ProfitFactor':>13}")
    print("  " + "─" * 63)
    for _, r in sp.iterrows():
        print(
            f"  {str(r['classification']):<18} "
            f"{int(r['trade_count']):>8,} "
            f"${r['avg_pnl']:>9.2f} "
            f"{r['win_rate_pct']:>8.1f}% "
            f"{r['profit_factor']:>13.2f}"
        )

    kw = cor.get("kruskal_wallis", {})
    print("\n  Kruskal-Wallis H-Test across sentiment groups:")
    print(f"    H = {kw.get('H', 'N/A')},  p = {kw.get('p', 'N/A')}")
    sp_corr = cor["spearman_value_pnl"]
    print(f"\n  Spearman ρ (F&G value ↔ Trade PnL): {sp_corr['rho']}")
    print("═" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperliquid Sentiment Analysis Pipeline")
    parser.add_argument("--no-plots",  action="store_true", help="Skip figure generation")
    parser.add_argument("--no-report", action="store_true", help="Skip markdown report generation")
    args = parser.parse_args()
    main(args)

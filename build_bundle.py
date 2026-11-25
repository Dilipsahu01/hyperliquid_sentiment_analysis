import os
from pathlib import Path

files_to_bundle = [
    ("README.md", "markdown"),
    ("docs/01_methodology_and_techniques.md", "markdown"),
    ("docs/02_long_term_prediction.md", "markdown"),
    ("docs/03_full_day_intraday_prediction.md", "markdown"),
    ("docs/04_data_factors_analysis.md", "markdown"),
    ("src/data.py", "python"),
    ("Master.md", "markdown"),
    ("report.md", "markdown"),
    ("results/audit_v4.md", "markdown"),
    ("results/audit_v3.md", "markdown"),
    ("results/audit_v2.md", "markdown"),
    ("results/audit.md", "markdown"),
    ("results/forecast_nov26_30.csv", "csv"),
    ("results/forecast_nov26_30_v4.csv", "csv"),
    ("results/holdout.json", "json"),
    ("results/audit_v4_meta.json", "json"),
    ("results/probabilistic_gap_results.json", "json"),
    ("results/dynamic_calibration_comparison.json", "json"),
    ("btc_pipeline/baselines.py", "python"),
    ("btc_pipeline/models.py", "python"),
    ("btc_pipeline/feature_engineering.py", "python"),
    ("btc_pipeline/audit_v4_pipeline.py", "python"),
    ("btc_pipeline/download_5m_data.py", "python"),
    ("btc_pipeline/trading_simulation.py", "python"),
    ("btc_pipeline/live_forward_logger.py", "python"),
    ("scripts/generate_forecast_and_spike_plots.py", "python"),
    ("scripts/generate_data_factor_analysis.py", "python"),
    ("scripts/gap_prediction_evaluation.py", "python"),
    ("scripts/probabilistic_gap_engine.py", "python"),
    ("scripts/dynamic_garch_gap_engine.py", "python"),
    ("tests/test_pipeline.py", "python"),
    ("tests/test_live_logger.py", "python")
]

content = [
    "# Claude Review Master Compilation: BTC Volatility Forecasting Pipeline (Round 4 Unified)\n",
    "> Comprehensive single-file bundle containing all research reports, audit evaluations, forecast CSVs, and Python implementation scripts for external review.\n",
    "## Table of Contents\n"
]

for p_str, lang in files_to_bundle:
    anchor = p_str.replace('/', '_').replace('.', '_').lower()
    content.append(f"- [{p_str}](#{anchor})")

content.append("\n---\n")

for p_str, lang in files_to_bundle:
    p = Path(p_str)
    if not p.exists():
        continue
    anchor = p_str.replace('/', '_').replace('.', '_').lower()
    content.append(f"\n# <a id=\"{anchor}\"></a>FILE: `{p_str}`\n")
    text = p.read_text(encoding="utf-8")
    content.append(f"```{lang}\n{text}\n```\n")

Path("claude-to-review.md").write_text("\n".join(content), encoding="utf-8")
print(f"Successfully generated claude-to-review.md with {len(files_to_bundle)} files ({Path('claude-to-review.md').stat().st_size:,} bytes).")

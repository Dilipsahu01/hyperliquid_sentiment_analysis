"""
config.py — Central configuration for the Hyperliquid × Fear & Greed Analysis.
All file paths, constants, and plot styling are defined here.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIG_DIR    = OUTPUT_DIR / "figures"
REP_DIR    = OUTPUT_DIR / "reports"

FEAR_GREED_PATH = DATA_DIR / "fear_greed_index.csv"
TRADES_PATH     = DATA_DIR / "historical_data.csv"
MERGED_PATH     = OUTPUT_DIR / "merged_data.csv"

# Ensure output dirs exist at import time
FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

# ── Sentiment ordering ──────────────────────────────────────────────────────
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

SENTIMENT_COLORS = {
    "Extreme Fear":  "#d62728",   # red
    "Fear":          "#ff7f0e",   # orange
    "Neutral":       "#bcbd22",   # yellow-green
    "Greed":         "#2ca02c",   # green
    "Extreme Greed": "#1f77b4",   # blue
}

SENTIMENT_PALETTE = [SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER]

# ── Plot defaults ───────────────────────────────────────────────────────────
FIGURE_DPI   = 150
FIGURE_STYLE = "seaborn-v0_8-whitegrid"

# ── Analysis thresholds ─────────────────────────────────────────────────────
OUTLIER_ZSCORE_THRESHOLD = 3.5   # for flagging extreme PnL outliers
MIN_TRADES_PER_ACCOUNT   = 10    # minimum trades to include account in ranking

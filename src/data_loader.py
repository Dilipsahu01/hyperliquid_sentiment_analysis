"""
data_loader.py — Data ingestion, cleaning, and merging pipeline.

Produces a single merged DataFrame with every trade enriched by the
Fear & Greed sentiment for that day.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    FEAR_GREED_PATH, TRADES_PATH, MERGED_PATH,
    SENTIMENT_ORDER
)


def _parse_trades(path: Path) -> pd.DataFrame:
    """Load and clean the Hyperliquid historical trades CSV."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    df["datetime"] = pd.to_datetime(
        df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    df["date"] = df["datetime"].dt.normalize()

    num_cols = ["Execution Price", "Size Tokens", "Size USD",
                "Closed PnL", "Fee", "Start Position"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_winner"] = df["Closed PnL"] > 0
    df["is_closer"] = df["Direction"].str.contains("Close", case=False, na=False)
    df["net_pnl"] = df["Closed PnL"] - df["Fee"].fillna(0)
    df["side_num"] = df["Side"].map({"BUY": 1, "SELL": -1}).fillna(0)

    if "Leverage" in df.columns:
        df["Leverage"] = pd.to_numeric(df["Leverage"], errors="coerce")

    df = df.dropna(subset=["date"])
    print(f"  [trades]   {len(df):,} rows loaded  |  "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def _parse_fear_greed(path: Path) -> pd.DataFrame:
    """Load and clean the Fear & Greed Index CSV."""
    fg = pd.read_csv(path)
    fg.columns = fg.columns.str.strip()
    fg["date"] = pd.to_datetime(fg["date"], errors="coerce").dt.normalize()
    fg["value"] = pd.to_numeric(fg["value"], errors="coerce")
    fg = fg.dropna(subset=["date", "value"])

    valid = set(SENTIMENT_ORDER)
    fg["classification"] = fg["classification"].str.strip()
    fg.loc[~fg["classification"].isin(valid), "classification"] = np.nan

    fg = fg.sort_values("date").drop_duplicates("date", keep="last")
    print(f"  [f&g idx]  {len(fg):,} rows loaded  |  "
          f"{fg['date'].min().date()} -> {fg['date'].max().date()}")
    return fg


def load_and_merge(save: bool = True) -> pd.DataFrame:
    """Full pipeline: load, clean, and merge on calendar date."""
    print("\n[DataLoader] Loading datasets...")
    trades = _parse_trades(TRADES_PATH)
    fg = _parse_fear_greed(FEAR_GREED_PATH)

    merged = trades.merge(
        fg[["date", "value", "classification"]],
        on="date",
        how="inner"
    )

    merged["classification"] = pd.Categorical(
        merged["classification"],
        categories=SENTIMENT_ORDER,
        ordered=True
    )

    overlap_pct = len(merged) / len(trades) * 100
    print(f"  [merged]   {len(merged):,} rows  "
          f"({overlap_pct:.1f}% of trades matched to sentiment)")
    print(f"             date range: "
          f"{merged['date'].min().date()} -> {merged['date'].max().date()}")

    if save:
        merged.to_csv(MERGED_PATH, index=False)
        print(f"  [saved]    {MERGED_PATH}")

    return merged

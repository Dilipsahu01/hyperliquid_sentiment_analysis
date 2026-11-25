#!/usr/bin/env python3
"""
fetch_1sec_btc.py — Downloads raw trade dumps and builds institutional-grade
1-second OHLCV data for BTC/USDT.

Target period: Nov 01, 2026 00:00:00 to Nov 25, 2026 00:00:00 UTC (24 full days).
Total bars: 24 * 86,400 = 2,073,600 1-second OHLCV candles.

Source: Binance Public Data Archive (data.binance.vision) — 100% Free
Outputs:
  - data/btc_1sec_nov2026.csv
  - data/btc_1sec_nov2026.parquet (if pyarrow available)
"""

import io
import time
import zipfile
import datetime
import requests
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SYMBOL = "BTCUSDT"
YEAR = 2026
MONTH = 11
START_DAY = 1
END_DAY = 24  # 1 Nov 00:00:00 to 25 Nov 00:00:00 (end of 24 Nov)

BASE_URL = f"https://data.binance.vision/data/spot/daily/trades/{SYMBOL}"


def process_day(day: int, prev_close: float = None) -> tuple[pd.DataFrame, float]:
    """Download, extract and resample 1 day of tick trades into 1-second OHLCV."""
    date_str = f"{YEAR}-{MONTH:02d}-{day:02d}"
    url = f"{BASE_URL}/{SYMBOL}-trades-{date_str}.zip"

    t0 = time.time()
    try:
        r = requests.get(url, timeout=45)
        if r.status_code == 404:
            print(f"  [{date_str}]  404 Not Found")
            return None, prev_close
        r.raise_for_status()
    except Exception as e:
        print(f"  [{date_str}]  Download error: {e}")
        return None, prev_close

    dl_time = time.time() - t0

    # Unpack from memory
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    fname = zf.namelist()[0]

    # Binance trade CSV: [trade_id, price, qty, quote_qty, time_us, is_buyer_maker, is_best_match]
    # time_us is in microseconds (16 digits)
    with zf.open(fname) as f:
        df = pd.read_csv(f, header=None, usecols=[1, 2, 4], names=["price", "qty", "time_us"])

    trade_count = len(df)
    # Convert microseconds to integer second
    df["sec"] = df["time_us"] // 1_000_000

    # 1-second aggregation
    g = df.groupby("sec")
    agg = g.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("qty", "sum"),
        trades=("price", "count"),
    )

    # Reindex to exact 86,400 seconds for complete continuous timeseries
    day_dt = datetime.datetime(YEAR, MONTH, day, 0, 0, 0, tzinfo=datetime.timezone.utc)
    day_start_sec = int(day_dt.timestamp())
    full_idx = np.arange(day_start_sec, day_start_sec + 86400, dtype=np.int64)

    agg = agg.reindex(full_idx)

    # Handle seconds with no trades (forward-fill previous close)
    if prev_close is not None and pd.isna(agg["close"].iloc[0]):
        agg.loc[full_idx[0], "close"] = prev_close

    agg["close"] = agg["close"].ffill()
    agg["open"] = agg["open"].fillna(agg["close"])
    agg["high"] = agg["high"].fillna(agg["close"])
    agg["low"] = agg["low"].fillna(agg["close"])
    agg["volume"] = agg["volume"].fillna(0.0)
    agg["trades"] = agg["trades"].fillna(0).astype(np.int32)

    last_close = agg["close"].iloc[-1]
    agg["epoch_sec"] = agg.index
    agg["timestamp_utc"] = pd.to_datetime(agg["epoch_sec"], unit="s", utc=True)

    agg = agg.reset_index(drop=True)
    cols = ["timestamp_utc", "epoch_sec", "open", "high", "low", "close", "volume", "trades"]
    agg = agg[cols]

    proc_time = time.time() - t0
    print(f"  [{date_str}]  {trade_count:>9,} trades → 86,400 1s bars (DL: {dl_time:.1f}s, Total: {proc_time:.1f}s) Close: ${last_close:,.2f}")
    return agg, last_close


def main():
    print("=" * 70)
    print("  BTC/USDT 1-Second OHLCV Ingestion Pipeline")
    print(f"  Range: {YEAR}-{MONTH:02d}-{START_DAY:02d} 00:00:00 → {YEAR}-{MONTH:02d}-{END_DAY+1:02d} 00:00:00 UTC")
    print(f"  Expected bars: {(END_DAY - START_DAY + 1) * 86400:,}")
    print("=" * 70 + "\n")

    days_frames = []
    prev_close = None
    t_start = time.time()

    for d in range(START_DAY, END_DAY + 1):
        df_day, prev_close = process_day(d, prev_close=prev_close)
        if df_day is not None:
            days_frames.append(df_day)

    if not days_frames:
        print(" No data processed.")
        return

    print("\n" + "─" * 70)
    print("  Concatenating and calculating quant returns …")
    full_df = pd.concat(days_frames, ignore_index=True)

    # Quant features: 1-second log returns and price changes
    full_df["log_return"] = np.log(full_df["close"] / full_df["close"].shift(1)).fillna(0.0)
    full_df["price_change"] = (full_df["close"] - full_df["open"]).round(4)

    # Save to CSV
    csv_path = DATA_DIR / "btc_1sec_nov2026.csv"
    print(f"  Writing CSV ({len(full_df):,} rows) → {csv_path.name} …")
    full_df.to_csv(csv_path, index=False)
    csv_size = csv_path.stat().st_size / (1024 * 1024)

    # Save to Parquet if possible
    parquet_path = DATA_DIR / "btc_1sec_nov2026.parquet"
    try:
        full_df.to_parquet(parquet_path, index=False)
        pq_size = parquet_path.stat().st_size / (1024 * 1024)
        pq_msg = f"& Parquet ({pq_size:.1f} MB)"
    except Exception:
        pq_msg = ""

    elapsed = time.time() - t_start
    print(f"\n   Complete in {elapsed:.1f}s!")
    print(f"   Total 1-Second Bars: {len(full_df):,}")
    print(f"   Output: {csv_path} ({csv_size:.1f} MB) {pq_msg}")
    print(f"   Date Range: {full_df['timestamp_utc'].min()} → {full_df['timestamp_utc'].max()} UTC")
    print("=" * 70)


if __name__ == "__main__":
    main()

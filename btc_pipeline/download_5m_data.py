#!/usr/bin/env python3
"""
download_5m_data.py
Downloads Binance BTCUSDT 5-minute klines from 2023-01-01 through the latest available hour on 2026-11-25.
Saves to data/btc_5m_2023_2026.parquet.
Also appends the latest live 1h klines to data/btc_1h_2023_2026.parquet.
"""

import io
import time
import zipfile
import datetime
import requests
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

COLS = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore"
]


def download_5m_klines():
    dest = DATA_DIR / "btc_5m_2023_2026.parquet"
    if dest.exists():
        print(f"[BTC 5m] Found existing {dest.name}")
        df = pd.read_parquet(dest)
        print(f"[BTC 5m] Loaded {len(df):,} bars from {df['open_time'].min()} to {df['open_time'].max()}")
        return df

    print("[BTC 5m] Downloading Binance spot 5m historical klines (2023-01 to 2026-11)...")
    records = []

    # 1. Monthly files: 2023-01 to 2026-10
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2026, 10, 1)
    cur = start_date
    while cur <= end_date:
        yr = cur.year
        mo = cur.month
        url = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-{yr}-{mo:02d}.zip"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                fn = zf.namelist()[0]
                with zf.open(fn) as f:
                    mdf = pd.read_csv(f, header=None)
                    records.append(mdf)
                print(f"  [Monthly 5m] {yr}-{mo:02d} OK ({len(mdf):,} bars)")
            else:
                print(f"  [Monthly 5m] {yr}-{mo:02d} returned {r.status_code}")
        except Exception as e:
            print(f"  [Monthly 5m] {yr}-{mo:02d} error: {e}")

        if mo == 12:
            cur = datetime.date(yr + 1, 1, 1)
        else:
            cur = datetime.date(yr, mo + 1, 1)

    # 2. Daily files for 2026-11: days 1 to 24
    for day in range(1, 25):
        url = f"https://data.binance.vision/data/spot/daily/klines/BTCUSDT/5m/BTCUSDT-5m-2026-11-{day:02d}.zip"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                fn = zf.namelist()[0]
                with zf.open(fn) as f:
                    ddf = pd.read_csv(f, header=None)
                    records.append(ddf)
                print(f"  [Daily 5m] 2026-11-{day:02d} OK ({len(ddf):,} bars)")
            else:
                print(f"  [Daily 5m] 2026-11-{day:02d} returned {r.status_code}")
        except Exception as e:
            print(f"  [Daily 5m] 2026-11-{day:02d} error: {e}")

    # 3. Live 5m bars for 2026-11-25 from Binance REST API
    print("  [Live REST 5m] Fetching 2026-11-25 intraday bars...")
    start_ts = int(datetime.datetime(2026, 11, 25, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    try:
        r = requests.get("https://api.binance.com/api/v3/klines", params={
            "symbol": "BTCUSDT",
            "interval": "5m",
            "startTime": start_ts,
            "limit": 1000
        }, timeout=15)
        if r.status_code == 200:
            live_bars = r.json()
            ldf = pd.DataFrame(live_bars)
            records.append(ldf)
            print(f"  [Live REST 5m] 2026-11-25 OK ({len(ldf):,} bars)")
    except Exception as e:
        print(f"  [Live REST 5m] error: {e}")

    full = pd.concat(records, ignore_index=True)
    full.columns = COLS[:len(full.columns)]
    full["open_time"] = pd.to_datetime(full["open_time_ms"].astype(np.int64), unit="ms", utc=True)
    full["close_time"] = pd.to_datetime(full["close_time_ms"].astype(np.int64), unit="ms", utc=True)

    for c in full.columns:
        if c not in ["open_time", "close_time"]:
            full[c] = pd.to_numeric(full[c], errors="coerce")
    if "ignore" in full.columns:
        full = full.drop(columns=["ignore"])

    full = full.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    full.to_parquet(dest, index=False)
    print(f"[BTC 5m] Saved {len(full):,} bars to {dest.name} ({full['open_time'].min()} to {full['open_time'].max()})")
    return full


def refresh_1h_klines():
    """Refresh 1h klines through latest hour of 2026-11-25."""
    dest = DATA_DIR / "btc_1h_2023_2026.parquet"
    df_1h = pd.read_parquet(dest)
    print(f"[BTC 1h] Current 1h bars: {len(df_1h):,} through {df_1h['open_time'].max()}")

    # Fetch live 1h bars starting from Nov 24
    start_ts = int(datetime.datetime(2026, 11, 24, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    try:
        r = requests.get("https://api.binance.com/api/v3/klines", params={
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": start_ts,
            "limit": 100
        }, timeout=15)
        if r.status_code == 200:
            live_bars = r.json()
            ldf = pd.DataFrame(live_bars)
            ldf.columns = COLS[:len(ldf.columns)]
            ldf["open_time"] = pd.to_datetime(ldf["open_time_ms"].astype(np.int64), unit="ms", utc=True)
            ldf["close_time"] = pd.to_datetime(ldf["close_time_ms"].astype(np.int64), unit="ms", utc=True)
            for c in ldf.columns:
                if c not in ["open_time", "close_time"]:
                    ldf[c] = pd.to_numeric(ldf[c], errors="coerce")
            if "ignore" in ldf.columns:
                ldf = ldf.drop(columns=["ignore"])

            combined = pd.concat([df_1h, ldf], ignore_index=True)
            combined = combined.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
            combined.to_parquet(dest, index=False)
            print(f"[BTC 1h] Refreshed: {len(combined):,} bars through {combined['open_time'].max()}")
    except Exception as e:
        print(f"[BTC 1h] Refresh error: {e}")


if __name__ == "__main__":
    download_5m_klines()
    refresh_1h_klines()

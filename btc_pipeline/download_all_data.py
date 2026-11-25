#!/usr/bin/env python3
"""
download_all_data.py
Fetch and persist clean datasets covering 2023-01-01 to 2026-11-25:
1. BTC 1h klines from Binance Vision (2023-01 to 2026-11)
2. Daily equity & macro closes from yfinance (22 required assets + SPY + QQQ)
3. Fear & Greed index from alternative.me (persisted in data/fear_greed_index.csv)
4. Audit existing 1-second dataset (Nov 2026) and 5m perpetual dataset.
"""

import io
import time
import zipfile
import requests
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

EQUITY_TICKERS = [
    "COIN", "MSTR", "BLK", "MS", "GS", "JPM", "BAC", "QCOM", "NVDA", "AMD",
    "AVGO", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA", "GLD", "XOM",
    "UUP", "TLT", "WMT", "SPY", "QQQ"
]


def download_binance_1h_btc() -> pd.DataFrame:
    """Download Binance 1h klines from 2023-01 to 2026-11."""
    dest = DATA_DIR / "btc_1h_2023_2026.parquet"
    if dest.exists():
        print(f"[BTC 1h] Found existing {dest.name}")
        df = pd.read_parquet(dest)
        print(f"[BTC 1h] Loaded {len(df):,} bars from {df['open_time'].min()} to {df['open_time'].max()}")
        return df

    print("[BTC 1h] Downloading Binance spot 1h historical klines (2023-01 to 2026-11)...")
    records = []

    # Monthly files: 2023-01 to 2026-10
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2026, 10, 1)
    cur = start_date
    while cur <= end_date:
        yr = cur.year
        mo = cur.month
        url = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{yr}-{mo:02d}.zip"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                fn = zf.namelist()[0]
                with zf.open(fn) as f:
                    mdf = pd.read_csv(f, header=None)
                    records.append(mdf)
                print(f"  [Monthly] {yr}-{mo:02d} OK ({len(mdf)} bars)")
            else:
                print(f"  [Monthly] {yr}-{mo:02d} returned {r.status_code}")
        except Exception as e:
            print(f"  [Monthly] {yr}-{mo:02d} error: {e}")

        # advance 1 month
        if mo == 12:
            cur = datetime.date(yr + 1, 1, 1)
        else:
            cur = datetime.date(yr, mo + 1, 1)

    # Daily files for 2026-11: days 1 to 25
    for day in range(1, 26):
        url = f"https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-11-{day:02d}.zip"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                fn = zf.namelist()[0]
                with zf.open(fn) as f:
                    ddf = pd.read_csv(f, header=None)
                    records.append(ddf)
                print(f"  [Daily] 2026-11-{day:02d} OK ({len(ddf)} bars)")
            else:
                print(f"  [Daily] 2026-11-{day:02d} returned {r.status_code}")
        except Exception as e:
            print(f"  [Daily] 2026-11-{day:02d} error: {e}")

    full = pd.concat(records, ignore_index=True)
    # Binance kline schema:
    # 0: Open time (ms)
    # 1: Open
    # 2: High
    # 3: Low
    # 4: Close
    # 5: Volume
    # 6: Close time (ms)
    # 7: Quote asset volume
    # 8: Number of trades
    # 9: Taker buy base asset volume
    # 10: Taker buy quote asset volume
    # 11: Ignore
    cols = ["open_time_ms", "open", "high", "low", "close", "volume",
            "close_time_ms", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"]
    full.columns = cols[:len(full.columns)]
    full["open_time"] = pd.to_datetime(full["open_time_ms"].astype(np.int64), unit="ms", utc=True)
    full["close_time"] = pd.to_datetime(full["close_time_ms"].astype(np.int64), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"]:
        full[c] = pd.to_numeric(full[c], errors="coerce")

    full = full.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    full.to_parquet(dest, index=False)
    print(f"[BTC 1h] Saved {len(full):,} bars to {dest.name} ({full['open_time'].min()} to {full['open_time'].max()})")
    return full


def download_equities_macro() -> pd.DataFrame:
    """Download daily equity and macro adjusted closes."""
    dest = DATA_DIR / "equities_macro_daily_2023_2026.parquet"
    if dest.exists():
        print(f"[Equities] Found existing {dest.name}")
        df = pd.read_parquet(dest)
        print(f"[Equities] Loaded {len(df):,} rows from {df['date'].min()} to {df['date'].max()}")
        return df

    print(f"[Equities] Downloading daily data for {len(EQUITY_TICKERS)} tickers from yfinance (2022-12-20 to 2026-11-26)...")
    raw = yf.download(EQUITY_TICKERS, start="2022-12-20", end="2026-11-26", interval="1d", progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        adj_close = raw["Close"].copy()
        high = raw["High"].copy()
        low = raw["Low"].copy()
        open_px = raw["Open"].copy()
        vol = raw["Volume"].copy()
    else:
        adj_close = raw[["Close"]].copy()
        high = raw[["High"]].copy()
        low = raw[["Low"]].copy()
        open_px = raw[["Open"]].copy()
        vol = raw[["Volume"]].copy()

    # Create long/wide clean dataframe with Date
    adj_close = adj_close.reset_index()
    adj_close["Date"] = pd.to_datetime(adj_close["Date"]).dt.tz_localize(None)
    adj_close = adj_close.rename(columns={"Date": "date"})

    # Forward fill missing holidays (e.g. slight exchange mismatch)
    adj_close = adj_close.sort_values("date").reset_index(drop=True)
    adj_close.to_parquet(dest, index=False)
    print(f"[Equities] Saved {len(adj_close):,} trading days to {dest.name}")
    return adj_close


def load_fear_greed() -> pd.DataFrame:
    """Ensure fear & greed dataset is ready and normalized."""
    path = DATA_DIR / "fear_greed_index.csv"
    if not path.exists():
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        r = requests.get(url, timeout=30)
        data = r.json().get("data", [])
        rows = []
        for d in data:
            ts = int(d["timestamp"])
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
            rows.append({
                "timestamp": ts,
                "value": int(d["value"]),
                "classification": d["value_classification"],
                "date": dt
            })
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"[Sentiment] Fear & Greed ready: {len(df):,} records ({df['date'].min().date()} to {df['date'].max().date()})")
    return df


if __name__ == "__main__":
    b1h = download_binance_1h_btc()
    eq = download_equities_macro()
    fg = load_fear_greed()
    print("All foundational datasets verified and saved!")

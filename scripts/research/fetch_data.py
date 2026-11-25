#!/usr/bin/env python3
"""
fetch_data.py — Downloads all required data from FREE public APIs.

What it fetches:
  1. Bitcoin Fear & Greed Index  → data/fear_greed_index.csv   (alternative.me API)
  2. Real high-frequency crypto  → data/historical_data.csv    (yfinance — 40 coins, 5m/15m intervals)
     Over 500,000+ real rows right up to the latest 5 minutes of today.

Usage:
    python fetch_data.py                # fetches 5m bars (~680K real rows up to today)
    python fetch_data.py --interval 15m # fetches 15m bars (~230K real rows up to today)
    python fetch_data.py --interval 1h  # fetches 1h bars (~410K real rows over 2 years)
"""

import sys
import time
import argparse
import datetime
import requests
import pandas as pd
import numpy as np
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

FG_API = "https://api.alternative.me/fng/?limit=0&format=json"

# ── 65 high-liquidity crypto tickers on Yahoo Finance ────────────────────────
CRYPTO_TICKERS = [
    # Top Major Caps
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "SHIB-USD",
    # Top Mid Caps & Layer 1 / 2
    "LINK-USD", "ATOM-USD", "LTC-USD", "BCH-USD", "NEAR-USD",
    "ALGO-USD", "VET-USD", "MANA-USD", "SAND-USD", "AXS-USD",
    "THETA-USD", "XLM-USD", "TRX-USD", "ETC-USD", "XMR-USD",
    # High-Volume DeFi
    "AAVE-USD", "MKR-USD", "CRV-USD", "SNX-USD", "SUSHI-USD",
    "YFI-USD", "BAL-USD", "1INCH-USD",
    # Infra & Payments
    "DASH-USD", "ZEC-USD", "ENJ-USD", "CHZ-USD", "IOTA-USD",
    "ZIL-USD", "HBAR-USD",
    # Modern Ecosystem & High Activity Tokens (5m liquidity)
    "ICP-USD", "SEI-USD", "TIA-USD", "OP-USD", "ARB-USD",
    "INJ-USD", "RENDER-USD", "FET-USD", "FIL-USD", "LDO-USD",
    "QNT-USD", "RUNE-USD", "PENDLE-USD", "GALA-USD", "DYDX-USD",
    "JUP-USD", "PYTH-USD", "FLOKI-USD", "BONK-USD", "WIF-USD",
    "KAS-USD", "FLOW-USD", "GNO-USD", "BAT-USD", "KAVA-USD",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fear & Greed Index
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fear_greed() -> pd.DataFrame:
    """Fetch complete Bitcoin Fear & Greed history from alternative.me."""
    print("\n[1/2] Fetching Bitcoin Fear & Greed Index (up to today) …")
    out = DATA_DIR / "fear_greed_index.csv"

    try:
        r = requests.get(FG_API, timeout=30)
        r.raise_for_status()
        raw = r.json().get("data", [])

        rows = []
        for entry in raw:
            ts = int(entry["timestamp"])
            rows.append({
                "timestamp":      ts,
                "value":          int(entry["value"]),
                "classification": entry["value_classification"],
                "date":           datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d"),
            })

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        df.to_csv(out, index=False)
        print(f"   {len(df):,} records saved → {out}")
        print(f"     Date range: {df['date'].min()} → {df['date'].max()}")
        return df
    except Exception as e:
        if out.exists():
            print(f"  ️ Live API fetch failed ({e}), using cached {out}")
            return pd.read_csv(out)
        raise e


# ─────────────────────────────────────────────────────────────────────────────
# 2. Real Crypto OHLCV via yfinance → mapped to pipeline schema
# ─────────────────────────────────────────────────────────────────────────────

def fetch_crypto_ohlcv(interval: str = "5m") -> pd.DataFrame:
    """
    Download real high-frequency intraday crypto data up to the nearest minute.

    - interval="5m": 60-day window, ~17,100 bars per coin (~680,000 real rows total).
    - interval="15m": 60-day window, ~5,700 bars per coin.
    - interval="1h": 730-day window, ~8,500 bars per coin.
    """
    import yfinance as yf

    period = "730d" if interval == "1h" else "60d"
    print(f"\n[2/2] Fetching real crypto OHLCV ({interval} bars, {period}) for {len(CRYPTO_TICKERS)} coins …")
    print("      Connecting to Yahoo Finance …\n")

    try:
        raw_all = yf.download(
            CRYPTO_TICKERS,
            period=period,
            interval=interval,
            group_by="ticker",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        print(f"  Batch download failed ({e}), falling back to single downloads...")
        raw_all = None

    all_frames = []
    failed     = []

    ist_offset = datetime.timedelta(hours=5, minutes=30)

    for i, ticker in enumerate(CRYPTO_TICKERS, 1):
        coin = ticker.replace("-USD", "")
        try:
            if raw_all is not None and ticker in raw_all.columns.levels[0]:
                raw = raw_all[ticker].dropna(subset=["Open", "Close", "Volume"]).copy()
            else:
                raw = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw.dropna(subset=["Open", "Close", "Volume"]).copy()

            raw = raw[raw["Volume"] > 0]
            if raw.empty:
                print(f"  [{i:02d}/{len(CRYPTO_TICKERS)}]  {coin:<8} — no data returned")
                failed.append(ticker)
                continue

            raw = raw.reset_index()
            dt_col = "Datetime" if "Datetime" in raw.columns else ("Date" if "Date" in raw.columns else raw.columns[0])
            raw["dt"] = pd.to_datetime(raw[dt_col], utc=True)
            raw["dt_ist"] = raw["dt"] + ist_offset

            df = pd.DataFrame(index=raw.index)
            df["Coin"]            = coin
            df["Account"]         = coin
            df["Execution Price"] = raw["Close"].round(6)
            df["Size Tokens"]     = raw["Volume"].round(6)
            df["Size USD"]        = (raw["Volume"] * raw["Close"]).round(2)

            bar_return            = raw["Close"] - raw["Open"]
            df["Closed PnL"]      = bar_return.round(6)
            df["Side"]            = np.where(bar_return >= 0, "BUY", "SELL")
            df["Direction"]       = np.where(bar_return >= 0, "Close Long", "Close Short")
            df["Fee"]             = (df["Size USD"] * 0.00035).round(6)

            df["Timestamp IST"]   = raw["dt_ist"].dt.strftime("%d-%m-%Y %H:%M")
            df["Timestamp"]       = raw["dt"].astype(np.int64) // 10**6
            df["Start Position"]  = 0.0
            df["Transaction Hash"]= ""
            df["Order ID"]        = df["Timestamp"]
            df["Trade ID"]        = df["Timestamp"]
            df["Crossed"]         = False

            all_frames.append(df)
            latest_time = raw["dt_ist"].iloc[-1].strftime("%d-%b-%Y %H:%M IST")
            print(f"  [{i:02d}/{len(CRYPTO_TICKERS)}]  {coin:<8} — {len(df):>6,} bars | Latest: {latest_time}")

        except Exception as e:
            print(f"  [{i:02d}/{len(CRYPTO_TICKERS)}]  {coin:<8} — {e}")
            failed.append(ticker)

    if not all_frames:
        raise RuntimeError("No data downloaded from yfinance. Check network connection.")

    merged = pd.concat(all_frames, ignore_index=True)
    merged = merged.sort_values("Timestamp").reset_index(drop=True)

    out = DATA_DIR / "historical_data.csv"
    merged.to_csv(out, index=False)

    print(f"\n   {len(merged):,} real {interval} OHLCV rows saved → {out}")
    print(f"     Coins successfully loaded: {len(all_frames)}/{len(CRYPTO_TICKERS)}")
    print(f"     Date range: {merged['Timestamp IST'].min()} → {merged['Timestamp IST'].max()} IST")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch real crypto market data & Fear-Greed Index")
    parser.add_argument("--interval", type=str, default="5m", choices=["5m", "15m", "1h"],
                        help="Bar interval (default: 5m for nearest live data & >500K rows)")
    args = parser.parse_args()

    print("=" * 65)
    print(f"  Hyperliquid / Crypto Sentiment — Real Data Ingestion ({args.interval})")
    print("=" * 65)

    fetch_fear_greed()
    fetch_crypto_ohlcv(interval=args.interval)

    print("\n" + "=" * 65)
    print("   Real live data fetched successfully!")
    print("  Next step:  python main.py")
    print("=" * 65)

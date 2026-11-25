#!/usr/bin/env python3
"""
btc_pipeline/live_forward_logger.py
Automated Multi-Venue Live Logging & Microstructure Attribution Engine:
Connects to:
1. Binance Spot REST API (BTC/USDT 5m klines, taker buy volume)
2. Binance USD-M Futures REST API (BTC/USDT mark price, volume)
3. Hyperliquid L1/L2 Mainnet API (BTC candleSnapshot, open interest, funding rate, trades)
4. Alternative.me Fear & Greed Sentiment API

Appends unified, deduplicated 5-minute snapshots to data/live_forward_log.parquet
for ongoing out-of-sample live performance tracking.
"""

import os
import sys
import json
import urllib.request
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_PARQUET = DATA_DIR / "live_forward_log.parquet"


def fetch_binance_spot_klines(limit: int = 30) -> List[dict]:
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = json.loads(r.read().decode())
    
    records = []
    for k in raw:
        ot = datetime.datetime.fromtimestamp(k[0]/1000, tz=datetime.timezone.utc)
        ct = datetime.datetime.fromtimestamp(k[6]/1000, tz=datetime.timezone.utc)
        v = float(k[5])
        tbv = float(k[9])
        tsv = max(0.0, v - tbv)
        records.append({
            "timestamp_utc": ot.strftime("%Y-%m-%d %H:%M:%S"),
            "open_time_ms": int(k[0]),
            "close_time_ms": int(k[6]),
            "spot_open": float(k[1]),
            "spot_high": float(k[2]),
            "spot_low": float(k[3]),
            "spot_close": float(k[4]),
            "spot_volume_btc": v,
            "spot_taker_buy_btc": tbv,
            "spot_taker_sell_btc": tsv,
            "spot_net_ofi_btc": tbv - tsv,
            "spot_taker_buy_ratio": tbv / v if v > 0 else 0.5
        })
    return records


def fetch_binance_futures_klines(limit: int = 30) -> Dict[str, dict]:
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&limit={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = json.loads(r.read().decode())
        
        fut_dict = {}
        for k in raw:
            ot = datetime.datetime.fromtimestamp(k[0]/1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            fut_dict[ot] = {
                "binance_fut_open": float(k[1]),
                "binance_fut_high": float(k[2]),
                "binance_fut_low": float(k[3]),
                "binance_fut_close": float(k[4]),
                "binance_fut_volume_btc": float(k[5])
            }
        return fut_dict
    except Exception as e:
        print(f"Warning: Binance Futures fetch failed ({e}), continuing with defaults.")
        return {}


def fetch_hyperliquid_candles_and_meta(start_dt: datetime.datetime) -> Dict[str, dict]:
    url = "https://api.hyperliquid.xyz/info"
    
    # 1. Candles
    payload_candles = json.dumps({
        "type": "candleSnapshot",
        "req": {
            "coin": "BTC",
            "interval": "5m",
            "startTime": int(start_dt.timestamp() * 1000)
        }
    }).encode()
    req_candles = urllib.request.Request(url, data=payload_candles, headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req_candles, timeout=10) as r:
        hl_candles = json.loads(r.read().decode())

    # 2. Perpetual metadata (Open Interest, Funding Rate, 24h Volume)
    payload_meta = json.dumps({"type": "metaAndAssetCtxs"}).encode()
    req_meta = urllib.request.Request(url, data=payload_meta, headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    oi_coins, funding_hr, day_vol = np.nan, np.nan, np.nan
    try:
        with urllib.request.urlopen(req_meta, timeout=10) as r:
            meta_ctx = json.loads(r.read().decode())
            universe = meta_ctx[0]["universe"]
            ctxs = meta_ctx[1]
            for i, u in enumerate(universe):
                if u["name"] == "BTC":
                    oi_coins = float(ctxs[i]["openInterest"])
                    funding_hr = float(ctxs[i]["funding"]) * 100.0
                    day_vol = float(ctxs[i]["dayNtlVlm"])
                    break
    except Exception as e:
        print(f"Warning: Hyperliquid metadata fetch failed ({e})")

    hl_dict = {}
    for c in hl_candles:
        ot = datetime.datetime.fromtimestamp(c["t"]/1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        hl_dict[ot] = {
            "hl_open": float(c["o"]),
            "hl_high": float(c["h"]),
            "hl_low": float(c["l"]),
            "hl_close": float(c["c"]),
            "hl_volume_btc": float(c["v"]),
            "hl_trades": int(c["n"]),
            "hl_open_interest_btc": oi_coins,
            "hl_funding_rate_hourly_pct": funding_hr,
            "hl_24h_notional_vol_usd": day_vol
        }
    return hl_dict


def fetch_fear_and_greed() -> dict:
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())["data"][0]
            return {
                "fng_value": int(data["value"]),
                "fng_classification": str(data["value_classification"])
            }
    except Exception as e:
        print(f"Warning: FNG fetch failed ({e})")
        return {"fng_value": 13, "fng_classification": "Extreme Fear"}


def assemble_multi_venue_snapshot(limit: int = 30) -> pd.DataFrame:
    """Fetches, merges, and derives cross-venue microstructure metrics."""
    spot_records = fetch_binance_spot_klines(limit=limit)
    if not spot_records:
        raise ValueError("Failed to fetch Binance Spot klines.")

    earliest_time = datetime.datetime.strptime(spot_records[0]["timestamp_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    fut_data = fetch_binance_futures_klines(limit=limit)
    hl_data = fetch_hyperliquid_candles_and_meta(earliest_time)
    fng = fetch_fear_and_greed()

    rows = []
    for s in spot_records:
        t = s["timestamp_utc"]
        fut = fut_data.get(t, {})
        hl = hl_data.get(t, {})

        row = dict(s)
        # Add Futures
        row["binance_fut_close"] = fut.get("binance_fut_close", np.nan)
        row["binance_fut_volume_btc"] = fut.get("binance_fut_volume_btc", np.nan)

        # Add Hyperliquid
        row["hl_open"] = hl.get("hl_open", np.nan)
        row["hl_high"] = hl.get("hl_high", np.nan)
        row["hl_low"] = hl.get("hl_low", np.nan)
        row["hl_close"] = hl.get("hl_close", np.nan)
        row["hl_volume_btc"] = hl.get("hl_volume_btc", np.nan)
        row["hl_trades"] = hl.get("hl_trades", 0)
        row["hl_open_interest_btc"] = hl.get("hl_open_interest_btc", np.nan)
        row["hl_funding_rate_hourly_pct"] = hl.get("hl_funding_rate_hourly_pct", np.nan)

        # Cross-venue derived basis
        p_spot = s["spot_close"]
        p_hl = row["hl_close"]
        p_fut = row["binance_fut_close"]

        if np.isfinite(p_hl) and p_spot > 0:
            row["basis_hl_spot_usd"] = p_hl - p_spot
            row["basis_hl_spot_bps"] = (p_hl - p_spot) / p_spot * 10000.0
            row["hl_to_spot_vol_ratio"] = row["hl_volume_btc"] / s["spot_volume_btc"] if s["spot_volume_btc"] > 0 else 1.0
        else:
            row["basis_hl_spot_usd"] = np.nan
            row["basis_hl_spot_bps"] = np.nan
            row["hl_to_spot_vol_ratio"] = np.nan

        if np.isfinite(p_fut) and p_spot > 0:
            row["basis_binance_fut_spot_usd"] = p_fut - p_spot
            row["basis_binance_fut_spot_bps"] = (p_fut - p_spot) / p_spot * 10000.0
        else:
            row["basis_binance_fut_spot_usd"] = np.nan
            row["basis_binance_fut_spot_bps"] = np.nan

        # Sentiment
        row["fng_value"] = fng["fng_value"]
        row["fng_classification"] = fng["fng_classification"]

        # Realized log return and range
        row["spot_log_ret"] = np.log(s["spot_close"] / s["spot_open"])
        row["spot_range_pct"] = (s["spot_high"] - s["spot_low"]) / s["spot_low"] * 100.0

        rows.append(row)

    df_new = pd.DataFrame(rows)
    return df_new


def update_live_forward_log(limit: int = 30) -> pd.DataFrame:
    """Updates data/live_forward_log.parquet idempotently (upserts by timestamp_utc)."""
    df_new = assemble_multi_venue_snapshot(limit=limit)

    if LOG_PARQUET.exists():
        df_old = pd.read_parquet(LOG_PARQUET)
        # Upsert: keep old records, replace/append with fresh snapshot
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=["timestamp_utc"], keep="last")
        df_merged = df_merged.sort_values("timestamp_utc").reset_index(drop=True)
    else:
        df_merged = df_new.sort_values("timestamp_utc").reset_index(drop=True)

    df_merged.to_parquet(LOG_PARQUET, index=False)
    print(f"Updated {LOG_PARQUET} (Total Snapshots: {len(df_merged)})")
    print(f"  Range: {df_merged['timestamp_utc'].iloc[0]} -> {df_merged['timestamp_utc'].iloc[-1]}")
    return df_merged


if __name__ == "__main__":
    n_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    update_live_forward_log(limit=n_limit)

#!/usr/bin/env python3
"""
feature_engineering.py - Single-Definition Clean Feature Engineering Pipeline.
All daily slices are aligned strictly to the US equity close (21:00 UTC).

Fixes applied:
1. Uniform high-frequency RV across all rows (no 1s override). Supports 1h and 5m RV.
2. Dropped raw price levels for 24 macro tickers (keeping only stationary daily returns).
3. Dropped duplicate columns (rv_d, rv_ann_d, range_pct_lag0, log_ret_24h, ofi_lag0, bv_d, jump_d, rel_jump_d).
4. Continuous crypto rolling windows (rv_7d, rv_30d, rv_ann_7d, rv_ann_30d).
5. Fixed target_dir_next to be NaN on the last unclosed day.
6. Computed rolling equity correlations on TRADING DAYS ONLY and forward-filled across weekends.
"""

import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_raw_datasets():
    """Load raw klines, equity prices, and fear & greed sentiment."""
    b1h_path = DATA_DIR / "btc_1h_2023_2026.parquet"
    eq_path = DATA_DIR / "equities_macro_daily_2023_2026.parquet"
    fg_path = DATA_DIR / "fear_greed_index.csv"

    b1h = pd.read_parquet(b1h_path)
    eq = pd.read_parquet(eq_path)
    fg = pd.read_csv(fg_path)

    return b1h, eq, fg


def load_raw_5m_dataset():
    """Load 5m klines dataset."""
    b5m_path = DATA_DIR / "btc_5m_2023_2026.parquet"
    b5m = pd.read_parquet(b5m_path)
    return b5m


def build_daily_btc_aggregates(b1h: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1h BTC data to 21:00 UTC equity close cutoff.
    Each day t represents [t-1 21:00:00 UTC, t 20:59:59 UTC].
    Uniform definition: 100% hourly-sampled realized variance across the entire history.
    """
    df = b1h.copy()
    mask_us = df["open_time_ms"] > 1e14
    if mask_us.any():
        df.loc[mask_us, "open_time_ms"] = df.loc[mask_us, "open_time_ms"] // 1000
        df.loc[df["close_time_ms"] > 1e14, "close_time_ms"] = df.loc[df["close_time_ms"] > 1e14, "close_time_ms"] // 1000
        df["open_time"] = pd.to_datetime(df["open_time_ms"].astype("int64"), unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time_ms"].astype("int64"), unit="ms", utc=True)

    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # 24-hour trading session ending at US equity close (21:00 UTC on date t).
    df["cutoff_date"] = (df["open_time"] + pd.Timedelta(hours=3)).dt.date

    def agg_day(sub):
        r = sub["log_ret"].dropna().values
        if len(r) < 20:
            return None
        rv = np.sum(r**2)
        # Bipower Variation
        bv = (np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])) if len(r) > 1 else rv
        # Jump component
        jump = max(0.0, rv - bv)
        rel_jump = jump / (rv + 1e-12)

        high_max = sub["high"].max()
        low_min = sub["low"].min()
        close_last = sub["close"].iloc[-1]
        open_first = sub["open"].iloc[0]
        range_pct = (high_max - low_min) / (low_min + 1e-12) * 100.0
        daily_ret = (close_last / open_first - 1.0) * 100.0

        vol = sub["volume"].sum()
        trades = sub["trades"].sum()
        taker_vol = sub["taker_buy_volume"].sum()
        ofi = (2 * taker_vol - vol) / (vol + 1e-12)

        return pd.Series({
            "rv": rv,
            "rv_ann": np.sqrt(rv * 365.0),
            "bv": bv,
            "jump": jump,
            "rel_jump": rel_jump,
            "range_pct": range_pct,
            "daily_ret": daily_ret,
            "open": open_first,
            "high": high_max,
            "low": low_min,
            "close": close_last,
            "volume": vol,
            "trades": trades,
            "ofi": ofi
        })

    daily = df.groupby("cutoff_date", group_keys=False).apply(agg_day).dropna().reset_index()
    daily["cutoff_date"] = pd.to_datetime(daily["cutoff_date"])
    return daily


def build_daily_btc_aggregates_5m(b5m: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 5m BTC data to 21:00 UTC equity close cutoff.
    Each day t represents [t-1 21:00:00 UTC, t 20:59:59 UTC] (288 bars).
    Computes 5-minute realized variance, bipower variation, and jumps.
    """
    df = b5m.copy()
    mask_us = df["open_time_ms"] > 1e14
    if mask_us.any():
        df.loc[mask_us, "open_time_ms"] = df.loc[mask_us, "open_time_ms"] // 1000
        df.loc[df["close_time_ms"] > 1e14, "close_time_ms"] = df.loc[df["close_time_ms"] > 1e14, "close_time_ms"] // 1000
        df["open_time"] = pd.to_datetime(df["open_time_ms"].astype("int64"), unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time_ms"].astype("int64"), unit="ms", utc=True)

    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["cutoff_date"] = (df["open_time"] + pd.Timedelta(hours=3)).dt.date

    def agg_day(sub):
        r = sub["log_ret"].dropna().values
        if len(r) < 200:
            return None
        rv = np.sum(r**2)
        bv = (np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])) if len(r) > 1 else rv
        jump = max(0.0, rv - bv)
        rel_jump = jump / (rv + 1e-12)

        high_max = sub["high"].max()
        low_min = sub["low"].min()
        close_last = sub["close"].iloc[-1]
        open_first = sub["open"].iloc[0]
        range_pct = (high_max - low_min) / (low_min + 1e-12) * 100.0
        daily_ret = (close_last / open_first - 1.0) * 100.0

        vol = sub["volume"].sum()
        trades = sub["trades"].sum()
        taker_vol = sub["taker_buy_volume"].sum()
        ofi = (2 * taker_vol - vol) / (vol + 1e-12)

        return pd.Series({
            "rv": rv,
            "rv_ann": np.sqrt(rv * 365.0),
            "bv": bv,
            "jump": jump,
            "rel_jump": rel_jump,
            "range_pct": range_pct,
            "daily_ret": daily_ret,
            "open": open_first,
            "high": high_max,
            "low": low_min,
            "close": close_last,
            "volume": vol,
            "trades": trades,
            "ofi": ofi
        })

    daily = df.groupby("cutoff_date", group_keys=False).apply(agg_day).dropna().reset_index()
    daily["cutoff_date"] = pd.to_datetime(daily["cutoff_date"])
    return daily


def build_full_feature_matrix(use_5m: bool = False) -> pd.DataFrame:
    """Build the final dataset aligned strictly to US equity close date t with clean features."""
    b1h, eq, fg = load_raw_datasets()
    if use_5m:
        b5m = load_raw_5m_dataset()
        daily_btc = build_daily_btc_aggregates_5m(b5m)
    else:
        daily_btc = build_daily_btc_aggregates(b1h)

    df = daily_btc.sort_values("cutoff_date").reset_index(drop=True)

    # 1. BTC Lagged Features & HAR components
    df["btc_ret_lag1"] = df["daily_ret"].shift(1)
    df["btc_ret_lag2"] = df["daily_ret"].shift(2)
    df["btc_ret_lag3"] = df["daily_ret"].shift(3)
    df["btc_ret_lag5"] = df["daily_ret"].shift(5)

    # HAR components: 5-day week, 22-day month, plus continuous 7-day crypto week and 30-day crypto month
    df["rv_w"] = df["rv"].rolling(5, min_periods=1).mean()
    df["rv_m"] = df["rv"].rolling(22, min_periods=1).mean()
    df["rv_7d"] = df["rv"].rolling(7, min_periods=1).mean()
    df["rv_30d"] = df["rv"].rolling(30, min_periods=1).mean()

    df["rv_ann_w"] = np.sqrt(df["rv_w"] * 365.0)
    df["rv_ann_m"] = np.sqrt(df["rv_m"] * 365.0)
    df["rv_ann_7d"] = np.sqrt(df["rv_7d"] * 365.0)
    df["rv_ann_30d"] = np.sqrt(df["rv_30d"] * 365.0)

    # Range and volume lags
    df["range_pct_lag1"] = df["range_pct"].shift(1)
    df["range_pct_lag2"] = df["range_pct"].shift(2)
    df["range_pct_lag3"] = df["range_pct"].shift(3)

    df["volume_log"] = np.log(df["volume"] + 1.0)
    df["volume_chg_1d"] = df["volume"].pct_change() * 100.0
    df["trades_chg_1d"] = df["trades"].pct_change() * 100.0

    # 2. Sentiment: Fear & Greed Index
    fg_clean = fg.copy()
    fg_clean["date"] = pd.to_datetime(fg_clean["timestamp"].astype("int64"), unit="s").dt.date
    fg_clean["date"] = pd.to_datetime(fg_clean["date"])
    fg_clean = fg_clean.drop_duplicates("date").sort_values("date")
    fg_clean["fng_val"] = fg_clean["value"]
    fg_clean["fng_diff1"] = fg_clean["fng_val"].diff(1)

    df = df.merge(fg_clean[["date", "fng_val", "fng_diff1"]], left_on="cutoff_date", right_on="date", how="left")
    df = df.drop(columns=["date"])
    df["fng_val"] = df["fng_val"].ffill()
    df["fng_diff1"] = df["fng_diff1"].fillna(0.0)

    # 3. Equities Macro Alignment (Returns only, dropping non-stationary price levels)
    eq_clean = eq.copy()
    eq_clean["date"] = pd.to_datetime(eq_clean["date"])
    eq_clean = eq_clean.drop_duplicates("date").sort_values("date")

    eq_tickers = [c for c in eq_clean.columns if c != "date"]
    eq_ret_df = pd.DataFrame({"date": eq_clean["date"]})
    for tick in eq_tickers:
        eq_ret_df[f"{tick}_ret"] = eq_clean[tick].pct_change() * 100.0

    eq_ret_df["equity_is_trading_day"] = 1.0

    df = df.merge(eq_ret_df, left_on="cutoff_date", right_on="date", how="left")
    df = df.drop(columns=["date"])

    # Forward fill flags; on non-trading days return is 0.0
    df["equity_is_stale"] = df["equity_is_trading_day"].isna().astype(float)
    df["equity_is_trading_day"] = df["equity_is_trading_day"].fillna(0.0)

    for tick in eq_tickers:
        df[f"{tick}_ret"] = df[f"{tick}_ret"].fillna(0.0)

    # 4. Rolling 20-day Correlations on TRADING DAYS ONLY
    trading_mask = df["equity_is_trading_day"] == 1.0
    trading_slice = df.loc[trading_mask, ["cutoff_date", "daily_ret", "QQQ_ret", "COIN_ret", "TLT_ret"]].copy()
    trading_slice["roll20_corr_qqq"] = trading_slice["daily_ret"].rolling(20, min_periods=15).corr(trading_slice["QQQ_ret"])
    trading_slice["roll20_corr_coin"] = trading_slice["daily_ret"].rolling(20, min_periods=15).corr(trading_slice["COIN_ret"])
    trading_slice["roll20_corr_tlt"] = trading_slice["daily_ret"].rolling(20, min_periods=15).corr(trading_slice["TLT_ret"])

    df = df.merge(trading_slice[["cutoff_date", "roll20_corr_qqq", "roll20_corr_coin", "roll20_corr_tlt"]], on="cutoff_date", how="left")
    df["roll20_corr_qqq"] = df["roll20_corr_qqq"].ffill().fillna(0.0)
    df["roll20_corr_coin"] = df["roll20_corr_coin"].ffill().fillna(0.0)
    df["roll20_corr_tlt"] = df["roll20_corr_tlt"].ffill().fillna(0.0)

    # 5. Calendar Dummies
    df["dow"] = df["cutoff_date"].dt.day_of_week
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(float)
    df["is_friday"] = (df["dow"] == 4).astype(float)
    df["is_sunday"] = (df["dow"] == 6).astype(float)
    df["is_monday"] = (df["dow"] == 0).astype(float)

    is_thursday = df["dow"] == 3
    is_nov = df["cutoff_date"].dt.month == 11
    day_num = df["cutoff_date"].dt.day
    df["is_thanksgiving"] = (is_nov & is_thursday & (day_num >= 22) & (day_num <= 28)).astype(float)
    df["is_black_friday"] = (is_nov & (df["dow"] == 4) & (day_num >= 23) & (day_num <= 29)).astype(float)

    # 6. Targets Construction
    df["target_rv_ann_next"] = df["rv_ann"].shift(-1)
    df["target_rv_next"] = df["rv"].shift(-1)
    df["target_range_next"] = df["range_pct"].shift(-1)
    
    # Direction target: strictly NaN on the last row
    dir_target = (df["close"].shift(-1) > df["close"]).astype(float)
    dir_target.iloc[-1] = np.nan
    df["target_dir_next"] = dir_target

    # Weekend-to-Monday return signal
    b1h_sorted = b1h.sort_values("open_time").reset_index(drop=True)
    b1h_sorted["open_time"] = pd.to_datetime(b1h_sorted["open_time"], utc=True)
    fri_dates = df[df["is_friday"] == 1.0]["cutoff_date"]
    weekend_signals = {}
    for f_date in fri_dates:
        m_date = f_date + pd.Timedelta(days=3)
        sub_f = b1h_sorted[b1h_sorted["open_time"] == pd.Timestamp(f_date).tz_localize("UTC") + pd.Timedelta(hours=20)]
        sub_m = b1h_sorted[b1h_sorted["open_time"] == pd.Timestamp(m_date).tz_localize("UTC") + pd.Timedelta(hours=14)]
        if not sub_f.empty and not sub_m.empty:
            p_f = sub_f["close"].iloc[0]
            p_m = sub_m["close"].iloc[0]
            weekend_signals[f_date] = (p_m / p_f - 1.0) * 100.0

    df["target_weekend_to_mon"] = df["cutoff_date"].map(weekend_signals)

    # Drop warm-up window where 30-day and 22-day rolling features are undefined
    df = df.dropna(subset=["rv_30d", "rv_m", "btc_ret_lag5"]).reset_index(drop=True)

    # Save clean dataset
    fname = "engineered_features_5m.parquet" if use_5m else "engineered_features_master.parquet"
    out_path = DATA_DIR / fname
    df.to_parquet(out_path, index=False)
    print(f"[Feature Engineering] Dataset built (5m={use_5m}): {len(df):,} days from {df['cutoff_date'].min().date()} to {df['cutoff_date'].max().date()}")
    print(f"[Feature Engineering] Saved to {out_path.name}")
    return df


if __name__ == "__main__":
    print("Building Hourly RV Master Dataset...")
    df_1h = build_full_feature_matrix(use_5m=False)
    print("Building 5-Minute RV Master Dataset...")
    df_5m = build_full_feature_matrix(use_5m=True)
    print("Done! 1h rows:", len(df_1h), "5m rows:", len(df_5m))

import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PARQUET = DATA_DIR / "live_forward_log.parquet"

def test_live_forward_log_structure():
    assert LOG_PARQUET.exists(), f"Log file missing: {LOG_PARQUET}"
    df = pd.read_parquet(LOG_PARQUET)
    assert len(df) >= 16, f"Expected >= 16 bars in live log, got {len(df)}"
    
    # Check essential cross-venue columns
    required_cols = [
        "timestamp_utc", "spot_close", "spot_volume_btc", "spot_net_ofi_btc",
        "hl_close", "hl_volume_btc", "basis_hl_spot_usd", "basis_hl_spot_bps",
        "fng_value"
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing required column: {c}"
        
    # Check monotonic timestamps and uniqueness
    assert df["timestamp_utc"].is_monotonic_increasing, "Timestamps must be strictly ascending"
    assert df["timestamp_utc"].nunique() == len(df), "No duplicate timestamps allowed in live log"
    
    # Check validity of prices and basis
    assert (df["spot_close"] > 0).all(), "Spot prices must be strictly positive"
    assert (df["hl_close"].dropna() > 0).all(), "Hyperliquid prices must be strictly positive"

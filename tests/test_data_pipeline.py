"""
test_data_pipeline.py

Unit tests for data partitioning, temporal integrity, and schema compliance
produced by src/data.py and btc_pipeline/live_forward_logger.py.
"""

from pathlib import Path
import pandas as pd
import pytest


def test_dataset_split_files_exist():
    """Verify that all three canonical dataset partitions exist on disk."""
    assert Path("data/train_split.parquet").exists(), "Missing data/train_split.parquet"
    assert Path("data/val_split.parquet").exists(), "Missing data/val_split.parquet"
    assert Path("data/holdout_test_split.parquet").exists(), "Missing data/holdout_test_split.parquet"


def test_split_temporal_monotonicity_and_no_leakage():
    """Verify strict temporal ordering with zero date overlaps across splits."""
    df_train = pd.read_parquet("data/train_split.parquet")
    df_val = pd.read_parquet("data/val_split.parquet")
    df_holdout = pd.read_parquet("data/holdout_test_split.parquet")

    # Temporal bounds
    max_train_date = pd.to_datetime(df_train["cutoff_date"]).max()
    min_val_date = pd.to_datetime(df_val["cutoff_date"]).min()
    max_val_date = pd.to_datetime(df_val["cutoff_date"]).max()
    min_holdout_date = pd.to_datetime(df_holdout["cutoff_date"]).min()

    assert max_train_date < min_val_date, (
        f"Train/Val date overlap or violation: {max_train_date} >= {min_val_date}"
    )
    assert max_val_date < min_holdout_date, (
        f"Val/Holdout date overlap or violation: {max_val_date} >= {min_holdout_date}"
    )


def test_holdout_split_row_count():
    """Ensure holdout set has exactly 23 daily rows (Nov 8-30, 2025)."""
    df_holdout = pd.read_parquet("data/holdout_test_split.parquet")
    assert len(df_holdout) == 23, f"Expected 23 holdout days, got {len(df_holdout)}"


def test_live_forward_log_schema_and_monotonicity():
    """Verify live forward streaming log schema and monotonic timestamps."""
    path_live = Path("data/live_forward_log.parquet")
    assert path_live.exists(), "Missing data/live_forward_log.parquet"

    df_live = pd.read_parquet(path_live)
    assert len(df_live) >= 10, f"Expected at least 10 live snapshots, got {len(df_live)}"

    expected_cols = [
        "timestamp_utc", "spot_close", "spot_net_ofi_btc",
        "hl_close", "basis_hl_spot_bps", "hl_to_spot_vol_ratio"
    ]
    for col in expected_cols:
        assert col in df_live.columns, f"Missing expected column '{col}' in live forward log"

    # Check timestamps are sorted ascending
    ts = pd.to_datetime(df_live["timestamp_utc"])
    assert ts.is_monotonic_increasing, "Live forward log timestamps must be strictly monotonic increasing"

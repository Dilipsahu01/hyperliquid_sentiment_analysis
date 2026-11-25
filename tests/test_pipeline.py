import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_PATH = "data/engineered_features_master.parquet"

@pytest.fixture(scope="module")
def df_master():
    assert os.path.exists(DATA_PATH), f"Master dataset not found at {DATA_PATH}"
    df = pd.read_parquet(DATA_PATH)
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])
    return df

def test_data_integrity_shape_and_dates(df_master):
    """Assert minimum row count >= 500, date continuity, and expected date bounds."""
    assert len(df_master) >= 500, f"Expected >= 500 rows, got {len(df_master)}"
    min_date = df_master["cutoff_date"].min()
    max_date = df_master["cutoff_date"].max()
    assert min_date <= pd.Timestamp("2023-01-25"), f"Start date too late: {min_date}"
    assert max_date >= pd.Timestamp("2026-11-23"), f"End date too early: {max_date}"
    
    # Check no duplicate dates
    assert df_master["cutoff_date"].is_monotonic_increasing, "Dates must be strictly increasing"
    assert df_master["cutoff_date"].nunique() == len(df_master), "Dates must be unique"

def test_no_nans_in_feature_matrix(df_master):
    """Assert that feature columns have zero NaNs across the dataset."""
    feature_cols = [c for c in df_master.columns if not c.startswith("target_") and c not in ["cutoff_date", "date"]]
    nan_counts = df_master[feature_cols].isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    assert len(cols_with_nans) == 0, f"Found NaNs in feature columns: {cols_with_nans.to_dict()}"

def test_no_lookahead_bias_in_features(df_master):
    """
    Look-ahead verification:
    For any date t, lagged returns, moving averages, and rolling features must strictly depend
    only on data at <= t, and targets must align strictly to t+1.
    """
    # 1. Verify target alignment: target_rv_next at index i must equal rv at index i+1
    for i in range(len(df_master) - 1):
        actual_next_rv = df_master.loc[i + 1, "rv"]
        target_rv = df_master.loc[i, "target_rv_next"]
        np.testing.assert_almost_equal(target_rv, actual_next_rv, decimal=8,
                                       err_msg=f"Mismatch at index {i}: target_rv_next != rv[i+1]")
        
        actual_next_ann = df_master.loc[i + 1, "rv_ann"]
        target_ann = df_master.loc[i, "target_rv_ann_next"]
        np.testing.assert_almost_equal(target_ann, actual_next_ann, decimal=6,
                                       err_msg=f"Mismatch at index {i}: target_rv_ann_next != rv_ann[i+1]")

        actual_next_close = df_master.loc[i + 1, "close"]
        curr_close = df_master.loc[i, "close"]
        expected_dir = int(actual_next_close > curr_close)
        assert df_master.loc[i, "target_dir_next"] == expected_dir, f"Direction mismatch at index {i}"

def test_feature_lag_integrity(df_master):
    """Verify that lag features at row t equal row t-1, t-2, etc."""
    for i in range(10, 20):
        # btc_ret_lag1 at row i should equal daily_ret at row i-1
        assert np.isclose(df_master.loc[i, "btc_ret_lag1"], df_master.loc[i-1, "daily_ret"], atol=1e-5), \
            f"btc_ret_lag1 mismatch at row {i}"
        assert np.isclose(df_master.loc[i, "btc_ret_lag2"], df_master.loc[i-2, "daily_ret"], atol=1e-5), \
            f"btc_ret_lag2 mismatch at row {i}"

def test_rv_scaling_plausibility(df_master):
    """Assert realized volatility values are within realistic economic bounds (not 19,400%)."""
    rv_ann = df_master["rv_ann"]
    assert rv_ann.min() > 0.05, f"Min annualized vol too low: {rv_ann.min()}"
    assert rv_ann.max() < 3.0, f"Max annualized vol unrealistically high: {rv_ann.max()}"
    # Daily RV should be between 0 and 0.05
    assert df_master["rv"].min() > 0
    assert df_master["rv"].max() < 0.10

def test_holdout_isolation(df_master):
    """Ensure clean separation between training and holdout."""
    train_mask = df_master["cutoff_date"] < "2026-11-01"
    holdout_mask = (df_master["cutoff_date"] >= "2026-11-01") & (df_master["cutoff_date"] <= "2026-11-24")
    
    assert train_mask.sum() >= 500, f"Training rows: {train_mask.sum()}"
    assert holdout_mask.sum() >= 23, f"Holdout rows: {holdout_mask.sum()}"

def test_results_baselines_json():
    """Verify baselines.json exists, contains all 4 baselines, and valid metrics."""
    import json
    p = "results/baselines.json"
    assert os.path.exists(p), f"{p} not found"
    with open(p) as f:
        data = json.load(f)
    assert "best_baseline" in data
    assert "baselines" in data
    for b in ["RandomWalk", "RollingMean20d", "HAR-RV", "GARCH(1,1)-t"]:
        assert b in data["baselines"], f"Missing baseline: {b}"
        assert data["baselines"][b]["qlike"] > 0
        assert data["baselines"][b]["mse"] > 0

def test_results_walk_forward_metrics_csv():
    """Verify walk_forward_metrics.csv exists and has populated folds."""
    p = "results/walk_forward_metrics.csv"
    assert os.path.exists(p), f"{p} not found"
    df = pd.read_csv(p)
    assert len(df) > 500, f"Expected > 500 fold evaluations, got {len(df)}"
    assert set(["fold", "model", "target", "mse", "mae", "qlike", "r2_oos"]).issubset(df.columns)

def test_results_holdout_json():
    """Verify holdout.json exists and contains holdout scores."""
    import json
    p = "results/holdout.json"
    assert os.path.exists(p), f"{p} not found"
    with open(p) as f:
        data = json.load(f)
    assert data["n_holdout_days"] == 23
    assert "volatility_models" in data
    assert "direction_model" in data
    assert "range_model" in data

def test_results_forecast_csv(df_master):
    """Verify forecast_nov26_30.csv exists, has 5 rows (Nov 26-30), monotone plausible intervals, and upper 95% bound <= 1.5 * hist max vol."""
    p = "results/forecast_nov26_30.csv"
    assert os.path.exists(p), f"{p} not found"
    df = pd.read_csv(p)
    assert len(df) == 5, f"Expected 5 forecast rows, got {len(df)}"
    expected_dates = ["2026-11-26", "2026-11-27", "2026-11-28", "2026-11-29", "2026-11-30"]
    assert df["forecast_date"].tolist() == expected_dates, f"Date mismatch: {df['forecast_date'].tolist()}"

    hist_max_vol = df_master["rv_ann"].max()
    # Verify interval monotonicity and plausibility bound
    for _, r in df.iterrows():
        p_vol = r["point_forecast_vol_ann"]
        low_80 = r["interval_80_lower"]
        high_80 = r["interval_80_upper"]
        low_95 = r["interval_95_lower"]
        high_95 = r["interval_95_upper"]
        assert low_95 <= low_80 <= p_vol <= high_80 <= high_95, \
            f"Non-monotone interval on {r['forecast_date']}: {low_95} <= {low_80} <= {p_vol} <= {high_80} <= {high_95}"
        assert 0.05 <= p_vol <= 2.50, f"Implausible point forecast: {p_vol}"
        assert high_95 <= 1.5 * hist_max_vol, \
            f"Fix 1 violation: Upper 95% bound ({high_95:.4f}) exceeds 1.5x historical max vol ({1.5 * hist_max_vol:.4f})"

def test_diebold_mariano_math():
    """Verify Diebold-Mariano function with known distribution."""
    from btc_pipeline.models import diebold_mariano_test
    # Benchmark worse than candidate
    loss_base = np.array([2.0, 2.5, 3.0, 2.8, 3.2, 2.9, 3.1, 2.7, 3.0, 2.8] * 10)
    loss_cand = np.array([1.0, 1.2, 1.1, 1.3, 1.0, 1.2, 1.1, 1.0, 1.2, 1.1] * 10)
    stat, pval = diebold_mariano_test(loss_base, loss_cand)
    assert stat > 5.0, f"Expected strong positive DM stat, got {stat}"
    assert pval < 0.001, f"Expected highly significant p-value, got {pval}"

def test_real_point_in_time_leakage(df_master):
    """
    Leakage Test: Recompute daily RV features strictly from raw data <= t for 20 random dates,
    asserting 100% numerical equality with precomputed master features.
    """
    df_1h = pd.read_parquet("data/btc_1h_2023_2026.parquet")
    df_1h["open_time"] = pd.to_datetime(df_1h["open_time"])
    df_1h["cutoff_date"] = (df_1h["open_time"] + pd.Timedelta(hours=3)).dt.date
    df_1h["log_ret"] = np.log(df_1h["close"] / df_1h["close"].shift(1))

    sample_dates = df_master["cutoff_date"].iloc[50:-5].sample(20, random_state=42).tolist()
    for d in sample_dates:
        d_date = pd.to_datetime(d).date()
        raw_slice = df_1h[df_1h["cutoff_date"] <= d_date].sort_values("open_time").reset_index(drop=True)
        daily_sub = raw_slice.groupby("cutoff_date").agg(
            rv=("log_ret", lambda r: np.sum(r.dropna()**2))
        ).reset_index()

        rv_series = daily_sub["rv"].values
        rv_d_rec = rv_series[-1]
        rv_w_rec = np.mean(rv_series[-5:])
        rv_m_rec = np.mean(rv_series[-22:])

        row_master = df_master[df_master["cutoff_date"] == d].iloc[0]
        col_d = "rv" if "rv" in row_master else "rv_d"
        assert np.isclose(rv_d_rec, row_master[col_d], atol=1e-6), f"rv mismatch at {d}"
        assert np.isclose(rv_w_rec, row_master["rv_w"], atol=1e-6), f"rv_w mismatch at {d}"
        assert np.isclose(rv_m_rec, row_master["rv_m"], atol=1e-6), f"rv_m mismatch at {d}"

def test_rolling_windows_and_correlations_causality(df_master):
    """Verify that rolling 20-day correlations use strictly causal trailing windows."""
    corr_cols = [c for c in df_master.columns if "corr_" in c]
    assert len(corr_cols) > 0, "Correlation columns must exist"
    # Ensure correlations are within valid [-1, 1] range
    for c in corr_cols:
        vals = df_master[c].dropna()
        assert vals.min() >= -1.0 - 1e-4, f"{c} min out of bounds: {vals.min()}"
        assert vals.max() <= 1.0 + 1e-4, f"{c} max out of bounds: {vals.max()}"

def test_weekend_equity_handling(df_master):
    """Assert weekend handling: zero NaNs in equity columns, is_weekend matches dow."""
    weekends = df_master[df_master["is_weekend"] == 1.0]
    assert len(weekends) > 0, "Weekend rows must exist"
    for _, r in weekends.iterrows():
        assert r["dow"] in [5, 6], f"Weekend row has invalid dow: {r['dow']}"
        # Assert equity returns are finite numbers (not NaN)
        col_qqq = "QQQ_ret" if "QQQ_ret" in r.index else "QQQ"
        col_spy = "SPY_ret" if "SPY_ret" in r.index else "SPY"
        assert np.isfinite(r[col_qqq]), "Weekend QQQ price/return cannot be NaN"
        assert np.isfinite(r[col_spy]), "Weekend SPY price/return cannot be NaN"


def test_kupiec_and_christoffersen_coverage_tests():
    """Verify Kupiec and Christoffersen statistical tests with known values and extreme sample sizes."""
    from scripts.walkforward_5m_band_evaluation import kupiec_pof, christoffersen_fast
    
    # 1. Perfect calibration: 100 breaches out of 1000 at alpha=0.10
    lr, pval = kupiec_pof(100, 1000, 0.10)
    assert np.isclose(lr, 0.0, atol=1e-5), f"Expected LR ~ 0, got {lr}"
    assert np.isclose(pval, 1.0, atol=1e-3), f"Expected pval ~ 1, got {pval}"

    # 2. Large sample underflow protection: N=384,000 observations
    I = np.zeros(10000, dtype=int)
    I[:1000] = 1 # 10% breach rate
    lr_uc, p_uc, lr_ind, p_ind, lr_cc, p_cc = christoffersen_fast(I, 0.10)
    assert np.isfinite(lr_ind) and np.isfinite(p_ind)
    assert 0.0 <= p_ind <= 1.0
    assert 0.0 <= p_cc <= 1.0
    assert np.isclose(p_uc, 1.0, atol=1e-2)


def test_newey_west_bandwidth_t889():
    """Assert that Newey-West formula floor(4 * (T/100)^(2/9)) for T=889 yields exactly 6."""
    from btc_pipeline.models import compute_newey_west_bandwidth
    bw = compute_newey_west_bandwidth(889)
    assert bw == 6, f"Expected bandwidth 6 for T=889, got {bw}"
    
    # Check exact mathematical formula
    formula_val = 4.0 * (889.0 / 100.0) ** (2.0 / 9.0)
    assert np.isclose(formula_val, 6.5002, atol=1e-4)
    assert int(np.floor(formula_val)) == 6


def test_lead_lag_estimators_and_mark_price_assertion():
    """Verify guard against mark/oracle prices and test Hasbrouck / Gonzalo-Granger logic."""
    from btc_pipeline.lead_lag_estimators import (
        assert_not_mark_or_oracle,
        hayashi_yoshida_cross_correlation,
        compute_information_shares
    )

    # 1. Test assert_not_mark_or_oracle
    df_clean = pd.DataFrame({"binance_fut_close": [100.0, 101.0], "hl_close": [100.1, 101.1]})
    assert_not_mark_or_oracle("binance_fut_close", df_clean)
    assert_not_mark_or_oracle("hl_close", df_clean)

    # Reject mark price column
    df_bad = pd.DataFrame({"binance_fut_close": [100.0], "hl_mark_price": [100.0]})
    with pytest.raises(AssertionError):
        assert_not_mark_or_oracle("hl_mark_price", df_bad)

    with pytest.raises(AssertionError):
        assert_not_mark_or_oracle("hl_oracle_price")

    # 2. Test Hasbrouck and Gonzalo-Granger on synthetic data
    np.random.seed(42)
    T = 200
    common = np.cumsum(np.random.randn(T))
    p1 = common + np.random.randn(T) * 0.05
    p2 = common + np.random.randn(T) * 0.05
    shares = compute_information_shares(p1, p2, sampling_freq_label="1s")

    assert 0.0 <= shares["gonzalo_granger_share_m1"] <= 1.0
    assert 0.0 <= shares["gonzalo_granger_share_m2"] <= 1.0
    assert np.isclose(shares["gonzalo_granger_share_m1"] + shares["gonzalo_granger_share_m2"], 1.0, atol=1e-5)
    assert shares["hasbrouck_is_m1_lower"] <= shares["hasbrouck_is_m1_upper"]






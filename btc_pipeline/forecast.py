"""
Multi-Step Probabilistic Forecasting Engine for BTC: Nov 26 - Nov 30, 2026.
Integrates calendar features:
- Nov 26: US Thanksgiving (Equities closed, thin liquidity, stale equity features)
- Nov 27: Black Friday (Half trading day)
- Nov 28: Saturday (BTC only, weekend)
- Nov 29: Sunday (BTC only, weekend)
- Nov 30: Monday (US equities reopen, weekend-to-Monday signal)

Produces results/forecast_nov26_30.csv with verified monotone prediction intervals.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from btc_pipeline.baselines import HARRVBaseline
from btc_pipeline.models import LassoModel, LightGBMVolatilityModel, DirectionClassificationModel, RangeRegressionModel, get_feature_columns

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_forecasts():
    print("=" * 70)
    print("GENERATING PROBABILISTIC FORECASTS (2026-11-26 TO 2026-11-30)")
    print("=" * 70)

    # 1. Load historical master data up to Nov 24, 2026
    df = pd.read_parquet("data/engineered_features_master.parquet")
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])

    # Load walk-forward metrics to calculate empirical residual quantiles
    wf_metrics = pd.read_csv(RESULTS_DIR / "walk_forward_metrics.csv")
    lasso_wf = wf_metrics[wf_metrics["model"] == "Lasso_a005"]
    lasso_rmse = np.sqrt(lasso_wf["mse"].mean())

    # Train winning models on full sample up to Nov 24
    print("Training winning model (Lasso_a005) and baselines on full history...")
    train_df = df[df["target_rv_ann_next"].notna()].copy()
    
    lasso = LassoModel(alpha=0.005).fit(train_df)
    har = HARRVBaseline().fit(train_df)
    dir_model = DirectionClassificationModel(C=0.1).fit(train_df)
    range_model = RangeRegressionModel(alpha=10.0).fit(train_df)

    # Calculate empirical residuals on historical data for interval calibration
    hist_preds = lasso.predict(train_df)
    hist_actuals = train_df["target_rv_ann_next"].values
    residuals = hist_actuals - hist_preds
    
    # Residual quantiles for calibrated intervals
    q10 = float(np.percentile(residuals, 10))
    q90 = float(np.percentile(residuals, 90))
    q025 = float(np.percentile(residuals, 2.5))
    q975 = float(np.percentile(residuals, 97.5))

    print(f"Empirical residual distribution (N={len(residuals)}):")
    print(f"  RMSE: {lasso_rmse:.4f}")
    print(f"  80% Interval: [{q10:+.4f}, {q90:+.4f}]")
    print(f"  95% Interval: [{q025:+.4f}, {q975:+.4f}]")

    # Step forward starting from the last observed row (2026-11-24)
    last_row = df[df["cutoff_date"] == "2026-11-24"].iloc[0].copy()
    
    # We step from Nov 24 to Nov 25 first
    # Nov 25 is Wednesday (normal day)
    # Target dates to forecast: Nov 26 to Nov 30
    calendar = [
        {"date": "2026-11-25", "day_name": "Wednesday", "equity_status": "Open", "equity_stale": 0, "dow": 2, "is_weekend": 0, "is_friday": 0, "is_sunday": 0, "is_monday": 0, "is_thanksgiving": 0, "is_black_friday": 0},
        {"date": "2026-11-26", "day_name": "Thursday (Thanksgiving)", "equity_status": "Closed", "equity_stale": 1, "dow": 3, "is_weekend": 0, "is_friday": 0, "is_sunday": 0, "is_monday": 0, "is_thanksgiving": 1, "is_black_friday": 0},
        {"date": "2026-11-27", "day_name": "Friday (Black Friday)", "equity_status": "Half Day", "equity_stale": 0, "dow": 4, "is_weekend": 0, "is_friday": 1, "is_sunday": 0, "is_monday": 0, "is_thanksgiving": 0, "is_black_friday": 1},
        {"date": "2026-11-28", "day_name": "Saturday", "equity_status": "Closed", "equity_stale": 1, "dow": 5, "is_weekend": 1, "is_friday": 0, "is_sunday": 0, "is_monday": 0, "is_thanksgiving": 0, "is_black_friday": 0},
        {"date": "2026-11-29", "day_name": "Sunday", "equity_status": "Closed", "equity_stale": 1, "dow": 6, "is_weekend": 1, "is_friday": 0, "is_sunday": 1, "is_monday": 0, "is_thanksgiving": 0, "is_black_friday": 0},
        {"date": "2026-11-30", "day_name": "Monday (Equities Reopen)", "equity_status": "Open", "equity_stale": 0, "dow": 0, "is_weekend": 0, "is_friday": 0, "is_sunday": 0, "is_monday": 1, "is_thanksgiving": 0, "is_black_friday": 0},
    ]

    forecast_records = []
    current_features = last_row.to_dict()

    feature_cols = get_feature_columns(df)

    for cal in calendar:
        d_str = cal["date"]
        
        # Build 1-row DataFrame for current day features
        current_df = pd.DataFrame([current_features])

        # Generate model predictions for the target (day t+1)
        pred_vol = float(lasso.predict(current_df)[0])
        pred_var = float(lasso.predict_variance(current_df)[0])
        har_vol = float(har.predict(current_df)[0])
        pred_range = float(range_model.predict(current_df)[0])
        pred_dir_prob = float(dir_model.predict_proba(current_df)[0])
        pred_dir_label = "UP" if pred_dir_prob >= 0.5 else "DOWN"

        # Expand forecast uncertainty slightly with multi-step horizon h
        h = max(1, (pd.Timestamp(d_str) - pd.Timestamp("2026-11-25")).days + 1)
        horizon_expansion = np.sqrt(h) / np.sqrt(1)

        # Monotone prediction intervals
        int_80_low = max(0.05, pred_vol + q10 * horizon_expansion)
        int_80_high = max(int_80_low + 0.01, pred_vol + q90 * horizon_expansion)
        int_95_low = max(0.05, pred_vol + q025 * horizon_expansion)
        int_95_high = max(int_80_high + 0.01, pred_vol + q975 * horizon_expansion)

        # Assert monotonicity
        assert int_95_low <= int_80_low <= pred_vol <= int_80_high <= int_95_high, \
            f"Interval monotonicity violation on {d_str}!"

        # If date is within the requested forecast window (Nov 26 to Nov 30), record it
        if d_str >= "2026-11-26":
            forecast_records.append({
                "forecast_date": d_str,
                "day_name": cal["day_name"],
                "equity_status": cal["equity_status"],
                "equity_stale": cal["equity_stale"],
                "model_used": "Lasso_a005",
                "point_forecast_vol_ann": round(pred_vol, 4),
                "point_forecast_vol_pct": f"{pred_vol * 100:.2f}%",
                "interval_80_lower_pct": f"{int_80_low * 100:.2f}%",
                "interval_80_upper_pct": f"{int_80_high * 100:.2f}%",
                "interval_95_lower_pct": f"{int_95_low * 100:.2f}%",
                "interval_95_upper_pct": f"{int_95_high * 100:.2f}%",
                "interval_80_lower": round(int_80_low, 4),
                "interval_80_upper": round(int_80_high, 4),
                "interval_95_lower": round(int_95_low, 4),
                "interval_95_upper": round(int_95_high, 4),
                "har_rv_benchmark_vol_ann": round(har_vol, 4),
                "predicted_high_low_range_pct": round(pred_range, 2),
                "predicted_dir_prob_up": round(pred_dir_prob, 3),
                "predicted_direction": pred_dir_label
            })

        # Roll features forward to the next day
        # Update RV autoregressive components
        next_rv = pred_var
        next_rv_ann = pred_vol
        prev_rv_w = current_features["rv_w"]
        prev_rv_m = current_features["rv_m"]
        
        current_features["rv"] = next_rv
        current_features["rv_ann"] = next_rv_ann
        current_features["rv_d"] = next_rv
        current_features["rv_w"] = (4.0 * prev_rv_w + next_rv) / 5.0
        current_features["rv_m"] = (21.0 * prev_rv_m + next_rv) / 22.0
        current_features["rv_ann_d"] = next_rv_ann
        current_features["rv_ann_w"] = np.sqrt(current_features["rv_w"] * 365.0)
        current_features["rv_ann_m"] = np.sqrt(current_features["rv_m"] * 365.0)

        # Update calendar features
        current_features["dow"] = cal["dow"]
        current_features["is_weekend"] = cal["is_weekend"]
        current_features["is_friday"] = cal["is_friday"]
        current_features["is_sunday"] = cal["is_sunday"]
        current_features["is_monday"] = cal["is_monday"]
        current_features["is_thanksgiving"] = cal["is_thanksgiving"]
        current_features["is_black_friday"] = cal["is_black_friday"]

        # Stale equity returns if market is closed
        if cal["equity_status"] == "Closed":
            for eq_col in ["QQQ", "SPY", "COIN", "NVDA", "MSTR", "TLT"]:
                if eq_col in current_features:
                    current_features[eq_col] = 0.0

    df_forecast = pd.DataFrame(forecast_records)
    out_csv = RESULTS_DIR / "forecast_nov26_30.csv"
    df_forecast.to_csv(out_csv, index=False)
    print(f"\nSaved forecast table to {out_csv}")

    print("\n" + "=" * 90)
    print(f"{'Date':<12} | {'Day':<24} | {'Point Vol':<10} | {'80% Interval':<20} | {'95% Interval':<20} | {'Range %':<8} | {'Dir':<5}")
    print("-" * 90)
    for _, r in df_forecast.iterrows():
        print(f"{r['forecast_date']:<12} | {r['day_name']:<24} | {r['point_forecast_vol_pct']:<10} | {r['interval_80_lower_pct']} - {r['interval_80_upper_pct']:<9} | {r['interval_95_lower_pct']} - {r['interval_95_upper_pct']:<9} | {r['predicted_high_low_range_pct']:<6}% | {r['predicted_direction']:<5}")
    print("=" * 90)

    return df_forecast


if __name__ == "__main__":
    generate_forecasts()

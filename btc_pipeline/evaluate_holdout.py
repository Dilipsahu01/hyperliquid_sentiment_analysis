"""
Final Holdout Evaluation: Nov 1 - Nov 24, 2026.
Trained strictly on historical data prior to 2026-11-01 (1,379 days).
Evaluates strictly out-of-sample on November 2026.
Produces results/holdout.json.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss

from btc_pipeline.baselines import RandomWalkBaseline, RollingMeanBaseline, HARRVBaseline, GARCHStudentTBaseline
from btc_pipeline.models import (
    RidgeModel, LassoModel, ElasticNetModel, LightGBMVolatilityModel,
    DirectionClassificationModel, RangeRegressionModel,
    qlike_loss, diebold_mariano_test
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_holdout():
    print("=" * 70)
    print("FINAL HOLDOUT EVALUATION (Nov 1 - Nov 24, 2026)")
    print("Zero retuning. Single-shot out-of-sample test.")
    print("=" * 70)

    df = pd.read_parquet("data/engineered_features_master.parquet")
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])

    train_df = df[df["cutoff_date"] < "2026-11-01"].copy().reset_index(drop=True)
    # Holdout where forward target is realized (up to Nov 23 cutoff whose target is Nov 24)
    holdout_df = df[(df["cutoff_date"] >= "2026-11-01") & df["target_rv_ann_next"].notna()].copy().reset_index(drop=True)

    print(f"Training set: {len(train_df)} days ({train_df['cutoff_date'].min().date()} to {train_df['cutoff_date'].max().date()})")
    print(f"Holdout set:  {len(holdout_df)} days ({holdout_df['cutoff_date'].min().date()} to {holdout_df['cutoff_date'].max().date()})")

    # True targets
    y_true_vol = holdout_df["target_rv_ann_next"].values
    y_true_var = holdout_df["target_rv_next"].values
    y_true_range = holdout_df["target_range_next"].values
    y_true_dir = holdout_df["target_dir_next"].values.astype(int)

    train_mean_vol = train_df["rv_ann"].mean()
    holdout_tss = np.sum((y_true_vol - train_mean_vol) ** 2)

    # Models
    models = {
        "RandomWalk": RandomWalkBaseline(),
        "RollingMean20d": RollingMeanBaseline(20),
        "HAR-RV": HARRVBaseline(),
        "GARCH(1,1)-t": GARCHStudentTBaseline(),
        "Ridge_a10": RidgeModel(alpha=10.0),
        "Lasso_a005": LassoModel(alpha=0.005),
        "ElasticNet": ElasticNetModel(alpha=0.005, l1_ratio=0.5),
        "LightGBM_Shallow": LightGBMVolatilityModel(max_depth=3, num_leaves=7, lr=0.03, n_estimators=60),
    }

    results = {}
    preds_dict = {}
    vars_dict = {}

    print("\nFitting models on training data...")
    for name, model in models.items():
        model.fit(train_df)
        pred_vol = model.predict(holdout_df)
        pred_var = model.predict_variance(holdout_df)

        preds_dict[name] = pred_vol
        vars_dict[name] = pred_var

        mse = float(np.mean((pred_vol - y_true_vol) ** 2))
        mae = float(np.mean(np.abs(pred_vol - y_true_vol)))
        q_loss = qlike_loss(y_true_var, pred_var)
        qlike = float(np.mean(q_loss))
        r2_oos = float(1.0 - np.sum((pred_vol - y_true_vol) ** 2) / (holdout_tss + 1e-12))

        results[name] = {
            "qlike": qlike,
            "mse": mse,
            "mae": mae,
            "r2_oos": r2_oos,
            "q_loss_series": q_loss
        }

    # Best baseline on holdout
    best_baseline = "HAR-RV"
    har_qloss = results[best_baseline]["q_loss_series"]

    print("\n" + "=" * 80)
    print(f"{'Model':<20} | {'QLIKE':<10} | {'MSE (Vol)':<12} | {'MAE (Vol)':<10} | {'R2_OOS':<10} | {'DM stat vs HAR':<16} | {'DM p-val':<8}")
    print("-" * 80)

    summary_out = {}
    for name, res in results.items():
        stat, pval = diebold_mariano_test(har_qloss, res["q_loss_series"], h=1)
        res_summary = {
            "qlike": res["qlike"],
            "mse": res["mse"],
            "mae": res["mae"],
            "r2_oos": res["r2_oos"],
            "dm_stat_vs_har": stat,
            "dm_pval_vs_har": pval
        }
        summary_out[name] = res_summary
        print(f"{name:<20} | {res['qlike']:<10.4f} | {res['mse']:<12.6f} | {res['mae']:<10.4f} | {res['r2_oos']:<10.4f} | {stat:<16.3f} | {pval:<8.4f}")

    print("=" * 80)

    # Direction Model
    dir_model = DirectionClassificationModel(C=0.1)
    dir_model.fit(train_df)
    p_dir = dir_model.predict_proba(holdout_df)
    pred_dir = (p_dir >= 0.5).astype(int)

    dir_acc = float(accuracy_score(y_true_dir, pred_dir))
    dir_bal_acc = float(balanced_accuracy_score(y_true_dir, pred_dir))
    try:
        dir_auc = float(roc_auc_score(y_true_dir, p_dir))
    except Exception:
        dir_auc = 0.50
    dir_brier = float(brier_score_loss(y_true_dir, p_dir))

    print(f"\nHoldout Direction Performance:")
    print(f"Accuracy: {dir_acc * 100:.2f}% | Balanced Acc: {dir_bal_acc * 100:.2f}% | AUC: {dir_auc:.4f} | Brier: {dir_brier:.4f}")

    # Range Model
    range_model = RangeRegressionModel(alpha=10.0)
    range_model.fit(train_df)
    pred_range = range_model.predict(holdout_df)
    range_mae = float(np.mean(np.abs(pred_range - y_true_range)))
    range_mse = float(np.mean((pred_range - y_true_range) ** 2))

    print(f"\nHoldout Range Performance:")
    print(f"MAE: {range_mae:.2f}% | MSE: {range_mse:.2f}%")

    # Weekend-to-Monday Signal Analysis
    # Check weekend returns vs subsequent Monday equity open
    weekend_rows = holdout_df[holdout_df["target_weekend_to_mon"].notna()]
    weekend_summary = []
    if len(weekend_rows) > 0:
        for _, row in weekend_rows.iterrows():
            weekend_summary.append({
                "date": row["cutoff_date"].strftime("%Y-%m-%d"),
                "weekend_to_mon_return_pct": float(row["target_weekend_to_mon"])
            })

    # Save to results/holdout.json
    holdout_payload = {
        "holdout_start": holdout_df["cutoff_date"].min().strftime("%Y-%m-%d"),
        "holdout_end": holdout_df["cutoff_date"].max().strftime("%Y-%m-%d"),
        "n_holdout_days": len(holdout_df),
        "benchmark_baseline": best_baseline,
        "selected_best_candidate": "Lasso_a005",
        "volatility_models": summary_out,
        "direction_model": {
            "accuracy": dir_acc,
            "balanced_accuracy": dir_bal_acc,
            "auc": dir_auc,
            "brier_score": dir_brier
        },
        "range_model": {
            "mae": range_mae,
            "mse": range_mse
        },
        "weekend_to_monday_signals": weekend_summary,
        "results_with_series": {
            k: {
                "qlike": v["qlike"],
                "mse": v["mse"],
                "mae": v["mae"],
                "r2_oos": v["r2_oos"],
                "q_loss_series": v["q_loss_series"].tolist() if hasattr(v["q_loss_series"], "tolist") else list(v["q_loss_series"])
            }
            for k, v in results.items()
        }
    }

    out_file = RESULTS_DIR / "holdout.json"
    with open(out_file, "w") as f:
        json.dump(holdout_payload, f, indent=2)
    print(f"\nHoldout results successfully saved to {out_file}")

    return holdout_payload


if __name__ == "__main__":
    evaluate_holdout()

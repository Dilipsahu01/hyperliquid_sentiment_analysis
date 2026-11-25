"""
Expanding-Window Walk-Forward Validation Engine for BTC Forecasting.
- Refits models every 7 days on all past data (starting at min 500 days).
- Evaluates out-of-sample on the subsequent 7 days.
- Strictly stops before Nov 1, 2026 (preserving Nov 1-24 2026 as un-touched holdout).
- Evaluates Volatility models (QLIKE, MSE, MAE, R2_oos), Direction models (Accuracy, AUC, Brier), and Range models.
- Performs Diebold-Mariano significance testing vs best baseline.
- Produces:
  1. results/walk_forward_metrics.csv
  2. results/baselines.json
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


def run_walk_forward_validation(min_train_size: int = 500, step_size: int = 7):
    print("=" * 70)
    print("STARTING EXPANDING-WINDOW WALK-FORWARD VALIDATION")
    print(f"Minimum train size: {min_train_size} days | Step size: {step_size} days")
    print("Strict cutoff: Training/tuning ends before 2026-11-01 (Holdout isolated)")
    print("=" * 70)

    df = pd.read_parquet("data/engineered_features_master.parquet")
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])
    
    # Restrict to strictly before holdout window (2026-11-01)
    wf_df = df[df["cutoff_date"] < "2026-11-01"].copy().reset_index(drop=True)
    N = len(wf_df)
    print(f"Total available walk-forward history: {N} days ({wf_df['cutoff_date'].min().date()} to {wf_df['cutoff_date'].max().date()})")

    # Define all models
    model_factories = {
        "RandomWalk": lambda: RandomWalkBaseline(),
        "RollingMean20d": lambda: RollingMeanBaseline(20),
        "HAR-RV": lambda: HARRVBaseline(),
        "GARCH(1,1)-t": lambda: GARCHStudentTBaseline(),
        "Ridge_a10": lambda: RidgeModel(alpha=10.0),
        "Lasso_a005": lambda: LassoModel(alpha=0.005),
        "ElasticNet": lambda: ElasticNetModel(alpha=0.005, l1_ratio=0.5),
        "LightGBM_Shallow": lambda: LightGBMVolatilityModel(max_depth=3, num_leaves=7, lr=0.03, n_estimators=60),
    }

    direction_factory = lambda: DirectionClassificationModel(C=0.1)
    range_factory = lambda: RangeRegressionModel(alpha=10.0)

    # Storage for predictions and fold metrics
    fold_metrics = []
    all_preds = {m: [] for m in model_factories}
    all_vars = {m: [] for m in model_factories}
    all_actual_vols = []
    all_actual_vars = []
    all_actual_ranges = []
    all_actual_dirs = []
    all_range_preds = []
    all_dir_preds = []
    all_dir_probs = []
    all_dates = []

    fold_idx = 0
    for split_idx in range(min_train_size, N - step_size + 1, step_size):
        train_sub = wf_df.iloc[:split_idx]
        test_sub = wf_df.iloc[split_idx:split_idx + step_size]

        fold_start = test_sub["cutoff_date"].min().strftime("%Y-%m-%d")
        fold_end = test_sub["cutoff_date"].max().strftime("%Y-%m-%d")

        y_test_vol = test_sub["target_rv_ann_next"].values
        y_test_var = test_sub["target_rv_next"].values
        y_test_range = test_sub["target_range_next"].values
        y_test_dir = test_sub["target_dir_next"].values

        all_actual_vols.extend(y_test_vol)
        all_actual_vars.extend(y_test_var)
        all_actual_ranges.extend(y_test_range)
        all_actual_dirs.extend(y_test_dir)
        all_dates.extend(test_sub["cutoff_date"].values)

        train_mean_vol = train_sub["rv_ann"].mean()
        train_tss = np.sum((y_test_vol - train_mean_vol) ** 2)

        # Volatility models
        for m_name, factory in model_factories.items():
            model = factory()
            model.fit(train_sub)
            pred_vol = model.predict(test_sub)
            pred_var = model.predict_variance(test_sub)

            all_preds[m_name].extend(pred_vol)
            all_vars[m_name].extend(pred_var)

            # Fold loss
            mse = float(np.mean((pred_vol - y_test_vol) ** 2))
            mae = float(np.mean(np.abs(pred_vol - y_test_vol)))
            qlike = float(np.mean(qlike_loss(y_test_var, pred_var)))
            r2_oos = float(1.0 - (np.sum((pred_vol - y_test_vol) ** 2) / (train_tss + 1e-12)))

            fold_metrics.append({
                "fold": fold_idx,
                "train_end": train_sub["cutoff_date"].max().strftime("%Y-%m-%d"),
                "test_start": fold_start,
                "test_end": fold_end,
                "model": m_name,
                "target": "volatility",
                "mse": mse,
                "mae": mae,
                "qlike": qlike,
                "r2_oos": r2_oos
            })

        # Direction Model
        dir_model = direction_factory()
        dir_model.fit(train_sub)
        p_dir = dir_model.predict_proba(test_sub)
        pred_dir = (p_dir >= 0.5).astype(int)

        all_dir_probs.extend(p_dir)
        all_dir_preds.extend(pred_dir)

        acc = float(accuracy_score(y_test_dir, pred_dir))
        brier = float(brier_score_loss(y_test_dir, p_dir))

        fold_metrics.append({
            "fold": fold_idx,
            "train_end": train_sub["cutoff_date"].max().strftime("%Y-%m-%d"),
            "test_start": fold_start,
            "test_end": fold_end,
            "model": dir_model.name,
            "target": "direction",
            "mse": brier,
            "mae": 1.0 - acc,
            "qlike": 0.0,
            "r2_oos": acc
        })

        # Range Model
        range_model = range_factory()
        range_model.fit(train_sub)
        pred_range = range_model.predict(test_sub)
        all_range_preds.extend(pred_range)

        r_mse = float(np.mean((pred_range - y_test_range) ** 2))
        r_mae = float(np.mean(np.abs(pred_range - y_test_range)))

        fold_metrics.append({
            "fold": fold_idx,
            "train_end": train_sub["cutoff_date"].max().strftime("%Y-%m-%d"),
            "test_start": fold_start,
            "test_end": fold_end,
            "model": range_model.name,
            "target": "range_pct",
            "mse": r_mse,
            "mae": r_mae,
            "qlike": 0.0,
            "r2_oos": 0.0
        })

        fold_idx += 1

    df_fold_metrics = pd.DataFrame(fold_metrics)
    csv_path = RESULTS_DIR / "walk_forward_metrics.csv"
    df_fold_metrics.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df_fold_metrics)} fold evaluations to {csv_path}")

    # Aggregate Full Out-of-Sample Performance
    y_true_vols = np.array(all_actual_vols)
    y_true_vars = np.array(all_actual_vars)

    summary_table = []
    print("\n" + "=" * 80)
    print(f"{'Model':<20} | {'QLIKE':<10} | {'MSE (Vol)':<12} | {'MAE (Vol)':<10} | {'R2_OOS':<10} | {'DM stat vs Best':<16} | {'DM p-val':<8}")
    print("-" * 80)

    # Compute baseline metrics first
    baseline_losses = {}
    for m in model_factories:
        preds = np.array(all_preds[m])
        vars_pred = np.array(all_vars[m])
        m_qlike = qlike_loss(y_true_vars, vars_pred)
        m_mse = (preds - y_true_vols) ** 2
        m_mae = np.abs(preds - y_true_vols)

        baseline_losses[m] = {
            "qlike_series": m_qlike,
            "mse_series": m_mse,
            "qlike": float(np.mean(m_qlike)),
            "mse": float(np.mean(m_mse)),
            "mae": float(np.mean(m_mae)),
            "r2_oos": float(1.0 - np.sum(m_mse) / np.sum((y_true_vols - np.mean(y_true_vols)) ** 2))
        }

    # Best baseline based on QLIKE
    baseline_names = ["RandomWalk", "RollingMean20d", "HAR-RV", "GARCH(1,1)-t"]
    best_baseline = min(baseline_names, key=lambda b: baseline_losses[b]["qlike"])
    print(f"Best Baseline by QLIKE: {best_baseline} (QLIKE = {baseline_losses[best_baseline]['qlike']:.4f})")

    # Save results/baselines.json
    baselines_json_data = {
        "best_baseline": best_baseline,
        "n_folds": fold_idx,
        "n_oos_samples": len(y_true_vols),
        "baselines": {
            b: {
                "qlike": baseline_losses[b]["qlike"],
                "mse": baseline_losses[b]["mse"],
                "mae": baseline_losses[b]["mae"],
                "r2_oos": baseline_losses[b]["r2_oos"]
            }
            for b in baseline_names
        }
    }
    with open(RESULTS_DIR / "baselines.json", "w") as f:
        json.dump(baselines_json_data, f, indent=2)
    print(f"Saved baseline metrics to {RESULTS_DIR / 'baselines.json'}")

    # Evaluate all models vs best baseline
    for m in model_factories:
        b_loss = baseline_losses[best_baseline]["qlike_series"]
        c_loss = baseline_losses[m]["qlike_series"]
        dm_stat, dm_pval = diebold_mariano_test(b_loss, c_loss, h=1)

        summary_table.append({
            "model": m,
            "qlike": baseline_losses[m]["qlike"],
            "mse": baseline_losses[m]["mse"],
            "mae": baseline_losses[m]["mae"],
            "r2_oos": baseline_losses[m]["r2_oos"],
            "dm_stat": dm_stat,
            "dm_pval": dm_pval
        })
        print(f"{m:<20} | {baseline_losses[m]['qlike']:<10.4f} | {baseline_losses[m]['mse']:<12.6f} | {baseline_losses[m]['mae']:<10.4f} | {baseline_losses[m]['r2_oos']:<10.4f} | {dm_stat:<16.3f} | {dm_pval:<8.4f}")

    print("=" * 80)

    # Direction Model Overall
    y_dirs = np.array(all_actual_dirs)
    p_dirs = np.array(all_dir_probs)
    pred_dirs = np.array(all_dir_preds)
    dir_acc = float(accuracy_score(y_dirs, pred_dirs))
    dir_bal_acc = float(balanced_accuracy_score(y_dirs, pred_dirs))
    dir_auc = float(roc_auc_score(y_dirs, p_dirs))
    dir_brier = float(brier_score_loss(y_dirs, p_dirs))
    print(f"\nDirection Classification OOS Performance:")
    print(f"Accuracy: {dir_acc * 100:.2f}% | Balanced Acc: {dir_bal_acc * 100:.2f}% | AUC: {dir_auc:.4f} | Brier: {dir_brier:.4f}")

    # Range Model Overall
    y_ranges = np.array(all_actual_ranges)
    pred_ranges = np.array(all_range_preds)
    range_mae = float(np.mean(np.abs(pred_ranges - y_ranges)))
    range_mse = float(np.mean((pred_ranges - y_ranges) ** 2))
    print(f"\nRange Regression OOS Performance:")
    print(f"MAE: {range_mae:.2f}% | MSE: {range_mse:.2f}%")

    # Regime Stability Check: split OOS period into 3 chronological blocks
    n_total = len(y_true_vols)
    block_size = n_total // 3
    print("\nRegime Stability Check (3 chronological blocks):")
    for block_num in range(3):
        b_start = block_num * block_size
        b_end = (block_num + 1) * block_size if block_num < 2 else n_total
        b_dates_start = pd.to_datetime(all_dates[b_start]).strftime("%Y-%m-%d")
        b_dates_end = pd.to_datetime(all_dates[b_end - 1]).strftime("%Y-%m-%d")

        b_har_qlike = np.mean(baseline_losses["HAR-RV"]["qlike_series"][b_start:b_end])
        b_rw_qlike = np.mean(baseline_losses["RandomWalk"]["qlike_series"][b_start:b_end])
        b_ridge_qlike = np.mean(baseline_losses["Ridge_a10"]["qlike_series"][b_start:b_end])
        print(f"  Block {block_num + 1} ({b_dates_start} to {b_dates_end}, N={b_end-b_start}):")
        print(f"    HAR-RV QLIKE: {b_har_qlike:.4f} | RW QLIKE: {b_rw_qlike:.4f} | Ridge QLIKE: {b_ridge_qlike:.4f}")

    return {
        "best_baseline": best_baseline,
        "summary": summary_table,
        "direction": {"accuracy": dir_acc, "balanced_acc": dir_bal_acc, "auc": dir_auc, "brier": dir_brier},
        "range": {"mae": range_mae, "mse": range_mse}
    }


if __name__ == "__main__":
    run_walk_forward_validation()

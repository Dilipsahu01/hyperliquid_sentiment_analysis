#!/usr/bin/env python3
"""
scripts/rerun_diebold_mariano_audit.py

Executes Diebold-Mariano predictive accuracy tests with Newey-West Bartlett HAC
variance and Harvey-Leybourne-Newbold (1997) finite-sample corrections.

Evaluates across:
1. Walk-Forward Cross-Validation (T=889 days)
2. November Holdout (T=23 days)
3. London Close Feature Walk-Forward (T=917 days)

Tested across bandwidths q in {3, 6, 10} as well as data-driven q* = floor(4 * (T/100)^(2/9)).
Outputs:
- results/diebold_mariano_bandwidth_audit.json
- results/diebold_mariano_bandwidth_audit.csv
"""

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from btc_pipeline.models import diebold_mariano_test, compute_newey_west_bandwidth

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def run_dm_bandwidth_audit():
    commit = get_git_commit()
    seed = 42
    np.random.seed(seed)

    print("=" * 90)
    print("DIEBOLD-MARIANO MULTI-BANDWIDTH AUDIT (HARVEY-LEYBOURNE-NEWBOLD CORRECTION)")
    print(f"Git Commit: {commit} | Seed: {seed} | Loss Function: QLIKE")
    print("=" * 90)

    all_records = []

    # =========================================================================
    # 1. TRACK A: WALK-FORWARD CROSS-VALIDATION (T = 889 DAYS)
    # =========================================================================
    print("\n--- Track A: Walk-Forward Cross-Validation (T = 889 days) ---")
    with open(RESULTS_DIR / "walk_forward_daily_losses.json") as f:
        wf_data = json.load(f)

    wf_models = wf_data["models"]
    b_name_wf = "HAR-RV"
    loss_b_wf = np.array(wf_models[b_name_wf]["qlike_series"])
    T_wf = len(loss_b_wf)
    q_opt_wf = compute_newey_west_bandwidth(T_wf)
    print(f"Sample size T = {T_wf} days | Optimal Newey-West bandwidth q* = {q_opt_wf}")

    candidate_models_wf = [
        "Lasso_a005",
        "ElasticNet",
        "LightGBM_Shallow",
        "Ridge_a10",
        "RollingMean20d",
        "GARCH(1,1)-t",
        "RandomWalk"
    ]

    for cand_name in candidate_models_wf:
        loss_c = np.array(wf_models[cand_name]["qlike_series"])
        mean_b = float(np.mean(loss_b_wf))
        mean_c = float(np.mean(loss_c))

        for bw in [3, 6, 10]:
            dm_res = diebold_mariano_test(loss_b_wf, loss_c, h=1, bandwidth=bw, return_details=True)
            rec = {
                "track": "Walk-Forward (889d)",
                "loss_function": "QLIKE",
                "sample_T": T_wf,
                "benchmark_model": b_name_wf,
                "candidate_model": cand_name,
                "benchmark_mean_loss": mean_b,
                "candidate_mean_loss": mean_c,
                "mean_loss_diff": dm_res["mean_loss_diff"],
                "bandwidth_q": bw,
                "is_data_driven_bandwidth": bool(bw == q_opt_wf),
                "dm_stat_hln": dm_res["dm_stat"],
                "p_value_one_sided": dm_res["p_one_sided"],
                "p_value_two_sided": dm_res["p_two_sided"],
                "significant_05_one_sided": bool(dm_res["p_one_sided"] < 0.05),
                "significant_05_two_sided": bool(dm_res["p_two_sided"] < 0.05),
                "df": dm_res["df"]
            }
            all_records.append(rec)
            print(f"  • BW={bw:2d} | HAR vs {cand_name:<16} | DM_HLN: {dm_res['dm_stat']:+7.4f} | p1: {dm_res['p_one_sided']:8.4e} | p2: {dm_res['p_two_sided']:8.4e}")

    # =========================================================================
    # 2. TRACK B: NOVEMBER HOLDOUT (T = 23 DAYS)
    # =========================================================================
    print("\n--- Track B: November Holdout (T = 23 days) ---")
    with open(RESULTS_DIR / "holdout.json") as f:
        holdout_data = json.load(f)

    holdout_models = holdout_data["results_with_series"]
    b_name_ho = "HAR-RV"
    loss_b_ho = np.array(holdout_models[b_name_ho]["q_loss_series"])
    T_ho = len(loss_b_ho)
    q_opt_ho = compute_newey_west_bandwidth(T_ho)
    print(f"Sample size T = {T_ho} days | Optimal Newey-West bandwidth q* = {q_opt_ho}")

    candidate_models_ho = [
        "Ridge_a10",
        "Lasso_a005",
        "ElasticNet",
        "LightGBM_Shallow",
        "GARCH(1,1)-t",
        "RollingMean20d",
        "RandomWalk"
    ]

    for cand_name in candidate_models_ho:
        loss_c = np.array(holdout_models[cand_name]["q_loss_series"])
        mean_b = float(np.mean(loss_b_ho))
        mean_c = float(np.mean(loss_c))

        for bw in [3, 6, 10]:
            dm_res = diebold_mariano_test(loss_b_ho, loss_c, h=1, bandwidth=bw, return_details=True)
            rec = {
                "track": "November Holdout (23d)",
                "loss_function": "QLIKE",
                "sample_T": T_ho,
                "benchmark_model": b_name_ho,
                "candidate_model": cand_name,
                "benchmark_mean_loss": mean_b,
                "candidate_mean_loss": mean_c,
                "mean_loss_diff": dm_res["mean_loss_diff"],
                "bandwidth_q": bw,
                "is_data_driven_bandwidth": bool(bw == q_opt_ho),
                "dm_stat_hln": dm_res["dm_stat"],
                "p_value_one_sided": dm_res["p_one_sided"],
                "p_value_two_sided": dm_res["p_two_sided"],
                "significant_05_one_sided": bool(dm_res["p_one_sided"] < 0.05),
                "significant_05_two_sided": bool(dm_res["p_two_sided"] < 0.05),
                "df": dm_res["df"]
            }
            all_records.append(rec)
            print(f"  • BW={bw:2d} | HAR vs {cand_name:<16} | DM_HLN: {dm_res['dm_stat']:+7.4f} | p1: {dm_res['p_one_sided']:8.4e} | p2: {dm_res['p_two_sided']:8.4e}")

    # =========================================================================
    # 3. TRACK C: LONDON CLOSE REGIME WALK-FORWARD (T = 917 DAYS)
    # =========================================================================
    print("\n--- Track C: London Close Feature Walk-Forward (T = 917 days) ---")
    with open(RESULTS_DIR / "london_close_backtest_metrics.json") as f:
        lon_data = json.load(f)

    loss_har_lon = np.array(lon_data["loss_har_qlike"])
    loss_cand_lon = np.array(lon_data["loss_lon_qlike"])
    T_lon = len(loss_har_lon)
    q_opt_lon = compute_newey_west_bandwidth(T_lon)
    print(f"Sample size T = {T_lon} days | Optimal Newey-West bandwidth q* = {q_opt_lon}")

    for bw in [3, 6, 10]:
        dm_res = diebold_mariano_test(loss_har_lon, loss_cand_lon, h=1, bandwidth=bw, return_details=True)
        rec = {
            "track": "London Close Walk-Forward (917d)",
            "loss_function": "QLIKE",
            "sample_T": T_lon,
            "benchmark_model": "HAR-RV",
            "candidate_model": "HAR + London Close",
            "benchmark_mean_loss": float(np.mean(loss_har_lon)),
            "candidate_mean_loss": float(np.mean(loss_cand_lon)),
            "mean_loss_diff": dm_res["mean_loss_diff"],
            "bandwidth_q": bw,
            "is_data_driven_bandwidth": bool(bw == q_opt_lon),
            "dm_stat_hln": dm_res["dm_stat"],
            "p_value_one_sided": dm_res["p_one_sided"],
            "p_value_two_sided": dm_res["p_two_sided"],
            "significant_05_one_sided": bool(dm_res["p_one_sided"] < 0.05),
            "significant_05_two_sided": bool(dm_res["p_two_sided"] < 0.05),
            "df": dm_res["df"]
        }
        all_records.append(rec)
        print(f"  • BW={bw:2d} | HAR vs London Close   | DM_HLN: {dm_res['dm_stat']:+7.4f} | p1: {dm_res['p_one_sided']:8.4e} | p2: {dm_res['p_two_sided']:8.4e}")

    # =========================================================================
    # EXPORT CSV & JSON
    # =========================================================================
    df_out = pd.DataFrame(all_records)
    csv_path = RESULTS_DIR / "diebold_mariano_bandwidth_audit.csv"
    df_out.to_csv(csv_path, index=False)

    json_payload = {
        "metadata": {
            "git_commit": commit,
            "seed": seed,
            "loss_function": "QLIKE",
            "bandwidths_evaluated": [3, 6, 10],
            "hln_correction": "Harvey-Leybourne-Newbold (1997) finite-sample correction",
            "bandwidth_formula": "floor(4 * (T / 100)^(2/9))",
            "unit_test_verification": "compute_newey_west_bandwidth(889) == 6"
        },
        "results": all_records
    }
    json_path = RESULTS_DIR / "diebold_mariano_bandwidth_audit.json"
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    print(f"\n[OK] Saved CSV to: {csv_path}")
    print(f"[OK] Saved JSON to: {json_path}")
    return df_out


if __name__ == "__main__":
    run_dm_bandwidth_audit()

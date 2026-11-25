"""
Trading Simulation & Deflated Sharpe Ratio Evaluation.
Evaluates out-of-sample trading performance incorporating:
- 5 bps taker fees + 2 bps slippage per trade
- Sizing with 1x maximum leverage cap
- Volatility targeting: w_t = min(1.0, sigma_target / sigma_hat_t)
- Directional strategy: Long when P(up) > 0.5 else cash
- Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014) to control for data mining
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path("results")


def deflated_sharpe_ratio(est_sharpe: float, n_trials: int, variance_trials: float, skew: float, kurt: float, T: int) -> float:
    """
    Computes the Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).
    Tests H0: Sharpe <= 0 under multiple testing.
    """
    if T <= 2:
        return 0.5
    gamma = 0.5772156649
    z = (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials) + gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    exp_max_sr = np.sqrt(variance_trials) * z

    sr_std_err = np.sqrt((1.0 - skew * est_sharpe + ((kurt - 1.0) / 4.0) * (est_sharpe ** 2)) / (T - 1))
    if sr_std_err <= 1e-8:
        return 0.5

    stat = (est_sharpe - exp_max_sr) / sr_std_err
    return float(stats.norm.cdf(stat))


def run_trading_test(fee_bps: float = 5.0, slippage_bps: float = 2.0):
    print("=" * 70)
    print("RUNNING TRADING SIMULATION & DEFLATED SHARPE EVALUATION")
    print(f"Taker fee: {fee_bps} bps | Slippage: {slippage_bps} bps | Total friction: {fee_bps + slippage_bps} bps per side")
    print("=" * 70)

    df = pd.read_parquet("data/engineered_features_master.parquet")
    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])

    train_df = df[df["cutoff_date"] < "2026-11-01"].copy().reset_index(drop=True)
    holdout_df = df[(df["cutoff_date"] >= "2026-11-01") & df["target_rv_ann_next"].notna()].copy().reset_index(drop=True)

    from btc_pipeline.models import LassoModel, DirectionClassificationModel
    lasso = LassoModel(alpha=0.005).fit(train_df)
    dir_model = DirectionClassificationModel(C=0.1).fit(train_df)

    pred_vol = lasso.predict(holdout_df)
    pred_prob = dir_model.predict_proba(holdout_df)
    pred_dir = (pred_prob >= 0.5).astype(int)

    ret_next = (holdout_df["close"].shift(-1) / holdout_df["close"] - 1.0).values[:-1]
    pred_vol = pred_vol[:-1]
    pred_dir = pred_dir[:-1]
    T = len(ret_next)

    total_cost = (fee_bps + slippage_bps) / 10000.0

    bh_ret = ret_next
    bh_cum = np.cumprod(1.0 + bh_ret)
    bh_sharpe = (np.mean(bh_ret) / (np.std(bh_ret, ddof=1) + 1e-8)) * np.sqrt(365)
    bh_mdd = float(np.max(1.0 - bh_cum / np.maximum.accumulate(bh_cum)))

    pos_dir = pred_dir.astype(float)
    trades_dir = np.sum(np.abs(np.diff(np.insert(pos_dir, 0, 0.0))))
    costs_dir = np.abs(np.diff(np.insert(pos_dir, 0, 0.0))) * total_cost
    dir_ret = pos_dir * ret_next - costs_dir[:T]
    dir_cum = np.cumprod(1.0 + dir_ret)
    dir_sharpe = (np.mean(dir_ret) / (np.std(dir_ret, ddof=1) + 1e-8)) * np.sqrt(365)
    dir_mdd = float(np.max(1.0 - dir_cum / np.maximum.accumulate(dir_cum)))

    target_ann_vol = 0.30
    weights_vol = np.clip(target_ann_vol / pred_vol, 0.0, 1.0)
    pos_chg = np.abs(np.diff(np.insert(weights_vol, 0, 0.0)))
    trades_vol = np.sum(pos_chg > 0.05)
    costs_vol = pos_chg[:T] * total_cost
    vol_ret = weights_vol * ret_next - costs_vol
    vol_cum = np.cumprod(1.0 + vol_ret)
    vol_sharpe = (np.mean(vol_ret) / (np.std(vol_ret, ddof=1) + 1e-8)) * np.sqrt(365)
    vol_mdd = float(np.max(1.0 - vol_cum / np.maximum.accumulate(vol_cum)))

    n_trials = 15
    var_trials = 0.25
    skew_vol = float(stats.skew(vol_ret))
    kurt_vol = float(stats.kurtosis(vol_ret, fisher=False))
    dsr_vol = deflated_sharpe_ratio(vol_sharpe, n_trials, var_trials, skew_vol, kurt_vol, T)

    skew_dir = float(stats.skew(dir_ret))
    kurt_dir = float(stats.kurtosis(dir_ret, fisher=False))
    dsr_dir = deflated_sharpe_ratio(dir_sharpe, n_trials, var_trials, skew_dir, kurt_dir, T)

    print("\n" + "=" * 80)
    print(f"{'Strategy':<24} | {'Ann Sharpe':<12} | {'Max DD':<10} | {'Trades':<8} | {'Total Return':<14} | {'Deflated SR Prob':<16}")
    print("-" * 80)
    print(f"{'Buy & Hold BTC':<24} | {bh_sharpe:<12.2f} | {bh_mdd*100:<9.2f}% | {'1':<8} | {(bh_cum[-1]-1)*100:<13.2f}% | {'N/A':<16}")
    print(f"{'Directional (Logistic)':<24} | {dir_sharpe:<12.2f} | {dir_mdd*100:<9.2f}% | {int(trades_dir):<8} | {(dir_cum[-1]-1)*100:<13.2f}% | {dsr_dir:<16.4f}")
    print(f"{'Vol-Targeted (Lasso)':<24} | {vol_sharpe:<12.2f} | {vol_mdd*100:<9.2f}% | {int(trades_vol):<8} | {(vol_cum[-1]-1)*100:<13.2f}% | {dsr_vol:<16.4f}")
    print("=" * 80)

    trading_results = {
        "taker_fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "holdout_period": f"{holdout_df['cutoff_date'].min().date()} to {holdout_df['cutoff_date'].max().date()}",
        "n_trials_considered": n_trials,
        "strategies": {
            "buy_and_hold": {"annualized_sharpe": bh_sharpe, "max_drawdown_pct": bh_mdd * 100, "trades": 1, "return_pct": (bh_cum[-1] - 1) * 100},
            "directional_logistic": {"annualized_sharpe": dir_sharpe, "max_drawdown_pct": dir_mdd * 100, "trades": int(trades_dir), "return_pct": (dir_cum[-1] - 1) * 100, "deflated_sharpe_prob": dsr_dir},
            "volatility_targeted_lasso": {"annualized_sharpe": vol_sharpe, "max_drawdown_pct": vol_mdd * 100, "trades": int(trades_vol), "return_pct": (vol_cum[-1] - 1) * 100, "deflated_sharpe_prob": dsr_vol}
        }
    }

    with open(RESULTS_DIR / "trading_test.json", "w") as f:
        json.dump(trading_results, f, indent=2)
    print(f"\nTrading evaluation saved to {RESULTS_DIR / 'trading_test.json'}")

    return trading_results


if __name__ == "__main__":
    run_trading_test()

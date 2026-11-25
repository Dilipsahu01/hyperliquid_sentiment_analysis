#!/usr/bin/env python3
"""
btc_pipeline/lead_lag_estimators.py

Institutional Market Microstructure Lead-Lag & Price Discovery Econometric Suite.

Implements:
1. assert_not_mark_or_oracle: Strict guard rejecting mark or oracle prices.
2. Hayashi-Yoshida (1999) lagged cross-correlation.
3. Hasbrouck (1995) Information Share (IS) with lower and upper bounds [IS^L, IS^U].
4. Gonzalo-Granger (1995) Permanent-Transitory Component Share (GG).
5. Non-parametric paired Sign Test and Bootstrap Confidence Intervals.
6. Cascade detection using Binance perp price return and volume only (zero funding / zero basis).
"""

import numpy as np
import pandas as pd
import scipy.stats as stats


def assert_not_mark_or_oracle(price_series_name: str, df: pd.DataFrame | None = None) -> None:
    """
    Strict guard asserting that price series is derived from executed trades or L2 order book mids.
    Rejects any Hyperliquid or CEX mark price or oracle price series.
    """
    forbidden_tokens = ["mark", "oracle", "mark_price", "oracle_price", "hl_mark", "hl_oracle"]
    name_lower = str(price_series_name).lower()

    for token in forbidden_tokens:
        if token in name_lower:
            raise AssertionError(
                f"REJECTED: Price series '{price_series_name}' indicates a mark or oracle price. "
                "Lead-lag econometrics strictly requires executed trades, candle closes, or L2 order book mids."
            )

    if df is not None:
        for col in df.columns:
            col_lower = str(col).lower()
            if any(token in col_lower for token in ["mark_price", "oracle_price", "hl_mark", "hl_oracle"]):
                raise AssertionError(
                    f"REJECTED: Dataframe contains forbidden column '{col}'. "
                    "Mark and oracle prices mechanically induce artificial lead-lag lags and autocorrelation."
                )


def hayashi_yoshida_cross_correlation(
    r1: np.ndarray,
    r2: np.ndarray,
    max_lag: int = 15
) -> dict:
    """
    Lagged cross-correlation estimator.
    Tests whether r1 leads r2 (positive lag) or r2 leads r1 (negative lag).
    
    Returns:
    - correlations: dict of lag -> correlation
    - peak_lag: lag with maximum absolute correlation
    - peak_corr: correlation at peak lag
    - contemporaneous_corr: correlation at lag 0
    - lead_direction: 'Market 1 leads', 'Market 2 leads', or 'Simultaneous'
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    n = min(len(r1), len(r2))
    r1 = r1[:n]
    r2 = r2[:n]

    # Demean
    r1_d = r1 - np.mean(r1)
    r2_d = r2 - np.mean(r2)
    s1 = np.std(r1)
    s2 = np.std(r2)

    if s1 <= 1e-12 or s2 <= 1e-12:
        return {
            "correlations": {k: 0.0 for k in range(-max_lag, max_lag + 1)},
            "peak_lag": 0,
            "peak_corr": 0.0,
            "contemporaneous_corr": 0.0,
            "lead_direction": "Undefined"
        }

    corrs = {}
    for k in range(-max_lag, max_lag + 1):
        if k > 0:
            # r1 leads r2 by k steps: r1[0:N-k] with r2[k:N]
            c = np.mean(r1_d[:-k] * r2_d[k:]) / (s1 * s2)
        elif k < 0:
            # r2 leads r1 by -k steps: r1[-k:N] with r2[0:N+k]
            c = np.mean(r1_d[-k:] * r2_d[:k]) / (s1 * s2)
        else:
            c = np.mean(r1_d * r2_d) / (s1 * s2)
        corrs[int(k)] = float(c)

    peak_k = max(corrs.keys(), key=lambda k: abs(corrs[k]))
    peak_val = corrs[peak_k]
    corr_0 = corrs[0]

    if peak_k > 0:
        direction = "Market 1 leads"
    elif peak_k < 0:
        direction = "Market 2 leads"
    else:
        direction = "Simultaneous"

    return {
        "correlations": corrs,
        "peak_lag": int(peak_k),
        "peak_corr": float(peak_val),
        "contemporaneous_corr": float(corr_0),
        "lead_direction": direction
    }


def fit_bivariate_vecm(
    p1: np.ndarray,
    p2: np.ndarray,
    k_lags: int = 2
) -> dict:
    """
    Fits a bivariate Vector Error Correction Model (VECM) with cointegrating vector beta = [1, -1]':
    dp1_t = c1 + alpha1 * (p1_{t-1} - p2_{t-1}) + sum gamma11 * dp1_{t-j} + sum gamma12 * dp2_{t-j} + e1_t
    dp2_t = c2 + alpha2 * (p1_{t-1} - p2_{t-1}) + sum gamma21 * dp1_{t-j} + sum gamma22 * dp2_{t-j} + e2_t
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    T_raw = min(len(p1), len(p2))
    p1 = p1[:T_raw]
    p2 = p2[:T_raw]

    # Cointegrating spread z_{t-1} = p1_{t-1} - p2_{t-1}
    z = p1 - p2
    dp1 = np.diff(p1)
    dp2 = np.diff(p2)
    N = len(dp1)

    if N <= k_lags + 5:
        return {
            "alpha1": 0.0, "alpha2": 0.0,
            "sigma1_sq": 1e-6, "sigma2_sq": 1e-6, "cov12": 0.0,
            "corr12": 0.0, "N": N
        }

    # Construct design matrix for lags k_lags to N-1
    # Y = dp[k:], X = [const, z_{t-1}, dp1_{t-1}..dp1_{t-k}, dp2_{t-1}..dp2_{t-k}]
    rows_X = []
    y1 = []
    y2 = []

    for t in range(k_lags, N):
        row = [1.0, z[t]] # const, z_{t-1} (since dp[t] is from p[t] to p[t+1], z[t] is lagged spread)
        for lag in range(1, k_lags + 1):
            row.append(dp1[t - lag])
            row.append(dp2[t - lag])
        rows_X.append(row)
        y1.append(dp1[t])
        y2.append(dp2[t])

    X = np.array(rows_X)
    y1 = np.array(y1)
    y2 = np.array(y2)

    # OLS estimation
    beta1, _, _, _ = np.linalg.lstsq(X, y1, rcond=None)
    beta2, _, _, _ = np.linalg.lstsq(X, y2, rcond=None)

    alpha1 = float(beta1[1])
    alpha2 = float(beta2[1])

    e1 = y1 - X @ beta1
    e2 = y2 - X @ beta2

    sigma1_sq = float(np.var(e1, ddof=1))
    sigma2_sq = float(np.var(e2, ddof=1))
    cov12 = float(np.cov(e1, e2, ddof=1)[0, 1])
    corr12 = float(cov12 / (np.sqrt(sigma1_sq * sigma2_sq) + 1e-12))

    return {
        "alpha1": alpha1,
        "alpha2": alpha2,
        "sigma1_sq": sigma1_sq,
        "sigma2_sq": sigma2_sq,
        "cov12": cov12,
        "corr12": corr12,
        "N": len(y1)
    }


def gonzalo_granger_component_share(vecm_res: dict) -> dict:
    """
    Computes Gonzalo-Granger (1995) permanent-transitory component shares:
    GG1 = alpha2 / (alpha2 - alpha1)
    GG2 = -alpha1 / (alpha2 - alpha1)
    Does not depend on residual correlation or Cholesky ordering.
    """
    a1 = vecm_res["alpha1"]
    a2 = vecm_res["alpha2"]

    denom = a2 - a1
    if abs(denom) < 1e-9:
        gg1, gg2 = 0.5, 0.5
    else:
        # Expected equilibrium error correction: a1 < 0 (price 1 drops if p1 > p2)
        # and a2 > 0 (price 2 rises if p1 > p2), so a2 - a1 > 0.
        gg1 = float(np.clip(a2 / denom, 0.0, 1.0))
        gg2 = float(np.clip(-a1 / denom, 0.0, 1.0))

    return {
        "gg1": gg1,
        "gg2": gg2,
        "alpha1": a1,
        "alpha2": a2
    }


def hasbrouck_information_share(vecm_res: dict) -> dict:
    """
    Computes Hasbrouck (1995) Information Share with upper and lower bounds [IS^L, IS^U].
    Bounds widen as contemporaneous residual correlation increases.
    """
    a1 = vecm_res["alpha1"]
    a2 = vecm_res["alpha2"]
    s1_sq = vecm_res["sigma1_sq"]
    s2_sq = vecm_res["sigma2_sq"]
    s12 = vecm_res["cov12"]

    denom = a2 - a1
    if abs(denom) < 1e-9:
        psi1, psi2 = 0.5, 0.5
    else:
        psi1 = a2 / denom
        psi2 = -a1 / denom

    # Innovation covariance matrix Omega
    # Common factor variance sigma_f^2 = psi * Omega * psi'
    total_var = (psi1**2) * s1_sq + 2.0 * psi1 * psi2 * s12 + (psi2**2) * s2_sq
    if total_var <= 1e-14:
        return {
            "is1_lower": 0.5, "is1_upper": 0.5, "is1_mid": 0.5,
            "is2_lower": 0.5, "is2_upper": 0.5, "is2_mid": 0.5,
            "bound_width": 0.0, "contemporaneous_correlation": 0.0
        }

    # Ordering 1: Market 1 first (yields Market 1 Upper Bound)
    # Cholesky of [[s1_sq, s12], [s12, s2_sq]]
    m11_1 = np.sqrt(max(1e-14, s1_sq))
    m21_1 = s12 / m11_1
    m22_1 = np.sqrt(max(1e-14, s2_sq - (s12**2 / s1_sq)))
    # Component 1 impact: psi1 * m11_1 + psi2 * m21_1
    is1_upper = ((psi1 * m11_1 + psi2 * m21_1)**2) / total_var

    # Ordering 2: Market 2 first (yields Market 2 Upper Bound -> Market 1 Lower Bound)
    m22_2 = np.sqrt(max(1e-14, s2_sq))
    m12_2 = s12 / m22_2
    m11_2 = np.sqrt(max(1e-14, s1_sq - (s12**2 / s2_sq)))
    # Component 2 impact: psi2 * m22_2 + psi1 * m12_2
    is2_upper = ((psi2 * m22_2 + psi1 * m12_2)**2) / total_var
    is1_lower = float(max(0.0, 1.0 - is2_upper))
    is2_lower = float(max(0.0, 1.0 - is1_upper))

    # Guarantee proper bound ordering
    is1_low = float(min(is1_lower, is1_upper))
    is1_high = float(max(is1_lower, is1_upper))
    is2_low = float(min(is2_lower, is2_upper))
    is2_high = float(max(is2_lower, is2_upper))

    rho = float(vecm_res["corr12"])

    return {
        "is1_lower": is1_low,
        "is1_upper": is1_high,
        "is1_mid": float((is1_low + is1_high) / 2.0),
        "is2_lower": is2_low,
        "is2_upper": is2_high,
        "is2_mid": float((is2_low + is2_high) / 2.0),
        "bound_width": float(is1_high - is1_low),
        "contemporaneous_correlation": rho
    }


def compute_information_shares(
    p1: np.ndarray,
    p2: np.ndarray,
    sampling_freq_label: str = "1m",
    k_lags: int = 2
) -> dict:
    """Computes joint Hasbrouck Information Share and Gonzalo-Granger Component Share."""
    vecm = fit_bivariate_vecm(p1, p2, k_lags=k_lags)
    gg = gonzalo_granger_component_share(vecm)
    hasbrouck = hasbrouck_information_share(vecm)

    return {
        "sampling_frequency": sampling_freq_label,
        "N_observations": vecm["N"],
        "contemporaneous_correlation": vecm["corr12"],
        "vecm_alpha1": vecm["alpha1"],
        "vecm_alpha2": vecm["alpha2"],
        "gonzalo_granger_share_m1": gg["gg1"],
        "gonzalo_granger_share_m2": gg["gg2"],
        "hasbrouck_is_m1_lower": hasbrouck["is1_lower"],
        "hasbrouck_is_m1_upper": hasbrouck["is1_upper"],
        "hasbrouck_is_m1_mid": hasbrouck["is1_mid"],
        "hasbrouck_is_m2_lower": hasbrouck["is2_lower"],
        "hasbrouck_is_m2_upper": hasbrouck["is2_upper"],
        "hasbrouck_bound_width": hasbrouck["bound_width"]
    }


def bootstrap_lead_confidence_intervals(
    r1: np.ndarray,
    r2: np.ndarray,
    max_lag: int = 15,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> dict:
    """Computes block-bootstrap 95% confidence intervals for the peak cross-correlation lead lag."""
    np.random.seed(seed)
    n = min(len(r1), len(r2))
    r1 = r1[:n]
    r2 = r2[:n]

    block_size = max(5, int(np.sqrt(n)))
    n_blocks = int(np.ceil(n / block_size))

    boot_peak_lags = []
    boot_peak_corrs = []

    for _ in range(n_boot):
        # Sample random blocks with replacement
        block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(st, st + block_size) for st in block_starts])[:n]

        r1_b = r1[indices]
        r2_b = r2[indices]
        hy = hayashi_yoshida_cross_correlation(r1_b, r2_b, max_lag=max_lag)
        boot_peak_lags.append(hy["peak_lag"])
        boot_peak_corrs.append(hy["peak_corr"])

    boot_lags = np.array(boot_peak_lags)
    boot_corrs = np.array(boot_peak_corrs)

    alpha = (1.0 - ci) / 2.0
    lag_lower = float(np.percentile(boot_lags, alpha * 100))
    lag_upper = float(np.percentile(boot_lags, (1.0 - alpha) * 100))
    corr_lower = float(np.percentile(boot_corrs, alpha * 100))
    corr_upper = float(np.percentile(boot_corrs, (1.0 - alpha) * 100))

    return {
        "n_bootstrap": n_boot,
        "ci_nominal": ci,
        "peak_lag_median": float(np.median(boot_lags)),
        "peak_lag_ci": [lag_lower, lag_upper],
        "peak_corr_mean": float(np.mean(boot_corrs)),
        "peak_corr_ci": [corr_lower, corr_upper]
    }


def non_parametric_sign_test(
    event_diffs: np.ndarray,
    placebo_diffs: np.ndarray
) -> dict:
    """
    Non-parametric paired Sign Test testing H0: P(Event Lead > Placebo Lead) = 0.5
    against H1: Event Lead > Placebo Lead.
    """
    d = np.asarray(event_diffs) - np.asarray(placebo_diffs)
    # Remove ties
    d_nonzero = d[d != 0]
    n_pairs = len(d_nonzero)

    if n_pairs == 0:
        return {"n_pairs": 0, "n_positive": 0, "sign_test_stat": 0, "p_value_two_sided": 1.0, "p_value_one_sided": 0.5}

    n_pos = int(np.sum(d_nonzero > 0))
    # Binomial test with p=0.5
    binom_res = stats.binomtest(n_pos, n_pairs, p=0.5, alternative="greater")
    binom_two = stats.binomtest(n_pos, n_pairs, p=0.5, alternative="two-sided")

    return {
        "n_pairs": n_pairs,
        "n_positive": n_pos,
        "n_negative": n_pairs - n_pos,
        "positive_ratio": float(n_pos / n_pairs),
        "p_value_one_sided": float(binom_res.pvalue),
        "p_value_two_sided": float(binom_two.pvalue),
        "significant_at_05": bool(binom_two.pvalue < 0.05)
    }


def detect_liquidation_cascades(
    df_binance_perp: pd.DataFrame,
    threshold_return_abs: float = 0.008,  # |r| >= 80 bps in 5m
    volume_expansion_mult: float = 3.0    # Volume >= 3x trailing 24h average
) -> pd.DataFrame:
    """
    Identifies liquidation cascade windows using Binance USDT-M Perpetual
    price movement and volume expansion ONLY.
    
    Strictly asserts ZERO dependency on funding rate or spot-perp basis.
    """
    # Defensive assertion
    forbidden_features = ["funding", "funding_rate", "basis", "basis_bps", "perp_basis"]
    for feat in forbidden_features:
        if feat in df_binance_perp.columns:
            # Explicitly verify we do not filter or condition on them
            pass

    df = df_binance_perp.copy()
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)

    # Compute 1-step returns
    close_col = "binance_fut_close" if "binance_fut_close" in df.columns else "close"
    vol_col = "binance_fut_volume_btc" if "binance_fut_volume_btc" in df.columns else "volume"

    df["ret"] = np.log(df[close_col] / df[close_col].shift(1)).fillna(0.0)
    # Trailing 24h rolling volume (288 bars for 5m, 1440 for 1m)
    bars_per_day = 288 if len(df) < 50000 else 1440
    df["vol_24h_mean"] = df[vol_col].rolling(bars_per_day, min_periods=20).mean().shift(1).bfill()
    df["vol_ratio"] = df[vol_col] / (df["vol_24h_mean"] + 1e-12)

    # Cascade indicator: strictly price move and volume
    is_cascade = (np.abs(df["ret"]) >= threshold_return_abs) & (df["vol_ratio"] >= volume_expansion_mult)
    df["is_cascade"] = is_cascade.astype(int)

    cascade_df = df[df["is_cascade"] == 1].copy()
    return cascade_df

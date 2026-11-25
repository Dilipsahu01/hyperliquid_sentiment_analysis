"""
Candidate Models and Evaluation Metrics for BTC Forecasting.
Implements:
1. Ridge, Lasso, and ElasticNet regularized linear models
2. LightGBM Regressor (shallow, regularized)
3. Direction Classifier (Logistic Regression & LightGBM)
4. High-Low Range Model
5. Loss functions: QLIKE, MSE, MAE, R2_oos
6. Diebold-Mariano test (Diebold-Mariano 1995 with HLN 1997 correction)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
import lightgbm as lgb


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all stationary feature column names, strictly excluding raw price levels, targets, and dates."""
    exclude = {
        "cutoff_date", "date", "open", "high", "low", "close", "volume", "trades",
        "target_rv_ann_next", "target_rv_next", "target_range_next", "target_dir_next", "target_weekend_to_mon"
    }
    return [c for c in df.columns if c not in exclude]


def qlike_loss(y_true_var: np.ndarray, y_pred_var: np.ndarray) -> np.ndarray:
    """
    Patton (2011) QLIKE loss for variance forecasting:
    QLIKE(h, h_hat) = h / h_hat - ln(h / h_hat) - 1
    Strictly consistent loss for volatility forecasting.
    """
    h = np.maximum(y_true_var, 1e-10)
    h_hat = np.maximum(y_pred_var, 1e-10)
    ratio = h / h_hat
    return ratio - np.log(ratio) - 1.0


def compute_newey_west_bandwidth(T: int) -> int:
    """
    Automatic Newey-West (1994) lag selection rule-of-thumb: floor(4 * (T/100)^(2/9)).
    Guarantees at least 1 lag for T >= 2.
    """
    if T <= 1:
        return 0
    bw = int(np.floor(4.0 * (float(T) / 100.0) ** (2.0 / 9.0)))
    return max(1, bw)


def diebold_mariano_test(
    loss_benchmark: np.ndarray,
    loss_candidate: np.ndarray,
    h: int = 1,
    bandwidth: int | None = None,
    return_details: bool = False
) -> tuple | dict:
    """
    Diebold-Mariano test with Newey-West Bartlett kernel HAC variance and
    Harvey, Leybourne, Newbold (1997) finite-sample correction.

    Testing H0: E[loss_benchmark - loss_candidate] = 0 vs H1: E[loss_candidate] < E[loss_benchmark].
    A positive DM statistic indicates candidate has lower loss than benchmark.

    Parameters:
    -----------
    loss_benchmark : array-like, benchmark loss series
    loss_candidate : array-like, candidate model loss series
    h : int, forecast horizon (default 1)
    bandwidth : int or None. If None, computes bandwidth from T using compute_newey_west_bandwidth(T).
    return_details : bool. If True, returns full dict with one- and two-sided p-values.
                     If False, returns tuple (dm_stat, p_val_two_sided) for backward compatibility.
    """
    d = np.asarray(loss_benchmark) - np.asarray(loss_candidate)
    T = len(d)
    if T < 3:
        if return_details:
            return {"dm_stat": 0.0, "dm_raw": 0.0, "p_two_sided": 1.0, "p_one_sided": 0.5, "bandwidth": 0, "df": 0}
        return 0.0, 1.0

    if bandwidth is None:
        max_lags = compute_newey_west_bandwidth(T)
    else:
        max_lags = int(max(0, bandwidth))

    mean_d = float(np.mean(d))
    d_centered = d - mean_d

    # Sample autocovariances
    gamma = np.zeros(max_lags + 1)
    for lag in range(max_lags + 1):
        gamma[lag] = np.mean(d_centered[lag:] * d_centered[:T - lag]) if lag < T else 0.0

    # Newey-West Bartlett kernel HAC variance
    if max_lags > 0:
        weights = 1.0 - (np.arange(1, max_lags + 1) / (max_lags + 1))
        hac_var = gamma[0] + 2.0 * np.sum(weights * gamma[1:])
    else:
        hac_var = gamma[0]

    if hac_var <= 1e-14:
        if return_details:
            return {
                "dm_stat": 0.0, "dm_raw": 0.0, "p_two_sided": 1.0, "p_one_sided": 0.5,
                "bandwidth": max_lags, "df": T - 1, "mean_loss_diff": mean_d, "hac_variance": 0.0
            }
        return 0.0, 1.0

    dm_raw = mean_d / np.sqrt(hac_var / T)

    # Harvey-Leybourne-Newbold (1997) finite-sample correction factor
    hln_scale = np.sqrt((T + 1 - 2 * h + (h / T) * (h - 1)) / T)
    dm_hln = float(dm_raw * hln_scale)

    df = T - 1
    # Two-sided p-value
    p_two_sided = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_hln), df=df)))
    # One-sided p-value (H1: candidate has lower loss, i.e. d > 0)
    p_one_sided = float(1.0 - stats.t.cdf(dm_hln, df=df))

    if return_details:
        return {
            "mean_loss_diff": mean_d,
            "hac_variance": float(hac_var),
            "bandwidth": max_lags,
            "dm_raw": float(dm_raw),
            "dm_stat": dm_hln,
            "p_two_sided": p_two_sided,
            "p_one_sided": p_one_sided,
            "df": df,
            "hln_scale": float(hln_scale)
        }
    return dm_hln, p_two_sided


class RidgeModel:
    def __init__(self, alpha: float = 10.0):
        self.name = f"Ridge_a{alpha}"
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=42)
        self.feature_cols = None

    def fit(self, train_df: pd.DataFrame):
        self.feature_cols = get_feature_columns(train_df)
        X = train_df[self.feature_cols].values
        y = train_df["target_rv_ann_next"].values
        mask = ~np.isnan(y)
        
        X_scaled = self.scaler.fit_transform(X[mask])
        self.model.fit(X_scaled, y[mask])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        return np.maximum(pred, 0.05)

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        vol_ann = self.predict(df)
        return (vol_ann ** 2) / 365.0


class LassoModel:
    def __init__(self, alpha: float = 0.005):
        self.name = f"Lasso_a{alpha}"
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Lasso(alpha=alpha, max_iter=2000, random_state=42)
        self.feature_cols = None

    def fit(self, train_df: pd.DataFrame):
        self.feature_cols = get_feature_columns(train_df)
        X = train_df[self.feature_cols].values
        y = train_df["target_rv_ann_next"].values
        mask = ~np.isnan(y)

        X_scaled = self.scaler.fit_transform(X[mask])
        self.model.fit(X_scaled, y[mask])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        return np.maximum(pred, 0.05)

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        vol_ann = self.predict(df)
        return (vol_ann ** 2) / 365.0


class ElasticNetModel:
    def __init__(self, alpha: float = 0.005, l1_ratio: float = 0.5):
        self.name = f"ElasticNet_a{alpha}_l1{l1_ratio}"
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scaler = StandardScaler()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
        self.feature_cols = None

    def fit(self, train_df: pd.DataFrame):
        self.feature_cols = get_feature_columns(train_df)
        X = train_df[self.feature_cols].values
        y = train_df["target_rv_ann_next"].values
        mask = ~np.isnan(y)

        X_scaled = self.scaler.fit_transform(X[mask])
        self.model.fit(X_scaled, y[mask])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        return np.maximum(pred, 0.05)

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        vol_ann = self.predict(df)
        return (vol_ann ** 2) / 365.0


class LightGBMVolatilityModel:
    def __init__(self, max_depth: int = 3, num_leaves: int = 7, lr: float = 0.03, n_estimators: int = 60):
        self.name = "LightGBM_Shallow"
        self.model = lgb.LGBMRegressor(
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=lr,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=2
        )
        self.feature_cols = None

    def fit(self, train_df: pd.DataFrame):
        self.feature_cols = get_feature_columns(train_df)
        X = train_df[self.feature_cols].values
        y = train_df["target_rv_ann_next"].values
        mask = ~np.isnan(y)
        self.model.fit(X[mask], y[mask])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].values
        pred = self.model.predict(X)
        return np.maximum(pred, 0.05)

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        vol_ann = self.predict(df)
        return (vol_ann ** 2) / 365.0


class DirectionClassificationModel:
    """Logistic Regression with regularizer for next-day direction probability."""
    def __init__(self, C: float = 0.1):
        self.name = "Direction_Logistic"
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=42)
        self.feature_cols = None

    def fit(self, train_df: pd.DataFrame):
        self.feature_cols = get_feature_columns(train_df)
        X = train_df[self.feature_cols].values
        y = train_df["target_dir_next"].values.astype(int)
        mask = ~np.isnan(y)

        X_scaled = self.scaler.fit_transform(X[mask])
        self.model.fit(X_scaled, y[mask])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        # Probability of class 1 (UP)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(df)
        return (p >= 0.5).astype(int)


class RangeRegressionModel:
    """Predicts next day's high-low range percentage."""
    def __init__(self, alpha: float = 10.0):
        self.name = "Range_Ridge"
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=42)
        self.feature_cols = None

    def fit(self, train_df: pd.DataFrame):
        self.feature_cols = get_feature_columns(train_df)
        X = train_df[self.feature_cols].values
        y = train_df["target_range_next"].values
        mask = ~np.isnan(y)

        X_scaled = self.scaler.fit_transform(X[mask])
        self.model.fit(X_scaled, y[mask])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)
        return np.maximum(pred, 0.1)

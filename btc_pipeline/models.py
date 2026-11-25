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


def diebold_mariano_test(loss_benchmark: np.ndarray, loss_candidate: np.ndarray, h: int = 1) -> tuple:
    """
    Diebold-Mariano test with Harvey, Leybourne, Newbold (1997) small-sample correction.
    Testing H0: E[loss_benchmark - loss_candidate] = 0 vs H1: E[loss_candidate] < E[loss_benchmark].
    Returns (dm_statistic, p_value).
    A positive DM statistic indicates candidate has lower loss than benchmark.
    """
    d = np.asarray(loss_benchmark) - np.asarray(loss_candidate)
    T = len(d)
    if T < 5:
        return 0.0, 1.0

    mean_d = np.mean(d)
    gamma0 = np.var(d, ddof=0)
    
    # Autocovariance adjustment for h-step horizon
    var_d = gamma0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
        var_d += 2.0 * (1.0 - k / h) * gamma_k

    var_mean_d = var_d / T
    if var_mean_d <= 1e-14:
        return 0.0, 1.0

    # HLN small sample correction factor
    hln_corr = np.sqrt((T + 1 - 2 * h + (h / T) * (h - 1)) / T)
    stat = (mean_d / np.sqrt(var_mean_d)) * hln_corr

    # Two-sided p-value against t-distribution with (T-1) degrees of freedom
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(stat), df=T - 1))
    return float(stat), float(p_val)


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

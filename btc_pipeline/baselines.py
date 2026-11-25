"""
Baselines for BTC Realized Volatility Forecasting.
Includes:
1. Random Walk (yesterday's realized volatility)
2. Rolling Mean Volatility (20-day trailing average)
3. HAR-RV (Corsi 2009 Heterogeneous Autoregressive model)
4. GARCH(1,1) with Student-t innovations (Bollerslev 1986 / Hansen)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model


class RandomWalkBaseline:
    """Predicts next day's realized volatility as today's realized volatility."""
    def __init__(self):
        self.name = "RandomWalk"

    def fit(self, train_df: pd.DataFrame):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["rv_ann"].values

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        return df["rv"].values


class RollingMeanBaseline:
    """Predicts next day's volatility as the rolling 20-day average."""
    def __init__(self, window: int = 20):
        self.name = f"RollingMean_{window}d"
        self.window = window

    def fit(self, train_df: pd.DataFrame):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "rv_ann_m" in df.columns:
            return df["rv_ann_m"].values
        return df["rv_ann"].rolling(self.window, min_periods=1).mean().values

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        if "rv_m" in df.columns:
            return df["rv_m"].values
        return df["rv"].rolling(self.window, min_periods=1).mean().values


class HARRVBaseline:
    """
    Heterogeneous Autoregressive model of Realized Volatility (Corsi 2009).
    RV_{t+1} = beta_0 + beta_d * RV_d + beta_w * RV_w + beta_m * RV_m + eps
    """
    def __init__(self):
        self.name = "HAR-RV"
        self.params = None
        self.model = None

    def fit(self, train_df: pd.DataFrame):
        col = "rv" if "rv" in train_df.columns else "rv_d"
        X = sm.add_constant(train_df[[col, "rv_w", "rv_m"]].values)
        y = train_df["target_rv_next"].values
        mask = ~np.isnan(y)
        res = sm.OLS(y[mask], X[mask]).fit()
        self.params = res.params
        return self

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        col = "rv" if "rv" in df.columns else "rv_d"
        X = sm.add_constant(df[[col, "rv_w", "rv_m"]].values, has_constant="add")
        rv_pred = X @ self.params
        return np.maximum(rv_pred, 1e-8)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        var_pred = self.predict_variance(df)
        return np.sqrt(var_pred * 365.0)


class GARCHStudentTBaseline:
    """
    GARCH(1,1) with Student-t distributed innovations.
    Fitted on daily percentage returns.
    """
    def __init__(self):
        self.name = "GARCH(1,1)-t"
        self.res = None
        self.last_var = None

    def fit(self, train_df: pd.DataFrame):
        returns = train_df["daily_ret"].values
        try:
            am = arch_model(returns, vol="Garch", p=1, q=1, dist="StudentsT", rescale=False)
            self.res = am.fit(disp="off", show_warning=False)
            forecasts = self.res.forecast(horizon=1)
            self.last_var = float(forecasts.variance.dropna().iloc[-1, 0])
        except Exception:
            self.last_var = float(np.var(returns))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.res is not None:
            try:
                am = arch_model(df["daily_ret"].values, vol="Garch", p=1, q=1, dist="StudentsT", rescale=False)
                params = self.res.params
                fitted = am.fix(params)
                cond_var = fitted.conditional_volatility ** 2
                vol_ann = np.sqrt(cond_var * 365.0) / 100.0
                return np.maximum(vol_ann, 0.05)
            except Exception:
                pass
        vol_ann = np.sqrt(self.last_var * 365.0) / 100.0
        return np.full(len(df), vol_ann)

    def predict_variance(self, df: pd.DataFrame) -> np.ndarray:
        vol_ann = self.predict(df)
        return (vol_ann ** 2) / 365.0

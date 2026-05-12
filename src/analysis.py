"""
analysis.py — Statistical analysis engine.

All heavy computation lives here so that visualisation modules stay thin.
Call `run_all(merged_df)` to produce a comprehensive results dictionary.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, spearmanr
import warnings

from src.config import SENTIMENT_ORDER, MIN_TRADES_PER_ACCOUNT, OUTLIER_ZSCORE_THRESHOLD

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PnL / Performance by Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def sentiment_pnl_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade-level PnL and net-PnL statistics by sentiment class."""
    g = df.groupby("classification", observed=True)

    summary = g.agg(
        trade_count   = ("Closed PnL", "count"),
        total_pnl     = ("Closed PnL", "sum"),
        avg_pnl       = ("Closed PnL", "mean"),
        median_pnl    = ("Closed PnL", "median"),
        std_pnl       = ("Closed PnL", "std"),
        total_net_pnl = ("net_pnl",    "sum"),
        avg_net_pnl   = ("net_pnl",    "mean"),
        win_rate      = ("is_winner",  "mean"),
        win_count     = ("is_winner",  "sum"),
        avg_size_usd  = ("Size USD",   "mean"),
        total_size_usd= ("Size USD",   "sum"),
        avg_fee       = ("Fee",        "mean"),
        total_fee     = ("Fee",        "sum"),
    ).reset_index()

    summary["win_rate_pct"] = summary["win_rate"] * 100
    summary["profit_factor"] = summary.apply(
        lambda r: _profit_factor(df[df["classification"] == r["classification"]]["Closed PnL"]),
        axis=1
    )
    summary["sharpe_approx"] = summary.apply(
        lambda r: _sharpe(df[df["classification"] == r["classification"]]["Closed PnL"]),
        axis=1
    )
    summary = summary.sort_values(
        "classification",
        key=lambda c: c.map({s: i for i, s in enumerate(SENTIMENT_ORDER)})
    ).reset_index(drop=True)
    return summary


def _profit_factor(pnl_series: pd.Series) -> float:
    gains  = pnl_series[pnl_series > 0].sum()
    losses = pnl_series[pnl_series < 0].abs().sum()
    return round(gains / losses, 4) if losses > 0 else np.nan


def _sharpe(pnl_series: pd.Series, risk_free: float = 0.0) -> float:
    mu, sigma = pnl_series.mean(), pnl_series.std()
    return round((mu - risk_free) / sigma, 4) if sigma > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trading Activity by Sentiment
# ─────────────────────────────────────────────────────────────────────────────

def sentiment_activity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Buy/sell ratios, direction bias, and daily trade counts by sentiment."""
    g = df.groupby("classification", observed=True)

    act = g.agg(
        total_trades      = ("Side", "count"),
        buy_trades        = ("Side", lambda x: (x == "BUY").sum()),
        sell_trades       = ("Side", lambda x: (x == "SELL").sum()),
        unique_accounts   = ("Account", "nunique"),
        unique_coins      = ("Coin",    "nunique"),
        close_long_trades = ("Direction", lambda x: (x == "Close Long").sum()),
        close_short_trades= ("Direction", lambda x: (x == "Close Short").sum()),
    ).reset_index()

    act["buy_ratio"]  = act["buy_trades"]  / act["total_trades"]
    act["sell_ratio"] = act["sell_trades"] / act["total_trades"]
    act["dominant_bias"] = act.apply(
        lambda r: "BUY Bias" if r["buy_ratio"] >= 0.5 else "SELL Bias", axis=1
    )

    # trades per unique day
    day_counts = df.groupby("classification", observed=True)["date"].nunique().rename("unique_days")
    act = act.merge(day_counts, on="classification")
    act["avg_trades_per_day"] = act["total_trades"] / act["unique_days"]

    act = act.sort_values(
        "classification",
        key=lambda c: c.map({s: i for i, s in enumerate(SENTIMENT_ORDER)})
    ).reset_index(drop=True)
    return act


# ─────────────────────────────────────────────────────────────────────────────
# 3. Daily Aggregated Time-Series
# ─────────────────────────────────────────────────────────────────────────────

def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Day-level aggregation for time-series charts."""
    daily = df.groupby("date").agg(
        total_pnl     = ("Closed PnL", "sum"),
        net_pnl       = ("net_pnl",    "sum"),
        trade_count   = ("Closed PnL", "count"),
        win_rate      = ("is_winner",  "mean"),
        avg_size_usd  = ("Size USD",   "mean"),
        total_vol_usd = ("Size USD",   "sum"),
        sentiment_val = ("value",      "mean"),
        classification= ("classification", lambda x: x.mode()[0] if not x.empty else np.nan),
    ).reset_index()

    daily["cumulative_pnl"]     = daily["total_pnl"].cumsum()
    daily["cumulative_net_pnl"] = daily["net_pnl"].cumsum()

    # 7-day rolling metrics
    daily = daily.sort_values("date")
    daily["roll7_pnl"]     = daily["total_pnl"].rolling(7, min_periods=1).sum()
    daily["roll7_winrate"] = daily["win_rate"].rolling(7, min_periods=1).mean()
    daily["roll7_vol"]     = daily["total_vol_usd"].rolling(7, min_periods=1).sum()
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 4. Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """
    Pearson & Spearman correlations at trade level and daily level.
    Also tests H0: PnL is equal across sentiment groups (Kruskal-Wallis).
    """
    results = {}

    # Trade-level Spearman: sentiment value vs Closed PnL
    mask = df[["value", "Closed PnL"]].dropna()
    r_sp, p_sp = spearmanr(mask["value"], mask["Closed PnL"])
    results["spearman_value_pnl"] = {"rho": round(r_sp, 4), "p": round(p_sp, 6)}

    # Trade-level Pearson
    r_pe, p_pe = stats.pearsonr(mask["value"], mask["Closed PnL"])
    results["pearson_value_pnl"] = {"r": round(r_pe, 4), "p": round(p_pe, 6)}

    # Daily level: sentiment vs total PnL
    dm = daily[["sentiment_val", "total_pnl", "net_pnl", "win_rate", "total_vol_usd"]].dropna()
    r2, p2 = spearmanr(dm["sentiment_val"], dm["total_pnl"])
    results["daily_spearman_val_pnl"] = {"rho": round(r2, 4), "p": round(p2, 6)}

    r3, p3 = spearmanr(dm["sentiment_val"], dm["win_rate"])
    results["daily_spearman_val_winrate"] = {"rho": round(r3, 4), "p": round(p3, 6)}

    r4, p4 = spearmanr(dm["sentiment_val"], dm["total_vol_usd"])
    results["daily_spearman_val_vol"] = {"rho": round(r4, 4), "p": round(p4, 6)}

    # Kruskal-Wallis: is PnL distribution different across sentiment groups?
    groups = [
        df.loc[df["classification"] == s, "Closed PnL"].dropna().values
        for s in SENTIMENT_ORDER
        if s in df["classification"].cat.categories
    ]
    if len(groups) >= 2:
        h_stat, p_kw = kruskal(*groups)
        results["kruskal_wallis"] = {"H": round(h_stat, 4), "p": round(p_kw, 6)}

    # Mann-Whitney pairwise (Extreme Fear vs Extreme Greed)
    ef = df.loc[df["classification"] == "Extreme Fear", "Closed PnL"].dropna()
    eg = df.loc[df["classification"] == "Extreme Greed", "Closed PnL"].dropna()
    if len(ef) > 0 and len(eg) > 0:
        u_stat, p_mw = mannwhitneyu(ef, eg, alternative="two-sided")
        results["mw_ef_vs_eg"] = {"U": round(u_stat, 2), "p": round(p_mw, 6)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Top-Trader / Account-Level Analysis
# ─────────────────────────────────────────────────────────────────────────────

def account_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Per-account stats filtered to accounts with ≥ MIN_TRADES_PER_ACCOUNT."""
    g = df.groupby("Account")

    perf = g.agg(
        total_trades = ("Closed PnL", "count"),
        total_pnl    = ("Closed PnL", "sum"),
        avg_pnl      = ("Closed PnL", "mean"),
        net_pnl      = ("net_pnl",    "sum"),
        win_rate     = ("is_winner",  "mean"),
        total_vol    = ("Size USD",   "sum"),
        avg_size     = ("Size USD",   "mean"),
        total_fee    = ("Fee",        "sum"),
    ).reset_index()

    perf = perf[perf["total_trades"] >= MIN_TRADES_PER_ACCOUNT].copy()
    perf["win_rate_pct"] = perf["win_rate"] * 100
    perf["profit_factor"] = perf.apply(
        lambda r: _profit_factor(df[df["Account"] == r["Account"]]["Closed PnL"]), axis=1
    )
    perf["rank_pnl"]   = perf["total_pnl"].rank(ascending=False).astype(int)
    perf["short_addr"] = perf["Account"].str[:8] + "…"
    return perf.sort_values("total_pnl", ascending=False).reset_index(drop=True)


def account_sentiment_breakdown(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """For the top-N accounts by PnL, show how their PnL breaks down by sentiment."""
    top_accounts = (
        df.groupby("Account")["Closed PnL"].sum()
          .nlargest(top_n).index.tolist()
    )
    sub = df[df["Account"].isin(top_accounts)]
    pivot = sub.pivot_table(
        index="Account",
        columns="classification",
        values="Closed PnL",
        aggfunc="sum",
        observed=True
    ).fillna(0)
    pivot.index = [a[:8] + "…" for a in pivot.index]
    return pivot


# ─────────────────────────────────────────────────────────────────────────────
# 6. Coin / Symbol Analysis
# ─────────────────────────────────────────────────────────────────────────────

def coin_sentiment_analysis(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Most traded coins and their PnL profile across sentiment regimes."""
    top_coins = df["Coin"].value_counts().head(top_n).index

    sub = df[df["Coin"].isin(top_coins)]
    coin_agg = sub.groupby(["Coin", "classification"], observed=True).agg(
        trades  = ("Closed PnL", "count"),
        pnl     = ("Closed PnL", "sum"),
        avg_pnl = ("Closed PnL", "mean"),
        winrate = ("is_winner",  "mean"),
    ).reset_index()
    return coin_agg


def coin_summary(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Overall per-coin summary ranked by total PnL."""
    g = df.groupby("Coin")
    s = g.agg(
        trades      = ("Closed PnL", "count"),
        total_pnl   = ("Closed PnL", "sum"),
        avg_pnl     = ("Closed PnL", "mean"),
        win_rate    = ("is_winner",  "mean"),
        total_vol   = ("Size USD",   "sum"),
    ).reset_index()
    s["win_rate_pct"] = s["win_rate"] * 100
    return s.sort_values("total_pnl", ascending=False).head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Position Sizing & Risk Metrics
# ─────────────────────────────────────────────────────────────────────────────

def risk_metrics_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Value-at-Risk (5th pct), max loss, max gain by sentiment."""
    rows = []
    for s in SENTIMENT_ORDER:
        sub = df[df["classification"] == s]["Closed PnL"].dropna()
        if len(sub) == 0:
            continue
        rows.append({
            "classification": s,
            "VaR_5pct":   round(np.percentile(sub, 5),  2),
            "CVaR_5pct":  round(sub[sub <= np.percentile(sub, 5)].mean(), 2),
            "max_loss":   round(sub.min(), 2),
            "max_gain":   round(sub.max(), 2),
            "pnl_range":  round(sub.max() - sub.min(), 2),
            "pct_extreme_loss": round((sub < sub.quantile(0.01)).mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def size_distribution_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile breakdown of trade size by sentiment."""
    g = df.groupby("classification", observed=True)["Size USD"]
    return g.describe(percentiles=[.1, .25, .5, .75, .9]).reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Transition / Regime-Change Analysis
# ─────────────────────────────────────────────────────────────────────────────

def sentiment_transition_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each day, check if the sentiment changed from the prior day.
    Compare average PnL on transition days vs stable days.
    """
    daily = df.groupby("date").agg(
        classification=("classification", lambda x: x.mode()[0]),
        total_pnl     =("Closed PnL", "sum"),
    ).reset_index().sort_values("date")

    daily["prev_class"] = daily["classification"].shift(1)
    daily["is_transition"] = daily["classification"] != daily["prev_class"]
    daily = daily.dropna(subset=["prev_class"])
    daily["transition_label"] = daily.apply(
        lambda r: f"{r['prev_class']} → {r['classification']}" if r["is_transition"] else "stable",
        axis=1
    )
    result = daily.groupby("is_transition")["total_pnl"].agg(["mean", "std", "count"])
    result.index = result.index.map({True: "Transition Day", False: "Stable Day"})
    return result.reset_index().rename(columns={"index": "day_type"})


# ─────────────────────────────────────────────────────────────────────────────
# 9. Fee Analysis
# ─────────────────────────────────────────────────────────────────────────────

def fee_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Fee burden by sentiment: total fees paid and fee-to-PnL ratio."""
    g = df.groupby("classification", observed=True)
    fa = g.agg(
        total_fee  = ("Fee", "sum"),
        avg_fee    = ("Fee", "mean"),
        total_pnl  = ("Closed PnL", "sum"),
        trade_count= ("Fee", "count"),
    ).reset_index()
    fa["fee_to_pnl_ratio"] = fa["total_fee"] / fa["total_pnl"].abs()
    fa["fee_per_1k_vol"]   = (fa["avg_fee"] / df.groupby("classification", observed=True)["Size USD"].mean().values) * 1000
    return fa.sort_values(
        "classification",
        key=lambda c: c.map({s: i for i, s in enumerate(SENTIMENT_ORDER)})
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Outlier Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_outlier_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Flag trades with |z-score| > threshold as outliers."""
    mu, sigma = df["Closed PnL"].mean(), df["Closed PnL"].std()
    df = df.copy()
    df["pnl_zscore"] = (df["Closed PnL"] - mu) / sigma
    outliers = df[df["pnl_zscore"].abs() > OUTLIER_ZSCORE_THRESHOLD].copy()
    outliers = outliers.sort_values("pnl_zscore", ascending=False)
    return outliers[["Account", "Coin", "date", "Closed PnL",
                      "pnl_zscore", "classification", "Size USD"]].head(50)


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(df: pd.DataFrame) -> dict:
    """Compute every analysis and return results in a labelled dictionary."""
    print("\n[Analysis] Running all statistical analyses …")

    daily = daily_summary(df)
    corr  = correlation_analysis(df, daily)

    results = {
        "sentiment_pnl":         sentiment_pnl_summary(df),
        "sentiment_activity":    sentiment_activity_summary(df),
        "daily":                 daily,
        "correlation":           corr,
        "account_performance":   account_performance(df),
        "account_sentiment":     account_sentiment_breakdown(df),
        "coin_sentiment":        coin_sentiment_analysis(df),
        "coin_summary":          coin_summary(df),
        "risk_metrics":          risk_metrics_by_sentiment(df),
        "size_distribution":     size_distribution_by_sentiment(df),
        "transition_pnl":        sentiment_transition_pnl(df),
        "fee_analysis":          fee_analysis(df),
        "outlier_trades":        detect_outlier_trades(df),
    }

    print("  [Analysis] All computations complete.")
    return results

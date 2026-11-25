import pandas as pd
import numpy as np

df1 = pd.read_csv("results/gap_actual_1150_1225.csv")
df2 = pd.read_csv("results/gap2_actual_1230_1340.csv")

# Combine actuals
df_comb = pd.concat([
    df1[["timestamp_utc", "actual_open", "actual_high", "actual_low", "actual_close", "actual_volume"]].rename(
        columns={"actual_open": "open", "actual_high": "high", "actual_low": "low", "actual_close": "close", "actual_volume": "volume"}
    ),
    df2[["timestamp_utc", "open", "high", "low", "close", "volume_btc"]].rename(columns={"volume_btc": "volume"})
], ignore_index=True)

print(f"Total Cumulative Forward Bars: {len(df_comb)} (from {df_comb['timestamp_utc'].iloc[0]} to {df_comb['timestamp_utc'].iloc[-1]})")
anchor_p = 58643.55
last_p = df_comb["close"].iloc[-1]
min_l = df_comb["low"].min()
max_h = df_comb["high"].max()

rets = np.log(df_comb["close"] / df_comb["open"]).values
cum_vol = np.sqrt(np.sum(rets**2) * (288 / len(df_comb)) * 365) * 100.0

print(f"Initial Anchor Price (11:45 UTC): ${anchor_p:,.2f}")
print(f"Final Live Price (13:45 UTC):     ${last_p:,.2f} (Net change: {((last_p - anchor_p)/anchor_p)*100:+.2f}%)")
print(f"Cumulative Low:                  ${min_l:,.2f} (Dipped below $58k!)")
print(f"Cumulative High:                 ${max_h:,.2f}")
print(f"Realized Volatility over 2 Hours: {cum_vol:.2f}% Annualized")
print(f"Total BTC Traded across 2 Hours: {df_comb['volume'].sum():,.2f} BTC")

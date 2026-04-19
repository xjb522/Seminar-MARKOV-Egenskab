import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tidyfinance as tf

print("succ")

# Import data
data_SP500 = tf.download_data(
    domain="stock_prices",
    symbols="^GSPC",
    start_date="2000-01-01",
    end_date="2026-02-28"
)

data_bonds_13_week = tf.download_data(
    domain="stock_prices",
    symbols="^IRX",
    start_date="2000-01-01",
    end_date="2026-02-28"
)

# S&P 500 monthly returns
sp500_m = (
    data_SP500
    .assign(date=lambda x: x["date"].dt.to_period("M").dt.to_timestamp())
    .groupby(["symbol", "date"], as_index=False)
    .agg(adjusted_close=("adjusted_close", "last"))
    .sort_values(["symbol", "date"])
    .assign(r_m=lambda x: x.groupby("symbol")["adjusted_close"].pct_change())
    .dropna(subset=["r_m"])
)

# IRX monthly yield -> monthly risk-free rate
irx_m = (
    data_bonds_13_week
    .assign(date=lambda x: x["date"].dt.to_period("M").dt.to_timestamp())
    .groupby(["symbol", "date"], as_index=False)
    .agg(irx=("adjusted_close", "last"))
    .sort_values(["symbol", "date"])
    .assign(r_f=lambda x: (1 + x["irx"] / 100) ** (1 / 12) - 1)
)

# Merge and compute excess returns
df = (
    sp500_m[["date", "r_m"]]
    .merge(irx_m[["date", "r_f"]], on="date", how="inner")
    .assign(r_excess=lambda x: x["r_m"] - x["r_f"])
    .dropna()
    .sort_values("date")
)

# Use date as index
df = df.set_index("date")

# Quick plot
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["r_excess"])
plt.title("Monthly S&P 500 Excess Returns")
plt.show()

# Final series for model
y = df["r_m"].astype(float).dropna()

print(y.isna().sum())
print(np.isinf(y).sum())
print(y.describe())

# Estimate 2-regime Markov-switching model
mod = sm.tsa.MarkovRegression(
    y,
    k_regimes=2,
    trend="c",
    switching_variance=True
)

res = mod.fit(em_iter=10, search_reps=5)
print(res.summary())

mod_ar = sm.tsa.MarkovAutoregression(
    y,
    k_regimes=2,
    order=1,
    trend="c",
    switching_variance=True
)

res_ar = mod_ar.fit(em_iter=10, search_reps=5)
print(res_ar.summary())
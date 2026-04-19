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
    .set_index("date")
)

# Vælg serie til modellerne
y = df["r_excess"].astype(float).dropna()

print(y.isna().sum())
print(np.isinf(y).sum())
print(y.describe())

# -----------------------------
# Model 1: MarkovRegression
# -----------------------------
mod = sm.tsa.MarkovRegression(
    y,
    k_regimes=2,
    trend="c",
    switching_variance=True
)

res = mod.fit(em_iter=10, search_reps=5)
print(res.summary())

# -----------------------------
# Model 2: MarkovAutoregression
# -----------------------------
mod_ar = sm.tsa.MarkovAutoregression(
    y,
    k_regimes=2,
    order=1,
    trend="c",
    switching_variance=True
)

res_ar = mod_ar.fit(em_iter=10, search_reps=5)
print(res_ar.summary())

# Smoothed probabilities
probs = res.smoothed_marginal_probabilities
probs_ar = res_ar.smoothed_marginal_probabilities


plt.clf
# -----------------------------
# Plot begge modeller
# -----------------------------
fig, axes = plt.subplots(
    3, 1, figsize=(14, 10), sharex=True,
    gridspec_kw={"height_ratios": [2, 1, 1]}
)

# Panel 1: returns
axes[0].plot(y.index, y, color="black", lw=1)
axes[0].set_title("Monthly S&P 500 Excess Returns")
axes[0].set_ylabel("Return")
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

# Panel 2: MarkovRegression
axes[1].plot(probs.index, probs[1], linestyle="--", lw=1.5)
axes[1].set_title("Smoothed Probability of Regime 2 - MarkovRegression")
axes[1].set_ylabel("Probability")
axes[1].set_ylim(0, 1)

# Panel 3: MarkovAutoregression
axes[2].plot(probs_ar.index, probs_ar[1], linestyle="--", lw=1.5)
axes[2].set_title("Smoothed Probability of Regime 2 - MarkovAutoregression")
axes[2].set_ylabel("Probability")
axes[2].set_ylim(0, 1)
axes[2].set_xlabel("Date")



plt.tight_layout()
plt.show()
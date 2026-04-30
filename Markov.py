import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tidyfinance as tf
from scipy.stats import norm
from statsmodels.tsa.ar_model import AutoReg
from arch import arch_model

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
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(sp500_m["date"], sp500_m["r_m"] * 100, linewidth=1)
ax.axhline(0, linestyle="--", linewidth=1)

ax.set_title("S&P 500 Monthly Returns (%)")
ax.set_xlabel("Date")
ax.set_ylabel("Monthly return (%)")

fig.tight_layout()


plt.show()

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

# I dit output er regime 0 høj-volatilitet:
high_vol_regime = 0

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
axes[1].plot(probs.index, probs[high_vol_regime], linestyle="--", lw=1.5)
axes[1].set_title("Smoothed Probability of High-Volatility Regime - MarkovRegression")
axes[1].set_ylabel("Probability")
axes[1].set_ylim(0, 1)

# Panel 3: MarkovAutoregression
axes[2].plot(probs_ar.index, probs_ar[high_vol_regime], linestyle="--", lw=1.5)
axes[2].set_title("Smoothed Probability of High-Volatility Regime - MarkovAutoregression")
axes[2].set_ylabel("Probability")
axes[2].set_ylim(0, 1)
axes[2].set_xlabel("Date")

plt.tight_layout()
plt.show()

# Estimerede parametre fra modellen
mu0 = res.params["const[0]"]
mu1 = res.params["const[1]"]
sigma0 = np.sqrt(res.params["sigma2[0]"])
sigma1 = np.sqrt(res.params["sigma2[1]"])

# Approksimativ regime-sandsynlighed fra smoothed probs
pi0 = probs[0].mean()
pi1 = probs[1].mean()

x = np.linspace(y.min() * 1.5, y.max() * 1.5, 1000)

f0 = norm.pdf(x, loc=mu0, scale=sigma0)
f1 = norm.pdf(x, loc=mu1, scale=sigma1)
f_mix = pi0 * f0 + pi1 * f1

plt.figure(figsize=(10, 6))
plt.plot(x, f0, "--", label="Regime 0 density")
plt.plot(x, f1, "--", label="Regime 1 density")
plt.plot(x, f_mix, color="black", linewidth=2, label="Mixture density")

plt.title("Estimated Two-Regime Mixture Density for S&P 500 Excess Returns")
plt.xlabel("Excess return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

ar1 = AutoReg(y, lags=1, old_names=False).fit()
print(ar1.summary())

arch1 = arch_model(y * 10, mean="Constant", vol="ARCH", p=1, q=0, dist="normal").fit(disp="off")
print(arch1.summary())

garch11 = arch_model(y *10 , mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
print(garch11.summary())

comparison = pd.DataFrame({
    "Model": [
        "MarkovRegression",
        "MarkovAutoregression",
        "AR(1)",
        "ARCH(1)",
        "GARCH(1,1)"
    ],
    "LogLik": [
        res.llf,
        res_ar.llf,
        ar1.llf,
        arch1.loglikelihood,
        garch11.loglikelihood
    ],
    "AIC": [
        res.aic,
        res_ar.aic,
        ar1.aic,
        arch1.aic,
        garch11.aic
    ],
    "BIC": [
        res.bic,
        res_ar.bic,
        ar1.bic,
        arch1.bic,
        garch11.bic
    ]
})

print(comparison.sort_values("AIC"))

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# GARCH volatility
ax[0].plot(y.index, garch11.conditional_volatility, label="GARCH volatility")
ax[0].set_title("GARCH Conditional Volatility")

# Markov regime probability
ax[1].plot(probs.index, probs[0], label="Prob high-vol regime", linestyle="--")
ax[1].set_title("Markov Regime Probability")

plt.tight_layout()
plt.show()
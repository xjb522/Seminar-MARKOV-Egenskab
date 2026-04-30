# ============================================================
# ACF/PACF + AR, MA, ARMA vs Markov Switching
# ============================================================

from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# ============================================================
# 0. Prepare data
# ============================================================

# Sørg for at y er en ren serie uden missing values
y = y.dropna()

# Split
split = int(len(y) * 0.8)

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Train sample:", y_train.index[0], "to", y_train.index[-1])
print("Test sample:", y_test.index[0], "to", y_test.index[-1])
print("Train observations:", len(y_train))
print("Test observations:", len(y_test))


# ============================================================
# 1. ACF and PACF analysis
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

plot_acf(y, lags=24, ax=axes[0])
axes[0].set_title("ACF of Monthly S&P 500 Excess Returns")

plot_pacf(y, lags=24, ax=axes[1], method="ywm")
axes[1].set_title("PACF of Monthly S&P 500 Excess Returns")

plt.tight_layout()
plt.show()


# ============================================================
# 2. AR(1) benchmark
# ============================================================

ar_model = AutoReg(y_train, lags=1, old_names=False).fit()

ar_forecast = ar_model.predict(
    start=len(y_train),
    end=len(y_train) + len(y_test) - 1
)

ar_forecast = pd.Series(ar_forecast.values, index=y_test.index)

rmse_ar = np.sqrt(np.mean((y_test - ar_forecast) ** 2))

print("\nAR(1) results")
print("AIC:", ar_model.aic)
print("BIC:", ar_model.bic)
print("RMSE:", rmse_ar)


# ============================================================
# 3. Markov Switching model
# ============================================================

mr = sm.tsa.MarkovRegression(
    y_train,
    k_regimes=2,
    trend="c",
    switching_variance=True
)

mr_res = mr.fit(
    em_iter=10,
    search_reps=5,
    disp=False
)

print("\nMarkov Switching results")
print("AIC:", mr_res.aic)
print("BIC:", mr_res.bic)
print("Log-likelihood:", mr_res.llf)


# ============================================================
# 4. Manual Markov forecast
# ============================================================

mu0 = mr_res.params["const[0]"]
mu1 = mr_res.params["const[1]"]

p00 = mr_res.params["p[0->0]"]
p10 = mr_res.params["p[1->0]"]

p01 = 1 - p00
p11 = 1 - p10

P = np.array([
    [p00, p01],
    [p10, p11]
])

print("\nTransition matrix:")
print(P)

last_probs = mr_res.filtered_marginal_probabilities.iloc[-1].values

markov_forecasts = []
current_probs = last_probs.copy()

for _ in range(len(y_test)):
    next_probs = current_probs @ P

    yhat = next_probs[0] * mu0 + next_probs[1] * mu1
    markov_forecasts.append(yhat)

    current_probs = next_probs

markov_forecast = pd.Series(markov_forecasts, index=y_test.index)

rmse_markov = np.sqrt(np.mean((y_test - markov_forecast) ** 2))

print("Markov RMSE:", rmse_markov)


# ============================================================
# 5. ARMA model selection
# ============================================================

arma_results = []

max_p = 5
max_q = 5

for p in range(max_p + 1):
    for q in range(max_q + 1):

        if p == 0 and q == 0:
            continue

        try:
            model = ARIMA(
                y_train,
                order=(p, 0, q),
                trend="c"
            )

            res = model.fit()

            arma_results.append({
                "p": p,
                "q": q,
                "AIC": res.aic,
                "BIC": res.bic,
                "LogLik": res.llf
            })

        except Exception as e:
            print(f"ARMA({p},{q}) failed: {e}")

arma_table = pd.DataFrame(arma_results)

arma_table = arma_table.sort_values("AIC").reset_index(drop=True)

print("\nTop 10 ARMA models by AIC:")
print(arma_table.head(10))


# ============================================================
# 6. Estimate best ARMA model
# ============================================================

best_row = arma_table.iloc[0]

best_p = int(best_row["p"])
best_q = int(best_row["q"])

print(f"\nBest ARMA model by AIC: ARMA({best_p},{best_q})")

best_arma = ARIMA(
    y_train,
    order=(best_p, 0, best_q),
    trend="c"
).fit()

print(best_arma.summary())


# ============================================================
# 7. ARMA out-of-sample forecast
# ============================================================

arma_forecast = best_arma.forecast(steps=len(y_test))
arma_forecast = pd.Series(arma_forecast.values, index=y_test.index)

rmse_arma = np.sqrt(np.mean((y_test - arma_forecast) ** 2))

print("\nOut-of-sample RMSE")
print("AR(1):", rmse_ar)
print("Markov Switching:", rmse_markov)
print(f"ARMA({best_p},{best_q}):", rmse_arma)


# ============================================================
# 8. Comparison table
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "AR(1)",
        "Markov Switching",
        f"ARMA({best_p},{best_q})"
    ],
    "LogLik": [
        ar_model.llf,
        mr_res.llf,
        best_arma.llf
    ],
    "AIC": [
        ar_model.aic,
        mr_res.aic,
        best_arma.aic
    ],
    "BIC": [
        ar_model.bic,
        mr_res.bic,
        best_arma.bic
    ],
    "Out-of-sample RMSE": [
        rmse_ar,
        rmse_markov,
        rmse_arma
    ]
})

print("\nModel comparison:")
print(comparison)


# ============================================================
# 9. Forecast error plot
# ============================================================

ar_error = y_test - ar_forecast
markov_error = y_test - markov_forecast
arma_error = y_test - arma_forecast

plt.figure(figsize=(12, 5))

plt.plot(ar_error.index, ar_error, lw=1.2, label="AR(1) forecast error")
plt.plot(markov_error.index, markov_error, lw=1.2, label="Markov forecast error")
plt.plot(arma_error.index, arma_error, lw=1.2, label=f"ARMA({best_p},{best_q}) forecast error")

plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)

plt.title("Out-of-Sample Forecast Errors")
plt.xlabel("Date")
plt.ylabel("Forecast error")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. Actual vs forecasts
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(y_test.index, y_test, color="black", lw=1.5, label="Actual excess return")
plt.plot(ar_forecast.index, ar_forecast, lw=1.2, label="AR(1) forecast")
plt.plot(markov_forecast.index, markov_forecast, lw=1.2, label="Markov forecast")
plt.plot(arma_forecast.index, arma_forecast, lw=1.2, label=f"ARMA({best_p},{best_q}) forecast")

plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)

plt.title("Actual vs Forecasted Excess Returns")
plt.xlabel("Date")
plt.ylabel("Excess return")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 11. Markov smoothed probabilities
# ============================================================

probs = mr_res.smoothed_marginal_probabilities

# Tjek hvilken regime der har højest volatilitet
sigma0 = mr_res.params["sigma2[0]"]
sigma1 = mr_res.params["sigma2[1]"]

if sigma1 > sigma0:
    high_vol_regime = 1
else:
    high_vol_regime = 0

print("\nHigh-volatility regime:", high_vol_regime)
print("Sigma regime 0:", np.sqrt(sigma0))
print("Sigma regime 1:", np.sqrt(sigma1))

plt.figure(figsize=(12, 5))

plt.plot(
    probs.index,
    probs[high_vol_regime],
    linestyle="--",
    lw=1.8,
    label="Probability of high-volatility regime"
)

plt.title("Markov Smoothed Probability of High-Volatility Regime")
plt.xlabel("Date")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()
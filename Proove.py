from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import tidyfinance as tf
from statsmodels.tsa.stattools import adfuller


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

Test_data = np.log(data_SP500["adjusted_close"])-np.log(data_SP500["adjusted_close"].shift(1))
Test_data= Test_data.dropna()

result = adfuller(Test_data)
print(result[0]) 
print(result[1])  # p-value
print(result[2])
print(result[3])
print(result[4])   


# =====================================================
# 1. DATA
# =====================================================
# Brug din egen stationære serie her
# Eksempel:
# y = data["r_excess"].dropna()
# eller:
# y = Test_data.dropna()

y = Test_data.dropna()

# Hvis Test_data er en DataFrame med én kolonne:
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# Sørg for at y er numerisk
y = pd.to_numeric(y, errors="coerce").dropna()

# =====================================================
# 2. TRAIN / TEST SPLIT
# =====================================================
split = int(len(y) * 0.8)

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Train observations:", len(y_train))
print("Test observations:", len(y_test))

# =====================================================
# 3. AR(1) MODEL
# =====================================================
ar_model = AutoReg(y_train, lags=1, old_names=False)
ar_res = ar_model.fit()

ar_forecast = ar_res.predict(
    start=len(y_train),
    end=len(y_train) + len(y_test) - 1,
    dynamic=False
)

ar_rmse = np.sqrt(mean_squared_error(y_test, ar_forecast))

# =====================================================
# 4. ARMA(1,1) MODEL
# ARMA(p,q) = ARIMA(p,0,q)
# =====================================================
arma_model = ARIMA(y_train, order=(1, 0, 1))
arma_res = arma_model.fit()

arma_forecast = arma_res.forecast(steps=len(y_test))

arma_rmse = np.sqrt(mean_squared_error(y_test, arma_forecast))

# =====================================================
# 5. MARKOV REGRESSION MODEL
# =====================================================
markov_model = sm.tsa.MarkovRegression(
    y_train,
    k_regimes=2,
    trend="c",
    switching_variance=True
)

markov_res = markov_model.fit(
    maxiter=1000,
    disp=False
)

# Simpel out-of-sample forecast:
# Vi bruger gennemsnittet af regime means vægtet med sidste smoothed probabilities

params = markov_res.params

print(markov_res.summary())

# Hent regime means
regime_means = []

for i in range(2):
    name = f"const[{i}]"
    if name in params.index:
        regime_means.append(params[name])
    else:
        regime_means.append(params.iloc[i])

regime_means = np.array(regime_means)

# Sidste sandsynlighed for hvert regime
last_probs = markov_res.smoothed_marginal_probabilities.iloc[-1].values

# Forecast = vægtet regime-gennemsnit
markov_one_step_forecast = np.sum(last_probs * regime_means)

markov_forecast = np.repeat(markov_one_step_forecast, len(y_test))

markov_rmse = np.sqrt(mean_squared_error(y_test, markov_forecast))

# =====================================================
# 6. MODEL COMPARISON TABLE
# =====================================================
comparison = pd.DataFrame({
    "Model": ["AR(1)", "ARMA(1,1)", "Markov Regression"],
    "AIC": [ar_res.aic, arma_res.aic, markov_res.aic],
    "BIC": [ar_res.bic, arma_res.bic, markov_res.bic],
    "RMSE": [ar_rmse, arma_rmse, markov_rmse]
})

print("\nModel comparison:")
print(comparison)

# =====================================================
# 7. PLOT FORECASTS
# =====================================================
plt.figure(figsize=(12, 6))

plt.plot(y_test.index, y_test, label="Actual returns")
plt.plot(y_test.index, ar_forecast, label="AR(1) forecast")
plt.plot(y_test.index, arma_forecast, label="ARMA(1,1) forecast")
plt.plot(y_test.index, markov_forecast, label="Markov forecast")

plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Forecast comparison: AR vs ARMA vs Markov Regression")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# 8. PLOT MARKOV REGIME PROBABILITIES
# =====================================================
plt.figure(figsize=(12, 6))

plt.plot(
    markov_res.smoothed_marginal_probabilities[0],
    label="Probability of Regime 0"
)

plt.plot(
    markov_res.smoothed_marginal_probabilities[1],
    label="Probability of Regime 1"
)

plt.title("Smoothed regime probabilities")
plt.xlabel("Date")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(Test_data, label="Returns")
plt.axhline(0, linestyle="--", linewidth=1)

plt.title("S&P 500 Returns (Test_data)")
plt.xlabel("Time")
plt.ylabel("Returns")
plt.legend()
plt.grid(True)

plt.show()


# Sørg for ren serie
y = pd.to_numeric(Test_data, errors="coerce").dropna()

# =========================
# AR(1)
# =========================
ar_model = AutoReg(y, lags=1, old_names=False)
ar_res = ar_model.fit()

ar_ll = ar_res.llf

# =========================
# ARMA(1,1)
# =========================
arma_model = ARIMA(y, order=(1,0,1))
arma_res = arma_model.fit()

arma_ll = arma_res.llf

# =========================
# MARKOV
# =========================
markov_model = sm.tsa.MarkovRegression(
    y,
    k_regimes=2,
    trend="c",
    switching_variance=True
)

markov_res = markov_model.fit(maxiter=1000, disp=False)

markov_ll = markov_res.llf

# =========================
# Sammenligning
# =========================
ll_comparison = pd.DataFrame({
    "Model": ["AR(1)", "ARMA(1,1)", "Markov"],
    "Log-Likelihood": [ar_ll, arma_ll, markov_ll]
})

print(ll_comparison)

### outliers analyse for stationaritet

y = pd.to_numeric(Test_data, errors="coerce").dropna()

# Beregn z-score
z_scores = (y - y.mean()) / y.std()

# Fjern outliers (typisk threshold = 3)
y_clean = y[np.abs(z_scores) < 3]

print("Original size:", len(y))
print("Cleaned size:", len(y_clean))

lower = y.quantile(0.01)
upper = y.quantile(0.99)

y_clean = y[(y >= lower) & (y <= upper)]

lower = y.quantile(0.01)
upper = y.quantile(0.99)

y_winsor = y.copy()
y_winsor[y_winsor < lower] = lower
y_winsor[y_winsor > upper] = upper

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(y)
plt.title("Original data")

plt.subplot(1,2,2)
plt.plot(y_clean)
plt.title("Without outliers")

plt.show()

### Split
split = int(len(y_clean) * 0.8)

y_train = y_clean.iloc[:split]
y_test = y_clean.iloc[split:]

# =========================
# AR(1)
# =========================
ar_res = AutoReg(y_train, lags=1, old_names=False).fit()

ar_forecast = ar_res.predict(
    start=len(y_train),
    end=len(y_train)+len(y_test)-1
)

ar_rmse = np.sqrt(mean_squared_error(y_test, ar_forecast))

# =========================
# ARMA(1,1)
# =========================
arma_res = ARIMA(y_train, order=(1,0,1)).fit()

arma_forecast = arma_res.forecast(steps=len(y_test))
arma_rmse = np.sqrt(mean_squared_error(y_test, arma_forecast))

# =========================
# MARKOV - 3 REGIMES
# =========================
markov_res = sm.tsa.MarkovRegression(
    y_train,
    k_regimes=3,
    trend="c",
    switching_variance=True
).fit(maxiter=1000, disp=False)

# Simpel forecast
params = markov_res.params

# Hent regime means for alle 3 regimer
regime_means = np.array([
    params.get("const[0]", params.iloc[0]),
    params.get("const[1]", params.iloc[1]),
    params.get("const[2]", params.iloc[2])
])

# Sidste sandsynlighed for hvert regime
last_probs = markov_res.smoothed_marginal_probabilities.iloc[-1].values

# Sikkerhedstjek
print("Regime means:", regime_means)
print("Last regime probabilities:", last_probs)
print("Sum of probabilities:", last_probs.sum())

# Forecast = sandsynlighedsvægtet gennemsnit
markov_forecast_value = np.sum(last_probs * regime_means)

markov_forecast = np.repeat(markov_forecast_value, len(y_test))

markov_rmse = np.sqrt(mean_squared_error(y_test, markov_forecast))

# =========================
# RESULTATER
# =========================
comparison_clean = pd.DataFrame({
    "Model": ["AR(1)", "ARMA(1,1)", "Markov"],
    "AIC": [ar_res.aic, arma_res.aic, markov_res.aic],
    "BIC": [ar_res.bic, arma_res.bic, markov_res.bic],
    "RMSE": [ar_rmse, arma_rmse, markov_rmse]
})

print(comparison_clean)
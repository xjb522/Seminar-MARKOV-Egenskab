from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Split
split = int(len(y) * 0.8)
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# ---------- AR(1) ----------
ar_model = AutoReg(y_train, lags=1, old_names=False).fit()
ar_forecast = ar_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

# ---------- MarkovRegression ----------
mr = sm.tsa.MarkovRegression(
    y_train,
    k_regimes=2,
    trend="c",
    switching_variance=True
)
mr_res = mr.fit(em_iter=10, search_reps=5)

# Parametre
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

# Start fra sidste in-sample regime-sandsynlighed
# filtered kan være bedre til forecast end smoothed
last_probs = mr_res.filtered_marginal_probabilities.iloc[-1].values
# hvis dine probabilities ligger som kolonner [0,1], så giver det en vektor [p0, p1]

markov_forecasts = []
current_probs = last_probs.copy()

for _ in range(len(y_test)):
    # næste periodes regime-sandsynlighed
    next_probs = current_probs @ P

    # forecast af mean
    yhat = next_probs[0] * mu0 + next_probs[1] * mu1
    markov_forecasts.append(yhat)

    # uden nye observationer glider man videre med de samme opdaterede sandsynligheder
    current_probs = next_probs

markov_forecast = pd.Series(markov_forecasts, index=y_test.index)

# ---------- RMSE uden sklearn ----------
rmse_ar = np.sqrt(np.mean((y_test - ar_forecast) ** 2))
rmse_markov = np.sqrt(np.mean((y_test - markov_forecast) ** 2))

print("AR RMSE:", rmse_ar)
print("Markov RMSE:", rmse_markov)

# -----------------------------------
# Sørg for at disse allerede findes:
# -----------------------------------
# y                     : hele excess return-serien
# garch11               : fitted GARCH-model
# probs                 : smoothed probabilities fra MarkovRegression
# y_test                : test sample
# ar_forecast           : AR forecast for test sample
# markov_forecast       : manuel Markov forecast for test sample
#
# Hvis dit high-vol regime er regime 1, så sæt:
high_vol_regime = 1
# Hvis det i din model er regime 0, så skift til 0

# -----------------------------------
# Forecast errors
# -----------------------------------
ar_error = y_test - ar_forecast
markov_error = y_test - markov_forecast

# -----------------------------------
# Figure
# -----------------------------------
fig, axes = plt.subplots(
    4, 1,
    figsize=(14, 12),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1.4, 1.4, 1.2]}
)

# =========================
# Panel 1: Excess returns
# =========================
axes[0].plot(y.index, y, color="black", lw=1)
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[0].set_title("Monthly S&P 500 Excess Returns")
axes[0].set_ylabel("Return")

# Optional: shade high-vol regimes
# Vi bruger Markov-sandsynligheder til at markere perioder,
# hvor high-vol regime sandsynlighed er over 0.5
high_regime_indicator = (probs[high_vol_regime] > 0.5).astype(int)

start = None
for i in range(len(high_regime_indicator)):
    if high_regime_indicator.iloc[i] == 1 and start is None:
        start = high_regime_indicator.index[i]
    elif high_regime_indicator.iloc[i] == 0 and start is not None:
        end = high_regime_indicator.index[i]
        axes[0].axvspan(start, end, alpha=0.15)
        start = None

# Hvis serien slutter i et high-vol regime
if start is not None:
    axes[0].axvspan(start, high_regime_indicator.index[-1], alpha=0.15)

# =========================
# Panel 2: GARCH volatility
# =========================
# Hvis du estimerede GARCH på y*10 eller y*100,
# så skal du evt. skalere tilbage:
# fx /10 eller /100.
# Hvis du brugte y direkte, så behold linjen som den er.
garch_vol = pd.Series(garch11.conditional_volatility, index=y.index)

axes[1].plot(garch_vol.index, garch_vol, lw=1.5, label="GARCH conditional volatility")
axes[1].set_title("GARCH Conditional Volatility")
axes[1].set_ylabel("Volatility")
axes[1].legend(loc="upper right")

# =========================
# Panel 3: Markov probability
# =========================
axes[2].plot(
    probs.index,
    probs[high_vol_regime],
    linestyle="--",
    lw=1.8,
    label="Probability of high-volatility regime"
)
axes[2].set_title("Markov Smoothed Probability of High-Volatility Regime")
axes[2].set_ylabel("Probability")
axes[2].set_ylim(0, 1)
axes[2].legend(loc="upper right")

# =========================
# Panel 4: Forecast errors
# =========================
axes[3].plot(ar_error.index, ar_error, lw=1.2, label="AR(1) forecast error")
axes[3].plot(markov_error.index, markov_error, lw=1.2, label="Markov forecast error")
axes[3].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[3].set_title("Out-of-Sample Forecast Errors")
axes[3].set_ylabel("Error")
axes[3].set_xlabel("Date")
axes[3].legend(loc="upper right")

plt.tight_layout()
plt.show()

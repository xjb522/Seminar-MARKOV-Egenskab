split = int(len(y) * 0.8)

y_train = y.iloc[:split]
y_test = y.iloc[split:]

ar_model = AutoReg(y_train, lags=1, old_names=False).fit()

markov_model = sm.tsa.MarkovRegression(
    y_train,
    k_regimes=2,
    trend="c",
    switching_variance=True
).fit()

ar_forecast = ar_model.predict(
    start=y_test.index[0],
    end=y_test.index[-1]
)

markov_pred = markov_model.predict(
    start=y_test.index[0],
    end=y_test.index[-1]
)

from sklearn.metrics import mean_squared_error
import numpy as np

rmse_ar = np.sqrt(mean_squared_error(y_test, ar_forecast))
rmse_markov = np.sqrt(mean_squared_error(y_test, markov_pred))

print("AR RMSE:", rmse_ar)
print("Markov RMSE:", rmse_markov)
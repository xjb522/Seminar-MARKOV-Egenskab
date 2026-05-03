from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg

# ============================================================
# 1. Select best AR(p) model by AIC
# ============================================================

ar_results = []

max_p = 12

for p in range(1, max_p + 1):
    try:
        ar_fit = AutoReg(y, lags=p, old_names=False).fit()

        ar_results.append({
            "Model": f"AR({p})",
            "p": p,
            "LogLik": ar_fit.llf,
            "AIC": ar_fit.aic,
            "BIC": ar_fit.bic
        })

    except Exception as e:
        print(f"AR({p}) failed: {e}")

ar_table = pd.DataFrame(ar_results).sort_values("AIC").reset_index(drop=True)

best_ar_p = int(ar_table.loc[0, "p"])
best_ar = AutoReg(y, lags=best_ar_p, old_names=False).fit()


# ============================================================
# 2. Select best ARMA(p,q) model by AIC
# ============================================================

arma_results = []

max_p = 5
max_q = 5

for p in range(max_p + 1):
    for q in range(max_q + 1):

        if p == 0 and q == 0:
            continue

        try:
            arma_fit = ARIMA(
                y,
                order=(p, 0, q),
                trend="c"
            ).fit()

            arma_results.append({
                "Model": f"ARMA({p},{q})",
                "p": p,
                "q": q,
                "LogLik": arma_fit.llf,
                "AIC": arma_fit.aic,
                "BIC": arma_fit.bic
            })

        except Exception as e:
            print(f"ARMA({p},{q}) failed: {e}")

arma_table = pd.DataFrame(arma_results).sort_values("AIC").reset_index(drop=True)

best_arma_p = int(arma_table.loc[0, "p"])
best_arma_q = int(arma_table.loc[0, "q"])

best_arma = ARIMA(
    y,
    order=(best_arma_p, 0, best_arma_q),
    trend="c"
).fit()


# ============================================================
# 3. Final in-sample comparison table
# Uses your existing Markov models:
# res    = MarkovRegression result
# res_ar = MarkovAutoregression result
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Markov Regression",
        "Markov Autoregression",
        f"AR({best_ar_p})",
        f"ARMA({best_arma_p},{best_arma_q})"
    ],
    "LogLik": [
        res.llf,
        res_ar.llf,
        best_ar.llf,
        best_arma.llf
    ],
    "AIC": [
        res.aic,
        res_ar.aic,
        best_ar.aic,
        best_arma.aic
    ],
    "BIC": [
        res.bic,
        res_ar.bic,
        best_ar.bic,
        best_arma.bic
    ]
})

comparison = comparison.sort_values("AIC").reset_index(drop=True)

print("\nBest AR models by AIC:")
print(ar_table.head(10))

print("\nBest ARMA models by AIC:")
print(arma_table.head(10))

print("\nFinal in-sample model comparison:")
print(comparison)


# ============================================================
# 4. Export LaTeX table
# ============================================================

latex_table = comparison.to_latex(
    index=False,
    float_format="%.3f",
    caption="In-sample model comparison",
    label="tab:in_sample_comparison"
)

print("\nLaTeX table:")
print(latex_table)
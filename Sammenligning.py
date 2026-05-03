import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

# ============================================================
# 0. Prepare data
# ============================================================

y = df["r_excess"].astype(float).dropna()

# Optional: make monthly frequency explicit
y = y.asfreq("MS")

# ============================================================
# 1. Estimate MarkovRegression
# ============================================================

mod_mr = sm.tsa.MarkovRegression(
    y,
    k_regimes=2,
    trend="c",
    switching_variance=True
)

res_mr = mod_mr.fit(
    em_iter=10,
    search_reps=5,
    disp=False
)

# ============================================================
# 2. Estimate MarkovAutoregression
# ============================================================

mod_mar = sm.tsa.MarkovAutoregression(
    y,
    k_regimes=2,
    order=1,
    trend="c",
    switching_variance=True
)

res_mar = mod_mar.fit(
    em_iter=10,
    search_reps=5,
    disp=False
)

# ============================================================
# 3. Select best AR(p) model by AIC
# ============================================================

ar_results = []

max_p = 12

for p in range(1, max_p + 1):
    try:
        res_ar = AutoReg(y, lags=p, old_names=False).fit()

        ar_results.append({
            "Model": f"AR({p})",
            "p": p,
            "LogLik": res_ar.llf,
            "AIC": res_ar.aic,
            "BIC": res_ar.bic
        })

    except Exception as e:
        print(f"AR({p}) failed: {e}")

ar_table = pd.DataFrame(ar_results).sort_values("AIC").reset_index(drop=True)

best_ar_p = int(ar_table.loc[0, "p"])
best_ar = AutoReg(y, lags=best_ar_p, old_names=False).fit()

# ============================================================
# 4. Select best ARMA(p,q) model by AIC
# ============================================================

arma_results = []

max_p = 5
max_q = 5

for p in range(max_p + 1):
    for q in range(max_q + 1):

        if p == 0 and q == 0:
            continue

        try:
            res_arma = ARIMA(
                y,
                order=(p, 0, q),
                trend="c"
            ).fit()

            arma_results.append({
                "Model": f"ARMA({p},{q})",
                "p": p,
                "q": q,
                "LogLik": res_arma.llf,
                "AIC": res_arma.aic,
                "BIC": res_arma.bic
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
# 5. Final in-sample comparison table
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "MarkovRegression",
        "MarkovAutoregression",
        f"AR({best_ar_p})",
        f"ARMA({best_arma_p},{best_arma_q})"
    ],
    "LogLik": [
        res_mr.llf,
        res_mar.llf,
        best_ar.llf,
        best_arma.llf
    ],
    "AIC": [
        res_mr.aic,
        res_mar.aic,
        best_ar.aic,
        best_arma.aic
    ],
    "BIC": [
        res_mr.bic,
        res_mar.bic,
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
# 6. Optional: export LaTeX table
# ============================================================

latex_table = comparison.to_latex(
    index=False,
    float_format="%.3f",
    caption="In-sample model comparison",
    label="tab:in_sample_comparison"
)

print("\nLaTeX table:")
print(latex_table)
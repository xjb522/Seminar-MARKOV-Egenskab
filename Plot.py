# =====================================================
# Compare crisis periods with Markov high-vol probability
# =====================================================

high_vol_regime = 0  # based on your output: regime 0 = high volatility

fig, axes = plt.subplots(
    2, 1,
    figsize=(14, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

# -----------------------------------------------------
# Panel 1: Monthly excess returns
# -----------------------------------------------------
axes[0].plot(
    y.index,
    y * 100,
    color="black",
    linewidth=1,
    label="Monthly excess returns"
)

axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

# Crisis windows
axes[0].axvspan(
    pd.Timestamp("2008-09-01"),
    pd.Timestamp("2009-06-01"),
    alpha=0.20,
    color="steelblue",
    label="Financial crisis"
)

axes[0].axvspan(
    pd.Timestamp("2020-02-01"),
    pd.Timestamp("2022-05-01"),
    alpha=0.18,
    color="red",
    label="COVID shock"
)

axes[0].set_title("S&P 500 Excess Returns and Known Market Stress Periods")
axes[0].set_ylabel("Monthly excess return (%)")
axes[0].legend(loc="upper right")

# -----------------------------------------------------
# Panel 2: Markov high-volatility probability
# -----------------------------------------------------
axes[1].plot(
    probs.index,
    probs[high_vol_regime],
    color="darkorange",
    linewidth=2,
    label="Smoothed probability: high-vol regime"
)

axes[1].axhline(
    0.5,
    color="gray",
    linestyle="--",
    linewidth=1,
    label="0.5 threshold"
)

# Same crisis windows
axes[1].axvspan(
    pd.Timestamp("2008-09-01"),
    pd.Timestamp("2009-06-01"),
    alpha=0.20,
    color="steelblue"
)

axes[1].axvspan(
    pd.Timestamp("2020-02-01"),
    pd.Timestamp("2022-05-01"),
    alpha=0.18,
    color="red"
)

axes[1].set_title("Markov-Switching Model: Inferred High-Volatility Regime")
axes[1].set_ylabel("Probability")
axes[1].set_xlabel("Date")
axes[1].set_ylim(0, 1)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()
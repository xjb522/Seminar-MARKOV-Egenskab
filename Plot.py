# -----------------------------
# Choose high-vol regime
# -----------------------------
# Based on your output: regime 0 has higher variance
high_vol_regime = 0
low_vol_regime = 1

p_high = probs[high_vol_regime]
p_low = probs[low_vol_regime]

# Most likely regime each month
regime_class = np.where(p_high > 0.5, "High volatility", "Low volatility")

plot_df = pd.DataFrame({
    "r_excess": y,
    "p_high": p_high,
    "p_low": p_low,
    "regime": regime_class
}).dropna()

# -----------------------------
# Helper function: shade regimes
# -----------------------------
def shade_regimes(ax, dates, regimes):
    current_regime = regimes.iloc[0]
    start_date = dates[0]

    for i in range(1, len(dates)):
        if regimes.iloc[i] != current_regime:
            end_date = dates[i]

            if current_regime == "High volatility":
                ax.axvspan(start_date, end_date, color="red", alpha=0.15)
            else:
                ax.axvspan(start_date, end_date, color="green", alpha=0.08)

            start_date = dates[i]
            current_regime = regimes.iloc[i]

    # last shaded area
    if current_regime == "High volatility":
        ax.axvspan(start_date, dates[-1], color="red", alpha=0.15)
    else:
        ax.axvspan(start_date, dates[-1], color="green", alpha=0.08)


# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(
    2, 1,
    figsize=(15, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

# =====================================================
# Panel 1: Excess returns + inferred regimes
# =====================================================

shade_regimes(
    axes[0],
    plot_df.index,
    plot_df["regime"]
)

axes[0].plot(
    plot_df.index,
    plot_df["r_excess"] * 100,
    color="black",
    linewidth=1,
    label="Monthly excess returns"
)

axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

# Known stress periods
axes[0].axvspan(
    pd.Timestamp("2008-09-01"),
    pd.Timestamp("2009-06-01"),
    color="blue",
    alpha=0.15,
    label="Financial crisis"
)

axes[0].axvspan(
    pd.Timestamp("2020-02-01"),
    pd.Timestamp("2021-12-01"),
    color="red",
    alpha=0.20,
    label="COVID shock"
)

# Optional labels
axes[0].annotate(
    "Financial crisis",
    xy=(pd.Timestamp("2008-10-01"), -15),
    xytext=(pd.Timestamp("2005-06-01"), -12),
    arrowprops=dict(arrowstyle="->", linewidth=1),
    fontsize=10
)

axes[0].annotate(
    "COVID-19 shock",
    xy=(pd.Timestamp("2020-03-01"), -12),
    xytext=(pd.Timestamp("2017-10-01"), -9),
    arrowprops=dict(arrowstyle="->", linewidth=1),
    fontsize=10
)

axes[0].set_title(
    "S&P 500 Excess Returns and Inferred Market Regimes",
    fontsize=14
)

axes[0].set_ylabel("Monthly excess return (%)")
axes[0].legend(loc="upper right")


# =====================================================
# Panel 2: Smoothed regime probabilities
# =====================================================

axes[1].plot(
    plot_df.index,
    plot_df["p_high"],
    color="red",
    linewidth=2,
    label="High-volatility regime"
)

axes[1].plot(
    plot_df.index,
    plot_df["p_low"],
    color="green",
    linewidth=2,
    label="Low-volatility regime"
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
    color="blue",
    alpha=0.15
)

axes[1].axvspan(
    pd.Timestamp("2020-02-01"),
    pd.Timestamp("2021-12-01"),
    color="red",
    alpha=0.20
)

axes[1].set_title(
    "Markov-Switching Model: Smoothed Regime Probabilities",
    fontsize=13
)

axes[1].set_ylabel("Probability")
axes[1].set_xlabel("Date")
axes[1].set_ylim(0, 1)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()
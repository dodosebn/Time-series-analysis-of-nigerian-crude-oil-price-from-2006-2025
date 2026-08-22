"""
=============================================================================
 TIME SERIES ANALYSIS OF MONTHLY CRUDE OIL PRICES IN NIGERIA (2006–2025)
 STAGE 2: Stationarity Analysis
=============================================================================
 Specific Objective ii:
   Investigate the time series properties with emphasis on stationarity.
   Tests used:
     • Augmented Dickey–Fuller (ADF)   – H₀: unit root (non-stationary)
     • Kwiatkowski–Phillips–Schmidt–Shin (KPSS) – H₀: stationary
     • Phillips–Perron (PP)            – H₀: unit root (non-stationary)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_excel("mydata.xlsx")
df = df.sort_values(["tyear", "tmonth"]).reset_index(drop=True)
df["date"] = pd.to_datetime(dict(year=df["tyear"], month=df["tmonth"], day=1))
df = df.set_index("date")
series = df["crudeOilPrice"].copy()
series.index.freq = "MS"

# First difference (removes trend / makes series stationary)
series_diff = series.diff().dropna()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER – FORMAT TEST RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def run_adf(ts, label=""):
    """Augmented Dickey-Fuller test."""
    result = adfuller(ts, autolag="AIC")
    stat, pval, lags, nobs, crit = result[0], result[1], result[2], result[3], result[4]
    decision = "REJECT H₀ → STATIONARY" if pval < 0.05 else "FAIL TO REJECT H₀ → NON-STATIONARY"
    print(f"\n{'─'*55}")
    print(f"  ADF Test — {label}")
    print(f"{'─'*55}")
    print(f"  Test Statistic : {stat:.4f}")
    print(f"  p-value        : {pval:.4f}")
    print(f"  Lags Used      : {lags}")
    print(f"  Obs. Used      : {nobs}")
    for key, val in crit.items():
        print(f"  Critical ({key})  : {val:.4f}")
    print(f"  Decision       : {decision}")
    return stat, pval, decision

def run_kpss(ts, label=""):
    """KPSS test (H₀: stationary)."""
    stat, pval, lags, crit = kpss(ts, regression="c", nlags="auto")
    decision = "REJECT H₀ → NON-STATIONARY" if pval < 0.05 else "FAIL TO REJECT H₀ → STATIONARY"
    print(f"\n{'─'*55}")
    print(f"  KPSS Test — {label}")
    print(f"{'─'*55}")
    print(f"  Test Statistic : {stat:.4f}")
    print(f"  p-value        : {pval:.4f}")
    print(f"  Lags Used      : {lags}")
    for key, val in crit.items():
        print(f"  Critical ({key})  : {val:.4f}")
    print(f"  Decision       : {decision}")
    return stat, pval, decision

# ─────────────────────────────────────────────────────────────────────────────
# 3.  RUN TESTS ON LEVEL SERIES
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  STATIONARITY TESTS — LEVEL SERIES (Original Prices)")
print("=" * 55)
adf_stat_lv, adf_pval_lv, adf_dec_lv = run_adf(series, "Level Series")
kpss_stat_lv, kpss_pval_lv, kpss_dec_lv = run_kpss(series, "Level Series")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  RUN TESTS ON FIRST-DIFFERENCED SERIES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STATIONARITY TESTS — FIRST-DIFFERENCED SERIES")
print("=" * 55)
adf_stat_d1, adf_pval_d1, adf_dec_d1 = run_adf(series_diff, "First Difference")
kpss_stat_d1, kpss_pval_d1, kpss_dec_d1 = run_kpss(series_diff, "First Difference")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  STATIONARITY TEST SUMMARY")
print("=" * 65)
print(f"  {'Series':<25} {'ADF p-val':<12} {'KPSS p-val':<13} {'Conclusion'}")
print(f"  {'-'*63}")
print(f"  {'Level (original)':<25} {adf_pval_lv:<12.4f} {kpss_pval_lv:<13.4f} {'I(1) – Non-stationary'}")
print(f"  {'First Difference':<25} {adf_pval_d1:<12.4f} {kpss_pval_d1:<13.4f} {'I(0) – Stationary'}")
print("=" * 65)
print("  Note: Series is integrated of order 1 → d = 1 in ARIMA(p,1,q)")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FIGURE 6 – LEVEL vs DIFFERENCED SERIES (side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(series.index, series.values, color="#1f77b4", linewidth=1.2)
axes[0].set_title("(a) Original Crude Oil Price Series [I(1) – Non-Stationary]",
                  fontsize=11, fontweight="bold")
axes[0].set_ylabel("Price (USD/bbl)", fontsize=10)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

axes[1].plot(series_diff.index, series_diff.values, color="#d62728", linewidth=1.0)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("(b) First-Differenced Series [I(0) – Stationary]",
                  fontsize=11, fontweight="bold")
axes[1].set_ylabel("Δ Price (USD/bbl)", fontsize=10)
axes[1].set_xlabel("Year", fontsize=10)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.suptitle("Stationarity Check: Original vs. First-Differenced Crude Oil Price",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("fig6_stationarity_series.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] fig6_stationarity_series.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FIGURE 7 – ACF & PACF PLOTS (Level + Differenced)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_acf(series, lags=40, ax=axes[0, 0], color="#1f77b4", alpha=0.05)
axes[0, 0].set_title("ACF — Original Series", fontsize=11, fontweight="bold")

plot_pacf(series, lags=40, ax=axes[0, 1], method="ywm", color="#1f77b4", alpha=0.05)
axes[0, 1].set_title("PACF — Original Series", fontsize=11, fontweight="bold")

plot_acf(series_diff, lags=40, ax=axes[1, 0], color="#d62728", alpha=0.05)
axes[1, 0].set_title("ACF — First-Differenced Series", fontsize=11, fontweight="bold")

plot_pacf(series_diff, lags=40, ax=axes[1, 1], method="ywm", color="#d62728", alpha=0.05)
axes[1, 1].set_title("PACF — First-Differenced Series", fontsize=11, fontweight="bold")

for ax in axes.flat:
    ax.set_xlabel("Lag", fontsize=9)
    ax.set_ylabel("Correlation", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

fig.suptitle("ACF and PACF Plots for Model Order Identification",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig7_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] fig7_acf_pacf.png")

print("\n✓ Stage 2 complete. Stationarity tests done and plots saved.")
print("  → Use d = 1 for the ARIMA model (proceed to Stage 3).")

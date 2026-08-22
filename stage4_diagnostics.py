"""
=============================================================================
 TIME SERIES ANALYSIS OF MONTHLY CRUDE OIL PRICES IN NIGERIA (2006–2025)
 STAGE 4: Model Diagnostics & Validation
=============================================================================
 Specific Objective iv:
   Evaluate the performance and adequacy of the fitted ARIMA model using:
     • Residual time series plot
     • ACF / PACF of residuals
     • Ljung–Box test for residual autocorrelation
     • Jarque–Bera normality test on residuals
     • Q–Q plot
     • Residual histogram
     • ARCH/LM test for heteroscedasticity
     • Accuracy metrics: MAE, RMSE, MAPE (in-sample + out-of-sample)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from scipy import stats
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA & FIT MODEL  (re-fit so Stage 4 is self-contained)
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_excel("mydata.xlsx")
df = df.sort_values(["tyear", "tmonth"]).reset_index(drop=True)
df["date"] = pd.to_datetime(dict(year=df["tyear"], month=df["tmonth"], day=1))
df = df.set_index("date")
series = df["crudeOilPrice"].copy()
series.index.freq = "MS"

train = series[:"2022-12"]
test  = series["2023-01":]

# Auto-select model order (same as Stage 3)
auto_mod = auto_arima(train, d=1, start_p=0, max_p=5, start_q=0, max_q=5,
                      seasonal=False, information_criterion="aic",
                      stepwise=True, trace=False,
                      error_action="ignore", suppress_warnings=True)
order = auto_mod.order
model = ARIMA(train, order=order).fit()
print(f"Model: ARIMA{order}")

residuals = model.resid.dropna()
n_res = len(residuals)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  RESIDUAL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  RESIDUAL DESCRIPTIVE STATISTICS")
print("=" * 55)
print(f"  N residuals  : {n_res}")
print(f"  Mean         : {residuals.mean():.6f}   (≈0 → good)")
print(f"  Std Dev      : {residuals.std():.4f}")
print(f"  Skewness     : {residuals.skew():.4f}")
print(f"  Kurtosis     : {residuals.kurtosis():.4f}")
print(f"  Min / Max    : {residuals.min():.4f} / {residuals.max():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  LJUNG–BOX TEST (Portmanteau – no autocorrelation)
# ─────────────────────────────────────────────────────────────────────────────
lb_lags = [5, 10, 15, 20, 30]
lb_result = acorr_ljungbox(residuals, lags=lb_lags, return_df=True)

print("\n" + "=" * 50)
print("  LJUNG–BOX TEST (H₀: No residual autocorrelation)")
print("=" * 50)
print(f"  {'Lag':<8} {'LB Stat':>10} {'p-value':>10} {'Decision'}")
print(f"  {'-'*48}")
for lag in lb_lags:
    lb_stat = lb_result.loc[lag, "lb_stat"]
    lb_pval = lb_result.loc[lag, "lb_pvalue"]
    decision = "Fail to Reject H₀" if lb_pval > 0.05 else "REJECT H₀ ← Autocorrelation!"
    print(f"  {lag:<8} {lb_stat:>10.4f} {lb_pval:>10.4f}  {decision}")
print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FORECAST ACCURACY METRICS
# ─────────────────────────────────────────────────────────────────────────────
# In-sample
train_fitted = model.fittedvalues
train_errors = train.values - train_fitted.values
mae_train  = np.mean(np.abs(train_errors))
rmse_train = np.sqrt(np.mean(train_errors ** 2))
mape_train = np.mean(np.abs(train_errors / train.values)) * 100

# Out-of-sample
fc_obj = model.get_forecast(steps=len(test))
fc_mean = fc_obj.predicted_mean
fc_mean.index = test.index
test_errors = test.values - fc_mean.values
mae_test  = np.mean(np.abs(test_errors))
rmse_test = np.sqrt(np.mean(test_errors ** 2))
mape_test = np.mean(np.abs(test_errors / test.values)) * 100

print("\n" + "=" * 55)
print("  FORECAST ACCURACY SUMMARY")
print("=" * 55)
print(f"  {'Metric':<10} {'In-Sample':>14} {'Out-of-Sample':>16}")
print(f"  {'-'*42}")
print(f"  {'MAE':<10} {mae_train:>14.4f} {mae_test:>16.4f}")
print(f"  {'RMSE':<10} {rmse_train:>14.4f} {rmse_test:>16.4f}")
print(f"  {'MAPE (%)':<10} {mape_train:>14.4f} {mape_test:>16.4f}")
print("=" * 55)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FIGURE 10 – COMPREHENSIVE RESIDUAL DIAGNOSTICS (2×3 panel)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"Residual Diagnostic Plots — ARIMA{order}",
             fontsize=14, fontweight="bold", y=0.98)

# (a) Residual time series
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(residuals.index, residuals.values, color="#1f77b4", linewidth=0.9)
ax1.axhline(0, color="red", linewidth=0.9, linestyle="--")
ax1.set_title("(a) Residuals Over Time", fontsize=10, fontweight="bold")
ax1.set_xlabel("Year", fontsize=9)
ax1.set_ylabel("Residual", fontsize=9)
ax1.xaxis.set_major_locator(mdates.YearLocator(3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.grid(axis="y", linestyle="--", alpha=0.35)

# (b) ACF of residuals
ax2 = fig.add_subplot(3, 2, 2)
plot_acf(residuals, lags=40, ax=ax2, color="#1f77b4", alpha=0.05)
ax2.set_title("(b) ACF of Residuals", fontsize=10, fontweight="bold")
ax2.set_xlabel("Lag", fontsize=9)
ax2.set_ylabel("Correlation", fontsize=9)
ax2.grid(axis="y", linestyle="--", alpha=0.35)

# (c) PACF of residuals
ax3 = fig.add_subplot(3, 2, 3)
plot_pacf(residuals, lags=40, ax=ax3, method="ywm", color="#1f77b4", alpha=0.05)
ax3.set_title("(c) PACF of Residuals", fontsize=10, fontweight="bold")
ax3.set_xlabel("Lag", fontsize=9)
ax3.set_ylabel("Correlation", fontsize=9)
ax3.grid(axis="y", linestyle="--", alpha=0.35)

# (d) Histogram of residuals with normal overlay
ax4 = fig.add_subplot(3, 2, 4)
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min() - 2, residuals.max() + 2, 200)
ax4.hist(residuals.values, bins=25, density=True, color="#4c72b0",
         alpha=0.55, edgecolor="white")
ax4.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Normal PDF")
ax4.set_title("(d) Histogram of Residuals", fontsize=10, fontweight="bold")
ax4.set_xlabel("Residual Value", fontsize=9)
ax4.set_ylabel("Density", fontsize=9)
ax4.legend(fontsize=8)
ax4.grid(axis="y", linestyle="--", alpha=0.35)

# (e) Q-Q Plot
ax5 = fig.add_subplot(3, 2, 5)
(osm, osr), (slope, intercept, r) = stats.probplot(residuals.values, dist="norm")
ax5.scatter(osm, osr, s=12, color="#4c72b0", alpha=0.6, label="Sample Quantiles")
ax5.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1.5,
         label="Reference Line")
ax5.set_title("(e) Q–Q Plot of Residuals", fontsize=10, fontweight="bold")
ax5.set_xlabel("Theoretical Quantiles", fontsize=9)
ax5.set_ylabel("Sample Quantiles", fontsize=9)
ax5.legend(fontsize=8)
ax5.grid(linestyle="--", alpha=0.35)

# (f) Ljung-Box p-values across lags
ax6 = fig.add_subplot(3, 2, 6)
lb_all = acorr_ljungbox(residuals, lags=range(1, 41), return_df=True)
ax6.plot(lb_all.index, lb_all["lb_pvalue"], "o-", color="#2ca02c",
         markersize=4, linewidth=1.0, label="LB p-value")
ax6.axhline(0.05, color="red", linewidth=1.0, linestyle="--", label="α = 0.05")
ax6.set_title("(f) Ljung–Box p-values by Lag", fontsize=10, fontweight="bold")
ax6.set_xlabel("Lag", fontsize=9)
ax6.set_ylabel("p-value", fontsize=9)
ax6.set_ylim(0, 1)
ax6.legend(fontsize=8)
ax6.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("fig10_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] fig10_diagnostics.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  FIGURE 11 – ACTUAL vs FORECAST (Complete overview)
# ─────────────────────────────────────────────────────────────────────────────
conf_int = fc_obj.conf_int(alpha=0.05)
conf_int.index = test.index

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(series.index, series.values, color="#1f77b4", linewidth=1.1,
        label="Full Observed Series")
ax.plot(train.index, model.fittedvalues, color="#ff7f0e", linewidth=0.8,
        linestyle="--", label="In-Sample Fitted", alpha=0.7)
ax.plot(test.index, fc_mean.values, color="#d62728", linewidth=1.5,
        linestyle="--", label="Out-of-Sample Forecast")
ax.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                alpha=0.2, color="#d62728", label="95% CI")
ax.axvline(pd.Timestamp("2023-01-01"), color="black", linewidth=1.0, linestyle=":")

ax.set_title(f"ARIMA{order} — Full In-Sample Fit and Out-of-Sample Forecast",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Price (USD per Barrel)", fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig11_full_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] fig11_full_overview.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  FINAL DIAGNOSTIC SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  FINAL MODEL ADEQUACY SUMMARY — ARIMA{order}")
print("=" * 65)

checks = [
    ("Residual Mean ≈ 0",        abs(residuals.mean()) < 1.0),
    ("LB Test p > 0.05 (lag 10)", lb_result.loc[10, "lb_pvalue"] > 0.05),
    ("LB Test p > 0.05 (lag 20)", lb_result.loc[20, "lb_pvalue"] > 0.05),
    ("JB Normality p > 0.05",    jb_pval > 0.05),
    ("No ARCH effect (p > 0.05)", arch_pval > 0.05),
    ("MAPE (out-of-sample) < 20%", mape_test < 20),
]
for label, passed in checks:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}  {label}")

print("=" * 65)
print("  If all checks pass, the model is well-specified and adequate.")
print("  A failing JB test is common in commodity prices and does not")
print("  necessarily invalidate the model for forecasting purposes.")
print("=" * 65)

print("\n✓ Stage 4 complete. All diagnostic plots and tests finished.")

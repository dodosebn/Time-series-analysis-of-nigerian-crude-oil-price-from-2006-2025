"""
=============================================================================
 TIME SERIES ANALYSIS OF MONTHLY CRUDE OIL PRICES IN NIGERIA (2006–2025)
 STAGE 3: ARIMA Model Identification & Fitting
=============================================================================
 Specific Objective iii:
   Fit appropriate ARIMA(p,d,q) models to the monthly crude oil price series.

 Strategy:
   • d = 1 (determined from Stage 2 stationarity analysis)
   • Use auto_arima (AIC-guided grid search) to identify optimal (p, q)
   • Also manually fit a candidate set and compare via AIC / BIC
   • Split data: Training (2006–2022) | Test (2023–2025) for validation
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
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

# ── Train / Test split ───────────────────────────────────────────────────────
train = series[:"2022-12"]
test  = series["2023-01":]
print(f"Training set : {train.index[0].strftime('%b %Y')} – {train.index[-1].strftime('%b %Y')}  ({len(train)} obs)")
print(f"Test set     : {test.index[0].strftime('%b %Y')} – {test.index[-1].strftime('%b %Y')}   ({len(test)} obs)")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  AUTO ARIMA (AIC-based search)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  AUTO ARIMA — Searching for Optimal (p, d, q)")
print("=" * 55)

auto_model = auto_arima(
    train,
    d=1,                    # fix d=1 from stationarity analysis
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    seasonal=False,         # non-seasonal ARIMA
    information_criterion="aic",
    stepwise=True,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
)
print(f"\n→ Best Model: ARIMA{auto_model.order}")
print(f"  AIC = {auto_model.aic():.4f}  |  BIC = {auto_model.bic():.4f}")

best_p, best_d, best_q = auto_model.order

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CANDIDATE MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CANDIDATE ARIMA MODELS — AIC / BIC COMPARISON")
print("=" * 60)
print(f"  {'Model':<16} {'AIC':>10} {'BIC':>10} {'Log-Lik':>12}")
print(f"  {'-'*50}")

candidate_orders = [
    (0, 1, 1), (1, 1, 0), (1, 1, 1),
    (2, 1, 0), (0, 1, 2), (2, 1, 1),
    (1, 1, 2), (2, 1, 2), (3, 1, 0),
    (0, 1, 3), (best_p, best_d, best_q),
]
# De-duplicate preserving order
seen = set()
unique_orders = []
for o in candidate_orders:
    if o not in seen:
        seen.add(o)
        unique_orders.append(o)

results = []
for order in unique_orders:
    try:
        m = ARIMA(train, order=order).fit()
        tag = " ← best (auto)" if order == (best_p, best_d, best_q) else ""
        print(f"  ARIMA{str(order):<12} {m.aic:>10.4f} {m.bic:>10.4f} {m.llf:>12.4f}{tag}")
        results.append({"order": order, "aic": m.aic, "bic": m.bic, "llf": m.llf})
    except Exception:
        pass

results_df = pd.DataFrame(results).sort_values("aic")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIT THE SELECTED MODEL ON TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────
selected_order = results_df.iloc[0]["order"]
print(f"\n→ Selected Model: ARIMA{selected_order}  (lowest AIC)")

fitted_model = ARIMA(train, order=selected_order).fit()
print(fitted_model.summary())

# ─────────────────────────────────────────────────────────────────────────────
# 5.  IN-SAMPLE FITTED VALUES PLOT
# ─────────────────────────────────────────────────────────────────────────────
fitted_vals = fitted_model.fittedvalues

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train.index, train.values, color="#1f77b4", linewidth=1.2,
        label="Observed (Training)")
ax.plot(fitted_vals.index, fitted_vals.values, color="#ff7f0e",
        linewidth=1.0, linestyle="--", label=f"Fitted ARIMA{selected_order}")
ax.set_title(f"ARIMA{selected_order} — Observed vs. In-Sample Fitted Values",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Price (USD per Barrel)", fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig8_fitted_vs_observed.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] fig8_fitted_vs_observed.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  OUT-OF-SAMPLE FORECAST (Test Period)
# ─────────────────────────────────────────────────────────────────────────────
n_forecast = len(test)
forecast_obj = fitted_model.get_forecast(steps=n_forecast)
forecast_mean = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int(alpha=0.05)   # 95% CI
forecast_mean.index = test.index
conf_int.index = test.index

# Accuracy metrics
mae  = np.mean(np.abs(test.values - forecast_mean.values))
rmse = np.sqrt(np.mean((test.values - forecast_mean.values) ** 2))
mape = np.mean(np.abs((test.values - forecast_mean.values) / test.values)) * 100

print("\n" + "=" * 50)
print("  OUT-OF-SAMPLE FORECAST ACCURACY (2023–2025)")
print("=" * 50)
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAPE : {mape:.4f}%")
print("=" * 50)

# Save forecast table
forecast_table = pd.DataFrame({
    "Date": test.index.strftime("%b %Y"),
    "Observed": test.values.round(2),
    "Forecast": forecast_mean.values.round(2),
    "Lower_95": conf_int.iloc[:, 0].values.round(2),
    "Upper_95": conf_int.iloc[:, 1].values.round(2),
    "Error": (test.values - forecast_mean.values).round(2),
})
print("\n  Forecast Table (first 12 rows):")
print(forecast_table.head(12).to_string(index=False))
forecast_table.to_csv("forecast_table.csv", index=False)
print("\n[Saved] forecast_table.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FIGURE 9 – FORECAST PLOT
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train.index[-48:], train.values[-48:], color="#1f77b4",
        linewidth=1.3, label="Training Data (last 4 yrs)")
ax.plot(test.index, test.values, color="#2ca02c", linewidth=1.5,
        label="Actual (Test Period)")
ax.plot(forecast_mean.index, forecast_mean.values, color="#d62728",
        linewidth=1.5, linestyle="--", label=f"Forecast — ARIMA{selected_order}")
ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                alpha=0.18, color="#d62728", label="95% Confidence Interval")

ax.axvline(pd.Timestamp("2023-01-01"), color="black", linewidth=1.0,
           linestyle=":", label="Train/Test Split")
ax.set_title(f"ARIMA{selected_order} — Out-of-Sample Forecast vs. Actual (2023–2025)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Price (USD per Barrel)", fontsize=11)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30, ha="right", fontsize=8)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig9_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] fig9_forecast.png")

print("\n✓ Stage 3 complete. Model fitted and forecast saved.")
print(f"  → Selected model: ARIMA{selected_order} | MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}%")

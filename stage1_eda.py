"""
=============================================================================
 TIME SERIES ANALYSIS OF MONTHLY CRUDE OIL PRICES IN NIGERIA (2006–2025)
 STAGE 1: Exploratory Data Analysis (EDA) & Trend/Pattern Examination
=============================================================================
 Specific Objective i:
   Examine the trends and patterns in monthly crude oil prices over 2006–2025.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_excel("mydata.xlsx")                          # adjust path if needed
df = df.sort_values(["tyear", "tmonth"]).reset_index(drop=True)
df["date"] = pd.to_datetime(dict(year=df["tyear"], month=df["tmonth"], day=1))
df = df.set_index("date")

series = df["crudeOilPrice"].copy()
series.index.freq = "MS"                                   # monthly start frequency

print("=" * 65)
print("  NIGERIA CRUDE OIL PRICE – DESCRIPTIVE STATISTICS (2006–2025)")
print("=" * 65)
print(f"  Observations : {len(series)}")
print(f"  Period       : {series.index[0].strftime('%B %Y')} – "
      f"{series.index[-1].strftime('%B %Y')}")
print(f"  Mean ($/bbl) : {series.mean():.2f}")
print(f"  Median       : {series.median():.2f}")
print(f"  Std Dev      : {series.std():.2f}")
print(f"  Min          : {series.min():.2f}  ({series.idxmin().strftime('%b %Y')})")
print(f"  Max          : {series.max():.2f}  ({series.idxmax().strftime('%b %Y')})")
print(f"  Skewness     : {series.skew():.4f}")
print(f"  Kurtosis     : {series.kurtosis():.4f}")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FIGURE 1 – TIME SERIES PLOT WITH ANNOTATED EVENTS
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(series.index, series.values, color="#1f77b4", linewidth=1.4, label="Crude Oil Price")
ax.fill_between(series.index, series.values, alpha=0.12, color="#1f77b4")

# Annotate key historical events
events = {
    "2008-07": ("2008\nPeak\n($138.7)", "above"),
    "2009-01": ("GFC\nCrash", "below"),
    "2014-06": ("2014\nSlump", "below"),
    "2020-04": ("COVID-19\nCrash\n($14.3)", "below"),
    "2022-06": ("2022\nSurge", "above"),
}
for dt_str, (label, pos) in events.items():
    dt = pd.Timestamp(dt_str)
    if dt in series.index:
        y = series[dt]
        offset = 12 if pos == "above" else -15
        ax.annotate(label, xy=(dt, y), xytext=(dt, y + offset),
                    fontsize=7.5, ha="center", color="darkred",
                    arrowprops=dict(arrowstyle="-", color="darkred", lw=0.8))

ax.set_title("Monthly Crude Oil Prices in Nigeria (2006–2025)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Price (USD per Barrel)", fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("fig1_time_series.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] fig1_time_series.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FIGURE 2 – SEASONAL (MONTHLY) BOX PLOTS
# ─────────────────────────────────────────────────────────────────────────────
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_data = [series[series.index.month == m].values for m in range(1, 13)]

fig, ax = plt.subplots(figsize=(12, 5))
bp = ax.boxplot(monthly_data, patch_artist=True,
                medianprops=dict(color="red", linewidth=1.8))
colors = plt.cm.tab20.colors
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticklabels(month_names, fontsize=10)
ax.set_title("Seasonal Box Plot: Monthly Crude Oil Prices by Month",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Price (USD per Barrel)", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig2_seasonal_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] fig2_seasonal_boxplot.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIGURE 3 – CLASSICAL DECOMPOSITION (Additive)
# ─────────────────────────────────────────────────────────────────────────────
decomp = seasonal_decompose(series, model="additive", period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
components = [
    (series, "Observed", "#1f77b4"),
    (decomp.trend, "Trend", "#ff7f0e"),
    (decomp.seasonal, "Seasonal", "#2ca02c"),
    (decomp.resid, "Residual", "#d62728"),
]
for ax, (data, title, color) in zip(axes, components):
    ax.plot(data.index, data.values, color=color, linewidth=1.2)
    ax.set_ylabel(title, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if title == "Residual":
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

axes[0].set_title("Classical Additive Decomposition of Monthly Crude Oil Prices",
                  fontsize=13, fontweight="bold", pad=10)
axes[-1].set_xlabel("Year", fontsize=11)
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("fig3_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] fig3_decomposition.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIGURE 4 – ANNUAL AVERAGE PRICE (Bar Chart)
# ─────────────────────────────────────────────────────────────────────────────
annual_avg = series.resample("YE").mean()
annual_avg.index = annual_avg.index.year

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(annual_avg.index, annual_avg.values,
              color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(annual_avg))),
              edgecolor="white", width=0.7)
for bar, val in zip(bars, annual_avg.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8)

ax.set_title("Annual Average Crude Oil Prices in Nigeria (2006–2025)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Average Price (USD per Barrel)", fontsize=11)
ax.set_xticks(annual_avg.index)
ax.set_xticklabels(annual_avg.index, rotation=45, fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig4_annual_average.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] fig4_annual_average.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FIGURE 5 – HISTOGRAM + KERNEL DENSITY ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────
from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(series.values, bins=25, density=True, color="#4c72b0",
        alpha=0.55, edgecolor="white", label="Observed")

kde = gaussian_kde(series.values)
x_range = np.linspace(series.min() - 5, series.max() + 5, 300)
ax.plot(x_range, kde(x_range), color="darkblue", linewidth=2, label="KDE")
ax.axvline(series.mean(),   color="red",    linestyle="--", linewidth=1.4,
           label=f"Mean = {series.mean():.1f}")
ax.axvline(series.median(), color="orange", linestyle="--", linewidth=1.4,
           label=f"Median = {series.median():.1f}")

ax.set_title("Distribution of Monthly Crude Oil Prices (2006–2025)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Price (USD per Barrel)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig5_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] fig5_distribution.png")

print("\n✓ Stage 1 complete. All five EDA figures saved successfully.")

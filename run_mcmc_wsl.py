"""
Run all MCMC sampling for the IEEE Bayesian Change-Point paper.
Designed for WSL Ubuntu where PyTensor's C backend works natively.
Combines scripts 04 (single CP), 05 (hierarchical), 06 (model comparison),
and 07 (paper figures).

Usage from Windows:
    wsl bash -c "cd /mnt/c/Users/TOSHIBA/Documents/IEEpaper && python3 run_mcmc_wsl.py"
"""

import os
import io
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Output dirs — write to the project's own data/traces and figures dirs
PROJECT_DIR = Path(__file__).resolve().parent
TRACES_DIR = PROJECT_DIR / "data" / "traces"
FIGURES_DIR = PROJECT_DIR / "figures"
DATA_DIR = PROJECT_DIR / "data"
TRACES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Zone definitions ─────────────────────────────────────────────────────

ZONE_ORDER = ["Sahel", "Guinea Savanna", "Coastal"]
ZONE_COLORS = {"Sahel": "#E74C3C", "Guinea Savanna": "#2ECC71", "Coastal": "#3498DB"}
ZONES = {
    "Sahel": [
        {"city": "Maiduguri", "lat": 11.846, "lon": 13.160},
        {"city": "Kano",      "lat": 12.000, "lon": 8.517},
        {"city": "Sokoto",    "lat": 13.060, "lon": 5.240},
    ],
    "Guinea Savanna": [
        {"city": "Abuja",  "lat": 9.058, "lon": 7.489},
        {"city": "Jos",    "lat": 9.897, "lon": 8.858},
        {"city": "Ilorin", "lat": 8.490, "lon": 4.542},
    ],
    "Coastal": [
        {"city": "Lagos",          "lat": 6.524, "lon": 3.379},
        {"city": "Port Harcourt",  "lat": 4.815, "lon": 7.050},
        {"city": "Calabar",        "lat": 4.976, "lon": 8.337},
    ],
}

IEEE_SINGLE_COL = (3.5, 2.5)
IEEE_DOUBLE_COL = (7.16, 3.0)
IEEE_DOUBLE_COL_TALL = (7.16, 5.0)


def get_all_stations():
    stations = []
    for zone, cities in ZONES.items():
        for c in cities:
            stations.append({**c, "zone": zone})
    return stations


# ── Load data ────────────────────────────────────────────────────────────

print("Loading data...")
annual_zone = pd.read_csv(DATA_DIR / "annual_zone_rainfall.csv")
print(f"  {len(annual_zone)} records, {annual_zone['zone'].nunique()} zones, "
      f"{annual_zone['year'].min()}-{annual_zone['year'].max()}")

# ── Imports that take time ───────────────────────────────────────────────

print("Importing PyMC (compiling C backend)...")
t_import = time.time()
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print(f"  PyMC {pm.__version__}, ArviZ {az.__version__} loaded in {time.time()-t_import:.1f}s")


def set_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "lines.linewidth": 1.0, "lines.markersize": 3,
    })


def savefig(fig, name, formats=("pdf", "png")):
    for fmt in formats:
        fig.savefig(FIGURES_DIR / f"{name}.{fmt}")
    plt.close(fig)


set_ieee_style()

# ═════════════════════════════════════════════════════════════════════════
# SCRIPT 04: Single Change-Point Models (per zone)
# ═════════════════════════════════════════════════════════════════════════

def build_single_changepoint_model(y, years):
    N = len(y)
    y_mean, y_std = float(y.mean()), float(y.std())
    with pm.Model(coords={"year": years}) as model:
        tau = pm.Normal("tau", mu=N / 2, sigma=N / 4)
        mu_1 = pm.Normal("mu_1", mu=y_mean, sigma=y_std * 2)
        mu_2 = pm.Normal("mu_2", mu=y_mean, sigma=y_std * 2)
        sigma = pm.HalfNormal("sigma", sigma=y_std)
        idx = np.arange(N, dtype="float64")
        weight = pm.math.sigmoid(5 * (idx - tau))
        mu = mu_1 * (1 - weight) + mu_2 * weight
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y, dims="year")
        pm.Deterministic("tau_year", tau + years[0])
        pm.Deterministic("shift_magnitude", mu_2 - mu_1)
    return model


print("\n" + "=" * 60)
print("SCRIPT 04: Single Change-Point Models")
print("=" * 60)

single_cp_traces = {}
t0_all = time.time()

for zone in ZONE_ORDER:
    zdf = annual_zone[annual_zone["zone"] == zone].sort_values("year")
    y = zdf["annual_precip_mm"].values.astype("float64")
    years = zdf["year"].values

    print(f"\n  Fitting {zone} ({len(y)} years, mean={y.mean():.0f} mm)...")
    t0 = time.time()
    model = build_single_changepoint_model(y, years)

    with model:
        trace = pm.sample(
            draws=4000, tune=2000, chains=4,
            target_accept=0.95, random_seed=42,
            return_inferencedata=True,
        )
        trace.extend(pm.sample_posterior_predictive(trace, random_seed=42))

    print(f"    Done in {time.time() - t0:.1f}s")
    summary = az.summary(trace, var_names=["tau_year", "mu_1", "mu_2",
                                            "sigma", "shift_magnitude"])
    print(summary)
    rhat_ok = (summary["r_hat"] < 1.01).all()
    ess_ok = (summary["ess_bulk"] > 400).all()
    print(f"    R-hat: {'PASS' if rhat_ok else 'FAIL'}  |  ESS: {'PASS' if ess_ok else 'FAIL'}")

    trace_path = TRACES_DIR / f"single_changepoint_{zone.replace(' ', '_')}.nc"
    trace.to_netcdf(str(trace_path))
    single_cp_traces[zone] = trace

print(f"\nScript 04 total: {time.time() - t0_all:.1f}s")
print("\nChange-Point Estimates (Median [94% HDI]):")
for zone in ZONE_ORDER:
    tau_s = single_cp_traces[zone].posterior["tau_year"].values.flatten()
    shift_s = single_cp_traces[zone].posterior["shift_magnitude"].values.flatten()
    hdi = az.hdi(tau_s, hdi_prob=0.94)
    print(f"  {zone}: {np.median(tau_s):.1f} [{hdi[0]:.1f}, {hdi[1]:.1f}], "
          f"shift = {np.median(shift_s):+.1f} mm/yr")


# ═════════════════════════════════════════════════════════════════════════
# SCRIPT 05: Hierarchical Change-Point Model
# ═════════════════════════════════════════════════════════════════════════

def prepare_zone_data(df):
    zone_data = []
    for zone in ZONE_ORDER:
        zdf = df[df["zone"] == zone].sort_values("year")
        y = zdf["annual_precip_mm"].values.astype("float64")
        zone_data.append({
            "name": zone, "y": y, "years": zdf["year"].values,
            "start_year": int(zdf["year"].min()),
            "y_mean": float(y.mean()), "y_std": float(y.std()),
        })
    return zone_data


def build_hierarchical_model(zone_data):
    zone_names = [z["name"] for z in zone_data]
    with pm.Model(coords={"zone": zone_names}) as model:
        tau_mu = pm.Normal("tau_mu", mu=30, sigma=10)
        tau_sigma = pm.HalfNormal("tau_sigma", sigma=5)
        tau_offset = pm.Normal("tau_offset", mu=0, sigma=1, dims="zone")
        tau_raw = pm.Deterministic("tau_raw", tau_mu + tau_sigma * tau_offset, dims="zone")
        steepness = pm.HalfNormal("steepness", sigma=5)

        prior_means = np.array([z["y_mean"] for z in zone_data])
        prior_stds = np.array([z["y_std"] for z in zone_data])
        mu_before = pm.Normal("mu_before", mu=prior_means, sigma=200, dims="zone")
        mu_after = pm.Normal("mu_after", mu=prior_means, sigma=200, dims="zone")
        sigma = pm.HalfNormal("sigma", sigma=prior_stds, dims="zone")

        for i, z in enumerate(zone_data):
            N = len(z["y"])
            idx = np.arange(N, dtype="float64")
            weight = pm.math.sigmoid(steepness * (idx - tau_raw[i]))
            mu_t = mu_before[i] * (1 - weight) + mu_after[i] * weight
            pm.Normal(f"obs_{z['name']}", mu=mu_t, sigma=sigma[i], observed=z["y"])

        start_years = np.array([z["start_year"] for z in zone_data])
        pm.Deterministic("tau_year", tau_raw + start_years, dims="zone")
        pm.Deterministic("shift_magnitude", mu_after - mu_before, dims="zone")
        pm.Deterministic("tau_year_group", tau_mu + start_years[0])
    return model


print("\n" + "=" * 60)
print("SCRIPT 05: Hierarchical Change-Point Model")
print("=" * 60)

zone_data = prepare_zone_data(annual_zone)
for z in zone_data:
    print(f"  {z['name']}: {len(z['y'])} years, mean={z['y_mean']:.0f} mm")

model = build_hierarchical_model(zone_data)

print("\nSampling hierarchical model...")
t0 = time.time()
with model:
    hier_trace = pm.sample(
        draws=4000, tune=2000, chains=4,
        target_accept=0.95, random_seed=42,
        return_inferencedata=True,
    )
    hier_trace.extend(pm.sample_posterior_predictive(hier_trace, random_seed=42))

print(f"Sampling took {time.time() - t0:.1f}s")

var_names = ["tau_year", "tau_year_group", "mu_before", "mu_after",
             "sigma", "shift_magnitude", "steepness", "tau_mu", "tau_sigma"]
summary = az.summary(hier_trace, var_names=var_names)
print("\nPosterior Summary:")
print(summary)
rhat_ok = (summary["r_hat"] < 1.01).all()
ess_ok = (summary["ess_bulk"] > 400).all()
print(f"\nR-hat: {'PASS' if rhat_ok else 'FAIL'}  |  ESS: {'PASS' if ess_ok else 'FAIL'}")

hier_trace.to_netcdf(str(TRACES_DIR / "hierarchical_changepoint.nc"))

print("\nHierarchical Results:")
tau_group = hier_trace.posterior["tau_year_group"].values.flatten()
print(f"  Group: {np.median(tau_group):.1f} "
      f"[{az.hdi(tau_group, hdi_prob=0.94)[0]:.1f}, "
      f"{az.hdi(tau_group, hdi_prob=0.94)[1]:.1f}]")
print(f"  tau_sigma: {hier_trace.posterior['tau_sigma'].values.flatten().mean():.2f}")
for zone in ZONE_ORDER:
    tau_z = hier_trace.posterior["tau_year"].sel(zone=zone).values.flatten()
    shift = hier_trace.posterior["shift_magnitude"].sel(zone=zone).values.flatten()
    print(f"  {zone}: tau={np.median(tau_z):.1f} "
          f"[{az.hdi(tau_z, hdi_prob=0.94)[0]:.1f}, "
          f"{az.hdi(tau_z, hdi_prob=0.94)[1]:.1f}], "
          f"shift={np.median(shift):+.1f} mm/yr")


# ═════════════════════════════════════════════════════════════════════════
# SCRIPT 06: Model Comparison (LOO-IC & WAIC)
# ═════════════════════════════════════════════════════════════════════════

def build_null_model(y):
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=float(y.mean()), sigma=float(y.std()) * 2)
        sigma = pm.HalfNormal("sigma", sigma=float(y.std()))
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return model

def build_one_cp_model(y):
    N = len(y)
    y_mean, y_std = float(y.mean()), float(y.std())
    with pm.Model() as model:
        tau = pm.Normal("tau", mu=N / 2, sigma=N / 4)
        mu_1 = pm.Normal("mu_1", mu=y_mean, sigma=y_std * 2)
        mu_2 = pm.Normal("mu_2", mu=y_mean, sigma=y_std * 2)
        sigma = pm.HalfNormal("sigma", sigma=y_std)
        idx = np.arange(N, dtype="float64")
        weight = pm.math.sigmoid(5 * (idx - tau))
        mu = mu_1 * (1 - weight) + mu_2 * weight
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return model

def build_two_cp_model(y):
    N = len(y)
    y_mean, y_std = float(y.mean()), float(y.std())
    with pm.Model() as model:
        tau_1 = pm.Normal("tau_1", mu=N / 3, sigma=N / 6)
        tau_2 = pm.Normal("tau_2", mu=2 * N / 3, sigma=N / 6)
        mu_1 = pm.Normal("mu_1", mu=y_mean, sigma=y_std * 2)
        mu_2 = pm.Normal("mu_2", mu=y_mean, sigma=y_std * 2)
        mu_3 = pm.Normal("mu_3", mu=y_mean, sigma=y_std * 2)
        sigma = pm.HalfNormal("sigma", sigma=y_std)
        idx = np.arange(N, dtype="float64")
        w1 = pm.math.sigmoid(5 * (idx - tau_1))
        w2 = pm.math.sigmoid(5 * (idx - tau_2))
        mu = mu_1 * (1 - w1) + mu_2 * w1 * (1 - w2) + mu_3 * w2
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return model

MODELS = {"null": build_null_model, "one_cp": build_one_cp_model, "two_cp": build_two_cp_model}

print("\n" + "=" * 60)
print("SCRIPT 06: Model Comparison (LOO-IC & WAIC)")
print("=" * 60)

all_comparisons_loo = {}
t0_all = time.time()

for zone in ZONE_ORDER:
    print(f"\n  Model comparison for {zone}")
    zdf = annual_zone[annual_zone["zone"] == zone].sort_values("year")
    y = zdf["annual_precip_mm"].values.astype("float64")

    traces = {}
    for model_name, builder in MODELS.items():
        print(f"    Fitting {model_name}...", end=" ", flush=True)
        t0 = time.time()
        mdl = builder(y)
        with mdl:
            tr = pm.sample(
                draws=2000, tune=1000, chains=4,
                target_accept=0.95, random_seed=42,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
            )
        traces[model_name] = tr
        safe_zone = zone.replace(" ", "_")
        tr.to_netcdf(str(TRACES_DIR / f"comparison_{safe_zone}_{model_name}.nc"))
        print(f"{time.time() - t0:.1f}s")

    comp_loo = az.compare(traces, ic="loo")
    comp_waic = az.compare(traces, ic="waic")
    all_comparisons_loo[zone] = comp_loo

    print(f"\n  {zone} LOO:")
    print(comp_loo)

    latex = comp_loo.to_latex(float_format="%.1f")
    safe_zone = zone.replace(" ", "_")
    with open(DATA_DIR / f"comparison_loo_{safe_zone}.tex", "w") as f:
        f.write(latex)

print(f"\nScript 06 total: {time.time() - t0_all:.1f}s")
print("\nBest Model per Zone (LOO):")
for zone in ZONE_ORDER:
    best = all_comparisons_loo[zone].index[0]
    elpd = all_comparisons_loo[zone].iloc[0]["elpd_loo"]
    print(f"  {zone}: {best} (ELPD-LOO = {elpd:.1f})")


# ═════════════════════════════════════════════════════════════════════════
# SCRIPT 07: Publication Figures
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SCRIPT 07: Generating Figures")
print("=" * 60)

# Fig 1: Study Area Map
fig, ax = plt.subplots(figsize=IEEE_SINGLE_COL)
nigeria_lon = [2.7, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
               11.0, 12.0, 13.0, 14.0, 14.5, 14.0, 13.5, 13.0, 12.5,
               12.0, 11.5, 11.0, 10.5, 10.0, 9.5, 9.0, 8.5, 8.0,
               7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.7]
nigeria_lat = [6.5, 6.0, 6.3, 6.0, 6.0, 5.5, 5.0, 4.5, 4.2, 4.5,
               4.8, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
               12.0, 12.5, 13.0, 13.5, 13.8, 13.5, 13.5, 13.5, 13.5,
               13.8, 13.8, 13.8, 13.5, 13.5, 13.0, 12.5, 12.0, 11.5,
               10.0, 6.5]
ax.plot(nigeria_lon, nigeria_lat, "k-", linewidth=1.0)
ax.fill(nigeria_lon, nigeria_lat, color="#f5f5f5", zorder=0)
ax.axhspan(4.0, 7.5, color=ZONE_COLORS["Coastal"], alpha=0.15, zorder=1)
ax.axhspan(7.5, 10.5, color=ZONE_COLORS["Guinea Savanna"], alpha=0.15, zorder=1)
ax.axhspan(10.5, 14.0, color=ZONE_COLORS["Sahel"], alpha=0.15, zorder=1)
for st in get_all_stations():
    color = ZONE_COLORS[st["zone"]]
    ax.plot(st["lon"], st["lat"], "o", color=color, markersize=5,
            markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax.annotate(st["city"], (st["lon"], st["lat"]),
                xytext=(4, 4), textcoords="offset points", fontsize=6, zorder=5)
ax.text(8, 12.5, "Sahel", fontsize=7, ha="center", style="italic",
        color=ZONE_COLORS["Sahel"], fontweight="bold")
ax.text(8, 9.0, "Guinea Savanna", fontsize=7, ha="center", style="italic",
        color=ZONE_COLORS["Guinea Savanna"], fontweight="bold")
ax.text(8, 6.0, "Coastal/Rainforest", fontsize=7, ha="center",
        style="italic", color=ZONE_COLORS["Coastal"], fontweight="bold")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_xlim(2, 15)
ax.set_ylim(3.5, 14.5)
ax.set_aspect("equal")
ax.set_title("(a) Study Area and Station Locations")
fig.tight_layout()
savefig(fig, "fig1_study_area")
print("  Saved fig1_study_area")

# Fig 2: Time Series with Posterior Change-Points
trace = hier_trace
fig, axes = plt.subplots(3, 1, figsize=IEEE_DOUBLE_COL_TALL, sharex=True)
for ax, zone in zip(axes, ZONE_ORDER):
    zdf = annual_zone[annual_zone["zone"] == zone].sort_values("year")
    years = zdf["year"].values
    precip = zdf["annual_precip_mm"].values
    color = ZONE_COLORS[zone]
    ax.bar(years, precip, color=color, alpha=0.4, width=0.8)
    tau_samples = trace.posterior["tau_year"].sel(zone=zone).values.flatten()
    ax2 = ax.twinx()
    ax2.hist(tau_samples, bins=50, density=True, color="gray", alpha=0.5, zorder=3)
    ax2.set_ylabel("P(change-point)", fontsize=6)
    ax2.tick_params(labelsize=6)
    mu_b = float(trace.posterior["mu_before"].sel(zone=zone).values.mean())
    mu_a = float(trace.posterior["mu_after"].sel(zone=zone).values.mean())
    tau_med = float(np.median(tau_samples))
    ax.axhline(mu_b, xmax=0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(mu_a, xmin=0.5, color="black", linestyle="-.", linewidth=0.8, alpha=0.7)
    ax.axvline(tau_med, color="red", linewidth=1.0, linestyle="-", alpha=0.8)
    ax.set_ylabel("Precipitation (mm)")
    ax.set_title(f"{zone} (change-point: {tau_med:.0f})")
axes[-1].set_xlabel("Year")
fig.suptitle("Annual Precipitation with Bayesian Change-Point Estimates", fontsize=10, y=1.01)
fig.tight_layout()
savefig(fig, "fig2_changepoint_timeseries")
print("  Saved fig2_changepoint_timeseries")

# Fig 3: Trace and Rank Plots
axes_arr = az.plot_trace(trace, var_names=["tau_year", "mu_before", "mu_after", "steepness"],
                         compact=True, figsize=(7.16, 8))
fig = axes_arr.ravel()[0].get_figure()
fig.tight_layout()
savefig(fig, "fig3_trace_plots")
print("  Saved fig3_trace_plots")

axes_arr = az.plot_rank(trace, var_names=["tau_year", "mu_before", "mu_after"], figsize=(7.16, 5))
fig = axes_arr.ravel()[0].get_figure()
fig.tight_layout()
savefig(fig, "fig3b_rank_plots")
print("  Saved fig3b_rank_plots")

# Fig 4: Forest Plot
fig, axes = plt.subplots(1, 2, figsize=IEEE_DOUBLE_COL)
az.plot_forest(trace, var_names=["tau_year"], combined=True, hdi_prob=0.94, ax=axes[0])
axes[0].set_title("Change-Point Year (94% HDI)")
axes[0].set_xlabel("Year")
az.plot_forest(trace, var_names=["shift_magnitude"], combined=True, hdi_prob=0.94, ax=axes[1])
axes[1].set_title("Shift Magnitude (94% HDI)")
axes[1].set_xlabel("mm/year")
fig.tight_layout()
savefig(fig, "fig4_forest")
print("  Saved fig4_forest")

# Fig 5: Posterior Predictive Checks
fig, axes = plt.subplots(1, 3, figsize=IEEE_DOUBLE_COL)
for ax, zone in zip(axes, ZONE_ORDER):
    obs_key = f"obs_{zone}"
    if obs_key in trace.posterior_predictive:
        az.plot_ppc(trace, observed_rug=True, var_names=[obs_key], ax=ax)
    ax.set_title(zone)
    ax.set_xlabel("Precip (mm)")
fig.suptitle("Posterior Predictive Checks", fontsize=10, y=1.02)
fig.tight_layout()
savefig(fig, "fig5_ppc")
print("  Saved fig5_ppc")

# Fig 6: Model Comparison
fig, axes = plt.subplots(1, 3, figsize=IEEE_DOUBLE_COL)
for ax, zone in zip(axes, ZONE_ORDER):
    safe_zone = zone.replace(" ", "_")
    try:
        traces_dict = {}
        for mname in ["null", "one_cp", "two_cp"]:
            tp = TRACES_DIR / f"comparison_{safe_zone}_{mname}.nc"
            if tp.exists():
                traces_dict[mname] = az.from_netcdf(str(tp))
        if traces_dict:
            comp = az.compare(traces_dict, ic="loo")
            az.plot_compare(comp, ax=ax)
    except Exception as e:
        ax.text(0.5, 0.5, f"{zone}\n{e}",
                ha="center", va="center", transform=ax.transAxes, fontsize=6)
    ax.set_title(zone, fontsize=8)
fig.suptitle("Model Comparison (LOO-IC)", fontsize=10, y=1.02)
fig.tight_layout()
savefig(fig, "fig6_model_comparison")
print("  Saved fig6_model_comparison")


# ═════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
print(f"\nTraces in: {TRACES_DIR}")
for f in sorted(TRACES_DIR.glob("*.nc")):
    print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"\nFigures in: {FIGURES_DIR}")
for f in sorted(FIGURES_DIR.glob("fig*")):
    print(f"  {f.name}")
print(f"\nComparison tables in: {DATA_DIR}")
for f in sorted(DATA_DIR.glob("comparison_*.tex")):
    print(f"  {f.name}")

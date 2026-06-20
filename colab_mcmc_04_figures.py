"""Step 4: Generate all publication figures from saved traces. ~2 min."""

print("\n" + "=" * 60)
print("STEP 4: Generating Figures from Saved Traces")
print("=" * 60)

# Load traces
hier_trace = az.from_netcdf(str(TRACES_DIR / "hierarchical_changepoint.nc"))
print("Loaded hierarchical trace")

hier_ppc = None
ppc_path = TRACES_DIR / "hierarchical_ppc.nc"
if ppc_path.exists():
    hier_ppc = az.from_netcdf(str(ppc_path))
    print("Loaded hierarchical PPC")

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
    tau_samples = trace["posterior"]["tau_year"].sel(zone=zone).values.flatten()
    ax2 = ax.twinx()
    ax2.hist(tau_samples, bins=50, density=True, color="gray", alpha=0.5, zorder=3)
    ax2.set_ylabel("P(change-point)", fontsize=6)
    ax2.tick_params(labelsize=6)
    mu_b = float(trace["posterior"]["mu_before"].sel(zone=zone).values.mean())
    mu_a = float(trace["posterior"]["mu_after"].sel(zone=zone).values.mean())
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
try:
    axes_arr = az.plot_trace(trace, var_names=["tau_year", "mu_before", "mu_after", "steepness"],
                             compact=True, figsize=(7.16, 8))
    fig = axes_arr.ravel()[0].get_figure()
    fig.tight_layout()
    savefig(fig, "fig3_trace_plots")
    print("  Saved fig3_trace_plots")
except Exception as e:
    print(f"  fig3_trace_plots skipped: {e}")

try:
    axes_arr = az.plot_rank(trace, var_names=["tau_year", "mu_before", "mu_after"], figsize=(7.16, 5))
    fig = axes_arr.ravel()[0].get_figure()
    fig.tight_layout()
    savefig(fig, "fig3b_rank_plots")
    print("  Saved fig3b_rank_plots")
except Exception as e:
    print(f"  fig3b_rank_plots skipped: {e}")

# Fig 4: Forest Plot
try:
    fig, axes = plt.subplots(1, 2, figsize=IEEE_DOUBLE_COL)
    az.plot_forest(trace, var_names=["tau_year"], combined=True, prob=0.94, ax=axes[0])
    axes[0].set_title("Change-Point Year (94% HDI)")
    axes[0].set_xlabel("Year")
    az.plot_forest(trace, var_names=["shift_magnitude"], combined=True, prob=0.94, ax=axes[1])
    axes[1].set_title("Shift Magnitude (94% HDI)")
    axes[1].set_xlabel("mm/year")
    fig.tight_layout()
    savefig(fig, "fig4_forest")
    print("  Saved fig4_forest")
except Exception as e:
    print(f"  fig4_forest skipped: {e}")

# Fig 5: Posterior Predictive Checks
try:
    fig, axes = plt.subplots(1, 3, figsize=IEEE_DOUBLE_COL)
    for ax, zone in zip(axes, ZONE_ORDER):
        obs_key = f"obs_{zone}"
        if hier_ppc is not None:
            pp_groups = list(hier_ppc.children) if hasattr(hier_ppc, 'children') else (list(hier_ppc.groups()) if hasattr(hier_ppc, 'groups') else [])
            if "posterior_predictive" in pp_groups:
                pp_data = hier_ppc["posterior_predictive"]
                if obs_key in pp_data:
                    ax.hist(pp_data[obs_key].values.flatten(), bins=30, density=True,
                            alpha=0.5, color="blue", label="Predicted")
                    zdf = annual_zone[annual_zone["zone"] == zone]
                    ax.hist(zdf["annual_precip_mm"].values, bins=15, density=True,
                            alpha=0.5, color="red", label="Observed")
                    ax.legend(fontsize=5)
        ax.set_title(zone)
        ax.set_xlabel("Precip (mm)")
    fig.suptitle("Posterior Predictive Checks", fontsize=10, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig5_ppc")
    print("  Saved fig5_ppc")
except Exception as e:
    print(f"  fig5_ppc skipped: {e}")

# Fig 6: Model Comparison
try:
    fig, axes = plt.subplots(1, 3, figsize=IEEE_DOUBLE_COL)
    for ax, zone in zip(axes, ZONE_ORDER):
        safe_zone = zone.replace(" ", "_")
        traces_dict = {}
        for mname in ["null", "one_cp", "two_cp"]:
            tp = TRACES_DIR / f"comparison_{safe_zone}_{mname}.nc"
            if tp.exists():
                traces_dict[mname] = az.from_netcdf(str(tp))
        if traces_dict:
            comp = az.compare(traces_dict, ic="loo")
            az.plot_compare(comp, ax=ax)
        ax.set_title(zone, fontsize=8)
    fig.suptitle("Model Comparison (LOO-IC)", fontsize=10, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig6_model_comparison")
    print("  Saved fig6_model_comparison")
except Exception as e:
    print(f"  fig6_model_comparison skipped: {e}")

# Zip everything
import shutil
shutil.make_archive("/content/mcmc_output", "zip", str(OUTPUT_DIR))

print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
print(f"\nTraces in: {TRACES_DIR}")
for f in sorted(TRACES_DIR.glob("*.nc")):
    print(f"  {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"\nFigures in: {FIGURES_DIR}")
for f in sorted(FIGURES_DIR.glob("fig*")):
    print(f"  {f.name}")
print(f"\nOutput archived to /content/mcmc_output.zip")

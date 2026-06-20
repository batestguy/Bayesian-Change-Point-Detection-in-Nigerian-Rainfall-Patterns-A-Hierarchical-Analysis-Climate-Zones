"""Step 2: Hierarchical change-point model. ~20 min."""

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
print("STEP 2: Hierarchical Change-Point Model")
print("=" * 60)

zone_data = prepare_zone_data(annual_zone)
model = build_hierarchical_model(zone_data)

print("\nSampling hierarchical model...")
t0 = time.time()
with model:
    hier_trace = pm.sample(draws=4000, tune=2000, chains=4,
                           target_accept=0.95, random_seed=42,
                           return_inferencedata=True)
    hier_ppc = pm.sample_posterior_predictive(hier_trace, random_seed=42)

print(f"Sampling took {time.time() - t0:.1f}s")

var_names = ["tau_year", "tau_year_group", "mu_before", "mu_after",
             "sigma", "shift_magnitude", "steepness", "tau_mu", "tau_sigma"]
summary = az.summary(hier_trace, var_names=var_names)
print("\nPosterior Summary:")
print(summary)
rhat_ok = (pd.to_numeric(summary["r_hat"], errors="coerce") < 1.01).all()
ess_ok = (pd.to_numeric(summary["ess_bulk"], errors="coerce") > 400).all()
print(f"\nR-hat: {'PASS' if rhat_ok else 'FAIL'}  |  ESS: {'PASS' if ess_ok else 'FAIL'}")

hier_trace.to_netcdf(str(TRACES_DIR / "hierarchical_changepoint.nc"))

# Save PPC
import pickle
pp_groups = list(hier_ppc.children) if hasattr(hier_ppc, 'children') else (list(hier_ppc.groups()) if hasattr(hier_ppc, 'groups') else [])
if "posterior_predictive" in pp_groups:
    hier_ppc.to_netcdf(str(TRACES_DIR / "hierarchical_ppc.nc"))
    print("Saved hierarchical PPC")

print("\nHierarchical Results:")
tau_group = hier_trace["posterior"]["tau_year_group"].values.flatten()
hdi_g = compute_hdi(tau_group, prob=0.94)
print(f"  Group: {np.median(tau_group):.1f} [{hdi_g[0]:.1f}, {hdi_g[1]:.1f}]")
print(f"  tau_sigma: {hier_trace['posterior']['tau_sigma'].values.flatten().mean():.2f}")
for zone in ZONE_ORDER:
    tau_z = hier_trace["posterior"]["tau_year"].sel(zone=zone).values.flatten()
    shift = hier_trace["posterior"]["shift_magnitude"].sel(zone=zone).values.flatten()
    hdi_z = compute_hdi(tau_z, prob=0.94)
    print(f"  {zone}: tau={np.median(tau_z):.1f} [{hdi_z[0]:.1f}, {hdi_z[1]:.1f}], "
          f"shift={np.median(shift):+.1f} mm/yr")
print("\nSTEP 2 COMPLETE")

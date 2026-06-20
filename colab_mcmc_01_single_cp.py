"""Step 1: Single change-point models (3 zones). ~30 min total."""

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
print("STEP 1: Single Change-Point Models")
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
        trace = pm.sample(draws=4000, tune=2000, chains=4,
                          target_accept=0.95, random_seed=42,
                          return_inferencedata=True)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    summary = az.summary(trace, var_names=["tau_year", "mu_1", "mu_2",
                                            "sigma", "shift_magnitude"])
    print(summary)
    rhat_ok = (pd.to_numeric(summary["r_hat"], errors="coerce") < 1.01).all()
    ess_ok = (pd.to_numeric(summary["ess_bulk"], errors="coerce") > 400).all()
    print(f"    R-hat: {'PASS' if rhat_ok else 'FAIL'}  |  ESS: {'PASS' if ess_ok else 'FAIL'}")

    trace.to_netcdf(str(TRACES_DIR / f"single_changepoint_{zone.replace(' ', '_')}.nc"))
    single_cp_traces[zone] = trace
    print(f"    Saved trace: single_changepoint_{zone.replace(' ', '_')}.nc")

print(f"\nStep 1 total: {time.time() - t0_all:.1f}s")
print("\nChange-Point Estimates (Median [94% HDI]):")
for zone in ZONE_ORDER:
    tau_s = single_cp_traces[zone]["posterior"]["tau_year"].values.flatten()
    shift_s = single_cp_traces[zone]["posterior"]["shift_magnitude"].values.flatten()
    hdi = compute_hdi(tau_s, prob=0.94)
    print(f"  {zone}: {np.median(tau_s):.1f} [{hdi[0]:.1f}, {hdi[1]:.1f}], "
          f"shift = {np.median(shift_s):+.1f} mm/yr")
print("\nSTEP 1 COMPLETE")

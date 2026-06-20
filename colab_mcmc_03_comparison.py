"""Step 3: Model comparison (null, 1-CP, 2-CP) x 3 zones. ~45 min."""

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
print("STEP 3: Model Comparison (LOO-IC & WAIC)")
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
            tr = pm.sample(draws=2000, tune=1000, chains=4,
                           target_accept=0.95, random_seed=42,
                           return_inferencedata=True)
            pm.compute_log_likelihood(tr)
        traces[model_name] = tr
        safe_zone = zone.replace(" ", "_")
        tr.to_netcdf(str(TRACES_DIR / f"comparison_{safe_zone}_{model_name}.nc"))
        print(f"{time.time() - t0:.1f}s  [saved]")

    comp_loo = az.compare(traces, ic="loo")
    all_comparisons_loo[zone] = comp_loo
    print(f"\n  {zone} LOO:")
    print(comp_loo)

    latex = comp_loo.to_latex(float_format="%.1f")
    safe_zone = zone.replace(" ", "_")
    with open(DATA_DIR / f"comparison_loo_{safe_zone}.tex", "w") as f:
        f.write(latex)

print(f"\nStep 3 total: {time.time() - t0_all:.1f}s")
print("\nBest Model per Zone (LOO):")
for zone in ZONE_ORDER:
    best = all_comparisons_loo[zone].index[0]
    elpd = all_comparisons_loo[zone].iloc[0]["elpd_loo"]
    print(f"  {zone}: {best} (ELPD-LOO = {elpd:.1f})")
print("\nSTEP 3 COMPLETE")

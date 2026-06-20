"""Hierarchical Bayesian change-point model: partial pooling across 3 climate zones."""

import os
os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float64,cxx="

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import DATA_DIR, TRACES_DIR, ZONE_ORDER, set_ieee_style
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt


def build_hierarchical_model(zone_data):
    """
    Hierarchical change-point model with partial pooling of change-point timing.

    Hyperpriors encode belief that zones may share a common climate transition,
    while allowing zone-specific deviations.

    tau_mu ~ Normal(30, 10)     # group-level change-point (~1980 if data starts 1950)
    tau_sigma ~ HalfNormal(5)   # how much zones can differ
    tau_z ~ Normal(tau_mu, tau_sigma)  for each zone z

    mu_before_z ~ Normal(y_bar_z, 200)
    mu_after_z  ~ Normal(y_bar_z, 200)
    sigma_z     ~ HalfNormal(100)

    y_z[t] ~ Normal(mu_before_z * (1-w) + mu_after_z * w, sigma_z)
    where w = sigmoid(steepness * (t - tau_z))
    """
    n_zones = len(zone_data)
    zone_names = [z["name"] for z in zone_data]

    with pm.Model(coords={"zone": zone_names}) as model:
        # === Hyperpriors ===
        tau_mu = pm.Normal("tau_mu", mu=30, sigma=10)
        tau_sigma = pm.HalfNormal("tau_sigma", sigma=5)

        # === Zone-level change-points (continuous for NUTS) ===
        tau_offset = pm.Normal("tau_offset", mu=0, sigma=1, dims="zone")
        tau_raw = pm.Deterministic("tau_raw", tau_mu + tau_sigma * tau_offset,
                                   dims="zone")

        # Steepness of transition (shared, learnable)
        steepness = pm.HalfNormal("steepness", sigma=5)

        # === Zone-level rainfall parameters ===
        prior_means = np.array([z["y_mean"] for z in zone_data])
        prior_stds = np.array([z["y_std"] for z in zone_data])

        mu_before = pm.Normal("mu_before", mu=prior_means, sigma=200,
                              dims="zone")
        mu_after = pm.Normal("mu_after", mu=prior_means, sigma=200,
                             dims="zone")
        sigma = pm.HalfNormal("sigma", sigma=prior_stds, dims="zone")

        # === Likelihood per zone ===
        for i, z in enumerate(zone_data):
            N = len(z["y"])
            idx = np.arange(N, dtype="float64")

            # Soft step function at zone-specific change-point
            weight = pm.math.sigmoid(steepness * (idx - tau_raw[i]))
            mu_t = mu_before[i] * (1 - weight) + mu_after[i] * weight

            pm.Normal(f"obs_{z['name']}", mu=mu_t, sigma=sigma[i],
                      observed=z["y"])

        # === Derived quantities ===
        start_years = np.array([z["start_year"] for z in zone_data])
        pm.Deterministic("tau_year", tau_raw + start_years, dims="zone")
        pm.Deterministic("shift_magnitude", mu_after - mu_before, dims="zone")
        pm.Deterministic("tau_year_group", tau_mu + start_years[0])

    return model


def prepare_zone_data(annual_zone):
    """Prepare data arrays for each zone."""
    zone_data = []
    for zone in ZONE_ORDER:
        zdf = annual_zone[annual_zone["zone"] == zone].sort_values("year")
        y = zdf["annual_precip_mm"].values.astype("float64")
        zone_data.append({
            "name": zone,
            "y": y,
            "years": zdf["year"].values,
            "start_year": int(zdf["year"].min()),
            "y_mean": float(y.mean()),
            "y_std": float(y.std()),
        })
    return zone_data


def main():
    set_ieee_style()
    annual_zone = pd.read_csv(DATA_DIR / "annual_zone_rainfall.csv")
    zone_data = prepare_zone_data(annual_zone)

    print("Building hierarchical change-point model...")
    for z in zone_data:
        print(f"  {z['name']}: {len(z['y'])} years, "
              f"mean={z['y_mean']:.0f} mm, std={z['y_std']:.0f} mm")

    model = build_hierarchical_model(zone_data)

    print("\nSampling (this takes ~10 minutes)...")
    with model:
        trace = pm.sample(
            draws=4000,
            tune=2000,
            chains=4,
            target_accept=0.95,
            random_seed=42,
            return_inferencedata=True,
        )
        trace.extend(pm.sample_posterior_predictive(trace, random_seed=42))

    # Diagnostics
    var_names = ["tau_year", "tau_year_group", "mu_before", "mu_after",
                 "sigma", "shift_magnitude", "steepness", "tau_mu", "tau_sigma"]
    summary = az.summary(trace, var_names=var_names)
    print("\nPosterior Summary:")
    print(summary)

    rhat_ok = (summary["r_hat"] < 1.01).all()
    ess_ok = (summary["ess_bulk"] > 400).all()
    print(f"\nR-hat check: {'PASS' if rhat_ok else 'FAIL'}")
    print(f"ESS check:   {'PASS' if ess_ok else 'FAIL'}")

    # Save trace
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = TRACES_DIR / "hierarchical_changepoint.nc"
    trace.to_netcdf(str(trace_path))
    print(f"Saved trace: {trace_path}")

    # Key results
    print("\n=== Hierarchical Change-Point Results ===")
    tau_group = trace.posterior["tau_year_group"].values.flatten()
    print(f"Group-level change-point: {np.median(tau_group):.1f} "
          f"[{az.hdi(tau_group, hdi_prob=0.94)[0]:.1f}, "
          f"{az.hdi(tau_group, hdi_prob=0.94)[1]:.1f}]")

    print(f"tau_sigma (zone spread): "
          f"{trace.posterior['tau_sigma'].values.flatten().mean():.2f} years")

    for i, zone in enumerate(ZONE_ORDER):
        tau_z = trace.posterior["tau_year"].sel(zone=zone).values.flatten()
        shift = trace.posterior["shift_magnitude"].sel(zone=zone).values.flatten()
        print(f"  {zone}: tau={np.median(tau_z):.1f} "
              f"[{az.hdi(tau_z, hdi_prob=0.94)[0]:.1f}, "
              f"{az.hdi(tau_z, hdi_prob=0.94)[1]:.1f}], "
              f"shift={np.median(shift):+.1f} mm/yr")


if __name__ == "__main__":
    main()

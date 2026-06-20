"""Single change-point model per zone using PyMC5."""

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


def build_single_changepoint_model(y, years, zone_name):
    """
    Single change-point model: rainfall regime shifts at unknown year tau.

    y[t] ~ Normal(mu, sigma)
    mu = mu_1  if t < tau
         mu_2  if t >= tau
    """
    N = len(y)
    y_mean, y_std = float(y.mean()), float(y.std())

    with pm.Model(coords={"year": years}) as model:
        # Change-point location in index space; sigmoid weight handles soft switch
        tau = pm.Normal("tau", mu=N / 2, sigma=N / 4)

        mu_1 = pm.Normal("mu_1", mu=y_mean, sigma=y_std * 2)
        mu_2 = pm.Normal("mu_2", mu=y_mean, sigma=y_std * 2)
        sigma = pm.HalfNormal("sigma", sigma=y_std)

        # Soft switch at change-point
        idx = np.arange(N, dtype="float64")
        weight = pm.math.sigmoid(5 * (idx - tau))
        mu = mu_1 * (1 - weight) + mu_2 * weight

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y, dims="year")

        # Derived: change-point year and magnitude of shift
        pm.Deterministic("tau_year", tau + years[0])
        pm.Deterministic("shift_magnitude", mu_2 - mu_1)

    return model


def fit_zone(zone, annual_zone):
    """Fit single change-point model for one zone."""
    zdf = annual_zone[annual_zone["zone"] == zone].sort_values("year")
    y = zdf["annual_precip_mm"].values.astype("float64")
    years = zdf["year"].values

    print(f"\n{'='*50}")
    print(f"Fitting single change-point model for {zone}")
    print(f"  Data: {len(y)} years ({years[0]}-{years[-1]})")
    print(f"  Mean: {y.mean():.1f} mm, Std: {y.std():.1f} mm")

    model = build_single_changepoint_model(y, years, zone)

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
    summary = az.summary(trace, var_names=["tau_year", "mu_1", "mu_2",
                                            "sigma", "shift_magnitude"])
    print(f"\n{zone} — Posterior Summary:")
    print(summary)

    rhat_ok = (summary["r_hat"] < 1.01).all()
    ess_ok = (summary["ess_bulk"] > 400).all()
    print(f"  R-hat check: {'PASS' if rhat_ok else 'FAIL'}")
    print(f"  ESS check:   {'PASS' if ess_ok else 'FAIL'}")

    # Save trace
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = TRACES_DIR / f"single_changepoint_{zone.replace(' ', '_')}.nc"
    trace.to_netcdf(str(trace_path))
    print(f"  Saved trace: {trace_path}")

    return trace


def main():
    set_ieee_style()
    annual_zone = pd.read_csv(DATA_DIR / "annual_zone_rainfall.csv")

    traces = {}
    for zone in ZONE_ORDER:
        traces[zone] = fit_zone(zone, annual_zone)

    print("\n\n=== Change-Point Estimates (Median [94% HDI]) ===")
    for zone in ZONE_ORDER:
        tau_samples = traces[zone].posterior["tau_year"].values.flatten()
        median = np.median(tau_samples)
        hdi = az.hdi(tau_samples, hdi_prob=0.94)
        shift = traces[zone].posterior["shift_magnitude"].values.flatten()
        shift_med = np.median(shift)
        print(f"  {zone}: {median:.1f} [{hdi[0]:.1f}, {hdi[1]:.1f}], "
              f"shift = {shift_med:+.1f} mm/yr")


if __name__ == "__main__":
    main()

"""Model comparison: null vs single vs two change-points, LOO-IC and WAIC."""

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


def build_null_model(y, zone_name):
    """No change-point: constant mean."""
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=float(y.mean()), sigma=float(y.std()) * 2)
        sigma = pm.HalfNormal("sigma", sigma=float(y.std()))
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    return model


def build_one_cp_model(y, zone_name):
    """Single change-point (same as 04 but standalone)."""
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


def build_two_cp_model(y, zone_name):
    """Two change-points: three regimes."""
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


MODELS = {
    "null": build_null_model,
    "one_cp": build_one_cp_model,
    "two_cp": build_two_cp_model,
}


def fit_and_compare(y, zone_name):
    """Fit all three models and compare via LOO and WAIC."""
    traces = {}

    for model_name, builder in MODELS.items():
        print(f"  Fitting {model_name}...")
        model = builder(y, zone_name)
        with model:
            trace = pm.sample(
                draws=2000,
                tune=1000,
                chains=4,
                target_accept=0.95,
                random_seed=42,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
            )
        traces[model_name] = trace

    # LOO comparison
    compare_loo = az.compare(traces, ic="loo")
    compare_waic = az.compare(traces, ic="waic")

    return traces, compare_loo, compare_waic


def main():
    set_ieee_style()
    annual_zone = pd.read_csv(DATA_DIR / "annual_zone_rainfall.csv")

    all_comparisons_loo = {}
    all_comparisons_waic = {}

    for zone in ZONE_ORDER:
        print(f"\n{'='*50}")
        print(f"Model comparison for {zone}")
        zdf = annual_zone[annual_zone["zone"] == zone].sort_values("year")
        y = zdf["annual_precip_mm"].values.astype("float64")

        traces, comp_loo, comp_waic = fit_and_compare(y, zone)
        all_comparisons_loo[zone] = comp_loo
        all_comparisons_waic[zone] = comp_waic

        print(f"\n{zone} — LOO Comparison:")
        print(comp_loo)
        print(f"\n{zone} — WAIC Comparison:")
        print(comp_waic)

    # Save comparison tables as LaTeX
    print("\n\n=== LaTeX Tables ===")
    for zone in ZONE_ORDER:
        latex = all_comparisons_loo[zone].to_latex(float_format="%.1f")
        safe_zone = zone.replace(" ", "_")
        table_path = DATA_DIR / f"comparison_loo_{safe_zone}.tex"
        with open(table_path, "w") as f:
            f.write(latex)
        print(f"Saved: {table_path}")

    # Summary
    print("\n=== Best Model per Zone (LOO) ===")
    for zone in ZONE_ORDER:
        best = all_comparisons_loo[zone].index[0]
        elpd = all_comparisons_loo[zone].iloc[0]["elpd_loo"]
        print(f"  {zone}: {best} (ELPD-LOO = {elpd:.1f})")


if __name__ == "__main__":
    main()

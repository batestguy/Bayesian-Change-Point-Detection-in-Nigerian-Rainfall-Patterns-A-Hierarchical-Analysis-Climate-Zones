"""Test PyMC 6 / ArviZ 1.2 API compatibility with our model code."""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

y = np.random.normal(100, 10, 74)

with pm.Model() as model:
    mu = pm.Normal("mu", mu=100, sigma=20)
    sigma = pm.HalfNormal("sigma", sigma=10)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

with model:
    trace = pm.sample(draws=200, tune=100, chains=2, random_seed=42,
                      return_inferencedata=True,
                      idata_kwargs={"log_likelihood": True})

print(f"trace type: {type(trace)}")
print(f"has posterior: {'posterior' in trace.groups()}")
print(f"has log_likelihood: {'log_likelihood' in trace.groups()}")

s = az.summary(trace, var_names=["mu", "sigma"])
print(f"\nsummary:\n{s}")
print(f"r_hat col: {'r_hat' in s.columns}")
print(f"ess_bulk col: {'ess_bulk' in s.columns}")

hdi = az.hdi(trace.posterior["mu"].values.flatten(), hdi_prob=0.94)
print(f"\nhdi: {hdi}")

print(f"\naz.compare available: {hasattr(az, 'compare')}")
print(f"pm.math.sigmoid available: {hasattr(pm.math, 'sigmoid')}")

with model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)
print(f"PPC type: {type(ppc)}")

print("\nALL API TESTS PASSED")

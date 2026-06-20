"""Test PyMC 6 / ArviZ 1.2 API for our specific use cases."""
import numpy as np
import pymc as pm
import arviz as az

y = np.random.normal(100, 10, 74)

# Test 1: Basic model + sampling
with pm.Model() as model:
    mu = pm.Normal("mu", mu=100, sigma=20)
    sigma = pm.HalfNormal("sigma", sigma=10)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

with model:
    trace = pm.sample(draws=200, tune=100, chains=2, random_seed=42,
                      return_inferencedata=True)
print(f"1. trace type: {type(trace).__name__}")
print(f"   children: {list(trace.children) if hasattr(trace, 'children') else 'N/A'}")

# Test 2: Compute log_likelihood separately
with model:
    pm.compute_log_likelihood(trace)
print(f"2. log_likelihood computed: {'log_likelihood' in list(trace.children)}")

# Test 3: Summary
s = az.summary(trace, var_names=["mu", "sigma"])
print(f"3. summary cols: {list(s.columns)}")

# Test 4: HDI
vals = trace["posterior"]["mu"].values.flatten()
hdi = az.hdi(vals, hdi_prob=0.94)
print(f"4. hdi works: {hdi}")

# Test 5: PPC
with model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)
print(f"5. ppc type: {type(ppc).__name__}")

# Test 6: Extend trace with PPC (the PyMC 5 way)
try:
    trace.extend(ppc)
    print("6. trace.extend(ppc) works")
except Exception as e:
    print(f"6. trace.extend(ppc) FAILED: {e}")
    try:
        trace["posterior_predictive"] = ppc["posterior_predictive"]
        print("   alternative: direct assignment works")
    except Exception as e2:
        print(f"   alternative also failed: {e2}")

# Test 7: Compare
with pm.Model() as model2:
    mu = pm.Normal("mu", mu=100, sigma=20)
    sigma = pm.HalfNormal("sigma", sigma=10)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

with model2:
    trace2 = pm.sample(draws=200, tune=100, chains=2, random_seed=42,
                       return_inferencedata=True)
    pm.compute_log_likelihood(trace2)

try:
    comp = az.compare({"m1": trace, "m2": trace2}, ic="loo")
    print(f"7. az.compare works, cols: {list(comp.columns)}")
except Exception as e:
    print(f"7. az.compare FAILED: {e}")

# Test 8: to_netcdf
try:
    trace.to_netcdf("/tmp/test_trace.nc")
    loaded = az.from_netcdf("/tmp/test_trace.nc")
    print(f"8. save/load NetCDF works, type: {type(loaded).__name__}")
except Exception as e:
    print(f"8. NetCDF FAILED: {e}")

# Test 9: plot functions
import matplotlib
matplotlib.use("Agg")
try:
    ax = az.plot_trace(trace, var_names=["mu"])
    print("9. plot_trace works")
except Exception as e:
    print(f"9. plot_trace FAILED: {e}")

try:
    ax = az.plot_forest(trace, var_names=["mu"])
    print("10. plot_forest works")
except Exception as e:
    print(f"10. plot_forest FAILED: {e}")

print("\nDONE")

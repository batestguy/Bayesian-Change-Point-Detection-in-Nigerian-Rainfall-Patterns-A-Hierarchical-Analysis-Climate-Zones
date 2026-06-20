"""Quick API test for PyMC 5.28.5 / ArviZ 0.22."""
import numpy as np
import pymc as pm
import arviz as az

y = np.random.normal(100, 10, 50)

with pm.Model() as m:
    mu = pm.Normal("mu", mu=100, sigma=20)
    sigma = pm.HalfNormal("sigma", sigma=10)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

with m:
    tr = pm.sample(draws=200, tune=100, chains=2, random_seed=42,
                   return_inferencedata=True)

# Test 1: compute_log_likelihood
print(f"1. has compute_log_likelihood: {hasattr(pm, 'compute_log_likelihood')}")
try:
    with m:
        pm.compute_log_likelihood(tr)
    print("   compute_log_likelihood: OK")
except Exception as e:
    print(f"   compute_log_likelihood FAILED: {e}")
    # Fallback: use idata_kwargs
    with m:
        tr2 = pm.sample(draws=200, tune=100, chains=2, random_seed=42,
                         return_inferencedata=True,
                         idata_kwargs={"log_likelihood": True})
    print("   idata_kwargs fallback: OK")

# Test 2: trace type and access
print(f"2. trace type: {type(tr).__name__}")
print(f"   has groups(): {hasattr(tr, 'groups')}")
if hasattr(tr, 'groups'):
    print(f"   groups: {list(tr.groups())}")
print(f"   has children: {hasattr(tr, 'children')}")

# Test 3: dict-style access
try:
    vals = tr["posterior"]["mu"].values.flatten()
    print(f"3. dict access: OK (mu mean = {vals.mean():.1f})")
except Exception as e:
    print(f"3. dict access FAILED: {e}")
    vals = tr.posterior["mu"].values.flatten()
    print(f"   attr access: OK (mu mean = {vals.mean():.1f})")

# Test 4: PPC
with m:
    ppc = pm.sample_posterior_predictive(tr, random_seed=42)
print(f"4. ppc type: {type(ppc).__name__}")
if hasattr(ppc, 'groups'):
    print(f"   ppc groups: {list(ppc.groups())}")
if hasattr(ppc, 'children'):
    print(f"   ppc children: {list(ppc.children)}")

# Test 5: az.compare
print(f"5. az.compare available: {hasattr(az, 'compare')}")

# Test 6: HDI
hdi = az.hdi(vals, hdi_prob=0.94)
print(f"6. hdi: {hdi}")

print("\nDONE")

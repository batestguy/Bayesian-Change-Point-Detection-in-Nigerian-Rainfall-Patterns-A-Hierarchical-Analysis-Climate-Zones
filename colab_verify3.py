"""Verify all imports work after kernel restart."""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
print(f"numpy {np.__version__}, pandas {pd.__version__}")
print(f"PyMC {pm.__version__}, ArviZ {az.__version__}")
print(f"sigmoid available: {hasattr(pm.math, 'sigmoid')}")
print("ALL IMPORTS OK")

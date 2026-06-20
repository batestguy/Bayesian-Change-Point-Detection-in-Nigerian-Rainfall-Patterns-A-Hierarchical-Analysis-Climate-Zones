"""Verify PyMC 6 is installed and importable."""
import pymc as pm
import arviz as az
import pytensor
import numpy as np
import pandas as pd
print(f"PyMC {pm.__version__}")
print(f"ArviZ {az.__version__}")
print(f"PyTensor {pytensor.__version__}")
print(f"NumPy {np.__version__}")
print(f"pandas {pd.__version__}")
print("ALL IMPORTS OK")

"""Install PyMC 6 on Colab session."""
import subprocess, sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "pymc>=6.0", "arviz>=1.0"
])

import pymc as pm
import arviz as az
import pytensor
print(f"PyMC {pm.__version__}, ArviZ {az.__version__}, PyTensor {pytensor.__version__}")
print("Install OK")

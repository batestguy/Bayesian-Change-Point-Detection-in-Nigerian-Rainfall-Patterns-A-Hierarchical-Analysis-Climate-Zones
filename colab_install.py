"""Install PyMC and dependencies on Colab."""
import subprocess
import sys

packages = ["pymc==5.8.0", "arviz==0.16.1"]
for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import pymc as pm
import arviz as az
print(f"\nPyMC {pm.__version__}, ArviZ {az.__version__} ready")

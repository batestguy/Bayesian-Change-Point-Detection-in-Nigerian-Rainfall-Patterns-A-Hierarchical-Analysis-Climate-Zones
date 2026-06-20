"""Force install PyMC deps with pip."""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytensor", "pymc==5.8.0", "arviz==0.16.1"])
print("Done installing")

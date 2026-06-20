"""Force reinstall pytensor and pymc properly."""
import subprocess, sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "pytensor>=2.15.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "pymc==5.8.0", "arviz==0.16.1"])

result = subprocess.run(["ls", "/usr/local/lib/python3.12/dist-packages/pytensor/"], capture_output=True, text=True)
print(f"pytensor dir exists: {result.returncode == 0}")
if result.returncode == 0:
    print(result.stdout[:300])

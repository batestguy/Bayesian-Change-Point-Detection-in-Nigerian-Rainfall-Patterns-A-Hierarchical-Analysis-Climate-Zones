"""Reinstall PyMC to match pytensor version, then verify."""
import subprocess, sys

print("Reinstalling PyMC (compatible with pytensor 3.x)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir",
    "pymc", "arviz>=0.16"
])

result = subprocess.run(["ls", "/usr/local/lib/python3.12/dist-packages/"],
                       capture_output=True, text=True)
for pkg in ['pymc', 'pytensor', 'arviz']:
    entries = [l for l in result.stdout.split('\n') if pkg in l.lower()]
    print(f"  {pkg}: {entries}")
print("Done. Restart kernel next.")

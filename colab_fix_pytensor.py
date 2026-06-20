"""Force reinstall pytensor and verify the actual package directory exists."""
import subprocess, sys, os

print("Current numpy/pandas versions:")
import numpy as np
import pandas as pd
print(f"  numpy={np.__version__}, pandas={pd.__version__}")

print("\nChecking pytensor state before fix:")
result = subprocess.run(["ls", "-la", "/usr/local/lib/python3.12/dist-packages/"],
                       capture_output=True, text=True)
pytensor_entries = [l for l in result.stdout.split('\n') if 'pytensor' in l.lower()]
print(f"  entries: {pytensor_entries}")

print("\nForce reinstalling pytensor...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir",
    "pytensor>=2.15.0"
])

print("\nChecking pytensor state after fix:")
result = subprocess.run(["ls", "-la", "/usr/local/lib/python3.12/dist-packages/"],
                       capture_output=True, text=True)
pytensor_entries = [l for l in result.stdout.split('\n') if 'pytensor' in l.lower()]
print(f"  entries: {pytensor_entries}")

result2 = subprocess.run(["ls", "/usr/local/lib/python3.12/dist-packages/pytensor/"],
                        capture_output=True, text=True)
if result2.returncode == 0:
    print(f"  pytensor dir contents (first 10): {result2.stdout.split(chr(10))[:10]}")
else:
    print(f"  pytensor dir STILL MISSING: {result2.stderr}")

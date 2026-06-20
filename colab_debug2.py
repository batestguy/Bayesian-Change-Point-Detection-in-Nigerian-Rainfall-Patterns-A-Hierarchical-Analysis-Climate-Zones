"""Debug where packages are actually installed."""
import sys, subprocess

result = subprocess.run(["ls", "/usr/local/lib/python3.12/dist-packages/"], capture_output=True, text=True)
matches = [l for l in result.stdout.split('\n') if 'pymc' in l.lower() or 'pytensor' in l.lower() or 'theano' in l.lower()]
print(f"dist-packages pytensor/pymc dirs: {matches}")

result2 = subprocess.run(["ls", "/env/python/"], capture_output=True, text=True)
print(f"\n/env/python contents: {result2.stdout[:500]}" if result2.returncode == 0 else "/env/python does not exist")

result3 = subprocess.run(["find", "/env", "-name", "pytensor", "-type", "d"], capture_output=True, text=True)
print(f"\npytensor in /env: {result3.stdout}" if result3.stdout else "pytensor not found in /env")

result4 = subprocess.run(["pip", "show", "pytensor"], capture_output=True, text=True)
print(f"\npip show pytensor: {result4.stdout[:300]}" if result4.returncode == 0 else f"\npip show failed: {result4.stderr[:200]}")

result5 = subprocess.run([sys.executable, "-c", "import pytensor; print(pytensor.__file__)"], capture_output=True, text=True)
print(f"\npython -c import pytensor: {result5.stdout}" if result5.returncode == 0 else f"\nimport failed: {result5.stderr[:200]}")

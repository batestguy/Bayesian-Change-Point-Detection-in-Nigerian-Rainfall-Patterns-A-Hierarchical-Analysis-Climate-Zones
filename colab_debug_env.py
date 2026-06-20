"""Debug Python environment on Colab."""
import sys
import subprocess
print(f"Python: {sys.executable}")
print(f"Path: {sys.path[:5]}")

result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
lines = [l for l in result.stdout.split('\n') if 'pymc' in l.lower() or 'pytensor' in l.lower() or 'theano' in l.lower()]
print(f"\nPyMC-related packages:\n" + '\n'.join(lines) if lines else "\nNo PyMC packages found in pip list")

result2 = subprocess.run([sys.executable, "-m", "pip", "show", "pytensor"], capture_output=True, text=True)
print(f"\npip show pytensor:\n{result2.stdout}" if result2.returncode == 0 else f"\npytensor not found: {result2.stderr}")

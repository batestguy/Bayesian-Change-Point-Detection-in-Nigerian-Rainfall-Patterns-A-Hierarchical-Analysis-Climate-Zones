"""Install pinned PyMC 5.8.0 with compatible pytensor."""
import subprocess, sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--force-reinstall", "--no-cache-dir", "-q",
    "pytensor>=2.15,<2.18",
    "pymc==5.8.0",
    "arviz==0.16.1",
])

result = subprocess.run(["ls", "/usr/local/lib/python3.12/dist-packages/"],
                       capture_output=True, text=True)
for pkg in ['pymc', 'pytensor', 'arviz']:
    entries = [l for l in result.stdout.split('\n') if pkg in l.lower()]
    print(f"  {pkg}: {entries}")

result2 = subprocess.run(["ls", "/usr/local/lib/python3.12/dist-packages/pytensor/"],
                        capture_output=True, text=True)
print(f"\npytensor dir exists: {result2.returncode == 0}")
if result2.returncode == 0:
    print(f"  files: {result2.stdout.split(chr(10))[:5]}")

"""Test pytensor import directly."""
import sys, site, os
site.addsitedir("/usr/local/lib/python3.12/dist-packages")
print(f"sys.path: {sys.path}")
print(f"\nChecking pytensor location:")
os.system("find /usr/local/lib/python3.12/dist-packages -name 'pytensor' -type d 2>/dev/null | head -5")
os.system("ls /usr/local/lib/python3.12/dist-packages/ | grep -i pytensor")
print("\nTrying import pytensor...")
try:
    import pytensor
    print(f"  SUCCESS: pytensor {pytensor.__version__}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

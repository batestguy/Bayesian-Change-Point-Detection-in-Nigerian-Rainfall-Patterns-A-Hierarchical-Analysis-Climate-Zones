"""Fix sys.path and verify import."""
import sys
import site
site.addsitedir("/usr/local/lib/python3.12/dist-packages")
print(f"Updated path includes dist-packages: {'/usr/local/lib/python3.12/dist-packages' in sys.path}")
import pymc as pm
import arviz as az
import pandas as pd
print(f"PyMC {pm.__version__}, ArviZ {az.__version__}")
df = pd.read_csv("/content/data/annual_zone_rainfall.csv")
print(f"Data: {len(df)} rows, zones: {df['zone'].unique().tolist()}")

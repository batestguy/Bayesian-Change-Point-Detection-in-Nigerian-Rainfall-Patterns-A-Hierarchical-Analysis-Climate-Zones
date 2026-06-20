"""Verify PyMC and data are ready on Colab."""
import pymc as pm
import arviz as az
import pandas as pd
print(f"PyMC {pm.__version__}, ArviZ {az.__version__}")
df = pd.read_csv("/content/data/annual_zone_rainfall.csv")
print(f"Data: {len(df)} rows, zones: {df['zone'].unique().tolist()}")
print(f"Years: {df['year'].min()}-{df['year'].max()}")

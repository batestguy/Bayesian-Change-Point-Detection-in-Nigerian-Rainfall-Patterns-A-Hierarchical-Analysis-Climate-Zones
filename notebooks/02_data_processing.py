"""Clean daily data and aggregate to annual/monthly zone-level precipitation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import DATA_DIR, ZONE_ORDER
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv(DATA_DIR / "rainfall_daily.csv", parse_dates=["date"])
    print(f"Loaded {len(df)} daily records")

    # Interpolate short gaps (<=7 days) per city
    for city in df["city"].unique():
        mask = df["city"] == city
        df.loc[mask, "precipitation_mm"] = (
            df.loc[mask, "precipitation_mm"]
            .interpolate(method="linear", limit=7)
        )
        df.loc[mask, "temp_mean_c"] = (
            df.loc[mask, "temp_mean_c"]
            .interpolate(method="linear", limit=7)
        )

    remaining_nulls = df["precipitation_mm"].isna().sum()
    print(f"Precipitation nulls after interpolation: {remaining_nulls}")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Annual total precipitation per city
    annual_city = (
        df.groupby(["year", "city", "zone"])["precipitation_mm"]
        .sum()
        .reset_index()
        .rename(columns={"precipitation_mm": "annual_precip_mm"})
    )

    # Zone-level annual mean (average of 3 cities)
    annual_zone = (
        annual_city.groupby(["year", "zone"])["annual_precip_mm"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "annual_precip_mm",
                         "std": "city_std",
                         "min": "city_min",
                         "max": "city_max"})
    )

    # Monthly total precipitation per city
    monthly_city = (
        df.groupby(["year", "month", "city", "zone"])["precipitation_mm"]
        .sum()
        .reset_index()
        .rename(columns={"precipitation_mm": "monthly_precip_mm"})
    )

    # Zone-level monthly mean
    monthly_zone = (
        monthly_city.groupby(["year", "month", "zone"])["monthly_precip_mm"]
        .mean()
        .reset_index()
    )

    # Save outputs
    annual_city.to_csv(DATA_DIR / "annual_city_rainfall.csv", index=False)
    annual_zone.to_csv(DATA_DIR / "annual_zone_rainfall.csv", index=False)
    monthly_zone.to_csv(DATA_DIR / "monthly_zone_rainfall.csv", index=False)

    print("\nAnnual zone rainfall summary:")
    for zone in ZONE_ORDER:
        zdf = annual_zone[annual_zone["zone"] == zone]
        print(f"  {zone}: {zdf['annual_precip_mm'].mean():.0f} mm/yr "
              f"(range {zdf['annual_precip_mm'].min():.0f}"
              f"-{zdf['annual_precip_mm'].max():.0f}), "
              f"years {zdf['year'].min()}-{zdf['year'].max()}")

    print(f"\nSaved: annual_city_rainfall.csv, annual_zone_rainfall.csv, "
          f"monthly_zone_rainfall.csv")


if __name__ == "__main__":
    main()

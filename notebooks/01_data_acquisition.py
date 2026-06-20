"""Fetch historical daily weather data from Open-Meteo for 9 Nigerian stations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_all_stations, fetch_city_data, DATA_DIR
import pandas as pd


def main():
    stations = get_all_stations()
    frames = []

    for i, st in enumerate(stations, 1):
        print(f"[{i}/{len(stations)}] Fetching {st['city']} ({st['zone']})...")
        df = fetch_city_data(st["city"], st["lat"], st["lon"],
                             start_year=1950, end_year=2023)
        df["zone"] = st["zone"]
        df["lat"] = st["lat"]
        df["lon"] = st["lon"]
        frames.append(df)
        print(f"  -> {len(df)} days fetched, "
              f"precip nulls: {df['precipitation_mm'].isna().sum()}")

    combined = pd.concat(frames, ignore_index=True)

    out_path = DATA_DIR / "rainfall_daily.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} rows to {out_path}")

    for zone in combined["zone"].unique():
        zdf = combined[combined["zone"] == zone]
        cities = zdf["city"].unique()
        year_range = (zdf["date"].dt.year.min(), zdf["date"].dt.year.max())
        print(f"  {zone}: {list(cities)}, years {year_range[0]}-{year_range[1]}")


if __name__ == "__main__":
    main()

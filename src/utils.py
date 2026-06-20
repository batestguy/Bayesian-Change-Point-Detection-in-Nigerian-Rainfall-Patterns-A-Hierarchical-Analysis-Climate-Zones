"""Shared utilities: zone definitions, Open-Meteo fetcher, IEEE plotting style."""

import time
import json
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRACES_DIR = DATA_DIR / "traces"
FIGURES_DIR = PROJECT_ROOT / "figures"

ZONES = {
    "Sahel": [
        {"city": "Maiduguri", "lat": 11.846, "lon": 13.160},
        {"city": "Kano",      "lat": 12.000, "lon": 8.517},
        {"city": "Sokoto",    "lat": 13.060, "lon": 5.240},
    ],
    "Guinea Savanna": [
        {"city": "Abuja",  "lat": 9.058, "lon": 7.489},
        {"city": "Jos",    "lat": 9.897, "lon": 8.858},
        {"city": "Ilorin", "lat": 8.490, "lon": 4.542},
    ],
    "Coastal": [
        {"city": "Lagos",          "lat": 6.524, "lon": 3.379},
        {"city": "Port Harcourt",  "lat": 4.815, "lon": 7.050},
        {"city": "Calabar",        "lat": 4.976, "lon": 8.337},
    ],
}

ZONE_COLORS = {
    "Sahel": "#E74C3C",
    "Guinea Savanna": "#2ECC71",
    "Coastal": "#3498DB",
}

ZONE_ORDER = ["Sahel", "Guinea Savanna", "Coastal"]


def get_all_stations():
    """Return flat list of dicts with city, lat, lon, zone."""
    stations = []
    for zone, cities in ZONES.items():
        for c in cities:
            stations.append({**c, "zone": zone})
    return stations


def fetch_open_meteo(lat, lon, start_date, end_date,
                     variables="precipitation_sum,temperature_2m_mean",
                     max_retries=5):
    """Fetch daily data from Open-Meteo Historical Weather API with retry."""
    params = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": variables,
        "timezone": "Africa/Lagos",
    })
    url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "IEEpaper-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def fetch_city_data(city_name, lat, lon, start_year=1950, end_year=2023,
                    chunk_years=10, cache=True):
    """Fetch data in chunks, with optional JSON caching."""
    all_dates, all_precip, all_temp = [], [], []

    for y0 in range(start_year, end_year + 1, chunk_years):
        y1 = min(y0 + chunk_years - 1, end_year)
        cache_path = RAW_DIR / f"{city_name}_{y0}_{y1}.json"

        if cache and cache_path.exists():
            with open(cache_path, "r") as f:
                data = json.load(f)
        else:
            data = fetch_open_meteo(lat, lon, f"{y0}-01-01", f"{y1}-12-31")
            if cache:
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(data, f)
            time.sleep(2)

        daily = data.get("daily", {})
        all_dates.extend(daily.get("time", []))
        all_precip.extend(daily.get("precipitation_sum", []))
        all_temp.extend(daily.get("temperature_2m_mean", []))

    df = pd.DataFrame({
        "date": pd.to_datetime(all_dates),
        "precipitation_mm": all_precip,
        "temp_mean_c": all_temp,
    })
    df["city"] = city_name
    return df


def set_ieee_style():
    """Configure matplotlib for IEEE two-column paper figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
    })


IEEE_SINGLE_COL = (3.5, 2.5)
IEEE_DOUBLE_COL = (7.16, 3.0)
IEEE_DOUBLE_COL_TALL = (7.16, 5.0)


def savefig(fig, name, formats=("pdf", "png")):
    """Save figure to figures/ directory in multiple formats."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(FIGURES_DIR / f"{name}.{fmt}")
    plt.close(fig)

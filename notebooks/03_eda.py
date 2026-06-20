"""Exploratory data analysis: time series, rolling means, decade distributions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (DATA_DIR, FIGURES_DIR, ZONE_ORDER, ZONE_COLORS,
                        set_ieee_style, savefig, IEEE_DOUBLE_COL,
                        IEEE_DOUBLE_COL_TALL, IEEE_SINGLE_COL)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_time_series(annual_zone):
    """Annual rainfall time series per zone with rolling means."""
    fig, axes = plt.subplots(3, 1, figsize=IEEE_DOUBLE_COL_TALL, sharex=True)

    for ax, zone in zip(axes, ZONE_ORDER):
        zdf = annual_zone[annual_zone["zone"] == zone].sort_values("year")
        color = ZONE_COLORS[zone]

        ax.bar(zdf["year"], zdf["annual_precip_mm"],
               color=color, alpha=0.4, width=0.8, label="Annual total")

        roll5 = zdf["annual_precip_mm"].rolling(5, center=True).mean()
        roll10 = zdf["annual_precip_mm"].rolling(10, center=True).mean()
        ax.plot(zdf["year"], roll5, color=color, linewidth=1.2,
                label="5-yr rolling mean")
        ax.plot(zdf["year"], roll10, color="black", linewidth=1.5,
                linestyle="--", label="10-yr rolling mean")

        ax.set_ylabel("Precipitation (mm)")
        ax.set_title(zone)
        ax.legend(loc="upper right", framealpha=0.8)

    axes[-1].set_xlabel("Year")
    fig.suptitle("Annual Precipitation by Climate Zone", fontsize=10, y=1.01)
    fig.tight_layout()
    savefig(fig, "eda_timeseries")
    print("Saved eda_timeseries")


def plot_decade_distributions(annual_zone):
    """Box plots of annual precipitation by decade for each zone."""
    annual_zone = annual_zone.copy()
    annual_zone["decade"] = (annual_zone["year"] // 10) * 10

    fig, axes = plt.subplots(1, 3, figsize=IEEE_DOUBLE_COL_TALL[::-1],
                             sharey=False)

    for ax, zone in zip(axes, ZONE_ORDER):
        zdf = annual_zone[annual_zone["zone"] == zone]
        decades = sorted(zdf["decade"].unique())
        data = [zdf[zdf["decade"] == d]["annual_precip_mm"].values
                for d in decades]
        labels = [str(d) + "s" for d in decades]

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(ZONE_COLORS[zone])
            patch.set_alpha(0.5)

        ax.set_title(zone)
        ax.set_xlabel("Decade")
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("Annual Precipitation (mm)")
    fig.suptitle("Precipitation Distribution by Decade", fontsize=10, y=1.02)
    fig.tight_layout()
    savefig(fig, "eda_decades")
    print("Saved eda_decades")


def plot_zone_comparison(annual_zone):
    """Overlay all three zones on a single plot."""
    fig, ax = plt.subplots(figsize=IEEE_DOUBLE_COL)

    for zone in ZONE_ORDER:
        zdf = annual_zone[annual_zone["zone"] == zone].sort_values("year")
        roll10 = zdf["annual_precip_mm"].rolling(10, center=True).mean()
        ax.plot(zdf["year"], roll10, color=ZONE_COLORS[zone],
                linewidth=1.5, label=zone)
        ax.fill_between(zdf["year"],
                        zdf["annual_precip_mm"].rolling(10, center=True).min(),
                        zdf["annual_precip_mm"].rolling(10, center=True).max(),
                        color=ZONE_COLORS[zone], alpha=0.15)

    ax.set_xlabel("Year")
    ax.set_ylabel("Precipitation (mm, 10-yr rolling)")
    ax.set_title("Zone Comparison: 10-Year Rolling Precipitation")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "eda_zone_comparison")
    print("Saved eda_zone_comparison")


def print_summary_stats(annual_zone):
    """Print key statistics for the paper."""
    print("\n=== Summary Statistics ===")
    for zone in ZONE_ORDER:
        zdf = annual_zone[annual_zone["zone"] == zone]
        print(f"\n{zone}:")
        print(f"  Mean: {zdf['annual_precip_mm'].mean():.1f} mm")
        print(f"  Std:  {zdf['annual_precip_mm'].std():.1f} mm")
        print(f"  CV:   {zdf['annual_precip_mm'].std() / zdf['annual_precip_mm'].mean():.3f}")
        print(f"  Min:  {zdf['annual_precip_mm'].min():.1f} mm "
              f"({zdf.loc[zdf['annual_precip_mm'].idxmin(), 'year']})")
        print(f"  Max:  {zdf['annual_precip_mm'].max():.1f} mm "
              f"({zdf.loc[zdf['annual_precip_mm'].idxmax(), 'year']})")


def main():
    set_ieee_style()
    annual_zone = pd.read_csv(DATA_DIR / "annual_zone_rainfall.csv")
    print(f"Loaded {len(annual_zone)} annual zone records")

    plot_time_series(annual_zone)
    plot_decade_distributions(annual_zone)
    plot_zone_comparison(annual_zone)
    print_summary_stats(annual_zone)


if __name__ == "__main__":
    main()

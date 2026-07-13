from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from geepers import gps
from geepers.io import XarrayReader


def create_comparison_plot(
    relative_comparison_csv: Path | str,
) -> tuple[plt.Figure, plt.Axes]:
    """Create scatter plot comparing GPS and InSAR measurements.

    Parameters
    ----------
    relative_comparison_csv : Path | str
        CSV file containing 'station', 'date', 'measurement', and 'value' columns

    Returns
    -------
    fig : plt.Figure
        The figure object
    ax : plt.Axes
        The axes object

    """
    df = pd.read_csv(relative_comparison_csv)
    df_wide = df.pivot_table(
        index=["station", "date"], columns="measurement", values="value"
    ).reset_index()

    valid_mask = ~(np.isnan(df_wide.los_gps) | np.isnan(df_wide.los_insar))
    gps_valid = df_wide.los_gps[valid_mask]
    insar_valid = df_wide.los_insar[valid_mask]

    rmse = np.sqrt(np.mean((gps_valid - insar_valid) ** 2))

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # Plot the scatter points
    ax.scatter(
        df_wide.los_insar,
        df_wide.los_gps,
        color="#2b8cbe",
        alpha=0.6,
        s=30,
        label=f"RMSE: {rmse:.1f} mm/yr",
    )

    # Plot the 1:1 line
    lims = np.array(
        [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
    )
    ax.plot(lims, lims, "gray", linestyle="-", alpha=0.8, zorder=0)

    ax.set_aspect("equal")

    # Set limits to be symmetric around zero
    max_val = max(abs(lims))
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("InSAR [mm/year]")
    ax.set_ylabel("GPS [mm/year]")
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        bbox_to_anchor=(0.02, 0.98),
        loc="upper left",
    )

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def create_rate_comparison_plot(
    rates: pd.DataFrame,
    quality_column: str = "similarity",
    quality_cmap: str = "viridis",
) -> tuple[plt.Figure, plt.Axes]:
    """Create a publication-quality scatter plot comparing GPS and InSAR rates.

    Parameters
    ----------
    rates : pd.DataFrame
        DataFrame with columns: station, gps_velocity, insar_velocity
    quality_column : str, optional
        Column to use for color mapping. Default is "similarity".
    quality_cmap : str, optional
        Colormap to use for color mapping. Default is "viridis".

    Returns
    -------
    fig : plt.Figure
        The figure object
    ax : plt.Axes
        The axes object

    """
    plt.style.use("seaborn-v0_8-paper")

    # Calculate correlation coefficient and RMSE
    rmse = np.sqrt(np.mean((rates.gps_velocity - rates.insar_velocity) ** 2))

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # Plot the scatter points
    scatter_img = ax.scatter(
        rates.insar_velocity,
        rates.gps_velocity,
        # color="#2b8cbe",
        c=rates[quality_column],
        cmap=quality_cmap,
        alpha=0.8,
        s=40,
    )

    # Plot the 1:1 line
    lims = np.array(
        [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
    )
    ax.plot(lims, lims, "gray", linestyle="-", alpha=0.8, zorder=0)

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Set limits to be symmetric around zero
    max_val = max(abs(lims))
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Add grid
    ax.grid(True, linestyle=":", alpha=0.6)

    # Set labels and title
    ax.set_xlabel("InSAR Rate [mm/year]")
    ax.set_ylabel("GPS Rate [mm/year]")

    # Add minor ticks
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
    label = f"RMSE: {rmse:.1f} mm/yr"
    ax.text(
        x=0.05,
        y=0.95,
        s=label,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )
    fig.colorbar(scatter_img, ax=ax, label=quality_column)
    return fig, ax


def plot_stations_on_map(
    filename: Path | str,
    units: str | None = None,
    stations: Sequence[str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot locations of GPS stations contained in `filename`.

    Parameters
    ----------
    filename : Path | str
        Path to the InSAR stack file.
    units : str | None, optional
        Units of the InSAR stack. Default is None.
    stations : Sequence[str] | None, optional
        List of station names to plot. Default is None.

    Returns
    -------
    fig : plt.Figure
        The figure object
    ax : plt.Axes
        The axes object

    """
    reader = XarrayReader.from_file(filename, units=units)
    gdf = gps.get_stations_within_image(reader)
    if stations is not None:
        gdf = gdf[gdf.name.isin(stations)]
    fig, ax = plt.subplots(figsize=(6, 5))
    da_lonlat = reader.da.rio.reproject("EPSG:4326")
    da_lonlat.plot.imshow(ax=ax, cmap="gray")
    gdf.plot(ax=ax, color="red", markersize=10)
    # Add labels from the names
    for name, geom in zip(gdf.name, gdf.geometry, strict=True):
        ax.text(geom.x, geom.y, name, fontsize=8)
    return fig, ax


def plot_rmse_vs_distance(
    rmse_df: pd.DataFrame,
    binned_df: pd.DataFrame | None = None,
    requirement=None,
    units: str = "m",
    ax=None,
    loglog: bool = True,
):
    """Structure function: pairwise differential RMSE vs separation.

    Parameters
    ----------
    rmse_df : pd.DataFrame
        Output of `geepers.analysis.pairwise_differential_rmse`.
    binned_df : pd.DataFrame, optional
        Output of `geepers.analysis.binned_rmse_profile`; adds the
        median-per-bin curve.
    requirement : callable, optional
        Requirement curve ``req(distance_km) -> threshold`` (same units
        as ``rmse``), drawn as a dashed line.
    units : str
        Unit label for the y axis. Default "m".
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    loglog : bool
        Use logarithmic x and y axes (the structure-function
        convention). Default True.

    Returns
    -------
    matplotlib.axes.Axes

    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        rmse_df["distance_km"],
        rmse_df["rmse"],
        s=25,
        c="0.2",
        alpha=0.6,
        zorder=3,
        label="station pairs",
    )
    if binned_df is not None and len(binned_df):
        ax.plot(
            binned_df["distance_km"],
            binned_df["rmse_median"],
            "r-o",
            ms=4,
            lw=1.5,
            zorder=4,
            label="median per bin",
        )
    if requirement is not None:
        dmin = max(rmse_df["distance_km"].min() * 0.9, 1e-3)
        d = np.linspace(dmin, rmse_df["distance_km"].max() * 1.02, 200)
        ax.plot(d, requirement(d), "k--", lw=1.5, label="requirement")

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Station separation (km)")
    ax.set_ylabel(f"Differential RMSE ({units})")
    ax.set_title("InSAR - GPS structure function")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax

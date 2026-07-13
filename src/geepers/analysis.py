"""Data analysis and comparison functions."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from pyproj import Geod

logger = logging.getLogger("geepers")

_GEOD = Geod(ellps="WGS84")


def create_tidy_df(station_to_merged_df: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-station dataframes into a tidy (long-form) dataframe.

    Parameters
    ----------
    station_to_merged_df
        Mapping from station name to a *wide* dataframe that contains one column
        per variable (e.g. ``los_gps``, ``los_insar``).

    Returns
    -------
    pandas.DataFrame
        Long-form dataframe with columns ``station``, ``date``, ``measurement``
        and ``value`` suitable for plotting with *seaborn* or *altair*.

    """
    dfs: list[pd.DataFrame] = []
    for station, df in station_to_merged_df.items():
        df_reset = df.reset_index(names="date")
        df_melted = pd.melt(
            df_reset, id_vars=["date"], var_name="measurement", value_name="value"
        )
        df_melted["id"] = station
        if df_melted.empty:
            logger.warning("No data for station %s", station)
            continue
        dfs.append(df_melted)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df[["id", "date", "measurement", "value"]]


def compare_relative_gps_insar(
    station_to_merged_df: Mapping[str, pd.DataFrame],
    *,
    reference_station: str,
) -> pd.DataFrame:
    """Compute relative displacement between all stations and a reference.

    The function subtracts the *GPS* and *InSAR* line-of-sight (LOS)
    displacements of *reference_station* from every other station, yielding
    time-series of relative motion.

    Parameters
    ----------
    station_to_merged_df
        Mapping from station name to merged GPS/InSAR dataframe produced by the
        main workflow.
    reference_station
        Name of the station to treat as the zero reference.

    Returns
    -------
    pandas.DataFrame
        Tidy dataframe with the relative series and their differences.

    """
    if reference_station not in station_to_merged_df:
        msg = f"Reference station '{reference_station}' not found."
        raise ValueError(msg)

    ref_df = station_to_merged_df[reference_station]
    results: list[pd.DataFrame] = []

    for station, df in station_to_merged_df.items():
        common_index = df.index.intersection(ref_df.index)
        if common_index.empty:
            logger.warning(
                "No common epochs between %s and %s", station, reference_station
            )
            continue

        station_df = df.loc[common_index]
        ref_df_aligned = ref_df.loc[common_index]

        relative_gps = station_df["los_gps"] - ref_df_aligned["los_gps"]
        relative_insar = station_df["los_insar"] - ref_df_aligned["los_insar"]
        difference = relative_insar - relative_gps

        results.append(
            pd.DataFrame(
                {
                    "id": station,
                    "date": common_index,
                    "relative_gps": relative_gps,
                    "relative_insar": relative_insar,
                    "difference": difference,
                }
            )
        )

    return pd.concat(results, ignore_index=True)


def pairwise_differential_rmse(
    station_to_merged_df: Mapping[str, pd.DataFrame],
    station_coords: Mapping[str, tuple[float, float]],
    *,
    min_common_dates: int = 3,
) -> pd.DataFrame:
    """Relative InSAR-GPS misfit for every station pair vs separation.

    For each unique pair (A, B), forms the *relative* displacement
    series between the two stations for GPS and InSAR separately -
    which cancels any common reference/datum - and reduces the misfit
    of the two relative series to an RMSE. Plotting `rmse` against
    `distance_km` gives the structure function used for OPERA DISP
    requirement verification.

    Parameters
    ----------
    station_to_merged_df
        Mapping from station name to a dataframe indexed by date with
        ``los_gps`` and ``los_insar`` columns (the merged tables built
        by the main workflow).
    station_coords
        Mapping from station name to (lon, lat) in degrees.
    min_common_dates : int
        Skip pairs with fewer common valid epochs. Default 3.

    Returns
    -------
    pd.DataFrame
        One row per pair: ``station1``, ``station2``, ``distance_km``,
        ``rmse`` (same units as the series), ``bias`` (mean of the
        differential misfit) and ``n_dates``.

    """
    names = [s for s in station_to_merged_df if s in station_coords]
    series = {
        s: station_to_merged_df[s][["los_gps", "los_insar"]].dropna()
        for s in names
    }

    records = []
    for i, s1 in enumerate(names):
        d1 = series[s1]
        for s2 in names[i + 1:]:
            d2 = series[s2]
            common = d1.index.intersection(d2.index)
            if len(common) < min_common_dates:
                continue
            gps_diff = (
                d2.loc[common, "los_gps"].to_numpy()
                - d1.loc[common, "los_gps"].to_numpy()
            )
            insar_diff = (
                d2.loc[common, "los_insar"].to_numpy()
                - d1.loc[common, "los_insar"].to_numpy()
            )
            resid = insar_diff - gps_diff
            lon1, lat1 = station_coords[s1]
            lon2, lat2 = station_coords[s2]
            dist_km = _GEOD.inv(lon1, lat1, lon2, lat2)[2] / 1000.0
            records.append(
                {
                    "station1": s1,
                    "station2": s2,
                    "distance_km": dist_km,
                    "rmse": float(np.sqrt(np.mean(resid**2))),
                    "bias": float(np.mean(resid)),
                    "n_dates": len(common),
                }
            )

    if not records:
        logger.warning("No station pairs with %d+ common dates", min_common_dates)
    return pd.DataFrame(
        records,
        columns=["station1", "station2", "distance_km", "rmse", "bias", "n_dates"],
    )


def binned_rmse_profile(
    rmse_df: pd.DataFrame,
    n_bins: int = 10,
    requirement: Callable[[np.ndarray], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Bin the pairwise RMSE by distance (median profile + compliance).

    Parameters
    ----------
    rmse_df : pd.DataFrame
        Output of `pairwise_differential_rmse`.
    n_bins : int
        Number of equal-width distance bins. Default 10.
    requirement : callable, optional
        Requirement curve ``req(distance_km) -> threshold`` in the same
        units as ``rmse`` (e.g. ``lambda d: (3 + 0.5*np.sqrt(d)) / 1000``
        for "3 mm + 0.5 mm * sqrt(km)" with series in meters). Adds a
        per-bin ``fraction_passing`` column.

    Returns
    -------
    pd.DataFrame
        One row per non-empty bin: ``distance_km`` (bin center),
        ``rmse_median``, ``rmse_mean``, ``rmse_p90``, ``n_pairs`` and,
        if a requirement was given, ``requirement`` and
        ``fraction_passing``.

    """
    if rmse_df.empty:
        return pd.DataFrame()
    d = rmse_df["distance_km"].to_numpy()
    r = rmse_df["rmse"].to_numpy()
    edges = np.linspace(0.0, d.max() * 1.001, n_bins + 1)
    which = np.digitize(d, edges) - 1

    rows = []
    for b in range(n_bins):
        sel = which == b
        if not sel.any():
            continue
        center = 0.5 * (edges[b] + edges[b + 1])
        row = {
            "distance_km": center,
            "rmse_median": float(np.median(r[sel])),
            "rmse_mean": float(np.mean(r[sel])),
            "rmse_p90": float(np.percentile(r[sel], 90)),
            "n_pairs": int(sel.sum()),
        }
        if requirement is not None:
            thresh = requirement(d[sel])
            row["requirement"] = float(np.median(requirement(np.array([center]))))
            row["fraction_passing"] = float(np.mean(r[sel] <= thresh))
        rows.append(row)
    return pd.DataFrame(rows)


def epoch_rmse(
    station_to_merged_df: Mapping[str, pd.DataFrame],
    min_stations: int = 3,
) -> pd.DataFrame:
    """Network-wide InSAR-GPS misfit per acquisition epoch.

    For each date, computes the spread of the per-station residuals
    ``los_insar - los_gps`` after removing the network median at that
    date (which absorbs any common datum/reference shift). Spikes in
    the result flag problem epochs: ionospheric storms, unwrapping
    failures, snow cover.

    Parameters
    ----------
    station_to_merged_df
        Mapping from station name to the merged per-station dataframe.
    min_stations : int
        Skip epochs observed by fewer stations. Default 3.

    Returns
    -------
    pd.DataFrame
        Indexed by date with columns ``rmse``, ``mad`` (1.4826-scaled)
        and ``n_stations``.

    """
    frames = []
    for station, df in station_to_merged_df.items():
        resid = (df["los_insar"] - df["los_gps"]).rename(station)
        frames.append(resid)
    wide = pd.concat(frames, axis=1)

    resid = wide.sub(wide.median(axis=1), axis=0)
    n = resid.notna().sum(axis=1)
    out = pd.DataFrame(
        {
            "rmse": np.sqrt((resid**2).mean(axis=1)),
            "mad": 1.4826 * (resid - resid.median(axis=1).values[:, None])
            .abs()
            .median(axis=1),
            "n_stations": n,
        }
    )
    return out[n >= min_stations]

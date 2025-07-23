"""Data analysis and comparison functions."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

logger = logging.getLogger("geepers")


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

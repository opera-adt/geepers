from dataclasses import asdict

import geopandas as gpd
import numpy as np
import pandas as pd

from .gps import read_station_llas
from .midas import MidasResult, midas
from .quality import compute_station_quality
from .schemas import DailyDispModel

EMPTY_MIDAS = MidasResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.array([]))


def calculate_rates(
    df: pd.DataFrame,
    outlier_threshold: float = 50,
    to_mm: bool = True,
) -> gpd.GeoDataFrame:
    """Calculate rates for each station from GPS and InSAR time series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: station, date, measurement, value
    outlier_threshold : float
        Remove measurements with absolute values greater than this
    to_mm : bool
        If True, output is in mm/year.
        Otherwise, units are no changed (meters/year)

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with GPS and InSAR rates for each station.
        If `to_mm` is True, output is in mm/year.
        Otherwise, units are no changed (meters/year).
        Includes sigma_los_mm column if uncertainty data is available.

    """
    # Convert date to datetime if it's not already
    df["date"] = pd.to_datetime(df["date"])

    # Remove obvious outliers
    df = df[abs(df["value"]) < outlier_threshold]

    # Optional validation for daily displacement data
    DailyDispModel.validate(df)

    # Pivot to get separate GPS and InSAR columns
    df_wide = df.pivot_table(
        index=["station", "date"], columns="measurement", values="value"
    ).reset_index()

    # Function to calculate rate for a single station's time series
    def calc_station_metrics(group: pd.DataFrame) -> pd.Series:
        # Convert dates to years since first measurement
        years = (group["date"] - group["date"].min()).dt.total_seconds() / (
            365.25 * 24 * 3600
        )

        # Start with nans for rates
        gps_velocity_l2 = insar_velocity = insar_velocity_l2 = np.nan
        const = 1000 if to_mm else 1

        # GPS rate
        gps_midas = EMPTY_MIDAS
        if not group["los_gps"].isna().all():
            mask = ~group["los_gps"].isna()
            # Calculate rates using least squares fit
            gps_velocity_l2 = (
                np.polyfit(years[mask], group["los_gps"][mask], 1)[0] * const
            )
            group_df = group[["date", "los_gps"]].dropna().set_index("date")
            gps_midas = const * _get_midas_rate(group_df)

        # InSAR rate
        if not group["los_insar"].isna().all():
            mask = ~group["los_insar"].isna()
            x, y = np.array(years[mask]), np.array(group["los_insar"][mask])
            insar_velocity_l2 = np.polyfit(x, y, 1)[0] * const
            group_df_insar = group[["date", "los_insar"]].dropna().set_index("date")
            insar_midas = const * _get_midas_rate(group_df_insar)
            insar_velocity = insar_midas.velocity

        # Compute station quality metrics
        station_df = group.set_index("date")
        quality = compute_station_quality(station_df)
        quality_dict = asdict(quality)

        midas_outputs = _dump_midas(gps_midas, prefix="gps_")

        # Compute LOS uncertainty
        # TODO: this is not implemented for the rate from the observations
        return pd.Series(
            {
                "difference": float(insar_velocity - gps_midas.velocity),
                "insar_velocity": float(insar_velocity),
                "insar_velocity_l2": float(insar_velocity_l2),
                "gps_velocity_l2": gps_velocity_l2,
                **quality_dict,
                **midas_outputs,
            }
        )

    # Calculate rates for each station
    rates = df_wide.groupby("station").apply(calc_station_metrics)

    # Get the longitude and latitude of each station
    gdf_stations = read_station_llas(to_geodataframe=True)
    rates = gpd.GeoDataFrame(
        rates,
        geometry=gdf_stations[gdf_stations.name.isin(rates.index)].geometry.tolist(),
    )

    return rates


# TODO: is this where a Pandera schema would be useful?
def _get_midas_rate(cur_df_measurement: pd.DataFrame) -> MidasResult:
    time_deltas = cur_df_measurement.index - cur_df_measurement.index[0]
    years = time_deltas.total_seconds() / (365.25 * 24 * 60 * 60)
    values = cur_df_measurement.values.squeeze()
    return midas(times=years.to_numpy(), values=values)


def _dump_midas(
    m: MidasResult,
    prefix: str = "",
    skip_cols: tuple[str, ...] = ("residuals", "reference_position"),
) -> dict[str, float]:
    return {f"{prefix}{k}": v for k, v in asdict(m).items() if k not in skip_cols}

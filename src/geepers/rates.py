from dataclasses import asdict

import geopandas as gpd
import numpy as np
import pandas as pd

from .gps_sources.unr import UnrSource
from .midas import MidasResult, midas
from .quality import compute_station_quality
from .schemas import RatesSchema

EMPTY_MIDAS = MidasResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.array([]))


def calculate_rates(
    df: pd.DataFrame,
    outlier_threshold: float = 50,
    use_midas_for_insar: bool = False,
    to_mm: bool = True,
) -> gpd.GeoDataFrame:
    """Calculate rates for each station from GPS and InSAR time series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: station, date, measurement, value
    outlier_threshold : float
        Remove measurements with absolute values greater than this
    use_midas_for_insar : bool
        If True, use MIDAS to calculate InSAR rate.
        Otherwise, use least squares fit.
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

    # Pivot to get separate GPS and InSAR columns
    df_wide = df.pivot_table(
        index=["id", "date"], columns="measurement", values="value"
    ).reset_index()

    # TODO: Make the schema to validate the pivoted data

    # Function to calculate rate for a single station's time series
    def calc_station_metrics(group: pd.DataFrame) -> pd.Series:
        # Convert dates to years since first measurement
        years = (group["date"] - group["date"].min()).dt.total_seconds() / (
            365.25 * 24 * 3600
        )

        # Start with nans for rates
        insar_velocity = np.nan
        const = 1000 if to_mm else 1

        # GPS rate
        gps_midas = EMPTY_MIDAS
        if not group["los_gps"].isna().all():
            mask = ~group["los_gps"].isna()
            group_df = group[["date", "los_gps"]].dropna().set_index("date")
            gps_midas = const * _get_midas_rate(group_df)

        # InSAR rate
        if not group["los_insar"].isna().all():
            mask = ~group["los_insar"].isna()
            if use_midas_for_insar:
                group_df_insar = group[["date", "los_insar"]].dropna().set_index("date")
                insar_midas = const * _get_midas_rate(group_df_insar)
                insar_velocity = insar_midas.velocity
            else:
                x, y = np.array(years[mask]), np.array(group["los_insar"][mask])
                insar_velocity = (
                    np.nan if len(x) < 2 else np.polyfit(x, y, 1)[0] * const
                )

        # Compute station quality metrics
        station_df = group.set_index("date")
        quality = compute_station_quality(station_df)
        quality_dict = asdict(quality)

        midas_outputs = _dump_midas(gps_midas, prefix="gps_")

        # Compute LOS uncertainty in the rates
        # TODO: this is not implemented for the rate from the observations
        return pd.Series(
            {
                "difference": float(insar_velocity - gps_midas.velocity),
                "insar_velocity": float(insar_velocity),
                **quality_dict,
                **midas_outputs,
            }
        )

    # Calculate rates for each station id
    rates = df_wide.groupby("id").apply(calc_station_metrics, include_groups=False)  # type: ignore[call-overload]
    # Get the longitude and latitude of each station
    unr_source = UnrSource()
    gdf_stations = unr_source.stations()
    rates = gpd.GeoDataFrame(
        rates,
        geometry=gdf_stations[gdf_stations.id.isin(rates.index)].geometry.tolist(),
    )

    rates = RatesSchema.validate(rates, lazy=True)

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

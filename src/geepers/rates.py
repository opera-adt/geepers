from dataclasses import asdict

import geopandas as gpd
import numpy as np
import pandas as pd

from .gps import read_station_llas
from .los import get_default_los_vector
from .midas import MidasResult, midas
from .quality import compute_station_quality
from .schemas import DailyDispModel
from .uncertainty import sigma_los

EMPTY_MIDAS = MidasResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.array([]))


def calculate_rates(
    df: pd.DataFrame,
    outlier_threshold: float = 50,
    to_mm: bool = True,
    los_vector: np.ndarray | None = None,
    satellite: str = "sentinel1_ascending",
    validate_daily_schema: bool = False,
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
    los_vector : np.ndarray, optional
        3-element LOS unit vector (u_east, u_north, u_up) for computing
        LOS uncertainty. If None, uses default vector for satellite.
    satellite : str, optional
        Satellite configuration for default LOS vector. Default is
        "sentinel1_ascending".
    validate_daily_schema : bool, optional
        Whether to validate daily displacement data against DailyDispModel.
        Default is False.

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

    # Set up LOS vector for uncertainty calculation
    if los_vector is None:
        los_vector = get_default_los_vector(satellite)

    # Optional validation for daily displacement data
    if validate_daily_schema:
        # Check if data looks like daily displacement data
        expected_cols = {"station", "date", "east_mm", "north_mm", "up_mm"}
        if expected_cols.issubset(set(df.columns)):
            try:
                DailyDispModel.validate(df, lazy=True)
            except Exception:
                # If validation fails, continue without error but log
                import logging

                logger = logging.getLogger("geepers")
                logger.warning("Daily displacement data validation failed")

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
            if sum(mask) > 2:  # Need at least 3 points for meaningful rate
                # Calculate rates using least squares fit
                gps_velocity_l2 = (
                    np.polyfit(years[mask], group["los_gps"][mask], 1)[0] * const
                )

                group_df = group[["date", "los_gps"]].dropna().set_index("date")
                gps_midas = const * _get_midas_rate(group_df)

        # InSAR rate
        if not group["los_insar"].isna().all():
            mask = ~group["los_insar"].isna()
            if sum(mask) > 2:  # Need at least 3 points for meaningful rate
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

        # Compute LOS uncertainty if sigma columns are available
        sigma_los_mm = np.nan
        has_sigma_cols = all(
            col in group.columns
            for col in ["sigma_east_mm", "sigma_north_mm", "sigma_up_mm"]
        )
        if has_sigma_cols:
            try:
                # Use the first available row with uncertainty data
                sigma_row = group.dropna(
                    subset=["sigma_east_mm", "sigma_north_mm", "sigma_up_mm"]
                )
                if not sigma_row.empty:
                    sigma_los_series = sigma_los(sigma_row.iloc[:1], los_vector)
                    sigma_los_mm = float(sigma_los_series.iloc[0])
            except (KeyError, IndexError):
                # If uncertainty computation fails, keep NaN
                pass

        return pd.Series(
            {
                "difference": float(insar_velocity - gps_midas.velocity),
                "insar_velocity": float(insar_velocity),
                "insar_velocity_l2": float(insar_velocity_l2),
                "gps_velocity_l2": gps_velocity_l2,
                "sigma_los_mm": sigma_los_mm,
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

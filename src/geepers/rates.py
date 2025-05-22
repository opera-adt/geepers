from dataclasses import asdict
from typing import Literal

import numpy as np
import pandas as pd

from .midas import MidasResult, midas

try:
    from ._robust_fit import robust_linear_fit
except ImportError:
    print("Failed to import `robust_linear_fit`: e")
    robust_linear_fit

EMPTY_MIDAS = MidasResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.array([]))


def calculate_rates(
    df: pd.DataFrame, outlier_threshold: float = 50, to_mm: bool = True
) -> pd.DataFrame:
    """Calculate rates for each station from GPS and InSAR time series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: station, date, measurement, value
    outlier_threshold : float
        Remove measurements with absolute values greater than this

    Returns
    -------
    pd.DataFrame
        DataFrame with GPS and InSAR rates for each station
        If `to_mm` is True, output is in mm/year.
        Otherwise, units are no changed (meters/year)
    """
    # Convert date to datetime if it's not already
    df["date"] = pd.to_datetime(df["date"])

    # Remove obvious outliers
    df = df[abs(df["value"]) < outlier_threshold]

    # Pivot to get separate GPS and InSAR columns
    df_wide = df.pivot_table(
        index=["station", "date"], columns="measurement", values="value"
    ).reset_index()

    # Function to calculate rate for a single station's time series
    def calc_station_metrics(group):
        # Convert dates to years since first measurement
        years = (group["date"] - group["date"].min()).dt.total_seconds() / (
            365.25 * 24 * 3600
        )
        time_span_years = years.iloc[-1] - years.iloc[0]

        # Start with nans for rates
        num_gps = gps_velocity_l2 = insar_velocity = insar_velocity_l2 = np.nan
        tcoh = similarity = np.nan
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
                gps_midas = const * _get_group_midas(group, "los_gps")
                num_gps = len(group["los_gps"][mask])

        # InSAR rate
        # insar_midas = EMPTY_MIDAS
        if not group["los_insar"].isna().all():
            mask = ~group["los_insar"].isna()
            if sum(mask) > 2:  # Need at least 3 points for meaningful rate
                x, y = np.array(years[mask]), np.array(group["los_insar"][mask])
                insar_velocity_l2 = np.polyfit(x, y, 1)[0] * const
                insar_velocity = const * robust_linear_fit(x, y)[0]
                # insar_midas = const * _get_group_midas(group, "los_insar")

        if not np.isnan(insar_velocity):
            tcoh = group["temporal_coherence"].dropna().iloc[0]
            similarity = group["similarity"].dropna().iloc[0]

        midas_outputs = _dump_midas(gps_midas, prefix="gps_")
        # midas_outputs |= _dump_midas(insar_midas, prefix="insar_")
        return pd.Series(
            {
                "difference": float(insar_velocity - gps_midas.velocity),
                "insar_velocity": float(insar_velocity),
                "insar_velocity_l2": float(insar_velocity_l2),
                "temporal_coherence": tcoh,
                "similarity": similarity,
                "num_gps_points": num_gps,
                "gps_time_span_years": time_span_years,
                "gps_velocity_l2": gps_velocity_l2,
                **midas_outputs,
            }
        )

    # Calculate rates for each station
    # breakpoint()
    # rates = df_wide.groupby("station").apply(calc_station_metrics).reset_index()
    rates = df_wide.groupby("station").apply(calc_station_metrics)

    # Remove stations where either rate is NaN
    # rates = rates.dropna()
    return rates


def get_midas_rate(
    df: pd.DataFrame, station: str, col: Literal["los_gps", "los_insar"]
) -> MidasResult:
    """Calculate the MIDAS rate for one gps station.

    `df` is the result from the result `combined_df` from `create_tidy_df`.
    """
    ddf = df[df.station == station]
    cur_df_measurement = ddf[ddf.measurement == col]
    return _get_midas_rate(cur_df_measurement)


def _get_group_midas(grouped_df, col: Literal["los_gps", "los_insar"]):
    df = grouped_df[["station", "date", col]].dropna()
    time_deltas = df.date - df.date.iloc[0]
    years = time_deltas.dt.total_seconds() / (365.25 * 24 * 60 * 60)
    values = df[col].to_numpy()
    return midas(times=years.to_numpy(), values=values)


def _get_midas_rate(cur_df_measurement):
    time_deltas = cur_df_measurement.date - cur_df_measurement.date.iloc[0]
    years = time_deltas.dt.total_seconds() / (365.25 * 24 * 60 * 60)
    values = cur_df_measurement.value.to_numpy()
    return midas(times=years.to_numpy(), values=values)


def _dump_midas(
    m: MidasResult,
    prefix: str = "",
    skip_cols: tuple[str] = ("residuals", "reference_position"),
) -> dict:
    return {f"{prefix}{k}": v for k, v in asdict(m).items() if k not in skip_cols}

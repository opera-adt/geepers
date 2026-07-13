"""Station quality metrics and computation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def gap_percentage(
    dates: ArrayLike,
    sampling_days: float = 1.0,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> float:
    """Percentage of missing epochs in a time series.

    Compares the number of observed epochs against the number expected
    on a regular grid with the given sampling interval.

    Parameters
    ----------
    dates : array-like of datetime64
        Observation epochs.
    sampling_days : float
        Nominal sampling interval in days. Default is 1 (daily).
    start, end : str or pd.Timestamp, optional
        Evaluate coverage over this window instead of the observed span
        (epochs outside are dropped; missing lead/tail time counts as
        gaps). Useful to grade stations against a campaign period.

    Returns
    -------
    float
        Gap percentage in [0, 100]. NaN if no epochs remain.

    """
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates))).sort_values()
    t0 = pd.Timestamp(start) if start is not None else dates[0]
    t1 = pd.Timestamp(end) if end is not None else dates[-1]
    dates = dates[(dates >= t0) & (dates <= t1)]
    if len(dates) == 0:
        return np.nan

    step = pd.Timedelta(days=sampling_days)
    n_expected = round((t1 - t0) / step) + 1
    # Snap observed epochs to the regular grid; count unique slots
    slots = np.round((dates - t0) / step).astype(int)
    n_present = len(np.unique(slots))
    return float(100.0 * (1.0 - n_present / n_expected))


@dataclass(slots=True)
class StationQuality:
    """Quality metrics for a GPS/InSAR station.

    Parameters
    ----------
    num_gps : int
        Number of valid GPS measurements at this station.
    gps_time_span_years : float
        Time span in years between first and last GPS measurement.
    temporal_coherence : float | None
        Mean temporal coherence value, or None if not available.
    similarity : float | None
        Mean phase similarity value, or None if not available.
    rms_misfit : float
        Root-mean-square misfit between GPS and InSAR measurements.

    """

    num_gps: int
    gps_time_span_years: float
    temporal_coherence: float | None
    similarity: float | None
    rms_misfit: float


def compute_station_quality(df: pd.DataFrame) -> StationQuality:
    """Compute quality metrics for a single station's merged GPS/InSAR data.

    Parameters
    ----------
    df : pd.DataFrame
        Merged GPS/InSAR dataframe for a single station with columns:
        - los_gps: GPS line-of-sight measurements
        - los_insar: InSAR line-of-sight measurements
        - temporal_coherence: optional temporal coherence values
        - similarity: optional phase similarity values

    Returns
    -------
    StationQuality
        Quality metrics for this station.

    """
    # GPS time span calculation
    valid_gps_mask = ~df["los_gps"].isna()
    num_gps = int(valid_gps_mask.sum())

    if num_gps > 0:
        gps_dates = df.index[valid_gps_mask]
        gps_time_span_years = float(
            (gps_dates.max() - gps_dates.min()).total_seconds() / (365.25 * 24 * 3600)
        )
    else:
        gps_time_span_years = 0.0

    # Temporal coherence (mean of available values)
    temporal_coherence = None
    if "temporal_coherence" in df.columns:
        tcoh_values = df["temporal_coherence"].dropna()
        if len(tcoh_values) > 0:
            temporal_coherence = float(tcoh_values.mean())

    # Similarity (mean of available values)
    similarity = None
    if "similarity" in df.columns:
        sim_values = df["similarity"].dropna()
        if len(sim_values) > 0:
            similarity = float(sim_values.mean())

    # RMS misfit between GPS and InSAR
    common_mask = ~(df["los_gps"].isna() | df["los_insar"].isna())
    if common_mask.sum() > 0:
        diff = df.loc[common_mask, "los_insar"] - df.loc[common_mask, "los_gps"]
        rms_misfit = float(np.sqrt(np.mean(diff**2)))
    else:
        rms_misfit = np.inf

    return StationQuality(
        num_gps=num_gps,
        gps_time_span_years=gps_time_span_years,
        temporal_coherence=temporal_coherence,
        similarity=similarity,
        rms_misfit=rms_misfit,
    )


def station_quality_to_dict(quality: StationQuality) -> dict[str, float | int | None]:
    """Convert StationQuality to a dictionary suitable for DataFrame construction."""
    return asdict(quality)


class InsufficientDataError(Exception):
    """Exception when there is insufficient data to determine a reference station."""


def select_gps_reference(
    station_to_merged_df: Mapping[str, pd.DataFrame],
    min_coverage_fraction: float = 0.8,
    coherence_priority: bool = True,
) -> str:
    """Pick a reference station when the user doesn't supply one.

    Parameters
    ----------
    station_to_merged_df
        Merged GPS/InSAR tables keyed by station name.
    min_coverage_fraction
        Minimum fraction of epochs required to consider a station.
        This is used to filter out stations with insufficient with InSAR data.
    coherence_priority
        If `True`, prefer highest mean temporal-coherence; otherwise, use RMSE.

    Returns
    -------
    str
        Name of the selected reference station.

    Raises
    ------
    InsufficientDataError
        If no stations have sufficient overlap with InSAR data.

    """
    # Compute quality metrics for all stations
    qualities = {
        station: compute_station_quality(df)
        for station, df in station_to_merged_df.items()
    }

    # Get total time for the InSAR data
    max_insar_time = pd.Series(
        [d.index.max() for d in station_to_merged_df.values()]
    ).max()
    min_insar_time = pd.Series(
        [d.index.min() for d in station_to_merged_df.values()]
    ).min()

    # los_insar
    total_time = max_insar_time - min_insar_time
    total_days = total_time.total_seconds() / (24 * 3600)

    # Filter stations with insufficient overlap
    candidate_stations = {
        station: quality
        for station, quality in qualities.items()
        if quality.num_gps >= (min_coverage_fraction * total_days)
    }

    if not candidate_stations:
        msg = (
            "Could not determine an automatic reference station "
            "(insufficient overlapping data)."
        )
        raise InsufficientDataError(msg)

    # Select best station based on quality metrics
    if coherence_priority:
        # Prefer highest temporal coherence, then lowest RMS misfit.
        # Use explicit None checks: `or` would treat a perfect 0.0 as missing.
        def _key(s: str) -> tuple[float, float]:
            q = qualities[s]
            tcoh = q.temporal_coherence if q.temporal_coherence is not None else -np.inf
            return (tcoh, -q.rms_misfit)

        best_station = max(candidate_stations, key=_key)
    else:
        # Prefer lowest RMS misfit
        best_station = min(candidate_stations, key=lambda s: qualities[s].rms_misfit)

    return best_station

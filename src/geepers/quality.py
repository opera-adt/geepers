"""Station quality metrics and computation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


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

    # Get total number of InSAR epochs across all stations
    # Use the maximum number of unique InSAR epochs from any station
    max_insar_epochs = max(
        df["los_insar"].notna().sum() for df in station_to_merged_df.values()
    )

    # Filter stations with insufficient overlap
    candidate_stations = {
        station: quality
        for station, quality in qualities.items()
        if quality.num_gps >= (min_coverage_fraction * max_insar_epochs)
    }

    if not candidate_stations:
        msg = (
            "Could not determine an automatic reference station "
            "(insufficient overlapping data)."
        )
        raise InsufficientDataError(msg)

    # Select best station based on quality metrics
    if coherence_priority:
        # Prefer highest temporal coherence, then lowest RMS misfit
        best_station = max(
            candidate_stations,
            key=lambda s: (
                qualities[s].temporal_coherence or -np.inf,
                -(qualities[s].rms_misfit or np.inf),
            ),
        )
    else:
        # Prefer lowest RMS misfit
        best_station = min(candidate_stations, key=lambda s: qualities[s].rms_misfit)

    return best_station

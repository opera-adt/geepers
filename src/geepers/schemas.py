"""Pandera schemas for GPS and InSAR data validation.

This module provides DataFrameModel classes to validate data at different stages
of the GPS-InSAR processing pipeline, ensuring consistent column names,
dtypes, units, and allowed ranges.
"""

import pandas as pd
from pandera.pandas import DataFrameModel, Field
from pandera.typing import Index, Series
from pandera.typing.geopandas import GeoSeries as GeoSeriesType

__all__ = [
    "MetadataSchema",
    "RatesSchema",
    "StationObservationSchema",
]

# Avoid zero standard deviations in uncertainty columns
EPS = 1e-9


class StationObservationSchema(DataFrameModel):
    """GNSS E/N/U observations for a single station."""

    date: pd.Timestamp = Field(coerce=True)
    east: Series[float]
    north: Series[float]
    up: Series[float]
    sigma_east: Series[float] = Field(ge=EPS)
    sigma_north: Series[float] = Field(ge=EPS)
    sigma_up: Series[float] = Field(ge=EPS)
    corr_en: Series[float] = Field(ge=-1, le=1)
    corr_eu: Series[float] = Field(ge=-1, le=1)
    corr_nu: Series[float] = Field(ge=-1, le=1)


class MetadataSchema(DataFrameModel):
    """Metadata for a single station."""

    station: Series[str] = Field(str_length={"min_value": 4, "max_value": 4})
    lat: Series[float] = Field(ge=-90, le=90)
    lon: Series[float] = Field(ge=-180, le=180)
    alt: Series[float]
    plate: Series[str] = Field(str_length={"min_value": 2, "max_value": 2})


class RatesSchema(DataFrameModel):
    """GNSS velocity rates comparison data."""

    geometry: GeoSeriesType
    station: Index[str] = Field(str_length={"min_value": 4, "max_value": 4})
    # GPS rates and uncertainties (mm/year)
    gps_velocity: Series[float] = Field(nullable=True)
    # InSAR rates and uncertainties (mm/year)
    insar_velocity: Series[float] = Field(nullable=True)
    difference: Series[float] = Field(nullable=True)
    # Number of GPS measurements used
    num_gps: Series[int] = Field(coerce=True)
    # GPS time span in years
    gps_time_span_years: Series[float]
    # Temporal coherence
    temporal_coherence: Series[float] = Field(ge=0, le=1, nullable=True)
    # Similarity
    similarity: Series[float] = Field(ge=-1, le=1, nullable=True)
    # RMS misfit
    rms_misfit: Series[float] = Field(nullable=True)
    # GPS outlier fraction
    gps_outlier_fraction: Series[float] = Field(nullable=True)
    # GPS velocity scatter
    gps_velocity_scatter: Series[float] = Field(nullable=True)
    # TODO: GPS rate uncertainty (mm/year)
    # gps_velocity_sigma: Series[float] = Field(ge=0, nullable=True)
    # TODO:
    # insar_velocity_sigma: Series[float] = Field(ge=0, nullable=True)
    # TODO: LOS uncertainty (mm/year)
    # sigma_los_mm: Series[float] = Field(ge=0, nullable=True)
    # Difference between GPS and InSAR (mm/year)

"""Pandera schemas for GPS and InSAR data validation.

This module provides DataFrameModel classes to validate data at different stages
of the GPS-InSAR processing pipeline, ensuring consistent column names,
dtypes, units, and allowed ranges.
"""

from __future__ import annotations

import pandas as pd
from pandera.pandas import DataFrameModel, Field

__all__ = [
    "DailyDispModel",
    "MetadataModel",
    "RatesModel",
    "RawObsModel",
]

# Avoid zero standard deviations in uncertainty columns
EPS = 1e-12


# DataFrameModel classes
class RawObsModel(DataFrameModel):
    """Typed schema for raw GPS observation data with uncertainty."""

    date: pd.Timestamp
    east: float
    north: float
    up: float
    sigma_east: float = Field(ge=EPS)
    sigma_north: float = Field(ge=EPS)
    sigma_up: float = Field(ge=EPS)
    corr_en: float = Field(ge=-1, le=1)
    corr_eu: float = Field(ge=-1, le=1)
    corr_nu: float = Field(ge=-1, le=1)

    class Config:
        strict = False


class DailyDispModel(DataFrameModel):
    """Typed schema for daily displacement data."""

    date: pd.Timestamp
    east_mm: float
    north_mm: float
    up_mm: float
    # Optional sigmas after uncertainty propagation
    sigma_east_mm: float | None = Field(ge=EPS, nullable=True)
    sigma_north_mm: float | None = Field(ge=EPS, nullable=True)
    sigma_up_mm: float | None = Field(ge=EPS, nullable=True)
    # Optional correlation coefficients
    corr_en: float | None = Field(ge=-1, le=1, nullable=True)
    corr_eu: float | None = Field(ge=-1, le=1, nullable=True)
    corr_nu: float | None = Field(ge=-1, le=1, nullable=True)

    class Config:
        strict = False


class MetadataModel(DataFrameModel):
    """Typed schema for station metadata."""

    station: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    alt: float  # altitude in meters
    plate: str | None = Field(nullable=True)

    class Config:
        strict = False


class RatesModel(DataFrameModel):
    """Typed schema for velocity rates comparison data."""

    station: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    alt: float
    # GPS rates and uncertainties (mm/year)
    gps_velocity: float | None = Field(nullable=True)
    gps_velocity_l2: float | None = Field(nullable=True)
    gps_velocity_sigma: float | None = Field(ge=0, nullable=True)
    # InSAR rates and uncertainties (mm/year)
    insar_velocity: float | None = Field(nullable=True)
    insar_velocity_l2: float | None = Field(nullable=True)
    insar_velocity_sigma: float | None = Field(ge=0, nullable=True)
    # LOS uncertainty (mm/year)
    sigma_los_mm: float | None = Field(ge=0, nullable=True)
    # Difference between GPS and InSAR (mm/year)
    difference: float | None = Field(nullable=True)

    class Config:
        strict = False

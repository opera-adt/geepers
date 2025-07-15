"""Pandera schemas for GPS and InSAR data validation.

This module provides DataFrameSchemas to validate data at different stages
of the GPS-InSAR processing pipeline, ensuring consistent column names,
dtypes, units, and allowed ranges.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

__all__ = [
    "DailyDispSchema",
    "MetadataSchema",
    "RatesSchema",
    "RawObsSchema",
]

# Avoid zero standard deviations in uncertainty columns
EPS = 1e-12


RawObsSchema = DataFrameSchema(
    {
        "station": Column(str, nullable=False),
        "time": Column("datetime64[ns]", nullable=False),
        "east": Column(float),
        "north": Column(float),
        "up": Column(float),
        "sigma_east": Column(float, checks=[Check.ge(EPS)]),
        "sigma_north": Column(float, checks=[Check.ge(EPS)]),
        "sigma_up": Column(float, checks=[Check.ge(EPS)]),
        "corr_en": Column(float, checks=[Check.ge(-1), Check.le(1)]),
        "corr_eu": Column(float, checks=[Check.ge(-1), Check.le(1)]),
        "corr_nu": Column(float, checks=[Check.ge(-1), Check.le(1)]),
        # Provider-specific quality flags may ride along unvalidated
    },
    strict=False,  # Allow additional provider-specific columns
)


DailyDispSchema = DataFrameSchema(
    {
        "station": Column(str, nullable=False),
        "date": Column("datetime64[ns]", nullable=False),
        "east_mm": Column(float),
        "north_mm": Column(float),
        "up_mm": Column(float),
        # Optional sigmas after uncertainty propagation
        "sigma_east_mm": Column(
            float, checks=[Check.ge(EPS)], nullable=True, required=False
        ),
        "sigma_north_mm": Column(
            float, checks=[Check.ge(EPS)], nullable=True, required=False
        ),
        "sigma_up_mm": Column(
            float, checks=[Check.ge(EPS)], nullable=True, required=False
        ),
        # Optional correlation coefficients
        "corr_en": Column(
            float, checks=[Check.ge(-1), Check.le(1)], nullable=True, required=False
        ),
        "corr_eu": Column(
            float, checks=[Check.ge(-1), Check.le(1)], nullable=True, required=False
        ),
        "corr_nu": Column(
            float, checks=[Check.ge(-1), Check.le(1)], nullable=True, required=False
        ),
    },
    strict=False,  # Allow additional quality metrics
)


MetadataSchema = DataFrameSchema(
    {
        "station": Column(str, nullable=False),
        "lat": Column(float, checks=[Check.ge(-90), Check.le(90)]),
        "lon": Column(float, checks=[Check.ge(-180), Check.le(180)]),
        "alt": Column(float),  # altitude in meters
        "plate": Column(str, nullable=True),  # tectonic plate code
    },
    strict=False,  # Allow additional metadata fields
)


RatesSchema = DataFrameSchema(
    {
        "station": Column(str, nullable=False),
        "lat": Column(float, checks=[Check.ge(-90), Check.le(90)]),
        "lon": Column(float, checks=[Check.ge(-180), Check.le(180)]),
        "alt": Column(float),
        # GPS rates and uncertainties (mm/year)
        "gps_velocity": Column(float, nullable=True),
        "gps_velocity_l2": Column(float, nullable=True),
        "gps_velocity_sigma": Column(float, checks=[Check.ge(0)], nullable=True),
        # InSAR rates and uncertainties (mm/year)
        "insar_velocity": Column(float, nullable=True),
        "insar_velocity_l2": Column(float, nullable=True),
        "insar_velocity_sigma": Column(float, checks=[Check.ge(0)], nullable=True),
        # LOS uncertainty (mm/year)
        "sigma_los_mm": Column(float, checks=[Check.ge(0)], nullable=True),
        # Difference between GPS and InSAR (mm/year)
        "difference": Column(float, nullable=True),
    },
    strict=False,  # Allow additional quality metrics and derived fields
)

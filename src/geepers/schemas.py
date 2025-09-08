"""Pandera schemas for GPS and InSAR data validation.

This module provides DataFrameModel classes to validate data at different stages
of the GPS-InSAR processing pipeline, ensuring consistent column names,
dtypes, units, and allowed ranges.
"""

from enum import StrEnum
from typing import Self

import pandas as pd
from pandera.pandas import DataFrameModel, Field
from pandera.typing import DataFrame, Index, Series
from pandera.typing.geopandas import GeoSeries as GeoSeriesType

__all__ = [
    "GPSUncertaintySchema",
    "GridCellSchema",
    "RatesSchema",
    "StationObservationSchema",
    "StationSchema",
]

# Avoid zero standard deviations in uncertainty columns
EPS = 1e-9


class Plate(StrEnum):
    AF = "AF"  # Africa
    AN = "AN"  # Antarctica
    AR = "AR"  # Arabia
    AU = "AU"  # Australia
    BG = "BG"  # Bering
    BU = "BU"  # Burma
    CA = "CA"  # Caribbean
    CO = "CO"  # Cocos
    EU = "EU"  # Eurasian
    IN = "IN"  # Indian
    MA = "MA"  # Mariana
    NA = "NA"  # North America
    NB = "NB"  # North Bismark
    NZ = "NZ"  # Nazca
    OK = "OK"  # Okhotsk
    ON = "ON"  # Okinawa
    PA = "PA"  # Pacific
    PM = "PM"  # Panama
    PS = "PS"  # Philippine Sea
    SA = "SA"  # South America
    SB = "SB"  # South Bismark
    SC = "SC"  # Scotia
    SL = "SL"  # Shetland
    SO = "SO"  # Somalia
    SU = "SU"  # Sunda
    WL = "WL"  # Woodlark


class PointSchema(DataFrameModel):
    """Metadata for a single station."""

    lat: Series[float] = Field(ge=-90, le=90)
    lon: Series[float] = Field(ge=-180, le=180)
    alt: Series[float]


class StationSchema(PointSchema):
    """Metadata for a single station."""

    id: Series[str] = Field(str_length={"min_value": 4, "max_value": 4})
    plate: Series[pd.StringDtype] = Field(isin=Plate, coerce=True)


class GridCellSchema(PointSchema):
    """Metadata for a single grid cell from UNR Grid."""

    id: Series[int] = Field(gt=0)


class GPSUncertaintySchema(DataFrameModel):
    """GPS uncertainty and correlation information."""

    sigma_east: Series[float] = Field(ge=EPS)
    sigma_north: Series[float] = Field(ge=EPS)
    sigma_up: Series[float] = Field(ge=EPS)
    corr_en: Series[float] = Field(ge=-1, le=1)
    corr_eu: Series[float] = Field(ge=-1, le=1)
    corr_nu: Series[float] = Field(ge=-1, le=1)


class StationObservationSchema(GPSUncertaintySchema):
    """GNSS E/N/U observations (in meters) with uncertainties for a single station."""

    date: pd.Timestamp = Field(coerce=True)
    east: Series[float]
    north: Series[float]
    up: Series[float]

    @classmethod
    def validate(
        cls,
        check_obj: pd.DataFrame,
        head: int | None = None,
        tail: int | None = None,
        sample: int | None = None,
        random_state: int | None = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> DataFrame[Self]:
        out = super().validate(
            check_obj=check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )
        # Document the units of the GPS columns in `.attrs`
        out.attrs["units"] = "meters"
        return out


class RatesSchema(DataFrameModel):
    """GNSS velocity rates comparison data."""

    geometry: GeoSeriesType
    id: Index[str] = Field(str_length={"min_value": 4, "max_value": 4})
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

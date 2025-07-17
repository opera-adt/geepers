"""GPS data handling and downloading functionality.

This module provides functions to download GPS station data, manage station
information, and handle GPS time series data.

.. deprecated::
    This module is deprecated. Use `geepers.gps_sources.unr.UnrSource` instead.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd

from .gps_sources.unr import UnrSource

if TYPE_CHECKING:
    from geepers.io import XarrayReader

    from ._types import PathOrStr

__all__ = [
    "get_stations_within_image",
    "load_station_enu",
    "read_station_llas",
    "station_lonlat",
]

# Create global UNR source instance
_unr_source = UnrSource()

# Deprecation warning message
_DEPRECATION_MSG = (
    "The 'geepers.gps' module is deprecated. "
    "Please use 'geepers.gps_sources.unr.UnrSource' instead."
)


def get_stations_within_image(
    reader: XarrayReader,
    mask_invalid: bool = True,
    bad_vals: Sequence[float] | None = None,
    exclude_stations: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    """Find GPS stations within a given geocoded image.

    .. deprecated::
        This function is deprecated. Use
        `geepers.gps_sources.unr.UnrSource.get_stations_within_image` instead.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    return _unr_source.get_stations_within_image(
        reader=reader,
        mask_invalid=mask_invalid,
        bad_vals=bad_vals,
        exclude_stations=exclude_stations,
    )


def load_station_enu(
    station_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    download_if_missing: bool = True,
    zero_by: str = "mean",
) -> pd.DataFrame:
    """Load GPS station data in the east-north-up (ENU) coordinate system.

    .. deprecated::
        This function is deprecated. Use
        `geepers.gps_sources.unr.UnrSource.load_station_enu` instead.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    return _unr_source.load_station_enu(
        station_name=station_name,
        start_date=start_date,
        end_date=end_date,
        download_if_missing=download_if_missing,
        zero_by=zero_by,
    )


def read_station_llas(
    filename: PathOrStr | None = None,
    to_geodataframe: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Get the station latitude, longitude, and altitude (LLA) data.

    .. deprecated::
        This function is deprecated. Use
        `geepers.gps_sources.unr.UnrSource.read_station_llas` instead.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    return _unr_source.read_station_llas(
        filename=filename,
        to_geodataframe=to_geodataframe,
    )


def station_lonlat(station_name: str) -> tuple[float, float]:
    """Get the longitude and latitude of a GPS station.

    .. deprecated::
        This function is deprecated. Use
        `geepers.gps_sources.unr.UnrSource.station_lonlat` instead.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    return _unr_source.station_lonlat(station_name)

# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Optional-dependency seams for the lean core.

The core of geepers (GNSS retrieval in `gps_sources`) must import with only
pandas, numpy, scipy, pyproj, requests and tqdm installed, because that is
what the operational DISP-CAL image carries. geopandas and pandera are
`[analysis]` extras: when they are present the core behaves exactly as
before (GeoDataFrames out, pandera validation on); when they are absent the
same functions return plain DataFrames and skip validation.

Everything that touches an optional package goes through this module so the
seam is in one place and testable.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger("geepers")

EXTRAS = {
    "geopandas": "analysis",
    "pandera": "analysis",
    "shapely": "grid",
    "xarray": "analysis",
    "rasterio": "analysis",
}


def has_module(name: str) -> bool:
    """Return True if `name` can be imported (without importing it)."""
    return importlib.util.find_spec(name) is not None


def require(name: str) -> Any:
    """Import `name`, or raise an ImportError that names the extra to install."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        extra = EXTRAS.get(name, "all")
        msg = (
            f"{name} is required for this call; install it with "
            f"'pip install geepers[{extra}]'"
        )
        raise ImportError(msg) from e


def validate(df: pd.DataFrame, schema_name: str, **kwargs: Any) -> pd.DataFrame:
    """Validate `df` against `geepers.schemas.<schema_name>` if pandera is installed.

    Without pandera the frame is returned unchanged. The one side effect the
    schemas have besides validation, ``attrs["units"] = "meters"`` on station
    observations, is reproduced so downstream code sees the same frame.
    """
    if has_module("pandera"):
        schemas = importlib.import_module("geepers.schemas")
        return getattr(schemas, schema_name).validate(df, **kwargs)
    if schema_name == "StationObservationSchema":
        df.attrs["units"] = "meters"
    return df


def to_point_frame(
    df: pd.DataFrame, lon: str = "lon", lat: str = "lat", crs: str = "EPSG:4326"
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Attach point geometries if geopandas is installed; else return `df`.

    The lon/lat columns stay in either case, so callers that only need
    coordinates work the same way with both return types.
    """
    if not has_module("geopandas"):
        return df
    gpd = importlib.import_module("geopandas")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs=crs)


def is_geo_frame(df: pd.DataFrame) -> bool:
    """Return True if `df` is a GeoDataFrame, without importing geopandas."""
    return type(df).__name__ == "GeoDataFrame" and has_module("geopandas")

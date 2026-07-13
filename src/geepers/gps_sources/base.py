"""Base class for GPS data sources."""

from __future__ import annotations

import difflib
import logging
import re
import urllib.error
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
import requests
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

from geepers import utils
from geepers._types import PathOrStr
from geepers.schemas import PointSchema

__all__ = ["BaseGpsSource", "validate_station_id"]

logger = logging.getLogger("geepers")

_STATION_ID_PATTERN = re.compile(r"[A-Za-z0-9_]{1,9}")


def validate_station_id(station_id: str) -> str:
    """Check that `station_id` is safe to embed in URLs and cache paths.

    Guards against path traversal (e.g. ``"../../etc"``), since station ids
    are interpolated into both download URLs and local cache filenames.

    Parameters
    ----------
    station_id : str
        The station identifier to check.

    Returns
    -------
    str
        The validated station id, unchanged.

    Raises
    ------
    ValueError
        If the id contains anything but 1-9 alphanumeric/underscore chars.

    """
    if not _STATION_ID_PATTERN.fullmatch(station_id):
        msg = f"Invalid station id: {station_id!r}"
        raise ValueError(msg)
    return station_id


class BaseGpsSource(ABC):
    """Base class for GPS data sources providing standardized interface."""

    def timeseries_many(
        self,
        /,
        ids: Iterable[str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        mask: gpd.GeoSeries | None = None,
        frame: Literal["ENU", "XYZ"] = "ENU",
        start_date: str | None = None,
        end_date: str | None = None,
        zero_by: Literal["mean", "start", "none"] = "mean",
        download_if_missing: bool = True,
        *,
        max_workers: int = 8,
        skip_errors: bool = True,
    ):
        if bbox is None and mask is None and ids is None:
            msg = "Must provide ids, bbox or mask"
            raise ValueError(msg)
        gdf_stations = self.stations(bbox=bbox, mask=mask)
        if bbox is not None or mask is not None or ids is None:
            ids = gdf_stations["id"]
        # Index by id for O(1) metadata lookups in `_load_one`
        station_rows = gdf_stations.set_index("id")

        # Function to load one id
        def _load_one(sid: str) -> pd.DataFrame | None:
            try:
                df = self.timeseries(
                    sid,
                    frame=frame,
                    start_date=start_date,
                    end_date=end_date,
                    zero_by=zero_by,
                    download_if_missing=download_if_missing,
                )
            except (requests.HTTPError, urllib.error.HTTPError) as e:
                # Some ids in the station lists have no data file on the
                # server (e.g. UNR grid points with too little data)
                if not skip_errors:
                    raise
                logger.warning("Skipping %s: %s", sid, e)
                return None
            df.insert(0, "id", sid)  # keep id as a column for melt/pivot
            row = station_rows.loc[sid]
            for col in ("lon", "lat", "alt", "geometry"):
                df[col] = row[col]
            return df

        # (Optional) parallel map
        if max_workers:
            dfs = thread_map(
                _load_one, ids, max_workers=max_workers, desc="Loading GPS data"
            )
        else:
            dfs = [_load_one(sid) for sid in tqdm(ids)]

        n_failed = sum(df is None for df in dfs)
        if n_failed:
            logger.warning("Skipped %d of %d ids with no data", n_failed, len(dfs))
        dfs = [df for df in dfs if df is not None]
        if not dfs:
            msg = "No time series could be loaded"
            raise ValueError(msg)
        big = pd.concat(dfs, ignore_index=True)

        return gpd.GeoDataFrame(big, geometry="geometry", crs="EPSG:4326")

    def __init__(self, cache_dir: PathOrStr | None = None):
        """Initialize the GPS data source.

        Parameters
        ----------
        cache_dir : PathOrStr, optional
            Base directory to store cached data.
            Default is None, which uses `utils.get_cache_dir()`.
            Subclasses create directories under this base directory.

        """
        if cache_dir is None:
            self._base_cache_dir = utils.get_cache_dir()
        else:
            self._base_cache_dir = Path(cache_dir)
        self._subdir = self.__class__.__name__.lower().replace("source", "")

    @property
    def _cache_dir(self) -> Path:
        """Cache directory for this source, created on first access.

        Creation is deferred so that instantiating a source (e.g. at module
        import time) has no filesystem side effects.
        """
        cache_dir = self._base_cache_dir / self._subdir
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    @abstractmethod
    def timeseries(
        self,
        station_id: str,
        /,
        frame: Literal["ENU", "XYZ"] = "ENU",
        start_date: str | None = None,
        end_date: str | None = None,
        zero_by: Literal["mean", "start", "none"] = "mean",
        download_if_missing: bool = True,
    ) -> pd.DataFrame:
        """Load GPS station time series data.

        Parameters
        ----------
        station_id : str
            The station identifier.
        frame : {"ENU", "XYZ"}, optional
            Coordinate frame for the data. Default is "ENU".
        start_date : str, optional
            Start date for data filtering (ISO format).
        end_date : str, optional
            End date for data filtering (ISO format).
        zero_by : Literal["mean", "start"], optional
            How to zero the data. Either "mean" or "start".
        download_if_missing : bool, optional
            Whether to download data if not found locally.

        Returns
        -------
        pd.DataFrame
            DataFrame validated against StationObservationSchema.

        """

    @abstractmethod
    def _read_station_data(self) -> gpd.GeoDataFrame:
        """Read raw station data from the source.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with station metadata including lon, lat, alt columns.

        """

    def stations(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        mask: gpd.GeoSeries | None = None,
    ) -> gpd.GeoDataFrame:
        """Get GPS stations, optionally filtered by spatial bounds.

        Parameters
        ----------
        bbox : tuple[float, float, float, float], optional
            Bounding box as (west, south, east, north) in degrees.
        mask : gpd.GeoSeries, optional
            Spatial mask to filter stations.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with station metadata including lon, lat, alt columns.

        """
        # Read data from source
        gdf = self._read_station_data()

        # Apply spatial filters and validate
        gdf = self._apply_spatial_filters(gdf, bbox, mask)

        return gdf

    def _apply_spatial_filters(
        self,
        gdf: gpd.GeoDataFrame,
        bbox: tuple[float, float, float, float] | None = None,
        mask: gpd.GeoSeries | None = None,
    ) -> gpd.GeoDataFrame:
        """Apply spatial filters to a GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame to filter.
        bbox : tuple[float, float, float, float], optional
            Bounding box as (west, south, east, north) in degrees.
        mask : gpd.GeoSeries, optional
            Spatial mask to filter stations.

        Returns
        -------
        gpd.GeoDataFrame
            Filtered GeoDataFrame.

        """
        # Apply bbox filter (coordinate slicing: much faster than a
        # geometric clip for point layers)
        if bbox is not None:
            west, south, east, north = bbox
            gdf = gdf.cx[west:east, south:north]

        # Apply mask filter
        if mask is not None:
            gdf = gdf[gdf.geometry.within(mask.union_all())]

        # Reset index for cleaner output
        gdf = gdf.reset_index(drop=True)

        # Validate basic point schema
        return PointSchema.validate(gdf, lazy=True)

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Filter DataFrame by date range.

        Accepts tz-aware date strings (e.g. UTC ISO timestamps); the tz is
        dropped so the bound compares against the tz-naive ``date`` column.
        """

        def _naive(value: str) -> pd.Timestamp:
            ts = pd.to_datetime(value)
            return ts.tz_localize(None) if ts.tzinfo is not None else ts

        if start_date:
            df = df[df["date"] >= _naive(start_date)]
        if end_date:
            df = df[df["date"] <= _naive(end_date)]
        return df

    def _zero_data(
        self,
        df: pd.DataFrame,
        zero_by: Literal["mean", "start", "none"] | None = "mean",
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Zero the data in a DataFrame ("none"/None leaves it as-is)."""
        if columns is None:
            columns = ["east", "north", "up"]
        if zero_by is None or zero_by.lower() == "none":
            return df
        if zero_by.lower() == "mean":
            mean_val = df[columns].mean()
            df.loc[:, columns] -= mean_val
        elif zero_by.lower() == "start":
            start_val = df[columns].iloc[:10].mean()
            df.loc[:, columns] -= start_val
        else:
            msg = "zero_by must be 'mean', 'start', or 'none'"
            raise ValueError(msg)
        return df

    def coordinates(self, station_id: str) -> tuple[float, float, float]:
        """Get coordinates for a single station.

        Parameters
        ----------
        station_id : str
            The station identifier.

        Returns
        -------
        tuple[float, float, float]
            Longitude, latitude, and altitude in degrees and meters.

        """
        stations_df = self.stations()
        station_id = station_id.upper()
        if station_id not in stations_df["id"].values:
            closest_names = difflib.get_close_matches(
                station_id, stations_df["id"], n=5
            )
            msg = f"No station named {station_id} found. Closest: {closest_names}"
            raise ValueError(msg)
        row = stations_df[stations_df["id"] == station_id].iloc[0]
        return row["lon"], row["lat"], row["alt"]

    def read_station_llas(
        self,
        to_geodataframe: bool = True,
        filename=None,  # noqa: ARG002
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """Read station location information.

        .. deprecated::
            Use `stations()` instead.
        """
        warnings.warn(
            "read_station_llas is deprecated. Use stations() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.stations()
        if not to_geodataframe:
            # Convert to regular DataFrame, drop geometry
            return pd.DataFrame(result.drop(columns="geometry"))
        return result

    def station_lonlat(self, station_id: str) -> tuple[float, float]:
        """Get longitude and latitude for a station.

        .. deprecated::
            Use `coordinates(station_id)[:2]` instead.
        """
        warnings.warn(
            "station_lonlat is deprecated. Use coordinates(station_id)[:2] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        lon, lat, _ = self.coordinates(station_id)
        return lon, lat

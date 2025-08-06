"""Base class for GPS data sources."""

from __future__ import annotations

import difflib
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

from geepers import utils
from geepers._types import PathOrStr
from geepers.schemas import PointSchema

__all__ = ["BaseGpsSource"]


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
        zero_by: Literal["mean", "start"] = "mean",
        download_if_missing: bool = True,
        *,
        max_workers: int = 8,
    ):
        if bbox is None and mask is None and ids is None:
            msg = "Must provide ids, bbox or mask"
            raise ValueError(msg)
        gdf_stations = self.stations()
        if bbox is not None:
            gdf_stations = self.stations(bbox=bbox)
            ids = gdf_stations["id"]
        elif mask is not None:
            gdf_stations = self.stations(mask=mask)
            ids = gdf_stations["id"]

        if ids is None:
            ids = gdf_stations["id"]

        # Function to load one id
        def _load_one(sid: str) -> pd.DataFrame:
            df = self.timeseries(
                sid,
                frame=frame,
                start_date=start_date,
                end_date=end_date,
                zero_by=zero_by,
                download_if_missing=download_if_missing,
            )
            df.insert(0, "id", sid)  # keep id as a column for melt/pivot
            row = gdf_stations[gdf_stations["id"] == sid]
            for col in ("lon", "lat", "alt", "geometry"):
                df[col] = row.iloc[0][col]
            return df

        # (Optional) parallel map
        if max_workers:
            dfs = thread_map(
                _load_one, ids, max_workers=max_workers, desc="Loading GPS data"
            )
        else:
            dfs = [_load_one(sid) for sid in tqdm(ids)]

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
        subdir = self.__class__.__name__.lower().replace("source", "")
        self._cache_dir = self._base_cache_dir / subdir
        self._cache_dir.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def timeseries(
        self,
        station_id: str,
        /,
        frame: Literal["ENU", "XYZ"] = "ENU",
        start_date: str | None = None,
        end_date: str | None = None,
        zero_by: Literal["mean", "start"] = "mean",
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
        # Apply bbox filter
        if bbox is not None:
            west, south, east, north = bbox
            bounds_poly = box(west, south, east, north)
            gdf = gdf.clip(bounds_poly)

        # Apply mask filter
        if mask is not None:
            gdf = gdf[gdf.geometry.within(mask.unary_union)]

        # Reset index for cleaner output
        gdf.reset_index(drop=True, inplace=True)

        # Validate basic point schema
        PointSchema.validate(gdf, lazy=True)

        return gdf

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        return df

    def _zero_data(
        self,
        df: pd.DataFrame,
        zero_by: Literal["mean", "start"] = "mean",
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Zero the data in a DataFrame."""
        if columns is None:
            columns = ["east", "north", "up"]
        if zero_by.lower() == "mean":
            mean_val = df[columns].mean()
            df.loc[:, columns] -= mean_val
        elif zero_by.lower() == "start":
            start_val = df[columns].iloc[:10].mean()
            df.loc[:, columns] -= start_val
        else:
            msg = "zero_by must be either 'mean' or 'start'"
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

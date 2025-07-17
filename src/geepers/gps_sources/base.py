"""Base class for GPS data sources."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Literal

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from geepers.schemas import PointSchema

__all__ = ["BaseGpsSource"]


class BaseGpsSource(ABC):
    """Base class for GPS data sources providing standardized interface."""

    @abstractmethod
    def timeseries(
        self,
        station_id: str,
        /,
        frame: Literal["ENU", "XYZ"] = "ENU",
        start_date: str | None = None,
        end_date: str | None = None,
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
        if station_id not in stations_df["name"].values:
            import difflib

            closest_names = difflib.get_close_matches(
                station_id, stations_df["name"], n=5
            )
            msg = f"No station named {station_id} found. Closest: {closest_names}"
            raise ValueError(msg)
        row = stations_df[stations_df["name"] == station_id].iloc[0]
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

    def station_lonlat(self, station_name: str) -> tuple[float, float]:
        """Get longitude and latitude for a station.

        .. deprecated::
            Use `coordinates(station_id)[:2]` instead.
        """
        warnings.warn(
            "station_lonlat is deprecated. Use coordinates(station_id)[:2] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        lon, lat, _ = self.coordinates(station_name)
        return lon, lat

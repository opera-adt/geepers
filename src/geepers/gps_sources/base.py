"""Base class for GPS data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
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

    def coordinate(self, station_id: str) -> tuple[float, float, float]:
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

    # Deprecated methods with warnings - remove in next minor release
    def load_station_enu(
        self,
        station_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
        download_if_missing: bool = True,
        zero_by: str = "mean",
    ) -> pd.DataFrame:
        """Load GPS station data in east-north-up coordinates.

        .. deprecated::
            Use `timeseries(station_id, frame="ENU")` instead.
        """
        import warnings

        warnings.warn(
            "load_station_enu is deprecated. Use timeseries(station_id, frame='ENU')"
            " instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.timeseries(
            station_name,
            frame="ENU",
            start_date=start_date,
            end_date=end_date,
            download_if_missing=download_if_missing,
        )

    def get_stations_within_image(
        self,
        reader,
        mask_invalid: bool = True,
        bad_vals: Sequence[float] | None = None,
        exclude_stations: Sequence[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """Find GPS stations within a geocoded image bounds.

        .. deprecated::
            Use `stations(bbox=...)` instead.
        """
        import warnings

        warnings.warn(
            "get_stations_within_image is deprecated. Use stations(bbox=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Convert reader bounds to bbox
        import rasterio.warp

        if reader.crs != "EPSG:4326":
            bounds = rasterio.warp.transform_bounds(
                reader.crs, "EPSG:4326", *reader.da.rio.bounds()
            )
        else:
            bounds = reader.da.rio.bounds()

        result = self.stations(bbox=bounds)

        # Apply additional filters if specified
        if exclude_stations is not None:
            result = result[~result["name"].isin(exclude_stations)]

        return result

    def read_station_llas(
        self,
        to_geodataframe: bool = False,
        **kwargs,
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """Read station location information.

        .. deprecated::
            Use `stations()` instead.
        """
        import warnings

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
            Use `coordinate(station_id)[:2]` instead.
        """
        import warnings

        warnings.warn(
            "station_lonlat is deprecated. Use coordinate(station_id)[:2] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        lon, lat, _ = self.coordinate(station_name)
        return lon, lat

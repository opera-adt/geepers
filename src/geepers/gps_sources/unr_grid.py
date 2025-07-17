"""UNR Grid GPS data source implementation."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

from geepers.schemas import GridCellSchema, StationObservationSchema
from geepers.utils import decimal_year_to_datetime

from .base import BaseGpsSource

if TYPE_CHECKING:
    pass

__all__ = ["UnrGridSource"]

LOOKUP_FILE_URL = "https://geodesy.unr.edu/grid_timeseries/grid_latlon_lookup.txt"
GRID_DATA_BASE_URL = "https://geodesy.unr.edu/grid_timeseries/time_variable_gridded/{plate}/{grid_id:06d}_{plate}.tenv8"
# https://geodesy.unr.edu/grid_timeseries/time_variable_gridded/NA/000007_NA.tenv8
# https://geodesy.unr.edu/grid_timeseries/time_variable_gridded/IGS14/000003_IGS14.tenv8


class UnrGridSource(BaseGpsSource):
    """UNR Grid GPS data source for gridded time series data."""

    def timeseries(
        self,
        station_id: str,
        /,
        frame: Literal["ENU", "XYZ"] = "ENU",
        start_date: str | None = None,
        end_date: str | None = None,
        zero_by: Literal["mean", "start"] = "mean",
        plate: Literal["NA", "PA", "IGS14"] = "IGS14",
        download_if_missing: bool = True,
    ) -> pd.DataFrame:
        """Load grid point time series data.

        Parameters
        ----------
        station_id : str | list[str]
            The grid point identifier(s) (6-digit string).
            If a list is provided, the data for all grid points will be loaded.
        frame : {"ENU", "XYZ"}, optional
            Coordinate frame for the data. Default is "ENU".
        start_date : str, optional
            Start date for data filtering (ISO format).
        end_date : str, optional
            End date for data filtering (ISO format).
        zero_by : Literal["mean", "start"], optional
            How to zero the data. Either "mean" or "start".
        plate : Literal["NA", "PA", "IGS14"], optional
            Plate for the data. Default is "IGS14".
        download_if_missing : bool, optional
            Whether to download data if not found locally.

        Returns
        -------
        pd.DataFrame
            DataFrame with ENU time series data validated against schema.

        Raises
        ------
        NotImplementedError
            Grid point ENU loading not yet implemented.

        """
        if frame == "XYZ":
            msg = "XYZ frame not supported for grid data"
            raise ValueError(msg)

        url = GRID_DATA_BASE_URL.format(plate=plate, grid_id=station_id)
        df = self.parse_data_file(url)
        df = self._zero_data(df, zero_by)
        return StationObservationSchema.validate(df, lazy=True)

    def _read_station_data(self) -> gpd.GeoDataFrame:
        """Read raw grid point data from the source.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with grid point metadata including lon, lat, alt columns.

        """
        df = self._read_grid_file()

        # Rename columns to match expected format
        df_out = df.reset_index()
        df_out = df_out.rename(
            columns={"grid_point": "id", "latitude": "lat", "longitude": "lon"}
        )
        df_out["alt"] = 0.0  # Grid points don't have altitude info

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df_out,
            geometry=gpd.points_from_xy(df_out.lon, df_out.lat),
            crs="EPSG:4326",
        )

        return gdf

    def stations(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        mask: gpd.GeoSeries | None = None,
    ) -> gpd.GeoDataFrame:
        """Get grid points, optionally filtered by spatial bounds.

        Parameters
        ----------
        bbox : tuple[float, float, float, float], optional
            Bounding box as (west, south, east, north) in degrees.
        mask : gpd.GeoSeries, optional
            Spatial mask to filter grid points.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with grid point metadata including lon, lat, alt columns.

        """
        # Get data using base class method
        gdf = super().stations(bbox, mask)

        # Apply grid-specific schema validation
        GridCellSchema.validate(gdf, lazy=True)

        return gdf

    def get_grid_geometry(self) -> gpd.GeoSeries:
        """Get the grid geometry.

        Returns
        -------
        gpd.GeoSeries
            GeoSeries with Point geometries for each grid cell.

        """
        df = self._read_grid_file()
        return gpd.GeoSeries.from_xy(df.longitude, df.latitude, crs="EPSG:4326")

    def list_remote_data_files(self) -> list[str]:
        """Retrieve available .tenv8 filenames from the UNR grid data directory.

        Returns
        -------
        list[str]
            Filenames matching the pattern '######_IGS14.tenv8'.

        """
        response = requests.get(GRID_DATA_BASE_URL)
        response.raise_for_status()

        # Extract .tenv8 filenames from HTML
        pattern = r"(\d{6}_IGS14\.tenv8)"
        matches = re.findall(pattern, response.text)
        return sorted(set(matches))

    def download_data_files(
        self,
        output_dir: Path,
        file_list: list[str] | None = None,
        max_workers: int = 8,
    ) -> None:
        """Download .tenv8 data files in parallel, showing progress.

        Parameters
        ----------
        output_dir : Path
            Directory to store downloaded data files.
        file_list : list[str], optional
            Specific filenames to download. If None, files are listed remotely.
        max_workers : int, optional
            Number of threads to use for downloading in parallel.

        """
        output_dir.mkdir(parents=True, exist_ok=True)
        if file_list is None:
            file_list = self.list_remote_data_files()

        def _download(fname: str) -> None:
            url = f"{GRID_DATA_BASE_URL}{fname}"
            dest = output_dir / fname
            if not dest.exists():
                resp = requests.get(url, stream=True)
                resp.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_download, fn): fn for fn in file_list}
            for _ in tqdm(
                as_completed(futures), total=len(futures), desc="Downloading data files"
            ):
                pass

    @staticmethod
    @lru_cache(maxsize=128)
    def _read_data_file(uri: str | Path) -> pd.DataFrame:
        df = pd.read_csv(
            uri,
            delim_whitespace=True,
            header=None,
            names=[
                "decimal_year",
                "east",
                "north",
                "up",
                "sigma_east",
                "sigma_north",
                "sigma_up",
                "rapid_flag",
            ],
        )
        return df

    def parse_data_file(self, uri: str | Path) -> pd.DataFrame:
        """Parse a .tenv8 time-series data file into a DataFrame.

        Parameters
        ----------
        uri : str | Path
            Path or URL to the .tenv8 file.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns validated against GPSUncertaintySchema.

        """
        df = self._read_data_file(uri)
        # Convert decimal year to datetime
        df["date"] = df["decimal_year"].apply(decimal_year_to_datetime)

        # Add placeholder correlation values (not in .tenv8 format)
        df["corr_en"] = 0.0
        df["corr_eu"] = 0.0
        df["corr_nu"] = 0.0

        # Select relevant columns for validation
        df_out = df[
            [
                "date",
                "east",
                "north",
                "up",
                "sigma_east",
                "sigma_north",
                "sigma_up",
                "corr_en",
                "corr_eu",
                "corr_nu",
            ]
        ]

        StationObservationSchema.validate(df_out, lazy=True)

        return df_out

    @staticmethod
    @lru_cache(maxsize=1)
    def _read_grid_file() -> pd.DataFrame:
        """Download and cache the UNR grid latitude/longitude lookup table."""
        df = pd.read_csv(
            LOOKUP_FILE_URL,
            delim_whitespace=True,
            names=["grid_point", "longitude", "latitude"],
        )
        return df.set_index("grid_point")


# Create instance for backward compatibility
_unr_grid_source = UnrGridSource()

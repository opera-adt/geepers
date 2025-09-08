"""UNR Grid GPS data source implementation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm.contrib.concurrent import thread_map

from geepers.schemas import GridCellSchema, StationObservationSchema
from geepers.utils import decimal_year_to_datetime

from .base import BaseGpsSource

if TYPE_CHECKING:
    pass

__all__ = ["UnrGridSource"]

LOOKUP_FILE_URL = "https://geodesy.unr.edu/grid_timeseries/grid_latlon_lookup.txt"
FILENAME_TEMPLATE = "{plate}/{grid_id:06d}_{plate}.tenv8"
GRID_DATA_BASE_URL = (
    f"https://geodesy.unr.edu/grid_timeseries/time_variable_gridded/{FILENAME_TEMPLATE}"
)
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
        download_if_missing: bool = True,
        *,
        plate: Literal["NA", "PA", "IGS14", "IGS20"] = "IGS14",
    ) -> pd.DataFrame:
        """Load grid point time series data.

        Parameters
        ----------
        station_id : str
            The grid point identifier (6-digit string).
        frame : {"ENU", "XYZ"}, optional
            Coordinate frame for the data. Default is "ENU".
        start_date : str, optional
            Start date for data filtering (ISO format). Currently not implemented.
        end_date : str, optional
            End date for data filtering (ISO format). Currently not implemented.
        zero_by : Literal["mean", "start"], optional
            How to zero the data. Either "mean" or "start".
        download_if_missing : bool, optional
            Whether to download data if not found locally. Currently not implemented.
        plate : Literal["NA", "PA", "IGS14", "IGS20"], optional
            Plate for the data. Default is "IGS14".

        Returns
        -------
        pd.DataFrame
            DataFrame with ENU time series data validated against schema.

        Raises
        ------
        ValueError
            If XYZ frame is requested (not supported for grid data).

        """
        if frame == "XYZ":
            msg = "XYZ frame not supported for grid data"
            raise ValueError(msg)

        # TODO: how to handle fetching/saving, vs using pandas to read...
        if download_if_missing:
            local_file = self._download_file(station_id, plate=plate)
            df = self.parse_data_file(local_file)
        else:
            uri = GRID_DATA_BASE_URL.format(plate=plate, grid_id=station_id)
            df = self.parse_data_file(uri)

        df = self._filter_by_date(df, start_date, end_date)
        df = self._zero_data(df, zero_by, columns=["east", "north", "up"])
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

    def _download_file(
        self,
        grid_id: str,
        plate: Literal["NA", "PA", "IGS14", "IGS20"] = "IGS14",
        output_dir: Path | None = None,
        session: requests.Session | None = None,
    ) -> Path:
        """Download ont .tenv8 data file.

        Parameters
        ----------
        grid_id: str
            Grid point ID to download.
        plate : Literal["NA", "PA", "IGS14", "IGS20], optional
            Plate for the data. Default is "IGS14".
        output_dir : Path | None, optional
            Directory to store downloaded data files.
            If None, the cache directory is used.
        session : requests.Session
            A shared requests.Session object.
            Can be used for retrying.

        Returns
        -------
        Path
            Paths to downloaded data files.

        """
        if output_dir is None:
            output_dir = self._cache_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        url = GRID_DATA_BASE_URL.format(plate=plate, grid_id=grid_id)
        dest = output_dir / url.rsplit("/", 1)[-1]
        if not dest.exists():
            if session is None:
                resp = requests.get(url)
            else:
                resp = session.get(url)
            resp.raise_for_status()
            with dest.open("wb") as f:
                f.write(resp.content)
        return dest

    def download_data_files(
        self,
        grid_id_list: list[str] | None = None,
        plate: Literal["NA", "PA", "IGS14", "IGS20"] = "IGS14",
        max_workers: int = 8,
        output_dir: Path | None = None,
    ) -> list[Path]:
        """Download .tenv8 data files in parallel, showing progress.

        Parameters
        ----------
        grid_id_list : list[str], optional
            Specific grid point IDs to download.
            If None, all grid points are downloaded.
        plate : Literal["NA", "PA", "IGS14", "IGS20"], optional
            Plate for the data. Default is "IGS14".
        max_workers : int, optional
            Number of threads to use for downloading in parallel.
        output_dir : Path | None, optional
            Directory to store downloaded data files.
            If None, the cache directory is used.

        Returns
        -------
        list[Path]
            List of paths to downloaded data files.

        """
        if output_dir is None:
            output_dir = self._cache_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        if grid_id_list is None:
            grid_id_list = self.stations().index.tolist()

        s = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=retries))

        return thread_map(
            self._download_file,
            grid_id_list,
            plate=plate,
            output_dir=output_dir,
            max_workers=max_workers,
            session=s,
            desc="Downloading data files",
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _read_data_file(uri: str | Path) -> pd.DataFrame:
        df = pd.read_csv(
            uri,
            sep=r"\s+",
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
        # UNR Grid is in millimeters instead of meters:
        df_out.loc[:, ["east", "north", "up"]] /= 1000

        StationObservationSchema.validate(df_out, lazy=True, inplace=True)

        return df_out

    @staticmethod
    @lru_cache(maxsize=1)
    def _read_grid_file() -> pd.DataFrame:
        """Download and cache the UNR grid latitude/longitude lookup table."""
        df = pd.read_csv(
            LOOKUP_FILE_URL,
            sep=r"\s+",
            names=["grid_point", "longitude", "latitude"],
        )
        return df.set_index("grid_point")


# Create instance for backward compatibility
_unr_grid_source = UnrGridSource()

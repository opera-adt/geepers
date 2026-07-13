"""UNR Grid GPS data source implementation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import geopandas as gpd
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm.contrib.concurrent import thread_map

from geepers.schemas import EPS, GridCellSchema, StationObservationSchema
from geepers.utils import decimal_years_to_datetimes

from .base import BaseGpsSource
from .unr import REQUEST_TIMEOUT

if TYPE_CHECKING:
    pass

__all__ = ["UnrGridSource"]

VALID_VERSIONS = {"0.1", "0.3"}
DEFAULT_VERSION = "0.3"
# Grid points are only available for time-variable positions in version 0.1.
# Note: "time_contsant_gridded" is the actual (misspelled) directory name on
# the UNR server, not a typo introduced here.
GRIDDED_TYPE_DIRS = {
    "constant": "time_contsant_gridded",
    "variable": "time_variable_gridded",
}
LOOKUP_FILE_URL = "https://geodesy.unr.edu/grid_timeseries/Version{version}/grid_latlon_lookup.txt"
FILENAME_TEMPLATE = "{plate}/{grid_id:06d}_{plate}.tenv8"
GRID_DATA_BASE_URL = (
    "https://geodesy.unr.edu/grid_timeseries/Version{version}/"
    "{gridded_dir}/{filename}"
)
# https://geodesy.unr.edu/grid_timeseries/Version0.3/time_variable_gridded/NA/000007_NA.tenv8
# https://geodesy.unr.edu/grid_timeseries/Version0.3/time_contsant_gridded/IGS20/000003_IGS20.tenv8


class UnrGridSource(BaseGpsSource):
    """UNR Grid GPS data source for gridded time series data."""
    def __init__(self,
                 version: Literal["0.1", "0.3"] = DEFAULT_VERSION,
                 gridded_type: Literal["constant", "variable"] = "variable",
                 cache_dir: Optional[str | Path] = None,):
        """Initialize UNR Grid GPS data source.

        Parameters
        ----------
        version : Literal["0.1", "0.3"], optional
            Version of the UNR grid data to use. Default is "0.3".
        gridded_type : Literal["constant", "variable"], optional
            Whether to use the time-constant or time-variable gridded
            products. Only available starting with version "0.3";
            version "0.1" only has time-variable data. Default is "variable".
        cache_dir : str | Path, optional
            Directory to cache downloaded data files.

        """
        # Initialize BaseGpsSource
        super().__init__(cache_dir=cache_dir)

        # Store UNR version
        self.version = version
        self.gridded_type = gridded_type

    def timeseries(
        self,
        station_id: str,
        /,
        frame: Literal["ENU", "XYZ"] = "ENU",
        start_date: str | None = None,
        end_date: str | None = None,
        zero_by: Literal["mean", "start", "none"] = "mean",
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
            local_file = self._download_file(station_id, plate=plate,
                                             version=self.version,
                                             gridded_type=self.gridded_type)
            df = self.parse_data_file(local_file)
        else:
            filename = FILENAME_TEMPLATE.format(plate=plate, grid_id=int(station_id))
            gridded_dir = GRIDDED_TYPE_DIRS[
                self.gridded_type if self.version != "0.1" else "variable"
            ]
            uri = GRID_DATA_BASE_URL.format(
                version=self.version, gridded_dir=gridded_dir, filename=filename
            )
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
        df = self._read_grid_file(version=self.version)

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
        version: Literal["0.1", "0.3"] = DEFAULT_VERSION,
        gridded_type: Literal["constant", "variable"] = "variable",
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
        version : Literal["0.1", "0.3"], optional
            Version of the UNR grid data to download.
        gridded_type : Literal["constant", "variable"], optional
            Whether to download the time-constant or time-variable
            gridded product. Version "0.1" only has time-variable data,
            so this is ignored when version is "0.1".

        Returns
        -------
        Path
            Paths to downloaded data files.

        """
        if output_dir is None:
            output_dir = self._cache_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if plate == "IGS14" and version == "0.3":
            # Warning: IGS14 plate is not available in UNR version 0.3.
            # Using IGS20 instead.
            plate = "IGS20"

        gridded_dir = GRIDDED_TYPE_DIRS[
            gridded_type if version != "0.1" else "variable"
        ]
        # Accept both int and zero-padded string ids ("000123" or 123)
        filename = FILENAME_TEMPLATE.format(plate=plate, grid_id=int(grid_id))
        url = GRID_DATA_BASE_URL.format(
            version=version,
            gridded_dir=gridded_dir,
            filename=filename
        )
        dest = output_dir / url.rsplit("/", 1)[-1]
        if not dest.exists():
            if session is None:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            else:
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
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
        version: Literal["0.1", "0.3"] = DEFAULT_VERSION,
        gridded_type: Literal["constant", "variable"] = "variable",
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
        version : Literal["0.1", "0.3"], optional
            Version of the UNR grid data to download. Default is "0.3".
        gridded_type : Literal["constant", "variable"], optional
            Whether to download the time-constant or time-variable
            gridded product. Ignored when version is "0.1", since that
            version only has time-variable data.

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
            version=version,
            gridded_type=gridded_type,
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
        # Copy so we never mutate the lru_cached DataFrame
        df = self._read_data_file(uri).copy()

        # Convert decimal year to datetime (vectorized)
        df["date"] = decimal_years_to_datetimes(df["decimal_year"])

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
        # UNR Grid is in millimeters instead of meters (values and sigmas):
        sigma_cols = ["sigma_east", "sigma_north", "sigma_up"]
        df_out.loc[:, ["east", "north", "up", *sigma_cols]] /= 1000

        # Time-constant gridded products can report exactly-zero sigmas,
        # which the schema rejects; clamp to its minimum
        df_out.loc[:, sigma_cols] = df_out[sigma_cols].clip(lower=EPS)

        return StationObservationSchema.validate(df_out, lazy=True)

    @staticmethod
    @lru_cache(maxsize=None)
    def _read_grid_file(version: str = DEFAULT_VERSION) -> pd.DataFrame:
        """Download and cache the UNR grid latitude/longitude lookup table."""
        if version not in VALID_VERSIONS:
            msg = (
                f"Unsupported version '{version}'. "
                f"Valid options are: {', '.join(VALID_VERSIONS)}."
                )
            raise ValueError(msg)

        url = LOOKUP_FILE_URL.format(version=version)
        df = pd.read_csv(
            url,
            sep=r"\s+",
            names=["grid_point", "longitude", "latitude"],
        )
        if version == "0.3":
            # Version 0.3 maps longitudes from 0 to 360
            # convert to -180 to 180, to be consistent with version 0.1
            df['longitude'] = ((df['longitude'] + 180) % 360) - 180
        return df.set_index("grid_point")


# Create instance for backward compatibility
_unr_grid_source = UnrGridSource()

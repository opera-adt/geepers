"""JPL Sideshow GPS data source implementation."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal

import geopandas as gpd
import numpy as np
import pandas as pd

from geepers.schemas import StationObservationSchema
from geepers.utils import decimal_year_to_datetime, get_cache_dir

from .base import BaseGpsSource

if TYPE_CHECKING:
    pass

__all__ = ["SideshowSource"]

# JPL Sideshow constants
SITE_LIST_URL = "https://sideshow.jpl.nasa.gov/post/tables/table2.html"
SITE_LIST_XYZ_URL = "https://sideshow.jpl.nasa.gov/post/tables/table1.html"
STATION_URL_BASE = (
    "https://sideshow.jpl.nasa.gov/pub/JPL_GPS_Timeseries/repro2018a/post/point/"
)
# e.g.
# https://sideshow.jpl.nasa.gov/pub/JPL_GPS_Timeseries/repro2018a/post/point/AB01.series
GPS_BASE_URL = f"{STATION_URL_BASE}{{station}}.series"
GPS_DIR = get_cache_dir() / "sideshow"
GPS_DIR.mkdir(exist_ok=True, parents=True)
STEPS_URL = "https://sideshow.jpl.nasa.gov/post/tables/table3.html"

logger = logging.getLogger("geepers")


class SideshowSource(BaseGpsSource):
    """JPL Sideshow GPS data source."""

    _names: Final[list[str]] = [
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

    def timeseries(
        self,
        station_id: str,
        /,
        frame: Literal["ENU", "XYZ"] = "ENU",
        start_date: str | None = None,
        end_date: str | None = None,
        zero_by: Literal["mean", "start"] = "mean",
        download_if_missing: bool = True,  # noqa: ARG002
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
        zero_by : str, optional
            How to zero the data. Either "mean" or "start".
        download_if_missing : bool, optional
            Whether to download data if not found locally.

        Returns
        -------
        pd.DataFrame
            DataFrame with ENU time series data validated against schema.

        """
        if frame == "XYZ":
            msg = "XYZ frame not supported for Sideshow data"
            raise ValueError(msg)

        df = self._read_series(station_id)
        # Replace decimal year with datetime
        df["date"] = df["decimal_year"].apply(decimal_year_to_datetime)
        df = df.drop(columns=["decimal_year"])
        # Move date to first column:
        df = df[["date", *df.columns[:-1].to_list()]]
        df = self._filter_by_date(df, start_date, end_date)
        df = self._zero_data(df, zero_by, columns=["east", "north", "up"])
        return StationObservationSchema.validate(df, lazy=True)

    @staticmethod
    @lru_cache(maxsize=1)
    def _read_series(station_id: str) -> pd.DataFrame:
        _raw_names = ["decimal_year"] + SideshowSource._names[1:]
        # https://sideshow.jpl.nasa.gov/post/tables/GNSS_Time_Series.pdf
        # Time Series and Residual Format
        # Column 1: Decimal_YR
        # Columns 2-4: East(m) North(m) Vert(m)
        # Columns 5-7: E_sig(m) N_sig(m) V_sig(m)
        # Columns 8-10: E_N_cor E_V_cor N_V_cor
        # Column 11: Time in Seconds past J2000
        # Columns 12-17: Time in YEAR MM DD HR MN SS
        series_url = GPS_BASE_URL.format(station=station_id)
        return pd.read_csv(
            series_url,
            sep=r"\s+",
            engine="c",
            header=None,
            names=_raw_names,
            usecols=list(range(len(_raw_names))),
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def _fetch_station_data() -> gpd.GeoDataFrame:
        """Download and cache the JPL Sideshow site list."""
        import warnings

        with warnings.catch_warnings(category=UserWarning, action="ignore"):
            return np.loadtxt(SITE_LIST_URL, comments="<", skiprows=9, dtype=str)

    def _read_station_data(self) -> gpd.GeoDataFrame:
        lines = self._fetch_station_data()
        stations = {
            "id": lines[::2, 0],
            "lat": lines[::2, 2].astype(float),
            "lon": lines[::2, 3].astype(float),
            "alt": lines[::2, 4].astype(float) / 1000,  # JPL height is in millimeters
            # next three columns are sigmas for the site position
        }
        df = pd.DataFrame(stations)
        df.loc[:, "lon"] = df.lon - (np.round(df.lon / 360) * 360)

        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
        )


# Create instance for backward compatibility
_sideshow_source = SideshowSource()

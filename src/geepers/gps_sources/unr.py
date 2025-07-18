"""UNR GPS data source implementation."""

from __future__ import annotations

import datetime
import logging
import re
from functools import cache
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from geepers import utils
from geepers._types import PathOrStr
from geepers.schemas import StationObservationSchema

from .base import BaseGpsSource

__all__ = ["UnrSource"]

# Constants
GPS_BASE_URL = "https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{station}.tenv3"
GPS_DIR = utils.get_cache_dir() / "unr"
GPS_DIR.mkdir(exist_ok=True, parents=True)
STATION_LLH_URL = "https://geodesy.unr.edu/NGLStationPages/llh.out"
STATION_LLH_FILE = str(GPS_DIR / "station_llh_all_{today}.csv")

logger = logging.getLogger("geepers")


class UnrSource(BaseGpsSource):
    """UNR GPS data source."""

    def timeseries(
        self,
        station_id: str,
        /,
        frame: Literal["ENU", "XYZ"] = "ENU",
        start_date: str | None = None,
        end_date: str | None = None,
        zero_by: Literal["mean", "start"] = "mean",
        download_if_missing: bool = True,
        plate_fixed: bool = False,
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
        zero_by : Literal["mean", "start"], optional
            How to zero the data. Either "mean" or "start".
        plate_fixed : bool, optional
            Whether to use plate-fixed coordinates.

        Returns
        -------
        pd.DataFrame
            DataFrame validated against StationObservationSchema.

        """
        if frame not in ["ENU", "XYZ"]:
            msg = f"Unsupported frame: {frame}. Use 'ENU' or 'XYZ'"
            raise ValueError(msg)

        station_id = station_id.upper()

        if plate_fixed and frame == "ENU":
            plate = self._get_station_plate(station_id)
            gps_data_file = GPS_DIR / f"{station_id}_{plate}.tenv3"
        else:
            if frame == "ENU":
                gps_data_file = GPS_DIR / f"{station_id}.tenv3"
            else:  # frame in ("XYZ")
                gps_data_file = GPS_DIR / f"{station_id}.txyz2"

        if not gps_data_file.exists():
            if download_if_missing:
                logger.info(f"Downloading {station_id} to {gps_data_file}")
                self.download_station_data(
                    station_id, frame=frame, plate_fixed=plate_fixed
                )
            else:
                msg = f"{gps_data_file} does not exist, download_if_missing = False"
                raise ValueError(msg)

        df = pd.read_csv(gps_data_file, sep=r"\s+", engine="c")
        df = self._clean_gps_df(
            df, start_date, end_date, coords="enu" if frame == "ENU" else "xyz"
        )

        if frame == "ENU" and zero_by:
            if zero_by.lower() == "mean":
                mean_val = df[["east", "north", "up"]].mean()
                df[["east", "north", "up"]] -= mean_val
            elif zero_by.lower() == "start":
                start_val = df[["east", "north", "up"]].iloc[:10].mean()
                df[["east", "north", "up"]] -= start_val
            else:
                msg = "zero_by must be either 'mean' or 'start'"
                raise ValueError(msg)

        if frame == "ENU":
            StationObservationSchema.validate(df, lazy=True)

        return df

    def _read_station_data(self) -> gpd.GeoDataFrame:
        """Read raw station data from the source.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with station metadata including lon, lat, alt columns.

        """
        today = datetime.date.today().strftime("%Y%m%d")
        filename = STATION_LLH_FILE.format(today=today)
        lla_path = Path(filename)

        try:
            df = pd.read_csv(lla_path, sep=r"\s+", engine="c", header=None)
        except FileNotFoundError:
            logger.info(f"Downloading from {STATION_LLH_URL} to {lla_path}")
            self._download_station_locations(lla_path, STATION_LLH_URL)
            df = pd.read_csv(lla_path, sep=r"\s+", engine="c", header=None)

        processed_stations = self.get_global_station_list()
        df = df[df[0].isin(processed_stations)]
        df.columns = ["id", "lat", "lon", "alt"]
        df.loc[:, "lon"] = df.lon - (np.round(df.lon / 360) * 360)

        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
        )

    def download_station_data(
        self,
        station_id: str,
        frame: Literal["ENU", "XYZ"] = "ENU",
        plate_fixed: bool = False,
    ) -> None:
        """Download GPS station data from the Nevada Geodetic Laboratory.

        Parameters
        ----------
        station_id : str
            The station identifier.
        frame : {"ENU", "XYZ"}, optional
            The coordinate system of the data to download. Default is "ENU".
        plate_fixed : bool, optional
            Whether to download plate-fixed data. Only applicable for "ENU" frame.

        """
        station_id = station_id.upper()

        if frame == "ENU":
            if plate_fixed:
                plate = self._get_station_plate(station_id)
                url = f"https://geodesy.unr.edu/gps_timeseries/tenv3/plates/{plate}/{station_id}.{plate}.tenv3"
                filename = GPS_DIR / f"{station_id}_{plate}.tenv3"
            else:
                url = GPS_BASE_URL.format(station=station_id)
                filename = GPS_DIR / f"{station_id}.tenv3"
        elif frame == "XYZ":
            url = (
                f"https://geodesy.unr.edu/gps_timeseries/txyz/IGS14/{station_id}.txyz2"
            )
            filename = GPS_DIR / f"{station_id}.txyz2"
        else:
            msg = "frame must be 'ENU' or 'XYZ'"
            raise ValueError(msg)

        response = requests.get(url)
        response.raise_for_status()

        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(response.text)
        logger.info(f"Saved {url} to {filename}")

    def _get_station_plate(self, station_id: str) -> str:
        """Get the tectonic plate for a given GPS station."""
        url = f"https://geodesy.unr.edu/NGLStationPages/stations/{station_id}.sta"
        response = requests.get(url)
        response.raise_for_status()

        match = re.search(r"tenv3\/plates\/(?P<plate>[A-Z]{2})", response.text)
        if not match:
            msg = f"Could not find plate name on {url}"
            raise ValueError(msg)
        return match.group("plate")

    def _clean_gps_df(
        self,
        df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
        coords: str = "enu",
    ) -> pd.DataFrame:
        """Clean and preprocess the GPS DataFrame."""
        df["date"] = pd.to_datetime(df["YYMMMDD"], format="%y%b%d")

        df = self._filter_by_date(df, start_date, end_date)

        if coords == "enu":
            df_integer = df[["_e0(m)", "____n0(m)", "u0(m)"]]
            df_out = df[
                [
                    "date",
                    "__east(m)",
                    "_north(m)",
                    "____up(m)",
                    "sig_e(m)",
                    "sig_n(m)",
                    "sig_u(m)",
                    "__corr_en",
                    "__corr_eu",
                    "__corr_nu",
                ]
            ]
            # Combine the integer e/n/u part with the fractional
            df_out.loc[:, ["__east(m)", "_north(m)", "____up(m)"]] += df_integer.values
        elif coords == "xyz":
            df_out = df[["date", "x", "y", "z"]]
        else:
            msg = "coords must be either 'enu' or 'xyz'"
            raise ValueError(msg)

        df_out = df_out.rename(columns=lambda s: s.replace("_", "").replace("(m)", ""))
        df_out = df_out.rename(
            columns={
                "sige": "sigma_east",
                "sign": "sigma_north",
                "sigu": "sigma_up",
                "corren": "corr_en",
                "correu": "corr_eu",
                "corrnu": "corr_nu",
            }
        )
        return df_out.reset_index(drop=True)

    def _download_station_locations(self, filename: PathOrStr, url: str) -> None:
        """Download the station location file from the Nevada Geodetic Laboratory."""
        resp = requests.get(url)
        resp.raise_for_status()

        with open(filename, "w") as f:
            f.write(resp.text)

    def get_global_station_list(self) -> list[str]:
        """Get the list of "processed" stations from UNR.

        Source: https://geodesy.unr.edu/NGLStationPages/GlobalStationList

        Note that this may be smaller than the lat/lon/alt list at
        https://geodesy.unr.edu/NGLStationPages/llh.out.
        """
        return self._read_global_station_list().values.ravel().tolist()

    @staticmethod
    @cache
    def _read_global_station_list() -> pd.DataFrame:
        """Read the global station list from UNR."""
        return pd.read_html(
            "https://geodesy.unr.edu/NGLStationPages/GlobalStationList"
        )[0]

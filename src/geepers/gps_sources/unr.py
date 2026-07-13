"""UNR GPS data source implementation."""

from __future__ import annotations

import datetime
import logging
from functools import cache
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from geepers._types import PathOrStr
from geepers.schemas import StationObservationSchema

from .base import BaseGpsSource, validate_station_id

__all__ = ["UnrSource"]

# Constants
GPS_BASE_URL = (
    # URLS dont match!!
    # Old one was
    # https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/TXKM.tenv3
    # New IGS20 is
    # https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/LAVR.tenv3
    "https://geodesy.unr.edu/gps_timeseries/{reference}/tenv3/{reference}/{station}.tenv3"
)
STATION_LLH_URL = "https://geodesy.unr.edu/NGLStationPages/llh.out"
STATION_LLH_FILENAME = "station_llh_all_{today}.csv"
STEPS_URL = "https://geodesy.unr.edu/NGLStationPages/steps.txt"
# Seconds before an HTTP download is abandoned (connect, read)
REQUEST_TIMEOUT = (10, 120)

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
        plate_name: str | None = None,
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
        plate_name : str, optional
            If `plate_fixed=True`, specify which plate to use.
            Some stations have multiple plates.
            If None, uses the first plate found.

        Returns
        -------
        pd.DataFrame
            DataFrame validated against StationObservationSchema.

        """
        if frame not in ["ENU", "XYZ"]:
            msg = f"Unsupported frame: {frame}. Use 'ENU' or 'XYZ'"
            raise ValueError(msg)

        station_id = validate_station_id(station_id.upper())

        plate = None
        if plate_fixed and frame == "ENU":
            plates = self._get_station_plates(station_id)
            if plate_name:
                if plate_name not in plates:
                    msg = (
                        f"Plate {plate_name} not found for {station_id}, which has"
                        f" plates {plates}"
                    )
                    raise ValueError(msg)
                plate = plate_name
            else:
                plate = plates[0]
            gps_data_file = self._cache_dir / plate / f"{station_id}.tenv3"
        else:
            if frame == "ENU":
                gps_data_file = self._cache_dir / f"{station_id}.tenv3"
            else:  # frame in ("XYZ")
                gps_data_file = self._cache_dir / f"{station_id}.txyz2"

        if not gps_data_file.exists():
            if download_if_missing:
                logger.info(f"Downloading {station_id} to {gps_data_file}")
                self.download_station_data(
                    station_id, frame=frame, plate_fixed=plate_fixed, plate=plate
                )
            else:
                msg = f"{gps_data_file} does not exist, download_if_missing = False"
                raise ValueError(msg)

        df = pd.read_csv(gps_data_file, sep=r"\s+", engine="c")
        df = self._clean_gps_df(
            df, start_date, end_date, coords="enu" if frame == "ENU" else "xyz"
        )

        if frame == "ENU" and zero_by:
            df = self._zero_data(df, zero_by, columns=["east", "north", "up"])

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
        lla_path = self._cache_dir / STATION_LLH_FILENAME.format(today=today)

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
        reference: Literal["IGS14", "IGS20"] = "IGS20",
        plate_fixed: bool = False,
        plate: str | None = None,
    ) -> None:
        """Download GPS station data from the Nevada Geodetic Laboratory.

        Parameters
        ----------
        station_id : str
            The station identifier.
        frame : {"ENU", "XYZ"}, optional
            The coordinate system of the data to download. Default is "ENU".
        reference : {"IGS14", "IGS20"}
            Geodetic reference of processed data.
        plate_fixed : bool, optional
            Whether to download plate-fixed data. Only applicable for "ENU" frame.
        plate : str, optional
            If using plate_fixed, specify which plate to use (in the case of a station
            on multiple plates).
            If None, uses the first plate from the UNR results.

        """
        station_id = validate_station_id(station_id.upper())

        if frame == "ENU":
            if plate_fixed:
                if plate is None:
                    plate = self._get_station_plates(station_id)[0]
                url = f"https://geodesy.unr.edu/gps_timeseries/tenv3/plates/{plate}/{station_id}.{plate}.tenv3"
                filename = self._cache_dir / plate / f"{station_id}.tenv3"
            else:
                url = GPS_BASE_URL.format(station=station_id, reference=reference)
                # Hack to get around bad url structure
                url = url.replace("gps_timeseries/IGS14", "gps_timeseries")
                filename = self._cache_dir / f"{station_id}.tenv3"
        elif frame == "XYZ":
            url = f"https://geodesy.unr.edu/gps_timeseries/txyz/{reference}/{station_id}.txyz2"
            filename = self._cache_dir / f"{station_id}.txyz2"
        else:
            msg = "frame must be 'ENU' or 'XYZ'"
            raise ValueError(msg)

        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(response.text)
        logger.info(f"Saved {url} to {filename}")

    def _get_station_plates(self, station_id: str) -> list[str]:
        """Get the tectonic plate(s) for a given GPS station."""
        plates = self._read_station_plates_table().get(station_id)
        if plates is None:
            msg = f"Failed to find {station_id} in the UNR station plates table"
            raise ValueError(msg)
        return plates

    @staticmethod
    @cache
    def _read_station_plates_table() -> dict[str, list[str]]:
        """Download and parse the UNR station -> plates table (cached)."""
        # A text file that gives the plate associated with each station.
        # The directory also contains files for each frame ("plate_??.txt" where
        # ?? is the 2-character plate designation) that list the stations
        # associated with each plate.
        url = "https://geodesy.unr.edu/gps_timeseries/Plates/sta_frames.txt"
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        table: dict[str, list[str]] = {}
        for line in response.text.splitlines():
            cur_id, *plates = line.split(" ")
            table[cur_id] = plates
        return table

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
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        with open(filename, "w") as f:
            f.write(resp.text)

    def steps(self, station_ids: list[str] | None = None) -> pd.DataFrame:
        """Fetch the UNR database of potential step epochs.

        Parses https://geodesy.unr.edu/NGLStationPages/steps.txt (format:
        https://geodesy.unr.edu/NGLStationPages/steps_readme.txt), which
        lists equipment changes (code 1) and earthquakes near the station
        (code 2).

        Parameters
        ----------
        station_ids : list of str, optional
            Only return steps for these stations.

        Returns
        -------
        pd.DataFrame
            Columns: ``id``, ``date`` (parsed datetime), ``code``,
            ``description``, plus for earthquake entries
            ``threshold_distance``, ``distance_from_eq`` and
            ``magnitude`` (NaN for equipment steps).

        """
        rows = self._read_steps_table()
        df = rows.copy()
        if station_ids is not None:
            wanted = {s.upper() for s in station_ids}
            df = df[df["id"].isin(wanted)].reset_index(drop=True)
        return df

    @staticmethod
    @cache
    def _read_steps_table() -> pd.DataFrame:
        """Download and parse the UNR steps file (cached)."""
        response = requests.get(STEPS_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        equipment, earthquakes = [], []
        for line in response.text.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            if len(parts) > 5:  # earthquake entries carry extra columns
                earthquakes.append(parts[:7])
            else:
                equipment.append(parts[:4])

        df_eq = pd.DataFrame(
            earthquakes,
            columns=[
                "id", "date", "code", "threshold_distance",
                "distance_from_eq", "magnitude", "description",
            ],
        )
        df_env = pd.DataFrame(
            equipment, columns=["id", "date", "code", "description"]
        )
        df = pd.concat([df_env, df_eq], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], format="%y%b%d")
        df["code"] = df["code"].astype(int)
        for col in ("threshold_distance", "distance_from_eq", "magnitude"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values(["id", "date"], ignore_index=True)

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

"""GPS data handling and downloading functionality.

This module provides functions to download GPS station data, manage station
information, and handle GPS time series data.
"""

from __future__ import annotations

import datetime
import difflib
import logging
import re
import warnings
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from shapely.geometry import box

from geepers import utils
from geepers.io import XarrayReader
from geepers.schemas import StationObservationSchema

from ._types import PathOrStr

__all__ = [
    "get_stations_within_image",
    "load_station_enu",
    "read_station_llas",
]

# Constants
GPS_BASE_URL = "https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{station}.tenv3"
GPS_DIR = utils.get_cache_dir()
GPS_DIR.mkdir(exist_ok=True, parents=True)
STATION_LLH_URL = "https://geodesy.unr.edu/NGLStationPages/llh.out"
STATION_LLH_FILE = str(GPS_DIR / "station_llh_all_{today}.csv")

logger = logging.getLogger("geepers")


def get_stations_within_image(
    reader: XarrayReader,
    mask_invalid: bool = True,
    bad_vals: Sequence[float] | None = None,
    exclude_stations: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    """Find GPS stations within a given geocoded image.

    Parameters
    ----------
    reader : XarrayReader
        Reader object containing the geocoded DataArray.
    bad_vals : list of float, optional
        Values (besides NaN) indicating no data. Default is [0].
    mask_invalid : bool, optional
        If True, don't return stations where the image value is NaN or in bad_vals.
        Default is True.
    exclude_stations : str or list of str, optional
        Station(s) to manually ignore. Default is None.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing information about the GPS stations within the image.

    Notes
    -----
    This function assumes the image is in a geographic coordinate system (lat/lon).

    """
    if bad_vals is None:
        bad_vals = [0]

    if reader.crs != "EPSG:4326":
        bounds = rasterio.warp.transform_bounds(
            reader.crs, "EPSG:4326", *reader.da.rio.bounds()
        )
    else:
        bounds = reader.da.rio.bounds()
    bounds_poly = box(*bounds)

    # Get all GPS stations
    gdf_all = read_station_llas(to_geodataframe=True)

    gdf_within = gdf_all.clip(bounds_poly)

    # TODO: this should probably be moved elsewhere
    # this is doing too much as is
    if mask_invalid:
        warnings.warn(
            "mask_invalid is deprecated implemented yet", UserWarning, stacklevel=2
        )

    # Exclude specified stations
    if exclude_stations:
        gdf_within = gdf_within[
            ~gdf_within.name.isin([s.upper() for s in exclude_stations])
        ]

    # Reset index for cleaner output
    gdf_within.reset_index(drop=True, inplace=True)
    return gdf_within


def download_station_data(
    station_name: str, coords: str = "enu", plate_fixed: bool = False
) -> None:
    """Download GPS station data from the Nevada Geodetic Laboratory.

    Parameters
    ----------
    station_name : str
        The name of the GPS station.
    coords : str, optional
        The coordinate system of the data to download. Either "enu" (east, north, up)
        or "xyz" (ECEF coordinates). Default is "enu".
    plate_fixed : bool, optional
        Whether to download plate-fixed data. Only applicable for "enu" coordinates.
        Default is False.

    Raises
    ------
    ValueError
        If an invalid coordinate system is specified.
    requests.HTTPError
        If the download request fails.

    """
    station_name = station_name.upper()
    plate = _get_station_plate(station_name)

    if coords == "enu":
        if plate_fixed:
            url = f"https://geodesy.unr.edu/gps_timeseries/tenv3/plates/{plate}/{station_name}.{plate}.tenv3"
            filename = GPS_DIR / f"{station_name}_{plate}.tenv3"
        else:
            url = GPS_BASE_URL.format(station=station_name)
            filename = GPS_DIR / f"{station_name}.tenv3"
    elif coords == "xyz":
        url = f"https://geodesy.unr.edu/gps_timeseries/txyz/IGS14/{station_name}.txyz2"
        filename = GPS_DIR / f"{station_name}.txyz2"
    else:
        msg = "coords must be either 'enu' or 'xyz'"
        raise ValueError(msg)

    response = requests.get(url)
    response.raise_for_status()

    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(response.text)
    logger.info(f"Saved {url} to {filename}")


def _get_station_plate(station_name: str) -> str:
    """Get the tectonic plate for a given GPS station.

    Parameters
    ----------
    station_name : str
        The name of the GPS station.

    Returns
    -------
    str
        The two-letter code for the tectonic plate.

    Raises
    ------
    ValueError
        If the plate information cannot be found for the station.
    requests.HTTPError
        If the request to fetch station information fails.

    """
    url = f"https://geodesy.unr.edu/NGLStationPages/stations/{station_name}.sta"
    response = requests.get(url)
    response.raise_for_status()

    match = re.search(r"tenv3\/plates\/(?P<plate>[A-Z]{2})", response.text)
    if not match:
        msg = f"Could not find plate name on {url}"
        raise ValueError(msg)
    return match.group("plate")


def load_station_enu(
    station_name: str,
    start_date: str | datetime.date | None = None,
    end_date: str | datetime.date | None = None,
    download_if_missing: bool = True,
    zero_by: str = "mean",  # TODO: remove, or change to Enum
) -> pd.DataFrame:
    """Load GPS station data in the east-north-up (ENU) coordinate system.

    Parameters
    ----------
    station_name : str
        The name of the GPS station.
    start_date : str or datetime.date, optional
        The start date for the data. If None, use all available data.
    end_date : str or datetime.date, optional
        The end date for the data. If None, use all available data.
    download_if_missing : bool, optional
        Whether to download the data if it's not found locally. Default is True.
    zero_by : str, optional
        How to zero the data. Either "mean" or "start". Default is "mean".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the ENU data for the specified station and date range.

    Raises
    ------
    ValueError
        If the station data file is not found and download_if_missing is False.

    """
    station_name = station_name.upper()
    gps_data_file = GPS_DIR / f"{station_name}.tenv3"

    if not gps_data_file.exists():
        if download_if_missing:
            logger.info(f"Downloading {station_name} to {gps_data_file}")
            download_station_data(station_name, coords="enu")
        else:
            msg = f"{gps_data_file} does not exist, download_if_missing = False"
            raise ValueError(msg)

    df = pd.read_csv(gps_data_file, sep=r"\s+", engine="c")
    # Add station name and standardize column names
    df = _clean_gps_df(df, start_date, end_date, coords="enu")

    if zero_by.lower() == "mean":
        mean_val = df[["east", "north", "up"]].mean()
        df[["east", "north", "up"]] -= mean_val
    elif zero_by.lower() == "start":
        start_val = df[["east", "north", "up"]].iloc[:10].mean()
        df[["east", "north", "up"]] -= start_val
    else:
        msg = "zero_by must be either 'mean' or 'start'"
        raise ValueError(msg)

    StationObservationSchema.validate(df, lazy=True)
    return df


def _clean_gps_df(
    df: pd.DataFrame,
    start_date: str | datetime.date | None = None,
    end_date: str | datetime.date | None = None,
    coords: str = "enu",
) -> pd.DataFrame:
    """Clean and preprocess the GPS DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The raw GPS DataFrame.
    start_date : str or datetime.date, optional
        The start date for the data. If None, use all available data.
    end_date : str or datetime.date, optional
        The end date for the data. If None, use all available data.
    coords : str, optional
        The coordinate system of the data. Either "enu" or "xyz". Default is "enu".

    Returns
    -------
    pd.DataFrame
        The cleaned GPS DataFrame.

    Raises
    ------
    ValueError
        If an invalid coordinate system is specified.

    Notes
    -----
    See here for .tenv3 format:
    https://geodesy.unr.edu/gps_timeseries/README_tenv3.txt

    """
    df["date"] = pd.to_datetime(df["YYMMMDD"], format="%y%b%d")

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    # NOTE: recently they added
    # _latitude(deg) _longitude(deg) __height(m)
    # TODO: check if we should use this here
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


def download_station_locations(filename: PathOrStr, url: str) -> None:
    """Download the station location file from the Nevada Geodetic Laboratory.

    Parameters
    ----------
    filename : str or Path
        The path to save the downloaded file.
    url : str
        The URL to download the station location file from.

    Raises
    ------
    requests.HTTPError
        If the download request fails.

    """
    resp = requests.get(url)
    resp.raise_for_status()

    with open(filename, "w") as f:
        f.write(resp.text)


def read_station_llas(
    filename: PathOrStr | None = None, to_geodataframe: bool = False
) -> pd.DataFrame:  # type: ignore[return-value]
    """Read the station latitude, longitude, and altitude (LLA) data.

    Parameters
    ----------
    filename : str or Path, optional
        The path to the station LLA file. If None, use the default location.
    to_geodataframe : bool, optional
        Whether to return a GeoDataFrame instead of a DataFrame. Default is False.

    Returns
    -------
    pd.DataFrame or geopandas.GeoDataFrame
        A DataFrame or GeoDataFrame containing the station LLA data.

    Raises
    ------
    FileNotFoundError
        If the station LLA file is not found and cannot be downloaded.

    """
    today = datetime.date.today().strftime("%Y%m%d")
    filename = filename or STATION_LLH_FILE.format(today=today)
    lla_path = Path(filename)

    # utils.remove_old_files(lla_path)
    # TODO: remove based on creation time

    try:
        df = pd.read_csv(lla_path, sep=r"\s+", engine="c", header=None)
    except FileNotFoundError:
        logger.info(f"Downloading from {STATION_LLH_URL} to {lla_path}")
        download_station_locations(lla_path, STATION_LLH_URL)
        df = pd.read_csv(lla_path, sep=r"\s+", engine="c", header=None)

    df.columns = ["name", "lat", "lon", "alt"]
    df.loc[:, "lon"] = df.lon - (np.round(df.lon / 360) * 360)

    if to_geodataframe:
        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
        )
    else:
        return df


def station_lonlat(station_name: str) -> tuple[float, float]:
    """Get the longitude and latitude of a GPS station.

    Parameters
    ----------
    station_name : str
        The name of the GPS station.

    Returns
    -------
    tuple[float, float]
        The longitude and latitude of the station in degrees.

    Raises
    ------
    ValueError
        If the station is not found in the LLA data.

    """
    df = read_station_llas()
    station_name = station_name.upper()
    if station_name not in df["name"].values:
        closest_names = difflib.get_close_matches(station_name, df["name"], n=5)
        msg = f"No station named {station_name} found. Closest: {closest_names}"
        raise ValueError(msg)
    _, lat, lon, _ = df[df["name"] == station_name].iloc[0]
    return lon, lat

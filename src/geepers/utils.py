import datetime
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np

from ._types import DateOrDatetime

DATE_FORMAT = "%Y%m%d"
DATETIME_FORMAT = "%Y%m%dT%H%M%S"


def datetime_to_float(dates: Sequence[DateOrDatetime]) -> np.ndarray:
    """Convert a sequence of datetime objects to a float representation.

    Output units are in days since the first item in `dates`.

    Parameters
    ----------
    dates : Sequence[DateOrDatetime]
        List of datetime objects to convert to floats

    Returns
    -------
    date_arr : np.array 1D
        The float representation of the datetime objects

    """
    sec_per_day = 60 * 60 * 24
    date_arr = np.asarray(dates).astype("datetime64[s]")
    # Reference the 0 to the first date
    date_arr = date_arr - date_arr[0]
    return date_arr.astype(float) / sec_per_day


def get_cache_dir(force_posix=False, app_name="geepers") -> Path:
    """Return the cache folder for the application.

    The directory is used to store gps timeseries data.
    The default behavior is to return whatever is most appropriate for the OS.


    the following folders could be returned:
    Mac OS X:
      ``~/Library/Application Support/geepers``
    Mac OS X (POSIX):
      ``~/.geepers``
    Unix:
      ``~/.cache/geepers``
    Unix (POSIX):
      ``~/.geepers``

    Parameters
    ----------
    force_posix : bool
        If this is set to `True` then on any POSIX system the folder will be stored
        in the home folder with a leading dot instead of the XDG config home or darwin's
        application support folder.
    app_name : str
        Name of the application (for naming the subfolder)

    Returns
    -------
    Path
        Folder to store cached data.

    Starting source:
    https://github.com/pallets/click/blob/ca5e1c3d75e95cbc70fa6ed51ef263592e9ac0d0/src/click/utils.py#L32

    """
    if force_posix:
        return Path(f"~/.{app_name}").expanduser()
    if sys.platform == "darwin":
        return Path("~/Library/Application Support").expanduser() / app_name
    else:
        base_path = Path(
            os.environ.get("XDG_CONFIG_HOME", Path("~/.cache").expanduser())
        )
        return base_path / app_name


def _get_path_from_gdal_str(name: Path | str) -> Path:
    s = str(name)
    if s.upper().startswith("DERIVED_SUBDATASET"):
        # like DERIVED_SUBDATASET:AMPLITUDE:slc_filepath.tif
        p = s.split(":")[-1].strip('"').strip("'")
    elif ":" in s and (s.upper().startswith("NETCDF") or s.upper().startswith("HDF")):
        # like NETCDF:"slc_filepath.nc":subdataset
        p = s.split(":")[1].strip('"').strip("'")
    else:
        # Whole thing is the path
        p = str(name)
    return Path(p)


def get_dates(filename: Path | str, fmt: str = DATE_FORMAT) -> list[datetime.datetime]:
    """Search for dates in the stem of `filename` matching `fmt`.

    Excludes dates that are not in the stem of `filename` (in the directories).

    Parameters
    ----------
    filename : Path or str
        Path or string to search for dates.
    fmt : str, optional
        Format of date to search for. Default is %Y%m%d

    Returns
    -------
    list[datetime.datetime]
        list of dates found in the stem of `filename` matching `fmt`.

    Examples
    --------
    >>> get_dates("/path/to/20191231.slc.tif")
    [datetime.datetime(2019, 12, 31, 0, 0)]
    >>> get_dates("S1A_IW_SLC__1SDV_20191231T000000_20191231T000000_032123_03B8F1_1C1D.nc")
    [datetime.datetime(2019, 12, 31, 0, 0), datetime.datetime(2019, 12, 31, 0, 0)]
    >>> get_dates("/not/a/date_named_file.tif")
    []

    """  # noqa: E501
    path = _get_path_from_gdal_str(filename)
    pattern = _date_format_to_regex(fmt)
    date_list = re.findall(pattern, path.name)
    if not date_list:
        return []
    return [datetime.datetime.strptime(d, fmt) for d in date_list]


def _date_format_to_regex(date_format: str) -> re.Pattern:
    r"""Convert a python date format string to a regular expression.

    Parameters
    ----------
    date_format : str
        Date format string, e.g. DATE_FORMAT

    Returns
    -------
    re.Pattern
        Regular expression that matches the date format string.

    Examples
    --------
    >>> pat2 = _date_format_to_regex("%Y%m%d").pattern
    >>> pat2 == re.compile(r'\d{4}\d{2}\d{2}').pattern
    True
    >>> pat = _date_format_to_regex("%Y-%m-%d").pattern
    >>> pat == re.compile(r'\d{4}\-\d{2}\-\d{2}').pattern
    True

    """
    # Escape any special characters in the date format string
    date_format = re.escape(date_format)

    # Replace each format specifier with a regular expression that matches it
    date_format = date_format.replace("%Y", r"\d{4}")
    date_format = date_format.replace("%y", r"\d{2}")
    date_format = date_format.replace("%m", r"\d{2}")
    date_format = date_format.replace("%d", r"\d{2}")
    date_format = date_format.replace("%H", r"\d{2}")
    date_format = date_format.replace("%M", r"\d{2}")
    date_format = date_format.replace("%S", r"\d{2}")
    date_format = date_format.replace("%j", r"\d{3}")

    # Return the resulting regular expression
    return re.compile(date_format)


def read_geo_csv(filename: Path | str) -> gpd.GeoDataFrame:
    """Read a CSV file with a geometry column."""
    df = gpd.read_file(filename)  # This just returns a pandas DataFrame
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df.geometry))


def decimal_year_to_datetime(decimal_year: float) -> datetime.datetime:
    """Convert a decimal year to a datetime object.

    See https://geodesy.unr.edu/NGLStationPages/DecimalYearConvention for
    more information, or https://geodesy.unr.edu/NGLStationPages/decyr.txt
    for a mapping from decimal year to datetime (with hour precision).

    Parameters
    ----------
    decimal_year : float
        Year expressed as a decimal (e.g., 2014.5).

    Returns
    -------
    datetime.datetime
        Corresponding calendar datetime (approximate to nearest day).

    """
    start_year = 1990
    seconds_per_year = 365.25 * 24 * 3600
    return datetime.datetime(1990, 1, 1) + datetime.timedelta(
        seconds=(decimal_year - start_year) * seconds_per_year
    )

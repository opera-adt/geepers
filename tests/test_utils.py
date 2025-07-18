import datetime
from pathlib import Path

import pandas as pd
import pytest

from geepers import utils


def test_date_format_to_regex():
    # Test date format strings with different specifiers and delimiters
    date_formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d-%m-%Y", "%m/%d/%Y", "%j0.%y"]
    matching_dates = [
        "2021-01-01",
        "2022/02/02",
        "20230103",
        "01-04-2024",
        "05/06/2025",
        "1300.23",
    ]
    for date_format, date in zip(date_formats, matching_dates, strict=False):
        pattern = utils._date_format_to_regex(date_format)

        # Test that the date matches the regular expression
        assert pattern.match(date) is not None

    # Test date formats that should not match the dates in "non_matching_dates"
    non_matching_dates = [
        "01-01-2021",
        "2022-02-03",
        "2022-03-04",
        "2022/05/06",
        "117.22",
        "23.0090",
    ]
    for date, date_format in zip(non_matching_dates, date_formats, strict=False):
        pattern = utils._date_format_to_regex(date_format)

        # Test that the date does not match the regular expression
        assert pattern.match(date) is None


def test_datetime_format_to_regex():
    # Check on a Sentinel-1-like datetime format
    date_format = "%Y%m%dT%H%M%S"
    date = "20221204T005230"
    pattern = utils._date_format_to_regex(date_format)

    # Test that the date matches the regular expression
    assert pattern.match(date)


def test_get_dates():
    assert utils.get_dates("20200303_20210101.int") == [
        datetime.datetime(2020, 3, 3),
        datetime.datetime(2021, 1, 1),
    ]

    assert utils.get_dates("20200303.slc")[0] == datetime.datetime(2020, 3, 3)
    assert utils.get_dates(Path("20200303.slc"))[0] == datetime.datetime(2020, 3, 3)
    # Check that it's the filename, not the path
    assert utils.get_dates(Path("/usr/19990101/asdf20200303.tif"))[
        0
    ] == datetime.datetime(2020, 3, 3)
    assert utils.get_dates("/usr/19990101/asdf20200303.tif")[0] == datetime.datetime(
        2020, 3, 3
    )
    assert utils.get_dates("/usr/19990101/20200303_20210101.int") == [
        datetime.datetime(2020, 3, 3),
        datetime.datetime(2021, 1, 1),
    ]

    assert utils.get_dates("/usr/19990101/notadate.tif") == []


def test_get_dates_with_format():
    # try other date formats
    fmt = "%Y-%m-%d"
    assert utils.get_dates("2020-03-03_2021-01-01.int", fmt) == [
        datetime.datetime(2020, 3, 3),
        datetime.datetime(2021, 1, 1),
    ]

    fmt = "%j0.%y"
    # check for TEC file format
    assert utils.get_dates("/usr/19990101/jplg0900.23i", fmt)[0] == datetime.datetime(
        2023, 3, 31, 0, 0
    )

    fmt = "%Y%m%dT%H%M%S"
    # Check the OPERA name
    fn = "OPERA_L2_CSLC-S1_T087-185678-IW2_20180210T232711Z_20230101T100506Z_S1A_VV_v1.0.h5"
    assert utils.get_dates(fn, fmt) == [
        datetime.datetime(2018, 2, 10, 23, 27, 11),
        datetime.datetime(2023, 1, 1, 10, 5, 6),
    ]

    # Check the Sentinel name
    fn = "S1A_IW_SLC__1SDV_20221204T005230_20221204T005257_046175_05873C_3B80.zip"
    assert utils.get_dates(fn, fmt) == [
        datetime.datetime(2022, 12, 4, 0, 52, 30),
        datetime.datetime(2022, 12, 4, 0, 52, 57),
    ]

    # Check without a format using default
    assert utils.get_dates(fn) == [
        datetime.datetime(2022, 12, 4, 0, 0, 0),
        datetime.datetime(2022, 12, 4, 0, 0, 0),
    ]


def test_get_dates_with_gdal_string():
    # Checks that is can parse 'NETCDF:"/path/to/file.nc":variable'
    assert utils.get_dates('NETCDF:"/usr/19990101/20200303_20210101.nc":variable') == [
        datetime.datetime(2020, 3, 3),
        datetime.datetime(2021, 1, 1),
    ]
    assert utils.get_dates(
        'NETCDF:"/usr/19990101/20200303_20210101.nc":"//variable/2"'
    ) == [
        datetime.datetime(2020, 3, 3),
        datetime.datetime(2021, 1, 1),
    ]
    # Check the derived dataset name too
    assert utils.get_dates(
        'DERIVED_SUBDATASET:AMPLITUDE:"/usr/19990101/20200303_20210101.int"'
    ) == [datetime.datetime(2020, 3, 3), datetime.datetime(2021, 1, 1)]


def test_read_geo_csv(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {
            "station": ["CHRS", "ERL2", "HILR"],
            "geometry": [
                Point(-156.05677, 19.72302),
                Point(-155.92682, 19.85227),
                Point(-155.04949, 19.71739),
            ],
        }
    )
    gdf.to_csv(tmp_path / "test.csv", index=False)
    gdf2 = utils.read_geo_csv(tmp_path / "test.csv")
    assert gdf.equals(gdf2)


class TestDecimalYearToDatetime:
    # Compare to the provided mapping
    # https://geodesy.unr.edu/NGLStationPages/decyr.txt
    @pytest.fixture
    def dec_year_df(self):
        columns = [
            "date",
            "decimalyr",
            "year",
            "mm",
            "dd",
            "hh",
            "day",
            "mjday",
            "week",
            "d",
            "J2000_sec",
        ]
        ddf = pd.read_csv(
            "https://geodesy.unr.edu/NGLStationPages/decyr.txt",
            sep=r"\s+",
            header=None,
            skiprows=1,
            names=columns,
        )
        ddf["datetime"] = ddf.apply(
            lambda row: datetime.datetime(row.year, row.mm, row.dd, row.hh), axis=1
        )
        return ddf

    def test_decimal_year_to_datetime(self, dec_year_df):
        dts = dec_year_df["decimalyr"].apply(utils.decimal_year_to_datetime)
        expected_dts = dec_year_df["datetime"]

        seconds_difference = (expected_dts - dts).dt.total_seconds()
        assert seconds_difference.max() < 3600  # within one hour

        # All dates should match
        assert (dec_year_df.datetime.dt.date != dts.dt.date).sum() == 0

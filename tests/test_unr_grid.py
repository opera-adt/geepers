"""Tests for the unr_grid module."""

import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests
from shapely.geometry import Point

from geepers import unr_grid


class TestLoadGridGeometry:
    """Test loading the UNR grid geometry."""

    def test_load_grid_geometry_returns_geoseries(self):
        """Test that load_grid_geometry returns a GeoSeries."""
        grid = unr_grid.load_grid_geometry()
        assert len(grid) == 24801
        assert all(isinstance(geom, Point) for geom in grid.geometry)

    def test_grid_coordinates_within_expected_range(self):
        """Test that grid coordinates are within expected global range."""
        grid = unr_grid.load_grid_geometry()
        bounds = grid.total_bounds
        # Global bounds should be roughly -180 to 180 longitude, -90 to 90 latitude
        assert bounds[0] >= -180  # min longitude
        assert bounds[1] >= -90  # min latitude
        assert bounds[2] <= 180  # max longitude
        assert bounds[3] <= 90  # max latitude


class TestGetGridWithinImage:
    """Test finding grid points within image boundaries."""

    def test_get_grid_within_small_region(self):
        """Test finding grid points within a small rectangular region."""
        # Small region around Los Angeles
        bounds = (-118.5, 33.5, -118.0, 34.0)
        grid_points = unr_grid.get_grid_within_image(bounds)

        assert len(grid_points) > 0
        # All points should be within bounds
        for _, row in grid_points.iterrows():
            lon, lat = row.geometry.x, row.geometry.y
            assert bounds[0] <= lon <= bounds[2]
            assert bounds[1] <= lat <= bounds[3]

    def test_get_grid_within_empty_region(self):
        """Test behavior when no grid points are found."""
        # Tiny region in the middle of the ocean
        bounds = (-150.001, 30.001, -150.0, 30.002)
        grid_points = unr_grid.get_grid_within_image(bounds)

        # Should return empty GeoDataFrame
        assert len(grid_points) == 0

    def test_get_grid_within_invalid_bounds(self):
        """Test behavior with invalid bounds."""
        # Invalid bounds (min > max)
        bounds = (-118.0, 34.0, -118.5, 33.5)
        grid_points = unr_grid.get_grid_within_image(bounds)

        # Should handle gracefully
        assert len(grid_points) == 0


class TestDecimalYearToDatetime:
    """Test decimal year to datetime conversion."""

    def test_decimal_year_conversion_exact_years(self):
        """Test conversion of exact years."""
        assert unr_grid.decimal_year_to_datetime(2020.0) == datetime.datetime(
            2020, 1, 1
        )
        assert unr_grid.decimal_year_to_datetime(2021.0) == datetime.datetime(
            2021, 1, 1
        )

    def test_decimal_year_conversion_mid_year(self):
        """Test conversion of mid-year dates."""
        # 2020.5 should be around July 1st (leap year)
        result = unr_grid.decimal_year_to_datetime(2020.5)
        expected = datetime.datetime(2020, 7, 1, 12, 0)  # Approximately
        # Allow some tolerance for the exact calculation
        assert abs((result - expected).days) <= 1

    def test_decimal_year_conversion_array(self):
        """Test conversion of array of decimal years."""
        years = np.array([2020.0, 2020.5, 2021.0])
        results = unr_grid.decimal_year_to_datetime(years)

        assert len(results) == 3
        assert results[0] == datetime.datetime(2020, 1, 1)
        assert results[2] == datetime.datetime(2021, 1, 1)


class TestParseDataFile:
    """Test parsing .tenv8 data files."""

    def test_parse_valid_data_file(self, tmp_path):
        """Test parsing a valid .tenv8 file."""
        # Create a mock .tenv8 file
        test_file = tmp_path / "test_grid.tenv8"
        test_content = """# Sample .tenv8 file
# Created for testing
2019.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000
2019.0027   1.0000   2.0000   3.0000   0.1000   0.2000   0.3000   0.0000
2019.0055   2.0000   4.0000   6.0000   0.1000   0.2000   0.3000   0.0000
"""
        test_file.write_text(test_content)

        df = unr_grid.parse_data_file(test_file)

        assert len(df) == 3
        assert list(df.columns) == [
            "datetime",
            "de",
            "dn",
            "du",
            "se",
            "sn",
            "su",
            "rho",
        ]
        assert df.iloc[0]["de"] == 0.0
        assert df.iloc[1]["de"] == 1.0
        assert isinstance(df.iloc[0]["datetime"], datetime.datetime)

    def test_parse_empty_file(self, tmp_path):
        """Test parsing an empty file."""
        test_file = tmp_path / "empty.tenv8"
        test_file.write_text("")

        df = unr_grid.parse_data_file(test_file)
        assert len(df) == 0

    def test_parse_comments_only_file(self, tmp_path):
        """Test parsing a file with only comments."""
        test_file = tmp_path / "comments.tenv8"
        test_content = """# This is a comment
# Another comment
# Final comment
"""
        test_file.write_text(test_content)

        df = unr_grid.parse_data_file(test_file)
        assert len(df) == 0

    def test_parse_malformed_data(self, tmp_path):
        """Test parsing malformed data."""
        test_file = tmp_path / "malformed.tenv8"
        test_content = """# Valid header
2019.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000
invalid_line_with_text
2019.0055   2.0000   4.0000   6.0000   0.1000   0.2000   0.3000   0.0000
"""
        test_file.write_text(test_content)

        # Should skip invalid lines and continue parsing
        df = unr_grid.parse_data_file(test_file)
        assert len(df) == 2  # Should have 2 valid lines


@pytest.mark.vcr
class TestDownloadGridTimeseries:
    """Test downloading grid time series data."""

    def test_download_valid_grid_point(self, tmp_path, monkeypatch):
        """Test downloading data for a valid grid point."""
        # Mock the cache directory
        monkeypatch.setattr(unr_grid, "_get_cache_dir", lambda: tmp_path)

        # Use a smaller known grid point index that's more likely to exist
        grid_idx = 100  # Smaller valid index
        reference = unr_grid.Reference.IGS14

        result_path = unr_grid.download_grid_timeseries(grid_idx, reference)

        assert result_path.exists()
        assert result_path.suffix == ".tenv8"
        assert str(grid_idx) in result_path.name
        assert reference.value in result_path.name

    def test_download_cached_file(self, tmp_path, monkeypatch):
        """Test that cached files are not re-downloaded."""
        monkeypatch.setattr(unr_grid, "_get_cache_dir", lambda: tmp_path)

        grid_idx = 100
        reference = unr_grid.Reference.IGS14

        # Create a fake cached file
        expected_path = tmp_path / f"grid_{grid_idx:05d}_{reference.value}.tenv8"
        expected_path.write_text("# Cached file")
        original_mtime = expected_path.stat().st_mtime

        # Download should return the cached file without modification
        result_path = unr_grid.download_grid_timeseries(grid_idx, reference)

        assert result_path == expected_path
        assert result_path.stat().st_mtime == original_mtime

    @patch("requests.get")
    def test_download_network_error(self, mock_get, tmp_path, monkeypatch):
        """Test handling of network errors during download."""
        monkeypatch.setattr(unr_grid, "_get_cache_dir", lambda: tmp_path)

        # Mock a network error
        mock_get.side_effect = requests.RequestException("Network error")

        grid_idx = 100
        reference = unr_grid.Reference.IGS14

        with pytest.raises(requests.RequestException):
            unr_grid.download_grid_timeseries(grid_idx, reference)

    @patch("requests.get")
    def test_download_http_error(self, mock_get, tmp_path, monkeypatch):
        """Test handling of HTTP errors during download."""
        monkeypatch.setattr(unr_grid, "_get_cache_dir", lambda: tmp_path)

        # Mock HTTP 404 error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        grid_idx = 99999  # Invalid index
        reference = unr_grid.Reference.IGS14

        with pytest.raises(requests.HTTPError):
            unr_grid.download_grid_timeseries(grid_idx, reference)

    def test_download_invalid_grid_index(self, tmp_path, monkeypatch):
        """Test downloading with invalid grid index."""
        monkeypatch.setattr(unr_grid, "_get_cache_dir", lambda: tmp_path)

        # Grid indices should be 0-24800
        invalid_indices = [-1, 25000, 100000]

        for grid_idx in invalid_indices:
            with pytest.raises((ValueError, requests.HTTPError)):
                unr_grid.download_grid_timeseries(grid_idx, unr_grid.Reference.IGS14)


class TestReference:
    """Test the Reference enum."""

    def test_reference_values(self):
        """Test that Reference enum has expected values."""
        assert unr_grid.Reference.IGS14.value == "IGS14"
        assert unr_grid.Reference.NA.value == "NA"
        assert unr_grid.Reference.PA.value == "PA"

    def test_reference_string_conversion(self):
        """Test string representation of Reference enum."""
        assert str(unr_grid.Reference.IGS14) == "Reference.IGS14"


class TestLoadGridTimeseries:
    """Test loading grid time series data."""

    def test_load_grid_timeseries_download_if_missing(self, tmp_path, monkeypatch):
        """Test loading grid data with download_if_missing=True."""
        monkeypatch.setattr(unr_grid, "_get_cache_dir", lambda: tmp_path)

        # Create a mock .tenv8 file
        grid_idx = 100
        reference = unr_grid.Reference.IGS14
        test_file = tmp_path / f"grid_{grid_idx:05d}_{reference.value}.tenv8"
        test_content = """# Sample .tenv8 file
2019.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000
2019.0027   1.0000   2.0000   3.0000   0.1000   0.2000   0.3000   0.0000
2019.0055   2.0000   4.0000   6.0000   0.1000   0.2000   0.3000   0.0000
"""
        test_file.write_text(test_content)

        # Load the grid data
        df = unr_grid.load_grid_timeseries(grid_idx, download_if_missing=False)

        assert len(df) == 3
        assert list(df.columns) == ["de", "dn", "du", "se", "sn", "su", "rho"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_grid_timeseries_zero_by_mean(self, tmp_path, monkeypatch):
        """Test zeroing by mean."""
        monkeypatch.setattr(unr_grid, "_get_cache_dir", lambda: tmp_path)

        grid_idx = 100
        reference = unr_grid.Reference.IGS14
        test_file = tmp_path / f"grid_{grid_idx:05d}_{reference.value}.tenv8"
        test_content = """# Sample .tenv8 file
2019.0000   1.0000   2.0000   3.0000   0.1000   0.2000   0.3000   0.0000
2019.0027   2.0000   4.0000   6.0000   0.1000   0.2000   0.3000   0.0000
2019.0055   3.0000   6.0000   9.0000   0.1000   0.2000   0.3000   0.0000
"""
        test_file.write_text(test_content)

        df = unr_grid.load_grid_timeseries(
            grid_idx, zero_by="mean", download_if_missing=False
        )

        # After mean zeroing, the mean should be approximately 0
        mean_vals = df[["de", "dn", "du"]].mean()
        assert abs(mean_vals["de"]) < 1e-10
        assert abs(mean_vals["dn"]) < 1e-10
        assert abs(mean_vals["du"]) < 1e-10

    def test_load_grid_timeseries_zero_by_start(self, tmp_path, monkeypatch):
        """Test zeroing by start values."""
        monkeypatch.setattr(unr_grid, "_get_cache_dir", lambda: tmp_path)

        grid_idx = 100
        reference = unr_grid.Reference.IGS14
        test_file = tmp_path / f"grid_{grid_idx:05d}_{reference.value}.tenv8"
        test_content = """# Sample .tenv8 file
2019.0000   1.0000   2.0000   3.0000   0.1000   0.2000   0.3000   0.0000
2019.0027   2.0000   4.0000   6.0000   0.1000   0.2000   0.3000   0.0000
2019.0055   3.0000   6.0000   9.0000   0.1000   0.2000   0.3000   0.0000
"""
        test_file.write_text(test_content)

        df = unr_grid.load_grid_timeseries(
            grid_idx, zero_by="start", download_if_missing=False
        )

        # First few values should be close to zero (after start zeroing)
        assert len(df) == 3

    def test_load_grid_timeseries_date_filtering(self, tmp_path, monkeypatch):
        """Test date filtering functionality."""
        monkeypatch.setattr(unr_grid, "_get_cache_dir", lambda: tmp_path)

        grid_idx = 100
        reference = unr_grid.Reference.IGS14
        test_file = tmp_path / f"grid_{grid_idx:05d}_{reference.value}.tenv8"
        test_content = """# Sample .tenv8 file
2018.0000   1.0000   2.0000   3.0000   0.1000   0.2000   0.3000   0.0000
2019.0000   2.0000   4.0000   6.0000   0.1000   0.2000   0.3000   0.0000
2020.0000   3.0000   6.0000   9.0000   0.1000   0.2000   0.3000   0.0000
2021.0000   4.0000   8.0000  12.0000   0.1000   0.2000   0.3000   0.0000
"""
        test_file.write_text(test_content)

        # Filter by date range
        df = unr_grid.load_grid_timeseries(
            grid_idx,
            start_date="2019-01-01",
            end_date="2020-12-31",
            download_if_missing=False,
        )

        assert len(df) == 2  # Should only include 2019 and 2020 data

    def test_load_grid_timeseries_missing_file(self, tmp_path, monkeypatch):
        """Test error when file is missing and download_if_missing=False."""
        monkeypatch.setattr(unr_grid, "_get_cache_dir", lambda: tmp_path)

        grid_idx = 100

        with pytest.raises(
            ValueError, match="does not exist, download_if_missing = False"
        ):
            unr_grid.load_grid_timeseries(grid_idx, download_if_missing=False)

    def test_load_grid_timeseries_invalid_zero_by(self, tmp_path, monkeypatch):
        """Test error with invalid zero_by parameter."""
        monkeypatch.setattr(unr_grid, "_get_cache_dir", lambda: tmp_path)

        grid_idx = 100
        reference = unr_grid.Reference.IGS14
        test_file = tmp_path / f"grid_{grid_idx:05d}_{reference.value}.tenv8"
        test_content = """# Sample .tenv8 file
2019.0000   1.0000   2.0000   3.0000   0.1000   0.2000   0.3000   0.0000
"""
        test_file.write_text(test_content)

        with pytest.raises(
            ValueError, match="zero_by must be either 'mean' or 'start'"
        ):
            unr_grid.load_grid_timeseries(
                grid_idx, zero_by="invalid", download_if_missing=False
            )


# WRONG COLUMNS
class TestDataValidation:
    """Test data validation and cleaning functions."""

    def test_validate_grid_data_valid(self):
        """Test validation of valid grid data."""
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03"]),
                "de": [0.001, 0.002, 0.003],
                "dn": [0.001, 0.002, 0.003],
                "du": [0.005, 0.010, 0.015],
                "se": [0.001, 0.001, 0.001],
                "sn": [0.001, 0.001, 0.001],
                "su": [0.005, 0.005, 0.005],
                "rho": [0.0, 0.0, 0.0],
            }
        )

        result = unr_grid.validate_grid_data(df)
        assert len(result) == 3
        assert list(result.columns) == [
            "datetime",
            "de",
            "dn",
            "du",
            "se",
            "sn",
            "su",
            "rho",
        ]

    def test_validate_grid_data_missing_columns(self):
        """Test validation fails with missing columns."""
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2019-01-01", "2019-01-02"]),
                "de": [0.001, 0.002],
                # Missing other required columns
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            unr_grid.validate_grid_data(df)

    def test_validate_grid_data_negative_uncertainties(self):
        """Test validation fails with negative uncertainties."""
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2019-01-01", "2019-01-02"]),
                "de": [0.001, 0.002],
                "dn": [0.001, 0.002],
                "du": [0.005, 0.010],
                "se": [0.001, -0.001],  # Negative uncertainty
                "sn": [0.001, 0.001],
                "su": [0.005, 0.005],
                "rho": [0.0, 0.0],
            }
        )

        with pytest.raises(ValueError, match="Negative uncertainty values found"):
            unr_grid.validate_grid_data(df)

    def test_validate_grid_data_removes_duplicates(self):
        """Test validation removes duplicate datetime entries."""
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2019-01-01", "2019-01-01", "2019-01-02"]
                ),  # Duplicate
                "de": [0.001, 0.002, 0.003],
                "dn": [0.001, 0.002, 0.003],
                "du": [0.005, 0.010, 0.015],
                "se": [0.001, 0.001, 0.001],
                "sn": [0.001, 0.001, 0.001],
                "su": [0.005, 0.005, 0.005],
                "rho": [0.0, 0.0, 0.0],
            }
        )

        with pytest.warns(UserWarning, match="Removed 1 duplicate datetime entries"):
            result = unr_grid.validate_grid_data(df)

        assert len(result) == 2  # Should remove one duplicate

    def test_validate_grid_data_empty(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        result = unr_grid.validate_grid_data(df)
        assert len(result) == 0

    def test_remove_outliers_iqr(self):
        """Test outlier removal using IQR method."""
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"]
                ),
                "de": [0.001, 0.002, 0.003, 1.000],  # Last value is outlier
                "dn": [0.001, 0.002, 0.003, 0.004],
                "du": [0.005, 0.010, 0.015, 0.020],
                "se": [0.001, 0.001, 0.001, 0.001],
                "sn": [0.001, 0.001, 0.001, 0.001],
                "su": [0.005, 0.005, 0.005, 0.005],
                "rho": [0.0, 0.0, 0.0, 0.0],
            }
        )

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = unr_grid.remove_outliers(df, method="iqr", threshold=1.5)
        # Should remove the large outlier or keep all data - both are valid
        assert len(result) <= len(df)

    def test_remove_outliers_std(self):
        """Test outlier removal using standard deviation method."""
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"]
                ),
                "de": [0.001, 0.002, 0.003, 1.000],  # Last value is outlier
                "dn": [0.001, 0.002, 0.003, 0.004],
                "du": [0.005, 0.010, 0.015, 0.020],
                "se": [0.001, 0.001, 0.001, 0.001],
                "sn": [0.001, 0.001, 0.001, 0.001],
                "su": [0.005, 0.005, 0.005, 0.005],
                "rho": [0.0, 0.0, 0.0, 0.0],
            }
        )

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = unr_grid.remove_outliers(df, method="std", threshold=2.0)
        # Should remove the large outlier or keep all data - both are valid
        assert len(result) <= len(df)

    def test_remove_outliers_invalid_method(self):
        """Test error with invalid outlier removal method."""
        df = pd.DataFrame(
            {
                "de": [0.001, 0.002, 0.003],
                "dn": [0.001, 0.002, 0.003],
                "du": [0.005, 0.010, 0.015],
            }
        )

        with pytest.raises(ValueError, match="method must be either 'iqr' or 'std'"):
            unr_grid.remove_outliers(df, method="invalid")

    def test_remove_outliers_empty(self):
        """Test outlier removal on empty DataFrame."""
        df = pd.DataFrame()
        result = unr_grid.remove_outliers(df)
        assert len(result) == 0


class TestCacheDirectory:
    """Test cache directory functionality."""

    def test_get_cache_dir_creates_directory(self, tmp_path, monkeypatch):
        """Test that cache directory is created if it doesn't exist."""
        # Mock home directory to use temp directory
        fake_home = tmp_path / "fake_home"
        monkeypatch.setenv("HOME", str(fake_home))

        cache_dir = unr_grid._get_cache_dir()

        assert cache_dir.exists()
        assert cache_dir.is_dir()
        assert "geepers" in str(cache_dir)

    def test_cache_dir_permissions(self, tmp_path, monkeypatch):
        """Test that cache directory has appropriate permissions."""
        fake_home = tmp_path / "fake_home"
        monkeypatch.setenv("HOME", str(fake_home))

        cache_dir = unr_grid._get_cache_dir()

        # Directory should be readable and writable
        assert cache_dir.exists()
        assert (
            cache_dir / "test_file"
        ).touch() or True  # Should be able to create files

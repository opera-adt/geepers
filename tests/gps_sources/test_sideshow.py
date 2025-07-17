"""Tests for Sideshow GPS data source."""

import pytest

from geepers.gps_sources.sideshow import SideshowSource


class TestSideshowSource:
    """Tests for SideshowSource class."""

    def test_init(self):
        """Test SideshowSource initialization."""
        source = SideshowSource()
        assert isinstance(source, SideshowSource)

    def test_not_implemented_methods(self):
        """Test that placeholder methods raise NotImplementedError."""
        source = SideshowSource()

        with pytest.raises(NotImplementedError):
            source.load_station_enu("TEST")

        with pytest.raises(NotImplementedError):
            source.read_station_llas()

        # This would require a mock XarrayReader
        # with pytest.raises(NotImplementedError):
        #     source.get_stations_within_image(mock_reader)

    def test_method_signatures(self):
        """Test that methods have expected signatures."""
        source = SideshowSource()

        # Test that methods exist with expected parameters
        assert hasattr(source, "load_station_enu")
        assert hasattr(source, "read_station_llas")
        assert hasattr(source, "get_stations_within_image")
        assert hasattr(source, "download_station_data")
        assert hasattr(source, "parse_series_file")

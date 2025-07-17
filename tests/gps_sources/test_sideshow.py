"""Tests for Sideshow GPS data source."""

from geepers.gps_sources.sideshow import SideshowSource


class TestSideshowSource:
    """Tests for SideshowSource class."""

    def test_init(self):
        """Test SideshowSource initialization."""
        source = SideshowSource()
        assert isinstance(source, SideshowSource)

"""Tests for UNR GPS data source."""

from geepers.gps_sources.unr import UnrSource


class TestUnrSource:
    """Tests for UnrSource class."""

    def test_init(self):
        """Test UnrSource initialization."""
        source = UnrSource()
        assert isinstance(source, UnrSource)

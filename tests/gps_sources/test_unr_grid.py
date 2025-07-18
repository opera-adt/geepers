"""Tests for UNR Grid GPS data source."""

from geepers.gps_sources.unr_grid import UnrGridSource


class TestUnrGridSource:
    """Tests for UnrGridSource class."""

    def test_init(self):
        """Test UnrGridSource initialization."""
        source = UnrGridSource()
        assert isinstance(source, UnrGridSource)

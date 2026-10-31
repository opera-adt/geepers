"""Tests for UNR GPS data source."""

import pandas as pd

from geepers.gps_sources.unr import UnrSource


class TestUnrSource:
    """Tests for UnrSource class."""

    def test_init(self):
        """Test UnrSource initialization."""
        source = UnrSource()
        assert isinstance(source, UnrSource)


def test_filter_by_date_keeps_the_start_day():
    """Regression: a start bound after midnight dropped that day's solution."""
    daily = pd.DataFrame({"date": pd.date_range("2016-09-25", "2016-09-30")})
    kept = UnrSource()._filter_by_date(
        daily, "2016-09-27T00:26:23Z", "2016-09-29T00:26:23Z"
    )
    assert kept["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2016-09-27",
        "2016-09-28",
        "2016-09-29",
    ]

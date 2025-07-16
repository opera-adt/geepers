"""Tests for the schemas module."""

from __future__ import annotations

import pandas as pd
import pytest

from geepers.schemas import DailyDispModel, MetadataModel, RatesModel, RawObsModel


class TestRawObsModel:
    """Tests for RawObsModel validation."""

    def test_valid_raw_obs_data(self):
        """Test validation with valid raw observation data."""
        df = pd.DataFrame(
            {
                "station": ["TEST", "TEST"],
                "time": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "east": [0.001, 0.002],
                "north": [0.003, 0.004],
                "up": [0.005, 0.006],
                "sigma_east": [0.001, 0.001],
                "sigma_north": [0.001, 0.001],
                "sigma_up": [0.002, 0.002],
                "corr_en": [0.1, 0.2],
                "corr_eu": [0.0, 0.1],
                "corr_nu": [0.0, 0.0],
            }
        )

        # Should not raise
        validated_df = RawObsModel.validate(df)
        assert len(validated_df) == 2
        assert validated_df["station"].iloc[0] == "TEST"

    def test_invalid_correlation_values(self):
        """Test validation fails with invalid correlation values."""
        df = pd.DataFrame(
            {
                "station": ["TEST"],
                "time": pd.to_datetime(["2023-01-01"]),
                "east": [0.001],
                "north": [0.003],
                "up": [0.005],
                "sigma_east": [0.001],
                "sigma_north": [0.001],
                "sigma_up": [0.002],
                "corr_en": [1.5],  # Invalid: > 1
                "corr_eu": [0.0],
                "corr_nu": [0.0],
            }
        )

        with pytest.raises(
            Exception, match="failed element-wise validator"
        ):  # Pandera will raise a validation error
            RawObsModel.validate(df)

    def test_zero_sigma_values(self):
        """Test validation fails with zero sigma values."""
        df = pd.DataFrame(
            {
                "station": ["TEST"],
                "time": pd.to_datetime(["2023-01-01"]),
                "east": [0.001],
                "north": [0.003],
                "up": [0.005],
                "sigma_east": [0.0],  # Invalid: should be > EPS
                "sigma_north": [0.001],
                "sigma_up": [0.002],
                "corr_en": [0.0],
                "corr_eu": [0.0],
                "corr_nu": [0.0],
            }
        )

        with pytest.raises(
            Exception, match="failed element-wise validator"
        ):  # Pandera will raise a validation error
            RawObsModel.validate(df)


class TestDailyDispModel:
    """Tests for DailyDispModel validation."""

    def test_valid_daily_data(self):
        """Test validation with valid daily displacement data."""
        df = pd.DataFrame(
            {
                "station": ["TEST", "TEST"],
                "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "east_mm": [1.0, 2.0],
                "north_mm": [3.0, 4.0],
                "up_mm": [5.0, 6.0],
                "sigma_east_mm": [1.0, 1.0],
                "sigma_north_mm": [1.0, 1.0],
                "sigma_up_mm": [2.0, 2.0],
            }
        )

        # Should not raise
        validated_df = DailyDispModel.validate(df)
        assert len(validated_df) == 2
        assert validated_df["station"].iloc[0] == "TEST"

    def test_optional_sigma_columns(self):
        """Test validation with missing optional sigma columns."""
        df = pd.DataFrame(
            {
                "station": ["TEST"],
                "date": pd.to_datetime(["2023-01-01"]),
                "east_mm": [1.0],
                "north_mm": [3.0],
                "up_mm": [5.0],
                # No sigma columns - should still validate
            }
        )

        # Should not raise
        validated_df = DailyDispModel.validate(df)
        assert len(validated_df) == 1


class TestMetadataModel:
    """Tests for MetadataModel validation."""

    def test_valid_metadata(self):
        """Test validation with valid metadata."""
        df = pd.DataFrame(
            {
                "station": ["TEST"],
                "lat": [34.0],
                "lon": [-118.0],
                "alt": [100.0],
                "plate": ["NA"],
            }
        )

        # Should not raise
        validated_df = MetadataModel.validate(df)
        assert len(validated_df) == 1
        assert validated_df["station"].iloc[0] == "TEST"

    def test_invalid_latitude(self):
        """Test validation fails with invalid latitude."""
        df = pd.DataFrame(
            {
                "station": ["TEST"],
                "lat": [95.0],  # Invalid: > 90
                "lon": [-118.0],
                "alt": [100.0],
            }
        )

        with pytest.raises(
            Exception, match="failed element-wise validator"
        ):  # Pandera will raise a validation error
            MetadataModel.validate(df)


class TestRatesModel:
    """Tests for RatesModel validation."""

    def test_valid_rates_data(self):
        """Test validation with valid rates data."""
        df = pd.DataFrame(
            {
                "station": ["TEST"],
                "lat": [34.0],
                "lon": [-118.0],
                "alt": [100.0],
                "gps_velocity": [1.5],
                "gps_velocity_l2": [1.6],
                "gps_velocity_sigma": [0.2],
                "insar_velocity": [1.4],
                "insar_velocity_l2": [1.3],
                "insar_velocity_sigma": [0.3],
                "sigma_los_mm": [0.25],
                "difference": [0.1],
            }
        )

        # Should not raise
        validated_df = RatesModel.validate(df)
        assert len(validated_df) == 1
        assert validated_df["station"].iloc[0] == "TEST"

    def test_negative_sigma_values(self):
        """Test validation fails with negative sigma values."""
        df = pd.DataFrame(
            {
                "station": ["TEST"],
                "lat": [34.0],
                "lon": [-118.0],
                "alt": [100.0],
                "gps_velocity": [1.5],
                "gps_velocity_sigma": [-0.2],  # Invalid: should be >= 0
                "insar_velocity": [1.4],
                "difference": [0.1],
            }
        )

        with pytest.raises(
            Exception, match="failed element-wise validator"
        ):  # Pandera will raise a validation error
            RatesModel.validate(df)

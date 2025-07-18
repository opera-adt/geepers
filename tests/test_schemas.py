"""Tests for the schemas module."""

from __future__ import annotations

import pandas as pd
import pytest
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point

from geepers.schemas import RatesSchema, StationObservationSchema, StationSchema


class TestRawObsModel:
    """Tests for RawObsModel validation."""

    def test_valid_raw_obs_data(self):
        """Test validation with valid raw observation data."""
        df = pd.DataFrame(
            {
                "id": ["TEST", "TEST"],
                "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
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
        validated_df = StationObservationSchema.validate(df)
        assert len(validated_df) == 2
        assert validated_df["id"].iloc[0] == "TEST"

    def test_invalid_correlation_values(self):
        """Test validation fails with invalid correlation values."""
        df = pd.DataFrame(
            {
                "id": ["TEST"],
                "date": pd.to_datetime(["2023-01-01"]),
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
            StationObservationSchema.validate(df)

    def test_zero_sigma_values(self):
        """Test validation fails with zero sigma values."""
        df = pd.DataFrame(
            {
                "id": ["TEST"],
                "date": pd.to_datetime(["2023-01-01"]),
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
            StationObservationSchema.validate(df)


class TestStationModel:
    """Tests for StationModel validation."""

    def test_valid_metadata(self):
        """Test validation with valid metadata."""
        df = pd.DataFrame(
            {
                "id": ["TEST"],
                "lat": [34.0],
                "lon": [-118.0],
                "alt": [100.0],
                "plate": ["NA"],
            }
        )

        # Should not raise
        validated_df = StationSchema.validate(df)
        assert len(validated_df) == 1
        assert validated_df["id"].iloc[0] == "TEST"

    def test_invalid_latitude(self):
        """Test validation fails with invalid latitude."""
        df = pd.DataFrame(
            {
                "id": ["TEST"],
                "lat": [95.0],  # Invalid: > 90
                "lon": [-118.0],
                "alt": [100.0],
                "plate": ["NA"],
            }
        )

        with pytest.raises(
            Exception, match="Column 'lat' failed element-wise validator number"
        ):  # Pandera will raise a validation error
            StationSchema.validate(df)


class TestRatesModel:
    """Tests for RatesModel validation."""

    def test_valid_rates_data(self):
        """Test validation with valid rates data."""
        df = GeoDataFrame(
            {
                "id": ["TEST"],
                "gps_velocity": [1.5],
                "gps_velocity_l2": [1.6],
                "gps_velocity_sigma": [0.2],
                "insar_velocity": [1.4],
                "insar_velocity_l2": [1.3],
                "insar_velocity_sigma": [0.3],
                "sigma_los_mm": [0.25],
                "num_gps": [400],
                "gps_time_span_years": [1.0],
                "temporal_coherence": [0.8],
                "similarity": [0.9],
                "rms_misfit": [0.1],
                "gps_outlier_fraction": [0.05],
                "gps_velocity_scatter": [0.2],
                "difference": [0.1],
                "geometry": GeoSeries([Point(-118.0, 34.0)], crs="EPSG:4326"),
            }
        ).set_index("id")

        # Should not raise
        validated_df = RatesSchema.validate(df)
        assert len(validated_df) == 1
        assert validated_df.index[0] == "TEST"

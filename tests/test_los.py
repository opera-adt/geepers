"""Tests for the los module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geepers.los import (
    DEFAULT_LOS_VECTORS,
    apply_los_vectors,
    compute_los_vector,
    get_default_los_vector,
)


class TestGetDefaultLosVector:
    """Tests for get_default_los_vector function."""

    def test_default_sentinel1_ascending(self):
        """Test default Sentinel-1 ascending LOS vector."""
        los_vec = get_default_los_vector("sentinel1_ascending")

        expected = np.array([-0.6, 0.0, 0.8])
        np.testing.assert_array_equal(los_vec, expected)

        # Should be unit vector (approximately)
        assert abs(np.linalg.norm(los_vec) - 1.0) < 0.1

    def test_default_sentinel1_descending(self):
        """Test default Sentinel-1 descending LOS vector."""
        los_vec = get_default_los_vector("sentinel1_descending")

        expected = np.array([0.6, 0.0, 0.8])
        np.testing.assert_array_equal(los_vec, expected)

        # Should be unit vector (approximately)
        assert abs(np.linalg.norm(los_vec) - 1.0) < 0.1

    def test_all_default_vectors(self):
        """Test that all default vectors are approximately unit vectors."""
        for satellite in DEFAULT_LOS_VECTORS:
            los_vec = get_default_los_vector(satellite)
            norm = np.linalg.norm(los_vec)
            assert abs(norm - 1.0) < 0.1, f"Vector for {satellite} is not unit: {norm}"

    def test_unknown_satellite(self):
        """Test error for unknown satellite."""
        with pytest.raises(KeyError):
            get_default_los_vector("unknown_satellite")

    def test_vector_is_copy(self):
        """Test that returned vector is a copy, not reference."""
        los_vec1 = get_default_los_vector("sentinel1_ascending")
        los_vec2 = get_default_los_vector("sentinel1_ascending")

        # Modify one vector
        los_vec1[0] = 999.0

        # Other vector should be unchanged
        assert los_vec2[0] == -0.6


class TestComputeLosVector:
    """Tests for compute_los_vector function."""

    def test_vertical_incidence(self):
        """Test LOS vector with vertical incidence."""
        los_vec = compute_los_vector(incidence_angle=0.0, heading=0.0, degrees=True)

        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(los_vec, expected)

    def test_horizontal_incidence(self):
        """Test LOS vector with horizontal incidence."""
        los_vec = compute_los_vector(
            incidence_angle=90.0, heading=0.0, degrees=True  # North
        )

        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(los_vec, expected)

    def test_east_heading(self):
        """Test LOS vector with east heading."""
        los_vec = compute_los_vector(
            incidence_angle=90.0, heading=90.0, degrees=True  # East
        )

        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(los_vec, expected)

    def test_radians_input(self):
        """Test LOS vector computation with radians."""
        los_vec = compute_los_vector(
            incidence_angle=np.pi / 2,  # 90 degrees
            heading=np.pi / 2,  # 90 degrees (east)
            degrees=False,
        )

        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(los_vec, expected)

    def test_typical_sar_geometry(self):
        """Test with typical SAR geometry."""
        # Typical Sentinel-1 incidence angle (~39 degrees)
        los_vec = compute_los_vector(
            incidence_angle=39.0, heading=0.0, degrees=True  # North (ascending)
        )

        # Should have positive north and up components
        assert los_vec[1] > 0  # North component
        assert los_vec[2] > 0  # Up component
        assert abs(los_vec[0]) < 0.1  # East component should be small

    def test_unit_vector_property(self):
        """Test that computed vectors are unit vectors."""
        for incidence in [0, 30, 45, 60, 90]:
            for heading in [0, 90, 180, 270]:
                los_vec = compute_los_vector(incidence, heading, degrees=True)
                norm = np.linalg.norm(los_vec)
                assert abs(norm - 1.0) < 1e-10, f"Not unit vector: {norm}"


class TestApplyLosVectors:
    """Tests for apply_los_vectors function."""

    def test_default_los_vector(self):
        """Test applying default LOS vector to stations."""
        df_stations = pd.DataFrame(
            {
                "station": ["TEST1", "TEST2"],
                "lat": [34.0, 35.0],
                "lon": [-118.0, -119.0],
            },
            index=["TEST1", "TEST2"],
        )

        result = apply_los_vectors(df_stations, satellite="sentinel1_ascending")

        # Should have LOS vector columns
        assert "los_east" in result.columns
        assert "los_north" in result.columns
        assert "los_up" in result.columns

        # All rows should have same LOS vector
        expected_los = get_default_los_vector("sentinel1_ascending")
        assert result["los_east"].iloc[0] == expected_los[0]
        assert result["los_north"].iloc[0] == expected_los[1]
        assert result["los_up"].iloc[0] == expected_los[2]

        # Check that both stations have same LOS vector
        assert result["los_east"].iloc[0] == result["los_east"].iloc[1]

    def test_station_specific_los_vectors(self):
        """Test applying station-specific LOS vectors."""
        df_stations = pd.DataFrame(
            {
                "station": ["TEST1", "TEST2"],
                "lat": [34.0, 35.0],
                "lon": [-118.0, -119.0],
                "incidence": [35.0, 40.0],
                "heading": [0.0, 90.0],
            },
            index=["TEST1", "TEST2"],
        )

        result = apply_los_vectors(
            df_stations, incidence_col="incidence", heading_col="heading"
        )

        # Should have LOS vector columns
        assert "los_east" in result.columns
        assert "los_north" in result.columns
        assert "los_up" in result.columns

        # Different stations should have different LOS vectors
        assert result["los_east"].iloc[0] != result["los_east"].iloc[1]
        assert result["los_north"].iloc[0] != result["los_north"].iloc[1]

    def test_missing_incidence_column(self):
        """Test error when incidence column is missing."""
        df_stations = pd.DataFrame(
            {
                "station": ["TEST1"],
                "lat": [34.0],
                "lon": [-118.0],
                "heading": [0.0],
                # Missing incidence column
            },
            index=["TEST1"],
        )

        with pytest.raises(KeyError):
            apply_los_vectors(
                df_stations, incidence_col="incidence", heading_col="heading"
            )

    def test_missing_heading_column(self):
        """Test error when heading column is missing."""
        df_stations = pd.DataFrame(
            {
                "station": ["TEST1"],
                "lat": [34.0],
                "lon": [-118.0],
                "incidence": [35.0],
                # Missing heading column
            },
            index=["TEST1"],
        )

        with pytest.raises(KeyError):
            apply_los_vectors(
                df_stations, incidence_col="incidence", heading_col="heading"
            )

    def test_partial_incidence_heading(self):
        """Test that providing only one of incidence/heading uses default."""
        df_stations = pd.DataFrame(
            {
                "station": ["TEST1"],
                "lat": [34.0],
                "lon": [-118.0],
                "incidence": [35.0],
                # No heading column
            },
            index=["TEST1"],
        )

        result = apply_los_vectors(
            df_stations,
            satellite="sentinel1_ascending",
            incidence_col="incidence",
            # No heading_col provided
        )

        # Should use default LOS vector
        expected_los = get_default_los_vector("sentinel1_ascending")
        assert result["los_east"].iloc[0] == expected_los[0]

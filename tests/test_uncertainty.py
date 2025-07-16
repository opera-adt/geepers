"""Tests for the uncertainty module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geepers.uncertainty import (
    UncertaintyData,
    build_covariance_matrix,
    get_sigma_los,
)


class TestUncertaintyData:
    """Tests for UncertaintyData Pydantic model."""

    def test_valid_uncertainty_data(self):
        """Test creating valid UncertaintyData model."""
        data = UncertaintyData(
            sigma_east=0.001,
            sigma_north=0.002,
            sigma_up=0.003,
            corr_en=0.1,
            corr_eu=0.2,
            corr_nu=0.3,
        )

        assert data.sigma_east == 0.001
        assert data.sigma_north == 0.002
        assert data.sigma_up == 0.003
        assert data.corr_en == 0.1
        assert data.corr_eu == 0.2
        assert data.corr_nu == 0.3

    def test_default_correlations(self):
        """Test default correlation values."""
        data = UncertaintyData(
            sigma_east=0.001,
            sigma_north=0.002,
            sigma_up=0.003,
        )

        assert data.corr_en == 0.0
        assert data.corr_eu == 0.0
        assert data.corr_nu == 0.0

    def test_invalid_sigma_values(self):
        """Test validation fails for non-positive sigma values."""
        with pytest.raises(ValueError, match="sigma"):
            UncertaintyData(
                sigma_east=-0.001,  # Invalid: negative
                sigma_north=0.002,
                sigma_up=0.003,
            )

    def test_invalid_correlation_values(self):
        """Test validation fails for invalid correlation values."""
        with pytest.raises(ValueError, match="correlation"):
            UncertaintyData(
                sigma_east=0.001,
                sigma_north=0.002,
                sigma_up=0.003,
                corr_en=1.5,  # Invalid: > 1
            )

    def test_from_dataframe_row(self):
        """Test creating UncertaintyData from DataFrame row."""
        row = pd.Series(
            {
                "sigma_east_mm": 1.0,
                "sigma_north_mm": 2.0,
                "sigma_up_mm": 3.0,
                "corr_en": 0.1,
                "corr_eu": 0.2,
                "corr_nu": 0.3,
            }
        )

        data = UncertaintyData.from_dataframe_row(row)

        assert data.sigma_east == 1.0
        assert data.sigma_north == 2.0
        assert data.sigma_up == 3.0
        assert data.corr_en == 0.1
        assert data.corr_eu == 0.2
        assert data.corr_nu == 0.3

    def test_from_dataframe_row_missing_correlations(self):
        """Test creating UncertaintyData from row with missing correlations."""
        row = pd.Series(
            {
                "sigma_east_mm": 1.0,
                "sigma_north_mm": 2.0,
                "sigma_up_mm": 3.0,
                # No correlation columns
            }
        )

        data = UncertaintyData.from_dataframe_row(row)

        assert data.sigma_east == 1.0
        assert data.sigma_north == 2.0
        assert data.sigma_up == 3.0
        assert data.corr_en == 0.0
        assert data.corr_eu == 0.0
        assert data.corr_nu == 0.0

    def test_to_covariance_matrix(self):
        """Test generating covariance matrix from UncertaintyData."""
        data = UncertaintyData(
            sigma_east=0.1,
            sigma_north=0.2,
            sigma_up=0.3,
            corr_en=0.5,
            corr_eu=0.3,
            corr_nu=0.2,
        )

        cov = data.to_covariance_matrix()

        # Check diagonal elements
        np.testing.assert_almost_equal(cov[0, 0], 0.01)  # sigma_east^2
        np.testing.assert_almost_equal(cov[1, 1], 0.04)  # sigma_north^2
        np.testing.assert_almost_equal(cov[2, 2], 0.09)  # sigma_up^2

        # Check off-diagonal elements
        np.testing.assert_almost_equal(
            cov[0, 1], 0.1 * 0.2 * 0.5
        )  # sigma_east * sigma_north * corr_en

        # Check symmetry
        np.testing.assert_array_equal(cov, cov.T)


class TestBuildCovarianceMatrix:
    """Tests for build_covariance_matrix function."""

    def test_diagonal_matrix(self):
        """Test building diagonal covariance matrix."""
        cov = build_covariance_matrix(
            sigma_east=0.1,
            sigma_north=0.2,
            sigma_up=0.3,
        )

        expected = np.array([[0.01, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.09]])

        np.testing.assert_array_almost_equal(cov, expected)

    def test_full_covariance_matrix(self):
        """Test building full covariance matrix with correlations."""
        cov = build_covariance_matrix(
            sigma_east=0.1,
            sigma_north=0.2,
            sigma_up=0.3,
            corr_en=0.5,
            corr_eu=0.3,
            corr_nu=0.2,
        )

        # Check diagonal elements
        np.testing.assert_almost_equal(cov[0, 0], 0.01)  # sigma_east^2
        np.testing.assert_almost_equal(cov[1, 1], 0.04)  # sigma_north^2
        np.testing.assert_almost_equal(cov[2, 2], 0.09)  # sigma_up^2

        # Check off-diagonal elements
        np.testing.assert_almost_equal(
            cov[0, 1], 0.1 * 0.2 * 0.5
        )  # sigma_east * sigma_north * corr_en
        np.testing.assert_almost_equal(
            cov[0, 2], 0.1 * 0.3 * 0.3
        )  # sigma_east * sigma_up * corr_eu
        np.testing.assert_almost_equal(
            cov[1, 2], 0.2 * 0.3 * 0.2
        )  # sigma_north * sigma_up * corr_nu

        # Check symmetry
        np.testing.assert_array_equal(cov, cov.T)


class TestSigmaLOS:
    """Tests for sigma_los function."""

    def test_analytical_case(self):
        """Test LOS sigma computation with analytical case."""
        # Simple case: vertical LOS vector, only up component matters
        df = pd.DataFrame(
            {
                "sigma_east_mm": [1.0],
                "sigma_north_mm": [2.0],
                "sigma_up_mm": [3.0],
                "corr_en": [0.0],
                "corr_eu": [0.0],
                "corr_nu": [0.0],
            }
        )

        los_vector = np.array([0.0, 0.0, 1.0])  # Pure vertical
        result = get_sigma_los(df, los_vector)

        # Should equal the up component sigma
        assert result.iloc[0] == 3.0

    def test_horizontal_los(self):
        """Test LOS sigma with horizontal vector."""
        df = pd.DataFrame(
            {
                "sigma_east_mm": [1.0],
                "sigma_north_mm": [2.0],
                "sigma_up_mm": [3.0],
                "corr_en": [0.0],
                "corr_eu": [0.0],
                "corr_nu": [0.0],
            }
        )

        los_vector = np.array([1.0, 0.0, 0.0])  # Pure east
        result = get_sigma_los(df, los_vector)

        # Should equal the east component sigma
        assert result.iloc[0] == 1.0

    def test_typical_sar_geometry(self):
        """Test with typical SAR geometry."""
        df = pd.DataFrame(
            {
                "sigma_east_mm": [1.0],
                "sigma_north_mm": [1.0],
                "sigma_up_mm": [2.0],
                "corr_en": [0.0],
                "corr_eu": [0.0],
                "corr_nu": [0.0],
            }
        )

        # Typical Sentinel-1 LOS vector (approximate)
        los_vector = np.array([-0.6, 0.0, 0.8])
        result = get_sigma_los(df, los_vector)

        # Should be somewhere between east and up sigmas
        assert 1.0 < result.iloc[0] < 2.0

    def test_missing_correlation_columns(self):
        """Test LOS sigma with missing correlation columns."""
        df = pd.DataFrame(
            {
                "sigma_east_mm": [1.0],
                "sigma_north_mm": [1.0],
                "sigma_up_mm": [2.0],
                # No correlation columns - should default to 0.0
            }
        )

        los_vector = np.array([0.0, 0.0, 1.0])  # Pure vertical
        result = get_sigma_los(df, los_vector)

        # Should still work and equal the up component sigma
        assert result.iloc[0] == 2.0

    def test_invalid_los_vector(self):
        """Test error with invalid LOS vector."""
        df = pd.DataFrame(
            {
                "sigma_east_mm": [1.0],
                "sigma_north_mm": [1.0],
                "sigma_up_mm": [2.0],
            }
        )

        los_vector = np.array([1.0, 0.0])  # Wrong size

        with pytest.raises(ValueError, match="Wrong size"):
            get_sigma_los(df, los_vector)

    def test_missing_sigma_columns(self):
        """Test error when required sigma columns are missing."""
        df = pd.DataFrame(
            {
                "sigma_east_mm": [1.0],
                # Missing sigma_north_mm and sigma_up_mm
            }
        )

        los_vector = np.array([0.0, 0.0, 1.0])

        with pytest.raises(KeyError):
            get_sigma_los(df, los_vector)

    def test_multiple_rows(self):
        """Test sigma_los with multiple rows."""
        df = pd.DataFrame(
            {
                "sigma_east_mm": [1.0, 2.0],
                "sigma_north_mm": [1.0, 2.0],
                "sigma_up_mm": [2.0, 3.0],
                "corr_en": [0.0, 0.0],
                "corr_eu": [0.0, 0.0],
                "corr_nu": [0.0, 0.0],
            }
        )

        los_vector = np.array([0.0, 0.0, 1.0])  # Pure vertical
        result = get_sigma_los(df, los_vector)

        assert len(result) == 2
        assert result.iloc[0] == 2.0  # First row up sigma
        assert result.iloc[1] == 3.0  # Second row up sigma

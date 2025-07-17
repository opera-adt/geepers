"""Tests for the uncertainty module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geepers.uncertainty import (
    build_covariance_matrix,
    get_sigma_los,
    get_sigma_los_df,
)


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
        np.testing.assert_almost_equal(cov[0, 1], 0.5 * 0.1 * 0.2)
        np.testing.assert_almost_equal(cov[0, 2], 0.3 * 0.1 * 0.3)
        np.testing.assert_almost_equal(cov[1, 2], 0.2 * 0.2 * 0.3)
        # Check symmetry
        np.testing.assert_array_equal(cov, cov.T)


class TestSigmaLOS:
    """Tests for sigma_los function."""

    def test_vertical_only(self):
        """Test LOS sigma computation with analytical case."""
        # Simple case: vertical LOS vector, only up component matters
        sigma_east = 1.0
        sigma_north = 2.0
        sigma_up = 3.0
        corr_en = 0.0
        corr_eu = 0.0
        corr_nu = 0.0

        los_vector = np.array([0.0, 0.0, 1.0])  # Pure vertical
        result = get_sigma_los(
            los_vector=los_vector,
            sigma_east=sigma_east,
            sigma_north=sigma_north,
            sigma_up=sigma_up,
            corr_en=corr_en,
            corr_eu=corr_eu,
            corr_nu=corr_nu,
        )

        # Should equal the up component sigma
        assert result == 3.0

    def test_vertical_only_df(self):
        """Test LOS sigma computation with analytical case."""
        # Simple case: vertical LOS vector, only up component matters
        df = pd.DataFrame(
            {
                "sigma_east": [1.0],
                "sigma_north": [2.0],
                "sigma_up": [3.0],
                "corr_en": [0.0],
                "corr_eu": [0.0],
                "corr_nu": [0.0],
            }
        )

        los_vector = np.array([0.0, 0.0, 1.0])  # Pure vertical
        result = get_sigma_los_df(
            df=df,
            los_vector=los_vector,
        )

        # Should equal the up component sigma
        assert result.iloc[0] == 3.0

    def test_horizontal_los(self):
        """Test LOS sigma with horizontal vector."""
        df = pd.DataFrame(
            {
                "sigma_east": [1.0],
                "sigma_north": [2.0],
                "sigma_up": [3.0],
                "corr_en": [0.0],
                "corr_eu": [0.0],
                "corr_nu": [0.0],
            }
        )

        los_vector = np.array([1.0, 0.0, 0.0])  # Pure east
        result = get_sigma_los_df(
            df=df,
            los_vector=los_vector,
        )

        # Should equal the east component sigma
        assert result.iloc[0] == 1.0

    def test_typical_sar_geometry(self):
        """Test with typical SAR geometry."""
        df = pd.DataFrame(
            {
                "sigma_east": [1.0],
                "sigma_north": [1.0],
                "sigma_up": [2.0],
                "corr_en": [0.0],
                "corr_eu": [0.0],
                "corr_nu": [0.0],
            }
        )

        # Typical Sentinel-1 LOS vector (approximate)
        los_vector = np.array([-0.6, 0.1, 0.75])
        result = get_sigma_los_df(df, los_vector)

        # Should be between the largest and smallest component sigmas
        assert 1.0 < result.iloc[0] < 2.0

    def test_invalid_los_vector(self):
        """Test error with invalid LOS vector."""
        df = pd.DataFrame(
            {
                "sigma_east": [1.0],
                "sigma_north": [1.0],
                "sigma_up": [2.0],
            }
        )

        los_vector = np.array([1.0, 0.0])  # Wrong size

        with pytest.raises(ValueError, match="los_vector must be a 3-element array"):
            get_sigma_los_df(df, los_vector)

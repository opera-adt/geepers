"""Uncertainty propagation and LOS sigma calculation utilities.

This module provides Pydantic models and functions for uncertainty propagation
and line-of-sight (LOS) sigma calculations for GPS and InSAR comparisons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

__all__ = [
    "UncertaintyData",
    "build_covariance_matrix",
    "get_sigma_los",
]


class UncertaintyData(BaseModel):
    """Pydantic model for ENU uncertainty data with correlations.

    This model represents the uncertainty information for a single observation,
    including standard deviations in East, North, Up directions and their
    correlation coefficients.
    """

    sigma_east: float = Field(gt=0, description="Standard deviation in east direction")
    sigma_north: float = Field(
        gt=0, description="Standard deviation in north direction"
    )
    sigma_up: float = Field(gt=0, description="Standard deviation in up direction")
    corr_en: float = Field(
        default=0.0, ge=-1, le=1, description="East-North correlation coefficient"
    )
    corr_eu: float = Field(
        default=0.0, ge=-1, le=1, description="East-Up correlation coefficient"
    )
    corr_nu: float = Field(
        default=0.0, ge=-1, le=1, description="North-Up correlation coefficient"
    )

    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> UncertaintyData:
        """Create UncertaintyData from a pandas DataFrame row.

        Parameters
        ----------
        row : pd.Series
            DataFrame row with uncertainty columns. Expected columns:
            sigma_east_mm, sigma_north_mm, sigma_up_mm, corr_en, corr_eu, corr_nu
            OR sigma_east, sigma_north, sigma_up, corr_en, corr_eu, corr_nu

        Returns
        -------
        UncertaintyData
            Validated uncertainty data model

        """
        # Try _mm columns first, fallback to raw column names
        if "sigma_east_mm" in row:
            sigma_east = row["sigma_east_mm"]
            sigma_north = row["sigma_north_mm"]
            sigma_up = row["sigma_up_mm"]
        else:
            sigma_east = row["sigma_east"]
            sigma_north = row["sigma_north"]
            sigma_up = row["sigma_up"]

        return cls(
            sigma_east=sigma_east,
            sigma_north=sigma_north,
            sigma_up=sigma_up,
            corr_en=row.get("corr_en", 0.0),
            corr_eu=row.get("corr_eu", 0.0),
            corr_nu=row.get("corr_nu", 0.0),
        )

    def to_covariance_matrix(self) -> np.ndarray:
        """Build 3x3 covariance matrix from this uncertainty data.

        Returns
        -------
        np.ndarray
            3x3 covariance matrix with structure:
            [[σ_E², σ_E*σ_N*ρ_EN, σ_E*σ_U*ρ_EU],
             [σ_E*σ_N*ρ_EN, σ_N², σ_N*σ_U*ρ_NU],
             [σ_E*σ_U*ρ_EU, σ_N*σ_U*ρ_NU, σ_U²]]

        """
        return build_covariance_matrix(
            self.sigma_east,
            self.sigma_north,
            self.sigma_up,
            self.corr_en,
            self.corr_eu,
            self.corr_nu,
        )


def build_covariance_matrix(
    sigma_east: float,
    sigma_north: float,
    sigma_up: float,
    corr_en: float = 0.0,
    corr_eu: float = 0.0,
    corr_nu: float = 0.0,
) -> np.ndarray:
    """Build 3x3 covariance matrix from standard deviations and correlations.

    Parameters
    ----------
    sigma_east : float
        Standard deviation in east direction.
    sigma_north : float
        Standard deviation in north direction.
    sigma_up : float
        Standard deviation in up direction.
    corr_en : float, optional
        Correlation coefficient between east and north. Default is 0.0.
    corr_eu : float, optional
        Correlation coefficient between east and up. Default is 0.0.
    corr_nu : float, optional
        Correlation coefficient between north and up. Default is 0.0.

    Returns
    -------
    np.ndarray
        3x3 covariance matrix with structure:
        [[σ_E², σ_E*σ_N*ρ_EN, σ_E*σ_U*ρ_EU],
         [σ_E*σ_N*ρ_EN, σ_N², σ_N*σ_U*ρ_NU],
         [σ_E*σ_U*ρ_EU, σ_N*σ_U*ρ_NU, σ_U²]]

    """
    # Build covariance matrix
    cov_matrix = np.array(
        [
            [
                sigma_east**2,
                sigma_east * sigma_north * corr_en,
                sigma_east * sigma_up * corr_eu,
            ],
            [
                sigma_east * sigma_north * corr_en,
                sigma_north**2,
                sigma_north * sigma_up * corr_nu,
            ],
            [
                sigma_east * sigma_up * corr_eu,
                sigma_north * sigma_up * corr_nu,
                sigma_up**2,
            ],
        ]
    )

    return cov_matrix


def get_sigma_los(
    df: pd.DataFrame,
    los_vector: np.ndarray | pd.Series,
) -> pd.Series:
    """Compute line-of-sight (LOS) uncertainty: u^T Σ u.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with standardized uncertainty columns
        Expected columns: sigma_east_mm, sigma_north_mm, sigma_up_mm, and
        optionally corr_en, corr_eu, corr_nu.
    los_vector : np.ndarray or pd.Series
        Unit vector toward the satellite/LOS direction of shape (3,).
        Components are (u_east, u_north, u_up).

    Returns
    -------
    pd.Series
        LOS uncertainty (σ_LOS) for each row in the DataFrame.

    Raises
    ------
    ValueError
        If los_vector is not a 3-element array.
    KeyError
        If required uncertainty columns are missing.

    Notes
    -----
    The LOS uncertainty is computed as:
    σ_LOS² = u^T Σ u
    where u is the unit LOS vector and Σ is the 3x3 ENU covariance matrix.

    """
    # Validate LOS vector
    los_vector = np.asarray(los_vector)
    if los_vector.shape != (3,):
        msg = f"los_vector must be a 3-element array, got shape {los_vector.shape}"
        raise ValueError(msg)

    # Compute LOS uncertainty for each row using UncertaintyData model
    los_uncertainties = []
    u = los_vector.reshape(3, 1)  # Column vector

    for _, row in df.iterrows():
        # Create UncertaintyData model from row (validates data)
        uncertainty_data = UncertaintyData.from_dataframe_row(row)

        # Get covariance matrix from the model
        cov_matrix = uncertainty_data.to_covariance_matrix()

        # Compute sigma_LOS^2 = u^T Sigma u
        los_var = (u.T @ cov_matrix @ u)[0, 0]
        los_uncertainties.append(np.sqrt(los_var))

    return pd.Series(los_uncertainties, index=df.index)

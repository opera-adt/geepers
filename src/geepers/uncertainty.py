"""Uncertainty propagation and LOS sigma calculation utilities.

This module provides Pydantic models and functions for uncertainty propagation
and line-of-sight (LOS) sigma calculations for GPS and InSAR comparisons.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from .schemas import StationUncertaintySchema

FloatOrArrayT = TypeVar("FloatOrArrayT", bound=float | pd.Series | np.ndarray)

__all__ = [
    "build_covariance_matrix",
    "get_sigma_los",
    "get_sigma_los_df",
]


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
    return np.array(
        [
            [
                sigma_east**2,
                corr_en * sigma_east * sigma_north,
                corr_eu * sigma_east * sigma_up,
            ],
            [
                corr_en * sigma_east * sigma_north,
                sigma_north**2,
                corr_nu * sigma_north * sigma_up,
            ],
            [
                corr_eu * sigma_east * sigma_up,
                corr_nu * sigma_north * sigma_up,
                sigma_up**2,
            ],
        ]
    )


def get_sigma_los(
    los_vector: np.ndarray | pd.Series,
    sigma_east: FloatOrArrayT,
    sigma_north: FloatOrArrayT,
    sigma_up: FloatOrArrayT,
    corr_en: FloatOrArrayT,
    corr_eu: FloatOrArrayT,
    corr_nu: FloatOrArrayT,
) -> FloatOrArrayT:
    """Compute line-of-sight (LOS) uncertainty: u^T Σ u.

    Parameters
    ----------
    los_vector : np.ndarray or pd.Series
        Unit vector toward the satellite/LOS direction of shape (3,).
        Components are (u_east, u_north, u_up).
    sigma_east : float
        Standard deviation in east direction.
    sigma_north : float
        Standard deviation in north direction.
    sigma_up : float
        Standard deviation in up direction.
    corr_en : float
        Correlation coefficient between east and north.
    corr_eu : float
        Correlation coefficient between east and up.
    corr_nu : float
        Correlation coefficient between north and up.

    Returns
    -------
    float
        LOS standard deviation (σ_LOS).

    """
    u_e, u_n, u_u = los_vector
    # For faster broadcasting, we unpack the u^T @ Sigma @ u formula
    # which can be verified with sympy:
    # In [18]: (u.T @ Sigma @ u)[0, 0]
    # Out[18]: u_e*(sigma_en*u_n + sigma_ev*u_v + sigma_e*u_e) + ...

    sigma_en = corr_en * sigma_east * sigma_north  # type: ignore[operator]
    sigma_eu = corr_eu * sigma_east * sigma_up  # type: ignore[operator]
    sigma_nu = corr_nu * sigma_north * sigma_up  # type: ignore[operator]
    los_variance = (
        u_e * (sigma_east**2 * u_e + sigma_en * u_n + sigma_eu * u_u)
        + u_n * (sigma_en * u_e + sigma_north**2 * u_n + sigma_nu * u_u)
        + u_u * (sigma_eu * u_e + sigma_nu * u_n + sigma_up**2 * u_u)
    )
    return np.sqrt(los_variance)


def get_sigma_los_df(
    df: DataFrame[StationUncertaintySchema],
    los_vector: np.ndarray | pd.Series,
) -> pd.Series:
    """Compute line-of-sight (LOS) uncertainty: u^T Σ u.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with standardized uncertainty columns
        Expected columns: sigma_east, sigma_north, sigma_up, corr_en, corr_eu, corr_nu.
    los_vector : np.ndarray or pd.Series
        Unit vector toward the satellite/LOS direction of shape (3,).
        Components are (u_east, u_north, u_up).

    Returns
    -------
    pd.Series
        LOS standard deviation (σ_LOS) for each row in the DataFrame.

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
    if np.asarray(los_vector).shape != (3,):
        msg = f"los_vector must be a 3-element array, got shape {los_vector.shape}"
        raise ValueError(msg)
    return get_sigma_los(
        los_vector=los_vector,
        sigma_east=df["sigma_east"],
        sigma_north=df["sigma_north"],
        sigma_up=df["sigma_up"],
        corr_en=df["corr_en"],
        corr_eu=df["corr_eu"],
        corr_nu=df["corr_nu"],
    )

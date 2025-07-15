"""Line-of-sight (LOS) vector utilities for GPS-InSAR comparisons.

This module provides default LOS unit vectors for common satellite configurations
and utilities to compute LOS vectors from incidence angles and headings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_LOS_VECTORS",
    "apply_los_vectors",
    "compute_los_vector",
    "get_default_los_vector",
]


# Default LOS unit vectors for common satellite configurations
# Convention: (u_east, u_north, u_up) where positive up is away from Earth
DEFAULT_LOS_VECTORS = {
    "sentinel1_ascending": np.array([-0.6, 0.0, 0.8]),  # Typical ascending pass
    "sentinel1_descending": np.array([0.6, 0.0, 0.8]),  # Typical descending pass
    "envisat_ascending": np.array([-0.6, 0.0, 0.8]),  # Similar to Sentinel-1
    "envisat_descending": np.array([0.6, 0.0, 0.8]),  # Similar to Sentinel-1
    "cosmo_ascending": np.array([-0.6, 0.0, 0.8]),  # Similar geometry
    "cosmo_descending": np.array([0.6, 0.0, 0.8]),  # Similar geometry
}


def get_default_los_vector(satellite: str = "sentinel1_ascending") -> np.ndarray:
    """Get a default LOS unit vector for common satellite configurations.

    Parameters
    ----------
    satellite : str, optional
        Satellite and pass direction. Default is "sentinel1_ascending".
        Available options: "sentinel1_ascending", "sentinel1_descending",
        "envisat_ascending", "envisat_descending", "cosmo_ascending",
        "cosmo_descending".

    Returns
    -------
    np.ndarray
        3-element LOS unit vector (u_east, u_north, u_up).

    Raises
    ------
    KeyError
        If the satellite configuration is not recognized.

    Notes
    -----
    These are typical values for mid-latitude regions. For precise analysis,
    use compute_los_vector() with actual incidence angles and headings.

    """
    if satellite not in DEFAULT_LOS_VECTORS:
        available = list(DEFAULT_LOS_VECTORS.keys())
        msg = f"Unknown satellite '{satellite}'. Available: {available}"
        raise KeyError(msg)

    return DEFAULT_LOS_VECTORS[satellite].copy()


def compute_los_vector(
    incidence_angle: float,
    heading: float,
    degrees: bool = True,
) -> np.ndarray:
    """Compute LOS unit vector from incidence angle and satellite heading.

    Parameters
    ----------
    incidence_angle : float
        Incidence angle (angle between radar beam and vertical).
    heading : float
        Satellite heading/azimuth angle (0° = north, 90° = east).
    degrees : bool, optional
        If True, angles are in degrees. If False, angles are in radians.
        Default is True.

    Returns
    -------
    np.ndarray
        3-element LOS unit vector (u_east, u_north, u_up).

    Notes
    -----
    The LOS vector points from the ground target toward the satellite.
    For SAR, this is typically computed as:
    - u_east = sin(incidence) * sin(heading)
    - u_north = sin(incidence) * cos(heading)
    - u_up = cos(incidence)

    """
    if degrees:
        incidence_rad = np.radians(incidence_angle)
        heading_rad = np.radians(heading)
    else:
        incidence_rad = incidence_angle
        heading_rad = heading

    # Compute LOS unit vector components
    u_east = np.sin(incidence_rad) * np.sin(heading_rad)
    u_north = np.sin(incidence_rad) * np.cos(heading_rad)
    u_up = np.cos(incidence_rad)

    return np.array([u_east, u_north, u_up])


def apply_los_vectors(
    df_stations: pd.DataFrame,
    satellite: str = "sentinel1_ascending",
    incidence_col: str | None = None,
    heading_col: str | None = None,
) -> pd.DataFrame:
    """Apply LOS vectors to a DataFrame of station locations.

    Parameters
    ----------
    df_stations : pd.DataFrame
        DataFrame with station information, indexed by station name.
    satellite : str, optional
        Default satellite configuration if incidence/heading not provided.
        Default is "sentinel1_ascending".
    incidence_col : str, optional
        Column name for incidence angles. If None, uses default LOS vector.
    heading_col : str, optional
        Column name for satellite headings. If None, uses default LOS vector.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: los_east, los_north, los_up.

    Notes
    -----
    If incidence_col and heading_col are provided, station-specific LOS vectors
    are computed. Otherwise, the default LOS vector is used for all stations.

    """
    df = df_stations.copy()

    if incidence_col and heading_col:
        # Compute station-specific LOS vectors
        if incidence_col not in df.columns:
            msg = f"Column '{incidence_col}' not found in DataFrame"
            raise KeyError(msg)
        if heading_col not in df.columns:
            msg = f"Column '{heading_col}' not found in DataFrame"
            raise KeyError(msg)

        los_vectors = []
        for _, row in df.iterrows():
            los_vec = compute_los_vector(row[incidence_col], row[heading_col])
            los_vectors.append(los_vec)

        los_array = np.array(los_vectors)
        df["los_east"] = los_array[:, 0]
        df["los_north"] = los_array[:, 1]
        df["los_up"] = los_array[:, 2]

    else:
        # Use default LOS vector for all stations
        default_los = get_default_los_vector(satellite)
        df["los_east"] = default_los[0]
        df["los_north"] = default_los[1]
        df["los_up"] = default_los[2]

    return df

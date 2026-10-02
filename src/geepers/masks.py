# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Masks for interpolated grids: keep only nodes supported by stations.

Every interpolator in geepers returns a value at every requested point,
however far from the data. These helpers flag the grid nodes that are
close enough to stations to be trusted, so extrapolated velocities (and
the strain rates derived from them) are not reported.

The two criteria follow ``verde.distance_mask`` and
``verde.convexhull_mask`` (Uieda, L., 2018, Verde: Processing and
gridding spatial data using Green's functions, JOSS, 3(29), 957,
https://doi.org/10.21105/joss.00957), computed here on the sphere /
in an equal-area plane from lon/lat input.

Example:
-------
>>> keep = distance_mask(lon, lat, lon_grid, lat_grid, 75)  # doctest: +SKIP
>>> ve_grid = np.where(keep, ve_grid, np.nan)               # doctest: +SKIP

"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from pyproj import Transformer
from scipy.spatial import cKDTree

__all__ = ["convex_hull_mask", "distance_mask"]

EARTH_RADIUS_KM = 6371.0


def _unit_vectors(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Geocentric unit vectors of points on a spherical Earth, shape (n, 3)."""
    lam, phi = np.radians(lon), np.radians(lat)
    return np.c_[np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]


def distance_mask(
    lon: ArrayLike,
    lat: ArrayLike,
    lon_new: ArrayLike,
    lat_new: ArrayLike,
    max_distance_km: float,
    *,
    min_stations: int = 1,
) -> np.ndarray:
    """Flag points that have enough stations within a given distance.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    lon_new, lat_new : array-like
        Points to test (e.g. grid nodes), any matching shape, degrees.
    max_distance_km : float
        Great-circle search radius in kilometers.
    min_stations : int
        Number of stations required within the radius. Default 1, i.e.
        the nearest station must be closer than `max_distance_km`.

    Returns
    -------
    np.ndarray of bool
        True where the point is supported by the stations; same shape as
        `lon_new`.

    Examples
    --------
    >>> distance_mask([0.0], [0.0], [0.1, 5.0], [0.0, 0.0], 100).tolist()
    [True, False]

    """
    lon_new = np.asarray(lon_new, float)
    lat_new = np.asarray(lat_new, float)
    shape = np.broadcast(lon_new, lat_new).shape
    stations = _unit_vectors(
        np.asarray(lon, float).ravel(), np.asarray(lat, float).ravel()
    )
    points = _unit_vectors(
        np.broadcast_to(lon_new, shape).ravel(), np.broadcast_to(lat_new, shape).ravel()
    )
    # A great-circle distance d corresponds to the chord 2 sin(d / 2R)
    # between unit vectors, so a Euclidean tree gives exact spherical radii
    chord = 2.0 * np.sin(min(max_distance_km / EARTH_RADIUS_KM, np.pi) / 2.0)
    dist, _ = cKDTree(stations).query(points, k=[min_stations])
    return (dist[:, 0] <= chord).reshape(shape)


def convex_hull_mask(
    lon: ArrayLike,
    lat: ArrayLike,
    lon_new: ArrayLike,
    lat_new: ArrayLike,
    *,
    buffer_km: float = 0.0,
) -> np.ndarray:
    """Flag points inside the convex hull of the stations.

    Inside the hull the field is interpolated; outside it is
    extrapolated.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    lon_new, lat_new : array-like
        Points to test (e.g. grid nodes), any matching shape, degrees.
    buffer_km : float
        Grow (positive) or shrink (negative) the hull by this many
        kilometers before testing. Default 0.

    Returns
    -------
    np.ndarray of bool
        True where the point lies inside the (buffered) hull; same shape
        as `lon_new`.

    """
    import shapely
    from shapely.geometry import MultiPoint

    lon = np.asarray(lon, float).ravel()
    lat = np.asarray(lat, float).ravel()
    lon_new = np.asarray(lon_new, float)
    lat_new = np.asarray(lat_new, float)
    shape = np.broadcast(lon_new, lat_new).shape

    # The hull is built in a plane; lon/lat edges would not be straight
    # on the ground and the buffer needs metric units
    transformer = Transformer.from_crs(
        "EPSG:4326",
        {
            "proj": "laea",
            "lat_0": float(lat.mean()),
            "lon_0": float(lon.mean()),
            "datum": "WGS84",
            "units": "km",
        },
        always_xy=True,
    )
    x, y = transformer.transform(lon, lat)
    hull = MultiPoint(np.c_[x, y]).convex_hull
    if buffer_km:
        hull = hull.buffer(buffer_km)
    x_new, y_new = transformer.transform(
        np.broadcast_to(lon_new, shape).ravel(), np.broadcast_to(lat_new, shape).ravel()
    )
    inside = shapely.intersects_xy(hull, x_new, y_new)
    return np.asarray(inside).reshape(shape)

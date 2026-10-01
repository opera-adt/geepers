"""Polynomial trend surfaces for remove-restore interpolation.

Least-squares collocation and the elastic spline model a *zero-mean*
signal: a regional offset or tilt left in the data is pulled toward zero
away from the stations and biases the empirical covariance. The standard
remedy is remove-restore: fit a low-degree polynomial surface, interpolate
the residuals, and add the surface back at the prediction points. For
horizontal velocities the rigid plate rotation plays that role (see
`geepers.euler`); this module covers scalar fields such as vertical
velocities, and any trend an Euler pole does not describe.

The surface is a full bivariate polynomial in local east/north
coordinates, as in ``verde.Trend`` (Uieda, L., 2018, Verde: Processing
and gridding spatial data using Green's functions, JOSS, 3(29), 957,
https://doi.org/10.21105/joss.00957).

Example:
-------
>>> surface = fit_polynomial_surface(lon, lat, vu, sigma_u, degree=1)  # doctest: +SKIP
>>> signal = interpolate(lon, lat, surface.residuals, lon_g, lat_g)    # doctest: +SKIP
>>> vu_g = signal + surface.predict(lon_g, lat_g)                      # doctest: +SKIP

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from pyproj import Transformer

__all__ = ["PolynomialSurface", "fit_polynomial_surface"]


def _powers(degree: int) -> list[tuple[int, int]]:
    """Exponent pairs (i, j) of every term x^i y^j with i + j <= degree."""
    return [(i, total - i) for total in range(degree + 1) for i in range(total, -1, -1)]


def _local_xy(
    lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float, scale_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled local east/north coordinates about (lon0, lat0)."""
    transformer = Transformer.from_crs(
        "EPSG:4326",
        {"proj": "aeqd", "lat_0": lat0, "lon_0": lon0, "datum": "WGS84", "units": "km"},
        always_xy=True,
    )
    x, y = transformer.transform(lon, lat)
    return np.asarray(x) / scale_km, np.asarray(y) / scale_km


def _design_matrix(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([x**i * y**j for i, j in _powers(degree)])


@dataclass
class PolynomialSurface:
    """A fitted polynomial trend surface.

    Attributes
    ----------
    coefficients : np.ndarray
        Coefficient of each term in `terms`, for coordinates in units of
        `scale_km`. The first is the value at (`lon0`, `lat0`).
    terms : list of (int, int)
        Powers ``(i, j)`` of the east/north term ``x**i * y**j``.
    degree : int
        Polynomial degree.
    lon0, lat0 : float
        Origin of the local coordinates, degrees.
    scale_km : float
        Kilometers per unit of the scaled coordinates (the network
        half-extent), which keeps the fit well conditioned.
    residuals : np.ndarray
        ``values - surface`` at the fitted points.

    """

    coefficients: np.ndarray
    terms: list[tuple[int, int]]
    degree: int
    lon0: float
    lat0: float
    scale_km: float
    residuals: np.ndarray

    def predict(self, lon: ArrayLike, lat: ArrayLike) -> np.ndarray:
        """Evaluate the surface.

        Parameters
        ----------
        lon, lat : array-like
            Points in degrees (any matching shape).

        Returns
        -------
        np.ndarray
            Surface values, same shape as the input.

        """
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        shape = np.broadcast(lon, lat).shape
        x, y = _local_xy(
            np.broadcast_to(lon, shape).ravel(),
            np.broadcast_to(lat, shape).ravel(),
            self.lon0,
            self.lat0,
            self.scale_km,
        )
        return (_design_matrix(x, y, self.degree) @ self.coefficients).reshape(shape)


def fit_polynomial_surface(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    sigmas: ArrayLike | None = None,
    *,
    degree: int = 1,
) -> PolynomialSurface:
    """Fit a polynomial trend surface to scattered values.

    Parameters
    ----------
    lon, lat : array-like
        Point coordinates in degrees.
    values : array-like
        Values at the points (e.g. vertical velocities). NaNs are not
        allowed.
    sigmas : array-like, optional
        1-sigma uncertainties; the fit is weighted by ``1 / sigma**2``.
    degree : int
        Polynomial degree: 0 a constant (weighted mean), 1 a tilted
        plane (default), 2 adds curvature. Keep it low - a high-degree
        surface absorbs the signal that is to be interpolated and
        diverges outside the network.

    Returns
    -------
    PolynomialSurface
        Call `PolynomialSurface.predict` to restore the trend at new
        points; `residuals` holds the de-trended values.

    Raises
    ------
    ValueError
        If `degree` is negative or there are fewer points than
        polynomial terms.

    """
    lon = np.asarray(lon, float).ravel()
    lat = np.asarray(lat, float).ravel()
    values = np.asarray(values, float).ravel()
    if degree < 0:
        msg = f"degree must be >= 0, got {degree}"
        raise ValueError(msg)
    terms = _powers(degree)
    if values.size < len(terms):
        msg = (
            f"A degree-{degree} surface has {len(terms)} terms but only "
            f"{values.size} points were given"
        )
        raise ValueError(msg)

    lon0, lat0 = float(lon.mean()), float(lat.mean())
    x_km, y_km = _local_xy(lon, lat, lon0, lat0, 1.0)
    # A single point (degree 0) has no extent; any positive scale works
    scale_km = float(max(np.abs(x_km).max(), np.abs(y_km).max())) or 1.0
    x, y = x_km / scale_km, y_km / scale_km

    design = _design_matrix(x, y, degree)
    a, b = design, values
    if sigmas is not None:
        w = 1.0 / np.asarray(sigmas, float).ravel()
        a, b = design * w[:, None], values * w
    coefficients, *_ = np.linalg.lstsq(a, b, rcond=None)

    return PolynomialSurface(
        coefficients=coefficients,
        terms=terms,
        degree=degree,
        lon0=lon0,
        lat0=lat0,
        scale_km=scale_km,
        residuals=values - design @ coefficients,
    )

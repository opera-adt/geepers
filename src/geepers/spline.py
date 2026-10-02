# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-FileCopyrightText: 2017 The Verde Developers (adapted parts; BSD-3-Clause, see LICENSES/)
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
r"""Elastically coupled spline interpolation of horizontal velocities.

Interpolates scattered east/north GNSS velocities *jointly*: the field is
modeled as the response of a thin elastic sheet to in-plane point forces
located at the stations, so the two components are coupled through
Poisson's ratio instead of being gridded independently. This is the
method behind GMT's ``gpsgridder`` and suits deforming zones, where the
rigid-rotation coupling of `geepers.collocation` does not apply.

The Green's functions and the damped least-squares fit follow

    Sandwell, D. T., & Wessel, P. (2016). Interpolation of 2-D vector
    data using constraints from elasticity. Geophysical Research
    Letters, 43(20), 10703-10709. https://doi.org/10.1002/2016GL070340

and the implementation is adapted from ``verde.VectorSpline2D`` (Uieda,
L., 2018, Verde: Processing and gridding spatial data using Green's
functions, JOSS, 3(29), 957, https://doi.org/10.21105/joss.00957;
BSD-3-Clause).

Unlike collocation, the spline has no formal prediction uncertainty;
use held-out stations to judge it and to choose `damping`.

Example:
-------
>>> spline = fit_vector_spline(lon, lat, ve, vn, se, sn, damping=0.1)  # doctest: +SKIP
>>> ve_grid, vn_grid = spline.predict(lon_grid, lat_grid)            # doctest: +SKIP

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from pyproj import Transformer
from scipy import linalg

__all__ = ["VectorSpline", "fit_vector_spline"]

# Prediction points per block, bounding the (points x forces) work arrays
_PREDICT_CHUNK = 20_000


def _greens_functions(
    dx: np.ndarray, dy: np.ndarray, mindist: float, poisson: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elastic Green's functions (Sandwell & Wessel, 2016, eq. 5).

    `dx`, `dy` are the offsets from a force to a point. Returns the
    east-east, north-north and east-north responses.
    """
    # The fudge distance keeps the logarithm and 1/r^2 finite at the forces
    r = np.hypot(dx, dy) + mindist
    ln_r = (3.0 - poisson) * np.log(r)
    over_r2 = (1.0 + poisson) / r**2
    return ln_r + over_r2 * dy**2, ln_r + over_r2 * dx**2, -over_r2 * dx * dy


def _jacobian(
    x: np.ndarray,
    y: np.ndarray,
    force_x: np.ndarray,
    force_y: np.ndarray,
    mindist: float,
    poisson: float,
) -> np.ndarray:
    """Design matrix mapping stacked (east, north) forces to stacked data."""
    g_ee, g_nn, g_en = _greens_functions(
        x[:, None] - force_x, y[:, None] - force_y, mindist, poisson
    )
    return np.block([[g_ee, g_en], [g_en, g_nn]])


@dataclass
class VectorSpline:
    """A fitted elastically coupled vector spline.

    Attributes
    ----------
    forces : np.ndarray
        Estimated point forces, east components stacked on north (2n,).
    force_x, force_y : np.ndarray
        Force locations in the working projection (km).
    lon0, lat0 : float
        Center of the working (oblique stereographic) projection, degrees.
    poisson : float
        Poisson's ratio coupling the components.
    mindist_km : float
        Fudge distance added to every force-point separation.
    residuals : np.ndarray
        Data-minus-model residuals at the stations, shape (n, 2), in true
        east/north.

    """

    forces: np.ndarray
    force_x: np.ndarray
    force_y: np.ndarray
    lon0: float
    lat0: float
    poisson: float
    mindist_km: float
    residuals: np.ndarray

    def predict(self, lon: ArrayLike, lat: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the spline at new points.

        Parameters
        ----------
        lon, lat : array-like
            Points in degrees (any matching shape).

        Returns
        -------
        east, north : np.ndarray
            Interpolated velocity components, same shape as the input and
            same units as the fitted data.

        """
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        shape = np.broadcast(lon, lat).shape
        lon, lat = (
            np.broadcast_to(lon, shape).ravel(),
            np.broadcast_to(lat, shape).ravel(),
        )
        x, y, gamma = _project(lon, lat, self.lon0, self.lat0)

        n = self.force_x.size
        f_east, f_north = self.forces[:n], self.forces[n:]
        vx, vy = np.empty(x.size), np.empty(x.size)
        for start in range(0, x.size, _PREDICT_CHUNK):
            sl = slice(start, start + _PREDICT_CHUNK)
            g_ee, g_nn, g_en = _greens_functions(
                x[sl, None] - self.force_x,
                y[sl, None] - self.force_y,
                self.mindist_km,
                self.poisson,
            )
            vx[sl] = g_ee @ f_east + g_en @ f_north
            vy[sl] = g_en @ f_east + g_nn @ f_north

        east, north = _grid_to_true(vx, vy, gamma)
        return east.reshape(shape), north.reshape(shape)


def _project(
    lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project to a conformal plane (km) and return the meridian convergence.

    The elastic Green's functions live on a plane. A conformal projection
    keeps angles, so true east/north differ from the grid axes only by a
    rotation `gamma` (true north measured clockwise from grid north),
    found here by displacing each point northward.
    """
    transformer = Transformer.from_crs(
        "EPSG:4326",
        {
            "proj": "stere",
            "lat_0": lat0,
            "lon_0": lon0,
            "datum": "WGS84",
            "units": "km",
        },
        always_xy=True,
    )
    x, y = transformer.transform(lon, lat)
    dlat = 1e-4
    # Step toward the equator near the pole so the probe stays on the globe
    step = np.where(lat + dlat > 90.0, -dlat, dlat)
    x2, y2 = transformer.transform(lon, lat + step)
    gamma = np.arctan2((x2 - x) * np.sign(step), (y2 - y) * np.sign(step))
    return np.asarray(x), np.asarray(y), gamma


def _true_to_grid(
    east: np.ndarray, north: np.ndarray, gamma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate true east/north components into grid x/y components."""
    c, s = np.cos(gamma), np.sin(gamma)
    return east * c + north * s, -east * s + north * c


def _grid_to_true(
    vx: np.ndarray, vy: np.ndarray, gamma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate grid x/y components back into true east/north components."""
    c, s = np.cos(gamma), np.sin(gamma)
    return vx * c - vy * s, vx * s + vy * c


def _solve_forces(
    jacobian: np.ndarray,
    data: np.ndarray,
    weights: np.ndarray | None,
    damping: float | None,
) -> np.ndarray:
    """Weighted, optionally damped least squares for the forces.

    Columns are scaled to unit standard deviation first (and unscaled
    afterwards) so `damping` has a comparable meaning across networks
    of different extent, as in verde.
    """
    scale = jacobian.std(axis=0)
    a = jacobian / scale
    b = data
    if weights is not None:
        sqrt_w = np.sqrt(weights)
        a, b = a * sqrt_w[:, None], b * sqrt_w
    if damping is not None:
        n_params = a.shape[1]
        a = np.vstack([a, np.sqrt(damping) * np.eye(n_params)])
        b = np.r_[b, np.zeros(n_params)]
    params, *_ = linalg.lstsq(a, b, check_finite=False)
    return params / scale


def fit_vector_spline(
    lon: ArrayLike,
    lat: ArrayLike,
    east: ArrayLike,
    north: ArrayLike,
    sigma_east: ArrayLike | None = None,
    sigma_north: ArrayLike | None = None,
    *,
    poisson: float = 0.5,
    damping: float | None = None,
    mindist_km: float = 10.0,
) -> VectorSpline:
    """Fit an elastically coupled spline to horizontal velocities.

    One in-plane point force is placed at every station and the forces
    are estimated by (weighted, damped) least squares so that their
    summed elastic response reproduces the observed velocities.

    Remove large-scale motion first (e.g. a rigid plate rotation with
    `geepers.euler.predict_plate_motion`) and restore it afterwards: the
    Green's functions grow logarithmically with distance, so a constant
    or rotational field is not represented well away from the stations.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    east, north : array-like
        Velocity components at the stations (any unit).
    sigma_east, sigma_north : array-like, optional
        1-sigma uncertainties; the fit is weighted by ``1 / sigma**2``.
        Give both or neither. Weights only matter when `damping` is set,
        since the undamped spline passes through every station.
    poisson : float
        Poisson's ratio coupling the components. 0.5 (default) is an
        incompressible sheet; -1 decouples east from north entirely.
    damping : float, optional
        Positive Tikhonov damping of the (scaled) forces. Larger values
        give a smoother field that no longer honors noisy data exactly.
        Default None: exact interpolation.
    mindist_km : float
        Fudge distance added to every force-point separation to keep the
        Green's functions finite at the stations; of the order of the
        station spacing. Default 10 km.

    Returns
    -------
    VectorSpline
        The fitted spline; call `VectorSpline.predict` to interpolate.

    Raises
    ------
    ValueError
        If only one of the two uncertainty arrays is given.

    """
    lon = np.asarray(lon, float).ravel()
    lat = np.asarray(lat, float).ravel()
    east = np.asarray(east, float).ravel()
    north = np.asarray(north, float).ravel()
    if (sigma_east is None) != (sigma_north is None):
        msg = "Give both sigma_east and sigma_north, or neither"
        raise ValueError(msg)

    lon0, lat0 = float(lon.mean()), float(lat.mean())
    x, y, gamma = _project(lon, lat, lon0, lat0)
    vx, vy = _true_to_grid(east, north, gamma)

    weights = None
    if sigma_east is not None:
        # The rotation by gamma is a few degrees at most, so the east/north
        # sigmas are used for the x/y components directly
        weights = (
            1.0
            / np.r_[
                np.asarray(sigma_east, float).ravel(),
                np.asarray(sigma_north, float).ravel(),
            ]
            ** 2
        )
        # Unit mean, so `damping` does not depend on the units of the sigmas
        weights /= weights.mean()

    jac = _jacobian(x, y, x, y, mindist_km, poisson)
    data = np.r_[vx, vy]
    forces = _solve_forces(jac, data, weights, damping)

    n = x.size
    fit = jac @ forces
    res_east, res_north = _grid_to_true(vx - fit[:n], vy - fit[n:], gamma)
    return VectorSpline(
        forces=forces,
        force_x=x,
        force_y=y,
        lon0=lon0,
        lat0=lat0,
        poisson=poisson,
        mindist_km=mindist_km,
        residuals=np.c_[res_east, res_north],
    )

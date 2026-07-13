"""Euler pole estimation and plate-motion prediction from GNSS velocities.

Rigid plate motion on a sphere is a rotation: the velocity of a point at
geocentric position :math:`\\mathbf{r}` is
:math:`\\mathbf{v} = \\boldsymbol{\\omega} \\times \\mathbf{r}`, linear in
the rotation vector :math:`\\boldsymbol{\\omega}`. This module estimates
:math:`\\boldsymbol{\\omega}` (equivalently the Euler pole position and
rate) from horizontal station velocities by weighted least squares, and
predicts the rigid-plate velocity anywhere — the remove/restore step of
plate-boundary-aware interpolation (see
`geepers.collocation.separate_plates`).

The formulation follows standard plate kinematics (e.g. Cox, A., &
Hart, R. B., 1986, *Plate Tectonics: How It Works*); implementation
informed by pyacs (J.-M. Nocquet,
https://github.com/JMNocquet/pyacs36) via the Venti GNSS toolbox by
Marin Govorcin.

Units convention: velocities in **mm/yr**, rates in **deg/Myr**.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from pyproj import Transformer

__all__ = [
    "EulerPole",
    "estimate_euler_pole",
    "pole_to_rotation_vector",
    "predict_plate_motion",
    "rotation_vector_to_pole",
]

logger = logging.getLogger("geepers")

_LONLAT_TO_ECEF = Transformer.from_crs(
    "EPSG:4326",
    {"proj": "geocent", "ellps": "GRS80", "datum": "WGS84"},
    always_xy=True,
)


def rotation_vector_to_pole(w: ArrayLike) -> tuple[float, float, float]:
    """Convert a rotation vector to an Euler pole.

    Parameters
    ----------
    w : array-like of float
        Rotation vector (wx, wy, wz) in rad/yr.

    Returns
    -------
    lon, lat : float
        Pole position in degrees.
    rate : float
        Angular rate in degrees per million years.

    """
    w = np.asarray(w, float)
    rate = float(np.linalg.norm(w))
    lat = 90.0 - np.degrees(np.arccos(w[2] / rate))
    lon = np.degrees(np.arctan2(w[1], w[0]))
    return float(lon), float(lat), float(np.degrees(rate) * 1e6)


def pole_to_rotation_vector(lon: float, lat: float, rate: float) -> np.ndarray:
    """Convert an Euler pole to a rotation vector.

    Parameters
    ----------
    lon, lat : float
        Pole position in degrees.
    rate : float
        Angular rate in degrees per million years.

    Returns
    -------
    np.ndarray
        Rotation vector (wx, wy, wz) in rad/yr.

    """
    w = np.radians(rate) * 1e-6
    lam, phi = np.radians(lon), np.radians(lat)
    return w * np.array(
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]
    )


def _ecef_to_enu_matrices(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Per-point rotation matrices geocentric XYZ -> local (E, N, U).

    Returns an (n, 3, 3) stack.
    """
    lam, phi = np.radians(lon), np.radians(lat)
    zero = np.zeros_like(lam)
    return np.stack(
        [
            np.stack([-np.sin(lam), np.cos(lam), zero], axis=-1),
            np.stack(
                [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
                axis=-1,
            ),
            np.stack(
                [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def _design_matrix(
    lon: np.ndarray, lat: np.ndarray, height: np.ndarray
) -> np.ndarray:
    """East/north observation rows for all stations, shape (2n, 3).

    Maps the rotation vector in micro-rad/yr to east/north velocities in
    mm/yr: rows are ``R (r x .)`` with positions in km.
    """
    x, y, z = _LONLAT_TO_ECEF.transform(lon, lat, height)
    x, y, z = (np.asarray(c) / 1000.0 for c in (x, y, z))  # m -> km
    zero = np.zeros_like(x)
    # v = omega x r: skew-symmetric matrix of r acting on omega, (n, 3, 3)
    skew = np.stack(
        [
            np.stack([zero, z, -y], axis=-1),
            np.stack([-z, zero, x], axis=-1),
            np.stack([y, -x, zero], axis=-1),
        ],
        axis=-2,
    )
    rows_en = (_ecef_to_enu_matrices(lon, lat) @ skew)[:, :2, :]  # (n, 2, 3)
    return rows_en.reshape(-1, 3)


@dataclass
class EulerPole:
    """An Euler pole with its estimation statistics.

    Attributes
    ----------
    lon, lat : float
        Pole position in degrees.
    rate : float
        Angular rate in degrees per million years.
    covariance : np.ndarray
        3x3 covariance of the rotation vector (rad^2/yr^2).
    rms, wrms : float
        (Weighted) RMS of the velocity residuals, mm/yr.
    reduced_chi2 : float
        sqrt(chi^2 / dof) of the fit; ~1 when the residual scatter
        matches the velocity uncertainties.
    dof : int
        Degrees of freedom (2 x n_stations - 3).

    """

    lon: float
    lat: float
    rate: float
    covariance: np.ndarray = field(default_factory=lambda: np.full((3, 3), np.nan))
    rms: float = np.nan
    wrms: float = np.nan
    reduced_chi2: float = np.nan
    dof: int = 0

    @property
    def rotation_vector(self) -> np.ndarray:
        """Rotation vector (wx, wy, wz) in rad/yr."""
        return pole_to_rotation_vector(self.lon, self.lat, self.rate)

    def uncertainty(self) -> dict[str, float]:
        """Error ellipse of the pole position and rate uncertainty.

        Returns
        -------
        dict
            ``semi_major_deg``, ``semi_minor_deg`` (1-sigma angular
            radii of the pole error ellipse), ``azimuth_deg``, and
            ``sigma_rate`` (deg/Myr).

        """
        R = _ecef_to_enu_matrices(
            np.atleast_1d(self.lon), np.atleast_1d(self.lat)
        )[0]
        cov_enu = R @ self.covariance @ R.T
        norm_w = np.linalg.norm(self.rotation_vector)

        s11, s12, s22 = cov_enu[0, 0], cov_enu[0, 1], cov_enu[1, 1]
        disc = np.sqrt((s11 - s22) ** 2 + 4 * s12**2)
        smaj = np.degrees(np.arctan(np.sqrt(0.5 * (s11 + s22 + disc)) / norm_w))
        smin = np.degrees(
            np.arctan(np.sqrt(max(0.5 * (s11 + s22 - disc), 0.0)) / norm_w)
        )
        azim = 0.5 * np.degrees(np.arctan2(2 * s12, s11 - s22))
        sigma_rate = float(np.degrees(np.sqrt(cov_enu[2, 2])) * 1e6)
        return {
            "semi_major_deg": float(smaj),
            "semi_minor_deg": float(smin),
            "azimuth_deg": float(azim),
            "sigma_rate": sigma_rate,
        }


def estimate_euler_pole(
    lon: ArrayLike,
    lat: ArrayLike,
    ve: ArrayLike,
    vn: ArrayLike,
    sigma_e: ArrayLike,
    sigma_n: ArrayLike,
    height: ArrayLike | None = None,
) -> EulerPole:
    """Estimate an Euler pole from horizontal GNSS velocities.

    Weighted least squares of ``v = omega x r`` over the east/north
    velocity components, weighting each station by its uncertainties
    (per-station east-north error correlations are ignored, as they are
    rarely reported).

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    ve, vn : array-like
        East and north velocities in **mm/yr** (in the frame the plate
        rotation is defined relative to, e.g. ITRF).
    sigma_e, sigma_n : array-like
        1-sigma velocity uncertainties in mm/yr.
    height : array-like, optional
        Ellipsoidal heights in meters (default 0; the effect on the
        pole is negligible for crustal heights).

    Returns
    -------
    EulerPole

    """
    lon = np.atleast_1d(np.asarray(lon, float))
    lat = np.atleast_1d(np.asarray(lat, float))
    ve = np.atleast_1d(np.asarray(ve, float))
    vn = np.atleast_1d(np.asarray(vn, float))
    sigma_e = np.atleast_1d(np.asarray(sigma_e, float))
    sigma_n = np.atleast_1d(np.asarray(sigma_n, float))
    hgt = (
        np.zeros_like(lon)
        if height is None
        else np.atleast_1d(np.asarray(height, float))
    )
    n = len(lon)
    if n < 2:
        msg = "Need at least 2 stations to estimate an Euler pole"
        raise ValueError(msg)

    A = _design_matrix(lon, lat, hgt)
    b = np.column_stack([ve, vn]).ravel()
    weights = np.column_stack([1 / sigma_e**2, 1 / sigma_n**2]).ravel()

    # Weighted normal equations (diagonal weight matrix)
    Aw = A * weights[:, None]
    Q = np.linalg.inv(Aw.T @ A)
    x = Q @ (Aw.T @ b)  # micro-rad/yr

    resid = b - A @ x
    chi2 = float(np.sum(weights * resid**2))
    dof = 2 * n - 3
    re, rn = resid[::2], resid[1::2]
    rms = float(np.sqrt(np.sum(re**2 + rn**2) / (2 * n)))
    wrms = float(
        np.sqrt(
            np.sum((re / sigma_e) ** 2 + (rn / sigma_n) ** 2)
            / np.sum(1 / sigma_e**2 + 1 / sigma_n**2)
        )
    )

    plon, plat, rate = rotation_vector_to_pole(x * 1e-6)
    logger.info(
        "Euler pole: lon=%.2f lat=%.2f rate=%.4f deg/Myr (wrms %.2f mm/yr)",
        plon, plat, rate, wrms,
    )
    return EulerPole(
        lon=plon,
        lat=plat,
        rate=rate,
        covariance=Q * 1e-12,  # (micro-rad/yr)^2 -> (rad/yr)^2
        rms=rms,
        wrms=wrms,
        reduced_chi2=float(np.sqrt(chi2 / dof)) if dof > 0 else np.nan,
        dof=dof,
    )


def predict_plate_motion(
    pole: EulerPole,
    lon: ArrayLike,
    lat: ArrayLike,
    height: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid-plate velocity predicted by an Euler pole.

    Use this to remove plate motion before interpolation across plate
    boundaries and to restore it afterward:
    ``ve_resid = ve - ve_plate``.

    Parameters
    ----------
    pole : EulerPole
        The rotation (e.g. from `estimate_euler_pole` or published
        ITRF plate-motion-model values).
    lon, lat : array-like
        Points in degrees.
    height : array-like, optional
        Ellipsoidal heights in meters.

    Returns
    -------
    ve, vn : np.ndarray
        East and north velocities in mm/yr.

    """
    lon = np.atleast_1d(np.asarray(lon, float))
    lat = np.atleast_1d(np.asarray(lat, float))
    hgt = (
        np.zeros_like(lon)
        if height is None
        else np.atleast_1d(np.asarray(height, float))
    )
    A = _design_matrix(lon, lat, hgt)
    v = A @ (pole.rotation_vector * 1e6)  # micro-rad/yr pairs with mm/yr
    return v[::2], v[1::2]

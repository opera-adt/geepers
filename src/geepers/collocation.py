"""Least-squares collocation (LSC) for GNSS velocity fields.

Interpolate scattered GNSS velocities (horizontal pairs or a single
vertical component) onto arbitrary points or regular grids, propagating
both the signal covariance and the per-station noise. Supports
heterogeneous ("moving-variance") signal scaling so that locally noisy
regions do not contaminate quiet ones.

Ported and refined from ``py3_hvlsc`` by Marin Govorcin; the angular
cross-covariance terms for velocity fields on the sphere follow the
classical LSC formulation (e.g. Moritz, H., 1980, *Advanced Physical
Geodesy*).

Example
-------
>>> emp = empirical_covariance(lon, lat, ve, vn, se, sn)   # doctest: +SKIP
>>> result = interpolate_velocities(                        # doctest: +SKIP
...     lon, lat, ve, vn, se, sn, lon_grid, lat_grid,
...     covariance_parameters=emp.parameters,
... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from pyproj import CRS, Geod, Transformer
from scipy import linalg, optimize, stats

__all__ = [
    "COVARIANCE_MODELS",
    "CollocationResult",
    "EmpiricalCovariance",
    "collocate",
    "create_regular_grid",
    "distance_matrix",
    "empirical_covariance",
    "interpolate_velocities",
    "noise_covariance",
    "ordinary_kriging",
    "predict",
    "separate_plates",
    "signal_covariance",
]

logger = logging.getLogger("geepers")

_GEOD = Geod(ellps="WGS84")

# Components of the 2x2 block covariance for a horizontal velocity field,
# in the block order used throughout this module.
_HORIZONTAL = ("ee", "en", "ne", "nn")


# ---------------------------------------------------------------------------
# Covariance model functions C(distance; C0, d0)
# ---------------------------------------------------------------------------
def _gauss_markov_1(dist, C0, d0):
    return C0 * np.exp(-dist / d0)


def _gauss_markov_2(dist, C0, d0):
    return C0 * np.exp(-(dist**2) / d0**2)


def _reilly(dist, C0, d0):
    return C0 * (1 - 0.5 * (dist / d0) ** 2) * np.exp(-0.5 * (dist / d0) ** 2)


def _markov_1(dist, C0, d0):
    return C0 * (1 + dist / d0) * np.exp(-dist / d0)


def _markov_2(dist, C0, d0):
    return C0 * (1 + dist / d0 + dist**2 / (3 * d0**2)) * np.exp(-dist / d0)


COVARIANCE_MODELS: dict[str, Callable] = {
    "gm1": _gauss_markov_1,
    "gm2": _gauss_markov_2,
    "reilly": _reilly,
    "markov1": _markov_1,
    "markov2": _markov_2,
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def distance_matrix(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
    unit: Literal["m", "km"] = "km",
) -> np.ndarray:
    """Geodesic distance matrix between two sets of points.

    Parameters
    ----------
    lon1, lat1 : np.ndarray
        Coordinates of the first point set (n points, degrees).
    lon2, lat2 : np.ndarray
        Coordinates of the second point set (m points, degrees).
    unit : {"m", "km"}
        Output unit. Default "km".

    Returns
    -------
    np.ndarray
        (m, n) matrix of WGS84 geodesic distances, with rows indexing the
        second point set (matching the block layout of the covariance
        builders).

    """
    lon1, lat1 = np.asarray(lon1, float), np.asarray(lat1, float)
    lon2, lat2 = np.asarray(lon2, float), np.asarray(lat2, float)
    L1, L2 = np.broadcast_arrays(lon1[None, :], lon2[:, None])
    B1, B2 = np.broadcast_arrays(lat1[None, :], lat2[:, None])
    dist_m = _GEOD.inv(L1, B1, L2, B2)[2]
    return dist_m * (1e-3 if unit == "km" else 1.0)


# Angular factors for the covariance of a rotational (angular) velocity
# field on the sphere; lon/lat inputs in degrees, broadcastable.
def _f_ee(lon1, lat1, lon2, lat2):
    b1, b2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon1 - lon2)
    return np.sin(b1) * np.sin(b2) * np.cos(dl) + np.cos(b1) * np.cos(b2)


def _f_en(lon1, lat1, lon2, lat2):
    return np.sin(np.radians(lon1 - lon2)) * np.sin(np.radians(lat1))


def _f_ne(lon1, lat1, lon2, lat2):
    return np.sin(np.radians(lon2 - lon1)) * np.sin(np.radians(lat2))


def _f_nn(lon1, lat1, lon2, lat2):
    return np.cos(np.radians(lon1 - lon2))


_ANGULAR = {"ee": _f_ee, "en": _f_en, "ne": _f_ne, "nn": _f_nn}


# ---------------------------------------------------------------------------
# Covariance builders
# ---------------------------------------------------------------------------
def signal_covariance(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
    parameters: np.ndarray,
    model: str = "gm1",
    components: tuple[str, ...] = _HORIZONTAL,
    cross_correlation: bool = True,
) -> np.ndarray:
    """Build the signal covariance matrix between two point sets.

    Parameters
    ----------
    lon1, lat1 : np.ndarray
        Coordinates of the first point set (n points, degrees).
    lon2, lat2 : np.ndarray
        Coordinates of the second point set (m points, degrees).
    parameters : np.ndarray
        Covariance function parameters, one row ``(C0, d0)`` per entry of
        `components` (a single row is broadcast to all). ``d0`` is in km.
    model : str
        Key into `COVARIANCE_MODELS`. Default "gm1"
        (first-order Gauss-Markov).
    components : tuple of str
        ``("ee", "en", "ne", "nn")`` for a horizontal velocity field
        (returns a 2x2 block matrix), or a single-element tuple (e.g.
        ``("up",)``) for a scalar field.
    cross_correlation : bool
        Multiply each block by the angular factor of a rotational field
        on the sphere and fill the off-diagonal blocks. If False, the
        off-diagonal blocks are zero. Ignored for scalar fields.

    Returns
    -------
    np.ndarray
        (m, n) matrix for scalar fields, (2m, 2n) block matrix otherwise.

    """
    func = COVARIANCE_MODELS[model]
    parameters = np.atleast_2d(np.asarray(parameters, float))
    if len(parameters) == 1:
        parameters = np.repeat(parameters, len(components), axis=0)

    dist = distance_matrix(lon1, lat1, lon2, lat2)
    scalar = len(components) == 1
    if scalar:
        C0, d0 = parameters[0][:2]
        return func(dist, C0, d0)

    L1, L2 = np.broadcast_arrays(
        np.asarray(lon1, float)[None, :], np.asarray(lon2, float)[:, None]
    )
    B1, B2 = np.broadcast_arrays(
        np.asarray(lat1, float)[None, :], np.asarray(lat2, float)[:, None]
    )

    blocks: dict[str, np.ndarray] = {}
    for comp, (C0, d0, *_) in zip(components, parameters, strict=True):
        C = func(dist, C0, d0)
        if cross_correlation:
            # NOTE: argument order (2 -> 1) matches the validated original
            C = C * _ANGULAR[comp](L2, B2, L1, B1)
        blocks[comp] = C

    if not cross_correlation:
        zero = np.zeros_like(blocks["ee"])
        blocks["en"] = blocks["ne"] = zero
    return np.block(
        [[blocks["ee"], blocks["en"]], [blocks["ne"], blocks["nn"]]]
    )


def noise_covariance(sigmas: np.ndarray) -> np.ndarray:
    """Diagonal noise covariance from per-observation standard deviations.

    Parameters
    ----------
    sigmas : np.ndarray
        1-sigma uncertainties; shape (n,) for a scalar field or (n, 2)
        for east/north (stacked column-wise to match the block layout).

    Returns
    -------
    np.ndarray
        Diagonal matrix of variances.

    """
    return np.diag(np.asarray(sigmas, float).ravel(order="F") ** 2)


# ---------------------------------------------------------------------------
# Empirical covariance estimation
# ---------------------------------------------------------------------------
@dataclass
class EmpiricalCovariance:
    """Fitted empirical covariance function.

    Attributes
    ----------
    parameters : np.ndarray
        ``(C0, d0)`` of the fitted model (d0 in km).
    uncertainties : np.ndarray
        1-sigma uncertainties of `parameters` (C0 is held fixed -> 0).
    misfit : float
        RMS misfit of the fit, normalized by C0.
    misfit_first_bins : float
        RMS misfit over the first three bins, normalized by C0.
    pearson : float
        Pearson correlation between binned and fitted covariances.
    bin_centers, bin_means, bin_stds : np.ndarray
        The binned covariogram used in the fit.

    """

    parameters: np.ndarray
    uncertainties: np.ndarray
    misfit: float
    misfit_first_bins: float
    pearson: float
    bin_centers: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_means: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_stds: np.ndarray = field(default_factory=lambda: np.array([]))


def _remove_outliers(arrays: list[np.ndarray], data: np.ndarray, nsigma: float):
    """Drop entries of every array where |data| > nsigma * rms(data)."""
    bad = np.abs(data) > nsigma * np.sqrt(np.mean(data**2))
    if bad.any():
        logger.info("Removing %d outliers (>%g x rms)", int(bad.sum()), nsigma)
        return [a[~bad] for a in arrays]
    return arrays


def empirical_covariance(
    lon: np.ndarray,
    lat: np.ndarray,
    data1: np.ndarray,
    data2: np.ndarray,
    noise1: np.ndarray,
    noise2: np.ndarray,
    *,
    bin_spacing_km: float = 50.0,
    outlier_nsigma: float = 3.0,
    max_range_km: float | None = None,
) -> EmpiricalCovariance:
    """Estimate an isotropic covariance function from scattered data.

    Bins the cross-products of the (outlier-cleaned) observations by
    separation distance, anchors the zero-lag variance at
    ``mean(data^2) - mean(noise^2)`` for each component, and fits a
    first-order Gauss-Markov model ``C0 * exp(-d / d0)`` for the
    correlation length ``d0`` (C0 held fixed). The fit is run over two
    ``d0`` search ranges and the better solution (by normalized misfit,
    short-range misfit, and Pearson correlation) is kept.

    Parameters
    ----------
    lon, lat : np.ndarray
        Station coordinates in degrees.
    data1, data2 : np.ndarray
        The two data components (e.g. east and north velocities). Pass
        the same array twice for a scalar (e.g. vertical) field.
    noise1, noise2 : np.ndarray
        Corresponding 1-sigma uncertainties.
    bin_spacing_km : float
        Base spacing of the distance bins (bins are ``2 x`` this wide).
    outlier_nsigma : float
        Observations with ``|data| > nsigma * rms`` are dropped.
    max_range_km : float, optional
        Ignore station pairs farther apart than this.

    Returns
    -------
    EmpiricalCovariance

    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    arrays = [np.asarray(a, float) for a in (lon, lat, data1, data2, noise1, noise2)]
    arrays = _remove_outliers(arrays, arrays[2], outlier_nsigma)
    arrays = _remove_outliers(arrays, arrays[3], outlier_nsigma)
    lon, lat, data1, data2, noise1, noise2 = arrays
    n = len(data1)

    dist = distance_matrix(lon, lat, lon, lat)
    dmax_km = (
        _GEOD.inv(lon.min(), lat.min(), lon.max(), lat.max())[2] * 1e-3
    )
    bin_edges = np.r_[[0.0, 0.001], np.arange(bin_spacing_km, dmax_km, 2 * bin_spacing_km)]
    if len(bin_edges) < 4:
        msg = (
            "Too few distance bins - decrease bin_spacing_km "
            f"(dmax = {dmax_km:.0f} km, spacing = {bin_spacing_km} km)"
        )
        raise ValueError(msg)
    if max_range_km:
        dist = np.where(dist > max_range_km, -1.0, dist)

    # Zero-lag variance, noise-corrected (average of both components)
    c0 = 0.5 * (np.mean(data1**2) - np.mean(noise1**2)) + 0.5 * (
        np.mean(data2**2) - np.mean(noise2**2)
    )
    s0 = float(np.std(np.c_[data1, data2]))

    # Binned covariogram from all cross-products
    cross = np.outer(data2, data1).ravel()
    d = dist.ravel()
    cross, d = cross[d > 0], d[d > 0]
    bin_sum, _, _ = stats.binned_statistic(d, cross, "sum", bins=bin_edges)
    count, _, _ = stats.binned_statistic(d, cross, "count", bins=bin_edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = bin_sum / (count - 1)
    stds, _, _ = stats.binned_statistic(d, cross, "std", bins=bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Keep well-populated bins; prepend the anchored zero-lag point
    keep = count >= 2 * n
    means, stds, centers = means[keep], stds[keep], centers[keep]
    c0_signed = -c0 if (len(means) and means[0] < 0) else c0
    means = np.r_[c0_signed, means]
    stds = np.r_[s0, stds]
    centers = np.r_[0.0, centers]

    # Fit d0 (C0 fixed) over two search ranges, keep the better result
    def model_fixed_c0(dd, d0):
        return _gauss_markov_1(dd, c0_signed, d0)

    candidates = []
    for lo, hi in ((bin_edges[2], bin_edges[3]), (0.0, bin_edges[-1])):
        try:
            coef, cov = optimize.curve_fit(
                model_fixed_c0, centers, means, bounds=(lo, hi)
            )
        except (RuntimeError, ValueError):
            continue
        d0, sd0 = float(coef[0]), float(np.sqrt(np.diag(cov))[0])
        fit = model_fixed_c0(centers, d0)
        mis = float(np.sqrt(np.mean((fit - means) ** 2)) / abs(c0_signed))
        mis3 = float(np.sqrt(np.mean((fit[:3] - means[:3]) ** 2)) / abs(c0_signed))
        pear = float(stats.pearsonr(means, fit)[0])
        candidates.append((d0, sd0, mis, mis3, pear))
    if not candidates:
        msg = "Covariance model fit failed for all d0 search ranges"
        raise RuntimeError(msg)

    if len(candidates) == 2:
        (d0a, _, misa, mis3a, peara), (d0b, _, misb, mis3b, pearb) = candidates
        if misb <= misa and mis3b <= mis3a:
            pick = 1
        elif (mis3b <= mis3a or misb <= misa) and round(pearb, 2) >= round(peara, 2):
            pick = 1 if d0b < d0a else 0
        else:
            pick = 0
    else:
        pick = 0
    d0, sd0, mis, mis3, pear = candidates[pick]

    logger.info(
        "Empirical covariance: C0 = %.5f, d0 = %.0f +/- %.0f km "
        "(misfit %.4f, pearson %.3f)",
        c0_signed, d0, sd0, mis, pear,
    )
    return EmpiricalCovariance(
        parameters=np.array([c0_signed, d0]),
        uncertainties=np.array([0.0, sd0]),
        misfit=mis,
        misfit_first_bins=mis3,
        pearson=pear,
        bin_centers=centers,
        bin_means=means,
        bin_stds=stds,
    )


# ---------------------------------------------------------------------------
# Collocation solvers
# ---------------------------------------------------------------------------
@dataclass
class CollocationResult:
    """Signal estimated by least-squares collocation.

    Attributes
    ----------
    signal : np.ndarray
        Estimated signal, shape (n_points, n_components).
    signal_sigma : np.ndarray
        1-sigma uncertainty of the signal, same shape.
    noise : np.ndarray | None
        Estimated noise at the observation points (None for predictions
        at new points).

    """

    signal: np.ndarray
    signal_sigma: np.ndarray
    noise: np.ndarray | None = None


def collocate(
    obs: np.ndarray, Css: np.ndarray, Cnn: np.ndarray
) -> tuple[CollocationResult, np.ndarray]:
    """Separate signal and noise at the observation points.

    Parameters
    ----------
    obs : np.ndarray
        Observations, shape (n, k) with one column per component.
    Css : np.ndarray
        Signal covariance at the observation points.
    Cnn : np.ndarray
        Noise covariance at the observation points.

    Returns
    -------
    result : CollocationResult
        Signal, its uncertainty, and the noise estimate.
    Czz_inv : np.ndarray
        Inverse of ``Css + Cnn``, reusable in `predict`.

    """
    obs = np.atleast_2d(np.asarray(obs, float))
    if obs.shape[0] == 1:
        obs = obs.T
    k = obs.shape[1]
    z = obs.ravel(order="F")

    Czz = Css + Cnn
    cho = linalg.cho_factor(Czz, lower=True, check_finite=False)
    Czz_inv = linalg.cho_solve(cho, np.eye(len(z)), check_finite=False)

    w = Czz_inv @ z
    signal = Css @ w
    noise = Cnn @ w
    sigma = np.sqrt(np.abs(np.diag(Css - Css @ Czz_inv @ Css)))

    logger.info("Collocation at %d observation points done", obs.shape[0])
    return (
        CollocationResult(
            signal=signal.reshape((-1, k), order="F"),
            signal_sigma=sigma.reshape((-1, k), order="F"),
            noise=noise.reshape((-1, k), order="F"),
        ),
        Czz_inv,
    )


def predict(
    obs: np.ndarray,
    Cps: np.ndarray,
    Cpp: np.ndarray,
    Czz_inv: np.ndarray,
) -> CollocationResult:
    """Predict the signal at new points.

    Parameters
    ----------
    obs : np.ndarray
        Observations, shape (n, k).
    Cps : np.ndarray
        Cross-covariance between prediction and observation points.
    Cpp : np.ndarray
        Signal covariance at the prediction points.
    Czz_inv : np.ndarray
        Inverse of ``Css + Cnn`` from `collocate`.

    Returns
    -------
    CollocationResult
        Signal and uncertainty at the prediction points.

    """
    obs = np.atleast_2d(np.asarray(obs, float))
    if obs.shape[0] == 1:
        obs = obs.T
    k = obs.shape[1]
    z = obs.ravel(order="F")

    G = Cps @ Czz_inv
    signal = G @ z
    sigma = np.sqrt(np.abs(np.diag(Cpp - G @ Cps.T)))

    logger.info("Collocation prediction at %d points done", Cpp.shape[0] // k)
    return CollocationResult(
        signal=signal.reshape((-1, k), order="F"),
        signal_sigma=sigma.reshape((-1, k), order="F"),
    )


def interpolate_velocities(
    lon: np.ndarray,
    lat: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    sigma_east: np.ndarray,
    sigma_north: np.ndarray,
    lon_new: np.ndarray,
    lat_new: np.ndarray,
    *,
    covariance_parameters: np.ndarray | None = None,
    model: str = "gm1",
    cross_correlation: bool = True,
) -> tuple[CollocationResult, CollocationResult]:
    """Interpolate horizontal velocities onto new points by collocation.

    Parameters
    ----------
    lon, lat : np.ndarray
        Station coordinates (degrees).
    east, north : np.ndarray
        Velocity components at the stations.
    sigma_east, sigma_north : np.ndarray
        1-sigma velocity uncertainties.
    lon_new, lat_new : np.ndarray
        Points to interpolate onto (degrees).
    covariance_parameters : np.ndarray, optional
        ``(C0, d0)`` rows for the signal covariance. If None, estimated
        with `empirical_covariance`.
    model : str
        Covariance model name. Default "gm1".
    cross_correlation : bool
        Use the on-sphere angular cross-covariance terms. Default True.

    Returns
    -------
    at_stations : CollocationResult
        Filtered signal/noise separation at the stations.
    at_new_points : CollocationResult
        Interpolated signal and uncertainty at `lon_new`/`lat_new`.

    """
    if covariance_parameters is None:
        emp = empirical_covariance(
            lon, lat, east, north, sigma_east, sigma_north
        )
        covariance_parameters = emp.parameters

    obs = np.c_[east, north]
    Css = signal_covariance(
        lon, lat, lon, lat, covariance_parameters,
        model=model, cross_correlation=cross_correlation,
    )
    Cnn = noise_covariance(np.c_[sigma_east, sigma_north])
    at_stations, Czz_inv = collocate(obs, Css, Cnn)

    Cps = signal_covariance(
        lon, lat, lon_new, lat_new, covariance_parameters,
        model=model, cross_correlation=cross_correlation,
    )
    Cpp = signal_covariance(
        lon_new, lat_new, lon_new, lat_new, covariance_parameters,
        model=model, cross_correlation=cross_correlation,
    )
    at_new = predict(obs, Cps, Cpp, Czz_inv)
    return at_stations, at_new


# ---------------------------------------------------------------------------
# Ordinary kriging
# ---------------------------------------------------------------------------
def ordinary_kriging(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    sigmas: np.ndarray,
    lon_new: np.ndarray,
    lat_new: np.ndarray,
    parameters: np.ndarray,
    model: str = "gm1",
) -> CollocationResult:
    """Ordinary kriging of a scalar field with measurement error.

    Solves the ordinary-kriging system with the fitted covariance model
    and a per-station nugget from `sigmas` (non-exact kriging: the
    surface is not forced through noisy observations). Unlike simple
    collocation, the unbiasedness constraint (weights sum to 1) makes
    the estimate independent of the (unknown) field mean, so the data
    do not need to be demeaned first.

    Parameters
    ----------
    lon, lat : np.ndarray
        Station coordinates (degrees).
    values : np.ndarray
        Station values (any unit).
    sigmas : np.ndarray
        1-sigma measurement uncertainties (adds a nugget to the
        diagonal of the kriging matrix).
    lon_new, lat_new : np.ndarray
        Points to predict at (degrees).
    parameters : np.ndarray
        ``(C0, d0)`` of the covariance model (d0 in km), e.g. from
        `empirical_covariance`.
    model : str
        Covariance model name from `COVARIANCE_MODELS`. Default "gm1".

    Returns
    -------
    CollocationResult
        Prediction and kriging standard deviation at the new points
        (shape (n, 1)).

    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    values = np.asarray(values, float)
    sigmas = np.asarray(sigmas, float)
    lon_new = np.atleast_1d(np.asarray(lon_new, float))
    lat_new = np.atleast_1d(np.asarray(lat_new, float))

    C0 = float(np.atleast_2d(parameters)[0][0])
    n = len(values)

    # Kriging matrix: [C + diag(sig^2), 1; 1^T, 0]
    K = np.empty((n + 1, n + 1))
    K[:n, :n] = signal_covariance(
        lon, lat, lon, lat, parameters, model=model, components=("v",)
    ) + np.diag(sigmas**2)
    K[n, :n] = K[:n, n] = 1.0
    K[n, n] = 0.0

    # RHS: covariances to every prediction point + the constraint row
    k = np.empty((n + 1, len(lon_new)))
    k[:n] = signal_covariance(
        lon_new, lat_new, lon, lat, parameters, model=model, components=("v",)
    )
    k[n] = 1.0

    sol = linalg.solve(K, k, assume_a="sym")
    w, mu = sol[:n], sol[n]
    est = w.T @ values
    var = C0 - np.einsum("ij,ij->j", w, k[:n]) - mu
    var = np.clip(var, 0.0, None)

    logger.info("Ordinary kriging at %d points done", len(lon_new))
    return CollocationResult(
        signal=est[:, None], signal_sigma=np.sqrt(var)[:, None]
    )


# ---------------------------------------------------------------------------
# Plate-boundary constraint: separate plates in a working coordinate space
# ---------------------------------------------------------------------------
def separate_plates(
    lon: np.ndarray,
    lat: np.ndarray,
    plates,
    *,
    min_separation_km: float = 1500.0,
    central_plate: int | None = None,
    max_iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate points on different tectonic plates apart.

    Interpolating across a plate boundary smears discontinuous motion.
    This helper builds a *working coordinate space* in which each
    plate's points are pushed away from the central plate (along the
    geodesic between plate centroids) until every plate pair is at
    least `min_separation_km` apart, so any distance-decaying
    covariance treats stations on different plates as uncorrelated.
    Interpolate in the moved coordinates, then attach the results back
    to the original ones.

    Parameters
    ----------
    lon, lat : np.ndarray
        Point coordinates in degrees (stations and/or grid nodes).
    plates : gpd.GeoDataFrame
        Plate polygons. Points are assigned by containment; points
        outside every polygon are assigned to the nearest plate.
    min_separation_km : float
        Minimum hull-to-hull separation to enforce. Default 1500 km
        (several times a typical covariance correlation length).
    central_plate : int, optional
        Index (row position in `plates`) of the plate to keep fixed.
        Default: the plate containing the most points.
    max_iterations : int
        Safety cap on the push-apart iterations.

    Returns
    -------
    lon_moved, lat_moved : np.ndarray
        Working coordinates, same order as the input.
    plate_index : np.ndarray of int
        Plate assignment of each point (row position in `plates`).

    """
    import shapely
    from shapely.geometry import MultiPoint

    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)

    # Assign each point to a plate (containment, else nearest)
    pts = shapely.points(lon, lat)
    geoms = np.asarray(plates.geometry.values)
    plate_index = np.full(len(lon), -1)
    for gi, poly in enumerate(geoms):
        inside = shapely.contains(poly, pts) & (plate_index < 0)
        plate_index[inside] = gi
    outside = plate_index < 0
    if outside.any():
        d = np.vstack([shapely.distance(poly, pts[outside]) for poly in geoms])
        plate_index[outside] = np.argmin(d, axis=0)

    used = np.unique(plate_index)
    if central_plate is None:
        central_plate = int(used[np.argmax([(plate_index == u).sum() for u in used])])
    logger.info(
        "Separating %d plates (central: %s)", len(used), central_plate
    )

    lon_m, lat_m = lon.copy(), lat.copy()

    def hull(k):
        return MultiPoint(np.c_[lon_m[plate_index == k], lat_m[plate_index == k]]).convex_hull

    def push(k_from: int, k_to: int, dist_km: float) -> None:
        """Move plate `k_to` points away from `k_from` along the centroid azimuth."""
        c1, c2 = hull(k_from).centroid, hull(k_to).centroid
        az, _, _ = _GEOD.inv(c1.x, c1.y, c2.x, c2.y)
        sel = plate_index == k_to
        lon_m[sel], lat_m[sel], _ = _GEOD.fwd(
            lon_m[sel], lat_m[sel], np.full(sel.sum(), az), np.full(sel.sum(), dist_km * 1e3)
        )

    # Initial push of every non-central plate
    for k in used:
        if k != central_plate:
            push(central_plate, int(k), min_separation_km)

    # Iterate until all plate-pair hulls are separated
    for _ in range(max_iterations):
        violations = []
        for i, ka in enumerate(used):
            for kb in used[i + 1:]:
                ha, hb = hull(int(ka)), hull(int(kb))
                bnd_a = np.array(
                    ha.exterior.coords if ha.geom_type == "Polygon" else ha.coords
                )
                bnd_b = np.array(
                    hb.exterior.coords if hb.geom_type == "Polygon" else hb.coords
                )
                gap = distance_matrix(
                    bnd_a[:, 0], bnd_a[:, 1], bnd_b[:, 0], bnd_b[:, 1]
                ).min()
                if gap < min_separation_km:
                    violations.append((int(ka), int(kb), min_separation_km - gap))
        if not violations:
            break
        for ka, kb, short in violations:
            mover = kb if ka == central_plate or kb != central_plate else ka
            anchor = ka if mover == kb else kb
            push(anchor, mover, short + 1.0)
    else:
        logger.warning("separate_plates did not converge in %d iterations", max_iterations)

    return lon_m, lat_m, plate_index


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def create_regular_grid(
    lon0: float,
    lat0: float,
    width_km: float,
    height_km: float,
    dx_km: float = 50.0,
    dy_km: float = 50.0,
    buffer_km: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a regular grid in an azimuthal-equidistant projection.

    Parameters
    ----------
    lon0, lat0 : float
        Grid center (degrees).
    width_km, height_km : float
        Grid extent in kilometers.
    dx_km, dy_km : float
        Grid spacing in kilometers.
    buffer_km : float
        Extra margin added to both extents.

    Returns
    -------
    lon, lat : np.ndarray
        Flattened grid node coordinates in degrees.

    """
    crs_aeqd = CRS(proj="aeqd", lon_0=lon0, lat_0=lat0, datum="WGS84", units="km")
    transformer = Transformer.from_crs(
        CRS(proj="latlong", datum="WGS84"), crs_aeqd, always_xy=True
    )
    w, h = width_km + buffer_km, height_km + buffer_km
    xs = np.arange(-w / 2 - dx_km, w / 2 + dx_km, dx_km)
    ys = np.arange(-h / 2 - dy_km, h / 2 + dy_km, dy_km)
    xi, yi = np.meshgrid(xs, ys)
    lon, lat = transformer.transform(xi.ravel(), yi.ravel(), direction="INVERSE")
    return np.asarray(lon), np.asarray(lat)

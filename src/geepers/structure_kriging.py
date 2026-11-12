# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Image station rates as a smooth surface with uncertainties.

GPS Imaging (`geepers.gps_imaging`) sets a point to the weighted median of the
stations linked to it. That is robust, but the surface steps wherever the median
changes station, and its uncertainty has no term for the distance to the stations.
This module interpolates the same rates by local kriging, using the quantity GPS
Imaging already measures: the spatial structure function (Hammond et al., 2016,
doi:10.1002/2016JB013458).

The structure function is the inverse of how the scatter between station rates grows
with their separation. That curve, measured from the stations, is a variogram: it says
how far the field can be trusted to be smooth, so the weights and the uncertainty of a
point both follow from it.

    g(d)      scatter^2 of one station against another d km away
    SSF(d)    sqrt(g(0) / g(d)), the curve GPS Imaging weights with
    a         local amplitude: the same curve, scaled to how rough the field is nearby

Use the median filter to decide which stations are outliers, then krige the remaining
raw rates. Kriging the filtered rates themselves is not safe.

With ``trend=True`` a local plane is part of the estimate: measure the structure on
rates with their local plane removed (`detrend`), and the plane is restored in the
same solve (universal kriging with a linear drift), so the uncertainty carries that of
the plane. Without it a tectonic gradient counts as roughness and the uncertainty is
too large.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from geepers.gps_imaging import Breaklines, crosses_breaklines

__all__ = ["detrend", "implied_ssf", "krige", "structure"]

EARTH_RADIUS_KM = 6371.0
KM_PER_DEGREE = 111.195
EDGES_KM = (0.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0)
N_NEIGHBORS = 20
# Extra candidates, for when breaklines remove some
N_SPARE = 10
MIN_NEIGHBORS = 3
# Fewer stations than this cannot hold a plane; the mean is used
MIN_FOR_TREND = 8
# On held-out UNR stations a plane helps within 30 km, is neutral to 150 km and hurts
# beyond
TREND_MAX_KM = 150.0
# A plane is used only if it at least halves the scatter of the stations
TREND_GAIN = 2.0
AMPLITUDE_LIMITS = (0.05, 50.0)
# Median of a chi-square variable with one degree of freedom
CHI2_MEDIAN = 0.4549
MIN_PAIRS = 20


def _unit(lon: ArrayLike, lat: ArrayLike) -> np.ndarray:
    """Return unit vectors, so that a k-d tree measures chord distance on the sphere."""
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def _robust_variance(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size <= MIN_PAIRS:
        return np.nan
    return float((1.4826 * np.median(np.abs(x - np.median(x)))) ** 2)


def structure(
    lon: ArrayLike, lat: ArrayLike, values: ArrayLike, edges_km: ArrayLike = EDGES_KM
) -> np.ndarray:
    """Measure how the scatter between stations grows with their separation.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    values : array-like
        Station values, such as rates in mm/yr. Pass `detrend` of them to krige with
        ``trend=True``.
    edges_km : array-like
        Edges of the separation bins in km.

    Returns
    -------
    np.ndarray
        (n_bins + 1, 2) of separation in km and the squared scatter of one station
        against another that far away, non-decreasing. The first row is zero lag, taken
        from the first bin. Scatter is robust (MAD): a few bad stations do not set it.

    """
    values = np.asarray(values, float)
    edges = np.asarray(edges_km, float)
    xyz = _unit(lon, lat)
    reach = 2 * np.sin(edges[-1] / EARTH_RADIUS_KM / 2)
    i, j = cKDTree(xyz).query_pairs(reach, output_type="ndarray").T
    separation = np.linalg.norm(xyz[i] - xyz[j], axis=1) * EARTH_RADIUS_KM
    difference = values[i] - values[j]
    g = [
        _robust_variance(difference[(separation > a) & (separation <= b)]) / 2
        for a, b in itertools.pairwise(edges)
    ]
    g = pd.Series(g).interpolate(limit_direction="both").to_numpy()
    g = np.maximum.accumulate(g)
    middle = (edges[:-1] + edges[1:]) / 2
    return np.c_[np.r_[0.0, middle], np.r_[g[0], g]]


def implied_ssf(g: np.ndarray) -> np.ndarray:
    """Return the spatial structure function a `structure` curve corresponds to.

    Parameters
    ----------
    g : np.ndarray
        Output of `structure`.

    Returns
    -------
    np.ndarray
        (n, 2) of separation in km and weight in (0, 1], comparable with
        `geepers.gps_imaging.make_ssf` and `geepers.gridding.atbd_ssf`.

    """
    return np.c_[g[:, 0], np.sqrt(g[0, 1] / g[:, 1])]


def _plane(lon: ArrayLike, lat: ArrayLike, lon0: float, lat0: float) -> np.ndarray:
    """Design matrix of a plane in local east and north, in units of 100 km."""
    x = (np.asarray(lon) - lon0) * np.cos(np.radians(lat0)) * KM_PER_DEGREE / 100.0
    y = (np.asarray(lat) - lat0) * KM_PER_DEGREE / 100.0
    return np.c_[np.ones(len(x)), x, y]


def detrend(
    lon: ArrayLike, lat: ArrayLike, values: ArrayLike, n_neighbors: int = N_NEIGHBORS
) -> np.ndarray:
    """Remove from each station the plane through its nearest stations.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    values : array-like
        Station values.
    n_neighbors : int
        Stations per plane, the station itself included.

    Returns
    -------
    np.ndarray
        The value of each station minus its local plane, for `structure`.

    """
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    values = np.asarray(values, float)
    xyz = _unit(lon, lat)
    _, index = cKDTree(xyz).query(xyz, k=min(n_neighbors, len(lon)))
    out = np.empty(len(values))
    for p, near in enumerate(index):
        design = _plane(lon[near], lat[near], lon[p], lat[p])
        beta = np.linalg.lstsq(design, values[near], rcond=None)[0]
        out[p] = values[p] - beta[0]
    return out


def _g(g: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Evaluate a structure curve, continued in a straight line beyond its last bin."""
    slope = (g[-1, 1] - g[-2, 1]) / (g[-1, 0] - g[-2, 0])
    beyond = g[-1, 1] + slope * (d - g[-1, 0])
    return np.where(d <= g[-1, 0], np.interp(d, g[:, 0], g[:, 1]), beyond)


def _solve(
    v: np.ndarray,
    s2: np.ndarray,
    d0: np.ndarray,
    between: np.ndarray,
    drift: np.ndarray,
    curve: np.ndarray,
    positive: bool,
) -> tuple[float, float, float, float]:
    """Krige one point: value, smooth and total variance, and local amplitude."""
    n, m = drift.shape
    shape = _g(curve, between)
    # Residuals about the drift tell how rough the field is here
    w0 = 1.0 / np.sqrt(s2 + curve[0, 1])
    fit = np.linalg.lstsq(drift * w0[:, None], v * w0, rcond=None)[0]
    residual = v - drift @ fit
    iu = np.triu_indices(n, 1)
    ratio = ((residual[:, None] - residual[None]) ** 2)[iu] / (2 * shape[iu])
    amplitude = np.median(ratio) / CHI2_MEDIAN * n / (n - m)
    amplitude = float(np.clip(amplitude, *AMPLITUDE_LIMITS))
    nugget = amplitude * curve[0, 1]

    # Variogram form of universal kriging. A station noisier than the local zero-lag
    # scatter gets the excess as its own error, so it weighs less. The drift rows make
    # the weights reproduce the drift exactly, and their multipliers put its uncertainty
    # into the variance.
    gamma0 = amplitude * _g(curve, d0)
    active = np.arange(n)
    while True:
        k = len(active)
        system = np.zeros((k + m, k + m))
        system[:k, :k] = amplitude * shape[np.ix_(active, active)]
        system[np.arange(k), np.arange(k)] = -np.maximum(s2[active] - nugget, 0.0)
        system[:k, k:] = drift[active]
        system[k:, :k] = drift[active].T
        solution = np.linalg.solve(system, np.r_[gamma0[active], 1.0, np.zeros(m - 1)])
        # About a constant mean, a negative weight means a station hidden behind nearer
        # ones is used to extrapolate their gradient, and the point can leave the range
        # of the stations. Such stations are dropped and the point solved again.
        negative = solution[:k] < 0
        few = k - negative.sum() < MIN_NEIGHBORS
        if not positive or m > 1 or not negative.any() or few:
            break
        active = active[~negative]
    weights = np.zeros(n)
    weights[active] = solution[:k]
    total = float(weights @ gamma0 + solution[k])
    if positive and m == 1 and (weights < 0).any():
        # Too few stations would be left to solve again: the negative weights are set to
        # zero and the rest rescaled. The variance of any weights that sum to one is
        # 2 w.gamma0 - w.Gamma.w, plus the stations' own excess error.
        weights = np.maximum(weights, 0.0)
        weights /= weights.sum()
        excess = np.maximum(s2 - nugget, 0.0)
        spread = weights @ (amplitude * shape) @ weights
        total = float(2 * weights @ gamma0 - spread + weights**2 @ excess)
    total = max(total, 0.0)
    # The smooth field cannot be known better than the mean of n noisy stations
    smooth = max(total - nugget, nugget / n)
    return float(weights @ v), smooth, total, amplitude


def krige(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    sigmas: ArrayLike,
    lon_new: ArrayLike,
    lat_new: ArrayLike,
    g: np.ndarray,
    *,
    breaklines: Breaklines | None = None,
    trend: bool = False,
    g_plain: np.ndarray | None = None,
    positive: bool = False,
    n_neighbors: int = N_NEIGHBORS,
) -> pd.DataFrame:
    """Krige station values onto points from their nearest stations.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    values, sigmas : array-like
        Station values and 1-sigma uncertainties.
    lon_new, lat_new : array-like
        Points to estimate, in degrees.
    g : np.ndarray
        `structure` of what is left about the drift: of `detrend` of the values with
        ``trend=True``, of the values themselves without.
    breaklines : sequence of array-like, optional
        Polylines of (lon, lat). Stations across them from a point are not used; see
        `geepers.gps_imaging.crosses_breaklines`.
    trend : bool
        Estimate about a local plane, where at least 8 stations lie within a mean of
        150 km. Elsewhere, and by default, about a constant mean.
    g_plain : np.ndarray, optional
        `structure` of the values themselves. If given with ``trend=True``, the plane is
        used only where it at least halves the scatter of the stations, and the constant
        mean with this curve elsewhere.
    positive : bool
        Keep the weights non-negative about a constant mean, and solve a point again
        about a constant mean if its plane estimate leaves the range of the stations.
        Stops extrapolation beyond the stations' own values far from them.
    n_neighbors : int
        Stations per point.

    Returns
    -------
    pd.DataFrame
        One row per point: ``value``; ``sigma``, of the smooth field (the kriging sigma
        without the local zero-lag scatter); ``sigma_total``, with it (what a station
        there would show; use it where ``planar`` is 1, since the scatter about a plane
        is unmodelled signal); ``amplitude``, the local roughness against the
        network's; ``n``, stations used; ``km``, their mean distance; ``planar``, 1 if
        the plane was used. NaN where fewer than 3 stations are on the point's side.

    """
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    values, sigmas = np.asarray(values, float), np.asarray(sigmas, float)
    lon_new = np.atleast_1d(np.asarray(lon_new, float))
    lat_new = np.atleast_1d(np.asarray(lat_new, float))
    xyz, new = _unit(lon, lat), _unit(lon_new, lat_new)
    chord, index = cKDTree(xyz).query(new, k=min(n_neighbors + N_SPARE, len(lon)))
    usable = np.ones(index.shape, bool)
    if breaklines:
        usable = ~crosses_breaklines(
            lon[index], lat[index], lon_new[:, None], lat_new[:, None], breaklines
        )

    columns = ["value", "sigma", "sigma_total", "amplitude", "n", "km", "planar"]
    out = np.full((len(lon_new), len(columns)), np.nan)
    for p in range(len(lon_new)):
        near = index[p][usable[p]][:n_neighbors]
        n = len(near)
        if n < MIN_NEIGHBORS:
            continue
        d0 = chord[p][usable[p]][:n_neighbors] * EARTH_RADIUS_KM
        apart = xyz[near][:, None] - xyz[near][None]
        between = np.linalg.norm(apart, axis=2) * EARTH_RADIUS_KM
        v, s2 = values[near], sigmas[near] ** 2
        design = _plane(lon[near], lat[near], lon_new[p], lat_new[p])
        # A plane is only trusted between stations: far from them it extrapolates
        planar = trend and n >= MIN_FOR_TREND and d0.mean() <= TREND_MAX_KM
        if planar and g_plain is not None:
            about_plane = v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
            planar = np.var(v) * (n - 3) > TREND_GAIN * np.sum(about_plane**2)
        for use_plane in (True, False) if planar else (False,):
            drift = design if use_plane else np.ones((n, 1))
            curve = g if use_plane or g_plain is None else g_plain
            value, smooth, total, amplitude = _solve(
                v, s2, d0, between, drift, curve, positive
            )
            if not (use_plane and positive) or v.min() <= value <= v.max():
                break
        out[p] = (
            value,
            np.sqrt(smooth),
            np.sqrt(total),
            amplitude,
            n,
            d0.mean(),
            use_plane,
        )
    return pd.DataFrame(out, columns=columns)

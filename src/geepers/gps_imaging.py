# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""GPS Imaging: robust median-spatial-filter interpolation of velocities.

Python port of the GPS Imaging MATLAB codes by Bill Hammond (Nevada
Geodetic Laboratory, University of Nevada, Reno), v6 Zenodo release.
The method interpolates scattered (typically vertical) GNSS velocities
with Delaunay-neighborhood *weighted medians*, weighted by a data-driven
spatial structure function (SSF) and the station uncertainties - robust
to outliers, tolerant of heterogeneous networks, and edge-preserving
(no explicit smoothing).

When using this method, please cite:

    Hammond, W. C., Blewitt, G., & Kreemer, C. (2016). GPS Imaging of
    vertical land motion in California and Nevada: Implications for
    Sierra Nevada uplift. Journal of Geophysical Research: Solid Earth,
    121(10), 7681-7703. doi:10.1002/2016JB013458

The optional robust-network-filter neighborhood expansion follows
Kreemer, C., Hammond, W. C., & Blewitt, G. (2020),
doi:10.1029/2020GL087976.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay

from geepers._optional import require

__all__ = [
    "DEFAULT_BREAKLINE_WEIGHT",
    "crosses_breaklines",
    "iterated_median_spatial_filter",
    "make_ssf",
    "median_spatial_filter",
    "msf_interpolate",
    "weighted_median",
]

logger = logging.getLogger("geepers")

Method = Literal["weighted_median", "median", "weighted_mean"]
Breaklines = Sequence[ArrayLike]

# Better than dropping crossing links on held-out stations at the San Andreas, and as
# good or better against published rates beside the Gulf of California. See the UNR
# grid notebook.
DEFAULT_BREAKLINE_WEIGHT = 0.25


def great_circle_degrees(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> np.ndarray:
    """Great-circle angular distance in degrees (spherical Earth)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    cosd = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))


def crosses_breaklines(
    lon: ArrayLike,
    lat: ArrayLike,
    lon_target: ArrayLike,
    lat_target: ArrayLike,
    breaklines: Breaklines,
) -> np.ndarray:
    """Find the baselines that cross a breakline, such as a fault or plate boundary.

    A baseline crossing a breakline joins points whose motion may be discontinuous,
    so the imaging functions drop it from their neighborhoods. Lines are straight in
    lon/lat, matching the triangulation the baselines come from. A baseline that ends
    exactly on a breakline, or runs around the end of one, does not cross it.

    Parameters
    ----------
    lon, lat : array-like
        One end of each baseline, in degrees.
    lon_target, lat_target : array-like
        The other end, in degrees; broadcast against `lon` and `lat`.
    breaklines : sequence of array-like
        Polylines, each (n_vertices, 2) of (lon, lat) in degrees.

    Returns
    -------
    np.ndarray
        Boolean array, True where the baseline crosses any breakline.

    """
    px, py, qx, qy = np.broadcast_arrays(
        *(
            np.atleast_1d(np.asarray(a, float))
            for a in (lon, lat, lon_target, lat_target)
        )
    )
    out = np.zeros(px.shape, bool)
    lines = [np.asarray(line, float) for line in breaklines]
    lines = [line for line in lines if len(line) >= 2]
    if not lines or not px.size:
        return out
    start = np.concatenate([line[:-1] for line in lines])
    end = np.concatenate([line[1:] for line in lines])
    # Most segments are far from any baseline, and this keeps the loop below short
    near = (
        (np.minimum(start[:, 0], end[:, 0]) <= max(px.max(), qx.max()))
        & (np.maximum(start[:, 0], end[:, 0]) >= min(px.min(), qx.min()))
        & (np.minimum(start[:, 1], end[:, 1]) <= max(py.max(), qy.max()))
        & (np.maximum(start[:, 1], end[:, 1]) >= min(py.min(), qy.min()))
    )
    for (ax, ay), (bx, by) in zip(start[near], end[near], strict=True):
        a_side = (qx - px) * (ay - py) - (qy - py) * (ax - px)
        b_side = (qx - px) * (by - py) - (qy - py) * (bx - px)
        p_side = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        q_side = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
        out |= (a_side * b_side < 0) & (p_side * q_side < 0)
    return out


def weighted_median(values: ArrayLike, weights: ArrayLike) -> float:
    """Weighted median: smallest x with cumulative weight >= 1/2.

    Parameters
    ----------
    values : array-like
        Observed values.
    weights : array-like
        Positive weights (normalized internally).

    Returns
    -------
    float

    """
    v = np.asarray(values, float).ravel()
    w = np.asarray(weights, float).ravel()
    if v.shape != w.shape:
        msg = "values and weights must have the same shape"
        raise ValueError(msg)
    order = np.argsort(v)
    w = w[order] / w.sum()
    return float(v[order][np.searchsorted(np.cumsum(w), 0.5)])


def weighted_medians(values: ArrayLike, weights: ArrayLike) -> np.ndarray:
    """Return `weighted_median` over the first axis, at once for all other axes.

    Parameters
    ----------
    values : array-like
        (station, ...) values.
    weights : array-like
        Positive weights, one per station (normalized internally).

    Returns
    -------
    np.ndarray
        The shape of `values` without the first axis.

    """
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    if w.shape != v.shape[:1]:
        msg = "weights must have one value per station"
        raise ValueError(msg)
    order = np.argsort(v, axis=0)
    cumulative = np.cumsum((w / w.sum())[order], axis=0)
    winner = np.take_along_axis(order, np.argmax(cumulative >= 0.5, axis=0)[None], 0)
    return np.take_along_axis(v, winner, 0)[0]


def make_ssf(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    sigmas: ArrayLike | None = None,
    *,
    max_difference: float = 10.0,
    log_bin_edges: np.ndarray | None = None,
) -> np.ndarray:
    """Spatial structure function (Hammond et al., 2016, eqs. 1-2).

    Bins the pairwise value differences by great-circle separation and
    computes the MAD (median absolute deviation about the bin median)
    per bin, forced to be non-decreasing with distance; the zero-lag bin
    is anchored at the median measurement uncertainty, or, without
    uncertainties, the curve is flat at 1 up to the first populated bin
    (Hammond et al., 2016, sec. 3.1). The inverted, normalized curve is
    used as the distance-weighting function of the median spatial filter.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    values : array-like
        Station values (e.g. vertical velocities, mm/yr).
    sigmas : array-like, optional
        1-sigma value uncertainties (anchor the zero-distance bin). If
        omitted, the near field is set to 1 instead.
    max_difference : float
        Pairs with ``|dv|`` larger than this are excluded. Default 10.
    log_bin_edges : np.ndarray, optional
        log10-degree bin edges. Default ``[-2, -1, -0.75, ..., 1.25]``.

    Returns
    -------
    np.ndarray
        (n_bins + 2, 2) array of (distance in degrees, SSF in [0, 1]),
        anchored at (0, 1) and (180, 0), monotonically non-increasing.

    """
    if log_bin_edges is None:
        log_bin_edges = np.r_[-2.0, np.arange(-1.0, 1.26, 0.25)]

    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    values = np.asarray(values, float)

    iu, ju = np.triu_indices(len(values), k=1)
    dv = values[iu] - values[ju]
    dist = great_circle_degrees(lat[iu], lon[iu], lat[ju], lon[ju])
    good = np.isfinite(dv) & (np.abs(dv) < max_difference)
    dv, dist = dv[good], dist[good]

    n_bins = len(log_bin_edges) - 1
    centers = 10 ** (0.5 * (log_bin_edges[:-1] + log_bin_edges[1:]))
    scatter = np.full(n_bins, np.nan)
    if sigmas is not None:
        # zero-lag: measurement noise floor
        scatter[0] = np.nanmedian(np.asarray(sigmas, float))
    for i in range(1, n_bins):
        in_bin = (dist >= 10 ** log_bin_edges[i]) & (dist <= 10 ** log_bin_edges[i + 1])
        mad = (
            np.median(np.abs(dv[in_bin] - np.median(dv[in_bin])))
            if in_bin.any()
            else np.nan
        )
        # Force non-decreasing scatter with distance (Hammond et al.)
        prev = scatter[i - 1]
        scatter[i] = (
            mad if np.isnan(prev) else (prev if np.isnan(mad) else max(prev, mad))
        )

    with np.errstate(divide="ignore"):
        ssf = 1.0 / scatter
    ssf[~np.isfinite(ssf)] = np.nan
    ssf /= np.nanmax(ssf) if np.isfinite(ssf).any() else 1.0
    # Bins before the first populated one (or a zero-scatter field): flat at 1
    ssf[np.isnan(ssf)] = 1.0
    # Taper the tail smoothly to zero
    ssf[-1] = 0.0
    ssf[-2] = 0.5 * (ssf[-3] + ssf[-1])
    ssf[-3] = 0.5 * (ssf[-4] + ssf[-2])

    return np.vstack([[0.0, 1.0], np.c_[centers, ssf], [180.0, 0.0]])


def _collapse_duplicates(
    lon: np.ndarray, lat: np.ndarray, values: np.ndarray, sigmas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Merge stations sharing identical coordinates (median value, min sigma).

    Zero-length baselines break the Delaunay triangulation; the returned
    index map allows scattering results back to the original stations.
    """
    key = np.stack([lon, lat], axis=1)
    _, first_idx, inverse = np.unique(
        key, axis=0, return_index=True, return_inverse=True
    )
    if len(first_idx) == len(lon):
        # Nothing to merge, which is nearly every call: skip the per-group medians
        groups = [first_idx[g : g + 1] for g in range(len(first_idx))]
        return (
            lon[first_idx],
            lat[first_idx],
            values[first_idx],
            sigmas[first_idx],
            groups,
        )
    groups = [np.flatnonzero(inverse == g) for g in range(len(first_idx))]
    lon2 = np.array([lon[g[0]] for g in groups])
    lat2 = np.array([lat[g[0]] for g in groups])
    v2 = np.array([np.median(values[g]) for g in groups])
    s2 = np.array([np.min(sigmas[g]) for g in groups])
    return lon2, lat2, v2, s2, groups


def tri_vertex_neighbors(tri: Delaunay) -> list[np.ndarray]:
    """For each point, the unique vertices of all simplices containing it."""
    pointer, others = tri.vertex_neighbor_vertices
    # A point in no simplex, which Qhull leaves out when coplanar, has no neighbors
    return [
        np.sort(np.r_[others[a:b], i]) if b > a else np.empty(0, int)
        for i, (a, b) in enumerate(itertools.pairwise(pointer))
    ]


def linked_stations(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_point: float,
    lat_point: float,
    neighbors: Sequence[np.ndarray],
) -> np.ndarray:
    """Return the stations linked to a point, as `msf_interpolate` chooses them.

    Use this to reuse one neighborhood for many values at the same stations. Stations
    must have distinct coordinates.

    Parameters
    ----------
    lon, lat : np.ndarray
        Station coordinates in degrees.
    lon_point, lat_point : float
        The point.
    neighbors : sequence of np.ndarray
        Per station, the vertices of the Delaunay triangles around it, itself included.

    Returns
    -------
    np.ndarray
        Station indices, empty if the local geometry leaves none.

    """
    nearest = int(np.argmin(great_circle_degrees(lat_point, lon_point, lat, lon)))
    around = neighbors[nearest]
    if around.size == 0:
        return np.empty(0, int)
    local = np.unique(np.concatenate([neighbors[k] for k in around]))
    if local.size <= 2:
        return np.empty(0, int)
    # A station exactly at the point stays linked, while the point takes its place
    same = (lon[local] == lon_point) & (lat[local] == lat_point)
    others = local[~same]
    point = others.size
    tri = Delaunay(np.c_[np.r_[lon[others], lon_point], np.r_[lat[others], lat_point]])
    linked = others[np.setdiff1d(tri_vertex_neighbors(tri)[point], [point])]
    return np.r_[linked, local[same]]


def _neighborhood_estimate(
    dist: np.ndarray,
    v: np.ndarray,
    sv: np.ndarray,
    ssf: np.ndarray | None,
    method: Method,
    link_scale: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Weighted estimate over one neighborhood; returns (value, weights)."""
    if method == "median":
        return float(np.median(v)), np.full(len(v), 1.0 / len(v))

    if ssf is None:
        with np.errstate(divide="ignore"):
            w = 1.0 / dist
        w[~np.isfinite(w)] = np.nanmax(w[np.isfinite(w)]) if np.isfinite(w).any() else 1
    else:
        w = np.interp(dist, ssf[:, 0], ssf[:, 1])
    w = w / sv
    if link_scale is not None:
        w = w * link_scale
    if np.all(w == 0):
        w = np.ones_like(w)
    w = w / w.sum()

    if method == "weighted_median":
        return weighted_median(v, w), w
    return float(np.sum(v * w)), w  # weighted_mean


def _check_breakline_weight(breakline_weight: float) -> None:
    if not 0.0 <= breakline_weight <= 1.0:
        msg = f"breakline_weight must be in [0, 1], got {breakline_weight}"
        raise ValueError(msg)


def median_spatial_filter(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    sigmas: ArrayLike,
    ssf: np.ndarray | None = None,
    *,
    method: Method = "weighted_median",
    robust_network: bool = False,
    breaklines: Breaklines | None = None,
    breakline_weight: float = DEFAULT_BREAKLINE_WEIGHT,
) -> np.ndarray:
    """Median spatial filter of a velocity field at the stations.

    Replaces each station value with the (weighted) median of its
    Delaunay neighborhood (station included), weighted by the SSF value
    at the neighbor distance divided by the neighbor uncertainty.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    values : array-like
        Station values.
    sigmas : array-like
        1-sigma uncertainties.
    ssf : np.ndarray, optional
        Spatial structure function from `make_ssf`. If None, 1/distance
        weighting is used.
    method : {"weighted_median", "median", "weighted_mean"}
        Neighborhood estimator. Default "weighted_median".
    robust_network : bool
        Expand each neighborhood with all stations closer than the
        median Delaunay-neighbor distance (Kreemer et al., 2020).
    breaklines : sequence of array-like, optional
        Polylines of (lon, lat). See `crosses_breaklines`.
    breakline_weight : float
        Multiplies the weight of links crossing `breaklines`, so that most of an
        estimate comes from one side. 0 drops them and 1 ignores the breaklines. The
        ``"median"`` method has no weights, so it always drops them.

    Returns
    -------
    np.ndarray
        Filtered values, same length/order as the input.

    """
    _check_breakline_weight(breakline_weight)
    if method == "median":
        breakline_weight = 0.0  # no weights to scale, so crossing links are dropped
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    values = np.asarray(values, float)
    sigmas = np.asarray(sigmas, float)

    lon2, lat2, v2, s2, groups = _collapse_duplicates(lon, lat, values, sigmas)
    tri = Delaunay(np.c_[lon2, lat2])
    neighborhoods = tri_vertex_neighbors(tri)

    out2 = np.full(len(lon2), np.nan)
    for i, iw in enumerate(neighborhoods):
        if iw.size == 0:
            continue
        idx = iw
        if robust_network:
            dist_all = great_circle_degrees(lat2[i], lon2[i], lat2, lon2)
            md = np.median(dist_all[idx])
            idx = np.union1d(idx, np.flatnonzero(dist_all <= md))
        scale = None
        if breaklines is not None and breakline_weight < 1:
            crossing = crosses_breaklines(
                lon2[idx], lat2[idx], lon2[i], lat2[i], breaklines
            )
            if breakline_weight == 0:
                idx = idx[~crossing]
            else:
                scale = np.where(crossing, breakline_weight, 1.0)
        dist = great_circle_degrees(lat2[i], lon2[i], lat2[idx], lon2[idx])
        out2[i], _ = _neighborhood_estimate(dist, v2[idx], s2[idx], ssf, method, scale)

    out = np.full(len(lon), np.nan)
    for i, g in enumerate(groups):
        out[g] = out2[i]
    return out


def iterated_median_spatial_filter(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    sigmas: ArrayLike,
    ssf: np.ndarray,
    *,
    n_iterations: int = 30,
    perturbation: float = 0.2,
    seed: int = 0,
    breaklines: Breaklines | None = None,
    breakline_weight: float = DEFAULT_BREAKLINE_WEIGHT,
) -> np.ndarray:
    """Median of repeated median spatial filters, each with a perturbed SSF.

    Makes the result insensitive to the exact structure function, as the OPERA
    DISP-CAL ATBD section 2.5.3 does for the UNR gridded product. Every SSF value
    is moved independently by up to `perturbation`, then kept within [0, 1] and
    non-increasing.

    Parameters
    ----------
    lon, lat, values, sigmas
        As in `median_spatial_filter`.
    ssf : np.ndarray
        Spatial structure function from `make_ssf`.
    n_iterations : int
        Number of filtered fields. Default 30.
    perturbation : float
        Largest fractional change of an SSF value. Default 0.2.
    seed : int
        Seed for the perturbations.
    breaklines : sequence of array-like, optional
        Passed to `median_spatial_filter`.
    breakline_weight : float
        Passed to `median_spatial_filter`.

    Returns
    -------
    np.ndarray
        Filtered values, same length/order as the input.

    """
    rng = np.random.default_rng(seed)
    fields = []
    for _ in range(n_iterations):
        moved = np.array(ssf, float)
        moved[1:-1, 1] *= 1 + perturbation * rng.uniform(-1, 1, len(moved) - 2)
        moved[:, 1] = np.minimum.accumulate(np.clip(moved[:, 1], 0, 1))
        fields.append(
            median_spatial_filter(
                lon,
                lat,
                values,
                sigmas,
                moved,
                breaklines=breaklines,
                breakline_weight=breakline_weight,
            )
        )
    return np.median(fields, axis=0)


def msf_interpolate(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    sigmas: ArrayLike,
    lon_new: ArrayLike,
    lat_new: ArrayLike,
    ssf: np.ndarray | None = None,
    *,
    method: Method = "weighted_median",
    robust_network: bool = False,
    breaklines: Breaklines | None = None,
    breakline_weight: float = DEFAULT_BREAKLINE_WEIGHT,
    background: ArrayLike | None = None,
    background_weight: ArrayLike | None = None,
) -> pd.DataFrame:
    """Interpolate station values onto new points with the MSF.

    For each evaluation point, the local level-2 Delaunay neighborhood
    of the nearest station is re-triangulated with the point inserted;
    the stations connected to it contribute a weighted-median estimate.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    values, sigmas : array-like
        Station values and 1-sigma uncertainties.
    lon_new, lat_new : array-like
        Evaluation points in degrees.
    ssf : np.ndarray, optional
        Spatial structure function from `make_ssf` (1/distance weights
        if None).
    method : {"weighted_median", "median", "weighted_mean"}
        Neighborhood estimator. Default "weighted_median".
    robust_network : bool
        Use the full network within the median neighbor distance instead
        of only Delaunay-connected stations (Kreemer et al., 2020).
    breaklines : sequence of array-like, optional
        Polylines of (lon, lat), see `crosses_breaklines`. A point with every link
        crossing them is triangulated again using the stations on its side.
    breakline_weight : float
        Multiplies the weight of links crossing `breaklines`, so that most of an
        estimate comes from one side. 0 drops them and 1 ignores the breaklines. A point
        with every link crossing keeps those links at this weight, alongside its own
        side's stations. The ``"median"`` method has no weights, so it always drops
        them.
    background : array-like, optional
        One extra datum per evaluation point, such as a plate motion or GIA model,
        inserted into the estimate. NaN means none at that point.
    background_weight : array-like, optional
        The weight of `background` in [0, 1]. Station weights are scaled to sum to one
        minus this. Requires `background` and a weighted `method`.

    Returns
    -------
    pd.DataFrame
        One row per evaluation point with columns ``value``,
        ``sigma_formal`` (propagated weights x sigmas), ``sigma_rms``
        (scatter of contributing stations about the estimate),
        ``sigma_robust`` (1.4826 x MAD of that scatter),
        ``sigma_median`` (median sigma of the contributing stations) and
        ``n_stations``. The sigmas describe the stations alone.

    Notes
    -----
    ``sigma_robust`` uses the median absolute deviation; the original
    MATLAB code calls ``mad()`` whose default is the *mean* absolute
    deviation, but scales by 1.4826, which is the consistency constant
    of the median-based MAD - we follow the evident intent.

    """
    _check_breakline_weight(breakline_weight)
    if method == "median":
        breakline_weight = 0.0  # no weights to scale, so crossing links are dropped
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    values = np.asarray(values, float)
    sigmas = np.asarray(sigmas, float)
    lon_new = np.atleast_1d(np.asarray(lon_new, float))
    lat_new = np.atleast_1d(np.asarray(lat_new, float))
    if (background is None) != (background_weight is None):
        msg = "background and background_weight must be given together"
        raise ValueError(msg)
    if background is not None and method == "median":
        msg = "a background needs a weighted method"
        raise ValueError(msg)
    extra = np.full(lon_new.shape, np.nan)
    extra_weight = np.zeros(lon_new.shape)
    if background is not None:
        extra = np.broadcast_to(np.asarray(background, float), lon_new.shape)
        extra_weight = np.broadcast_to(
            np.asarray(background_weight, float), lon_new.shape
        )

    lon2, lat2, v2, s2, _ = _collapse_duplicates(lon, lat, values, sigmas)
    tri0 = Delaunay(np.c_[lon2, lat2])
    nbr0 = tri_vertex_neighbors(tri0)

    rows = []
    for lo, la, bg, bg_w in zip(lon_new, lat_new, extra, extra_weight, strict=True):
        has_bg = np.isfinite(bg) and bg_w > 0
        row = {
            "value": bg if has_bg else np.nan,
            "sigma_formal": np.nan,
            "sigma_rms": np.nan,
            "sigma_robust": np.nan,
            "sigma_median": np.nan,
            "n_stations": 0,
        }
        if robust_network:
            iloc = np.arange(len(lon2))
        else:
            # Level-2 Delaunay neighborhood of the nearest station
            dist0 = great_circle_degrees(la, lo, lat2, lon2)
            imin = int(np.argmin(dist0))
            level1 = nbr0[imin]
            iloc = np.unique(np.concatenate([nbr0[k] for k in level1]))
        if len(iloc) <= 2:
            rows.append(row)
            continue

        # Insert the evaluation point and re-triangulate locally.
        # If it coincides with a station, that station moves to the
        # evaluation slot (its data still contributes).
        li, bi, vi, si = lon2[iloc], lat2[iloc], v2[iloc], s2[iloc]
        coincident = np.flatnonzero((li == lo) & (bi == la))
        if coincident.size:
            keep = np.setdiff1d(np.arange(len(li)), coincident)
            vpt, spt = vi[coincident[0]], si[coincident[0]]
            li, bi, vi, si = li[keep], bi[keep], vi[keep], si[keep]
            li, bi = np.r_[li, lo], np.r_[bi, la]
            vi, si = np.r_[vi, vpt], np.r_[si, spt]
            point_has_data = True
        else:
            li, bi = np.r_[li, lo], np.r_[bi, la]
            point_has_data = False

        q = len(li) - 1  # index of the evaluation point
        try:
            tri = Delaunay(np.c_[li, bi])
        except Exception:  # degenerate local geometry
            rows.append(row)
            continue
        iw = tri_vertex_neighbors(tri)[q]
        if robust_network:
            dista = great_circle_degrees(la, lo, bi, li)
            md = np.median(dista[iw])
            iw = np.union1d(iw, np.flatnonzero(dista <= md))
        if not point_has_data:
            iw = np.setdiff1d(iw, [q])
        scale = None
        if breaklines is not None and breakline_weight < 1:
            crossing = crosses_breaklines(li[iw], bi[iw], lo, la, breaklines)
            if crossing.all():
                far = iw
                li_far, bi_far, vi_far, si_far = li[far], bi[far], vi[far], si[far]
                li, bi, vi, si, iw = _same_side_neighbors(
                    lon2, lat2, v2, s2, lo, la, breaklines
                )
                if breakline_weight > 0:
                    # Scaling every link alike would change nothing
                    scale = np.r_[np.ones(iw.size), np.full(far.size, breakline_weight)]
                    iw = np.r_[iw, li.size + np.arange(far.size)]
                    li, bi = np.r_[li, li_far], np.r_[bi, bi_far]
                    vi, si = np.r_[vi, vi_far], np.r_[si, si_far]
            elif breakline_weight == 0:
                iw = iw[~crossing]
            else:
                scale = np.where(crossing, breakline_weight, 1.0)
        if iw.size == 0:
            rows.append(row)
            continue

        dist = great_circle_degrees(la, lo, bi[iw], li[iw])
        est, w = _neighborhood_estimate(dist, vi[iw], si[iw], ssf, method, scale)

        if has_bg:
            data = np.r_[vi[iw], bg]
            scaled = np.r_[w * (1 - bg_w), bg_w]
            est = (
                weighted_median(data, scaled)
                if method == "weighted_median"
                else float(np.sum(data * scaled))
            )

        resid = vi[iw] - est
        row.update(
            value=est,
            sigma_formal=float(np.sqrt(np.sum((w * si[iw]) ** 2))),
            sigma_rms=float(np.sqrt(np.mean(resid**2))),
            sigma_robust=float(1.4826 * np.median(np.abs(resid - np.median(resid)))),
            sigma_median=float(np.median(si[iw])),
            n_stations=int(iw.size),
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _same_side_neighbors(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    sigmas: np.ndarray,
    lon0: float,
    lat0: float,
    breaklines: Breaklines,
    max_stations: int = 45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Delaunay neighbors of a point among only the stations it can reach."""
    empty = np.array([], dtype=int)
    reachable = np.flatnonzero(~crosses_breaklines(lon, lat, lon0, lat0, breaklines))
    dist = great_circle_degrees(lat0, lon0, lat[reachable], lon[reachable])
    idx = reachable[np.argsort(dist)[:max_stations]]
    li, bi, vi, si = lon[idx], lat[idx], values[idx], sigmas[idx]
    if idx.size == 0:
        return li, bi, vi, si, empty
    if idx.size < 3:
        return li, bi, vi, si, np.arange(idx.size)
    try:
        tri = Delaunay(np.c_[np.r_[li, lon0], np.r_[bi, lat0]])
    except Exception:  # degenerate local geometry
        return li, bi, vi, si, np.arange(idx.size)
    return (
        li,
        bi,
        vi,
        si,
        np.setdiff1d(tri_vertex_neighbors(tri)[idx.size], [idx.size]),
    )


def reinterpolate_nodes(
    nodes: pd.DataFrame,
    exclude: Any,
    columns: tuple[tuple[str, str], ...] = (
        ("ve", "sigma_ve"),
        ("vn", "sigma_vn"),
        ("vu", "sigma_vu"),
    ),
    *,
    lon: str = "lon",
    lat: str = "lat",
    ssf: np.ndarray | None = None,
    method: Method = "weighted_median",
    robust_network: bool = False,
) -> pd.DataFrame:
    """Replace grid-node values inside exclusion areas by MSF estimates from outside.

    GNSS-side remove-restore for the calibration grid (Venti PRD R-G5): the
    UNR grid is a 25 km median-filtered interpolation, so a fast-subsiding
    basin or a coseismic area leaks into the surrounding nodes. Nodes inside
    the curated defo/event polygons are dropped from the input and
    re-estimated with the median spatial filter from the nodes outside, so the
    calibration surface sees only the regional field there.

    Parameters
    ----------
    nodes : pd.DataFrame
        Grid nodes with `lon`, `lat` and the value/sigma columns.
    exclude : shapely geometry, iterable of geometries, or GeoSeries/GeoDataFrame
        Exclusion areas (any object with ``union_all()`` or ``.geometry``
        works). Requires shapely (``geepers[grid]``).
    columns : tuple of (value, sigma) column-name pairs
        Pairs to re-estimate; pairs whose columns are absent are skipped.
    lon, lat : str
        Coordinate column names.
    ssf, method, robust_network
        Passed to `msf_interpolate`.

    Returns
    -------
    pd.DataFrame
        A copy of `nodes` with the inside values replaced, their sigmas set to
        the robust scatter of the contributing nodes (formal sigma where the
        scatter is undefined), and a boolean ``reinterpolated`` column.

    """
    shapely = require("shapely")
    if hasattr(exclude, "union_all"):
        geom = exclude.union_all()
    elif hasattr(exclude, "geometry"):
        geom = exclude.geometry.union_all()
    elif isinstance(exclude, shapely.Geometry):
        geom = exclude
    else:
        geom = shapely.union_all(list(exclude))

    x = nodes[lon].to_numpy(float)
    y = nodes[lat].to_numpy(float)
    inside = np.asarray(shapely.contains_xy(geom, x, y), dtype=bool)

    out = nodes.copy()
    out["reinterpolated"] = inside
    if not inside.any():
        return out
    if inside.all():
        msg = "every node lies inside the exclusion areas; nothing to interpolate from"
        raise ValueError(msg)

    for value_col, sigma_col in columns:
        if value_col not in out.columns or sigma_col not in out.columns:
            continue
        est = msf_interpolate(
            x[~inside],
            y[~inside],
            out.loc[~inside, value_col].to_numpy(float),
            out.loc[~inside, sigma_col].to_numpy(float),
            x[inside],
            y[inside],
            ssf,
            method=method,
            robust_network=robust_network,
        )
        sigma = est["sigma_robust"].to_numpy(float)
        formal = est["sigma_formal"].to_numpy(float)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, formal)
        out.loc[inside, value_col] = est["value"].to_numpy(float)
        out.loc[inside, sigma_col] = sigma
    logger.info(
        "Re-interpolated %d of %d grid nodes inside the exclusion areas",
        int(inside.sum()),
        inside.size,
    )
    return out

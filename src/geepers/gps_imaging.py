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

import logging
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay

__all__ = [
    "make_ssf",
    "median_spatial_filter",
    "msf_interpolate",
    "weighted_median",
]

logger = logging.getLogger("geepers")

Method = Literal["weighted_median", "median", "weighted_mean"]


def great_circle_degrees(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> np.ndarray:
    """Great-circle angular distance in degrees (spherical Earth)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    cosd = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))


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


def make_ssf(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    sigmas: ArrayLike,
    *,
    max_difference: float = 10.0,
    log_bin_edges: np.ndarray | None = None,
) -> np.ndarray:
    """Spatial structure function (Hammond et al., 2016, eqs. 1-2).

    Bins the pairwise value differences by great-circle separation and
    computes the MAD (median absolute deviation about the bin median)
    per bin, forced to be non-decreasing with distance; the zero-lag bin
    is anchored at the median measurement uncertainty. The inverted,
    normalized curve is used as the distance-weighting function of the
    median spatial filter.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    values : array-like
        Station values (e.g. vertical velocities, mm/yr).
    sigmas : array-like
        1-sigma value uncertainties (anchor the zero-distance bin).
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
    sigmas = np.asarray(sigmas, float)

    iu, ju = np.triu_indices(len(values), k=1)
    dv = values[iu] - values[ju]
    dist = great_circle_degrees(lat[iu], lon[iu], lat[ju], lon[ju])
    good = np.isfinite(dv) & (np.abs(dv) < max_difference)
    dv, dist = dv[good], dist[good]

    n_bins = len(log_bin_edges) - 1
    centers = 10 ** (0.5 * (log_bin_edges[:-1] + log_bin_edges[1:]))
    scatter = np.full(n_bins, np.nan)
    scatter[0] = np.nanmedian(sigmas)  # zero-lag: measurement noise floor
    for i in range(1, n_bins):
        in_bin = (dist >= 10 ** log_bin_edges[i]) & (dist <= 10 ** log_bin_edges[i + 1])
        mad = (
            np.median(np.abs(dv[in_bin] - np.median(dv[in_bin])))
            if in_bin.any()
            else np.nan
        )
        # Force non-decreasing scatter with distance (Hammond et al.)
        scatter[i] = np.nanmax(np.r_[scatter[:i], mad])

    ssf = 1.0 / scatter
    ssf /= np.nanmax(ssf)
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
    groups = [np.flatnonzero(inverse == g) for g in range(len(first_idx))]
    lon2 = np.array([lon[g[0]] for g in groups])
    lat2 = np.array([lat[g[0]] for g in groups])
    v2 = np.array([np.median(values[g]) for g in groups])
    s2 = np.array([np.min(sigmas[g]) for g in groups])
    return lon2, lat2, v2, s2, groups


def _tri_vertex_neighbors(tri: Delaunay) -> list[np.ndarray]:
    """For each point, the unique vertices of all simplices containing it."""
    n = tri.points.shape[0]
    sets: list[set[int]] = [set() for _ in range(n)]
    for simplex in tri.simplices:
        for i in simplex:
            sets[i].update(int(j) for j in simplex)
    return [np.fromiter(sorted(s), int) for s in sets]


def _neighborhood_estimate(
    dist: np.ndarray,
    v: np.ndarray,
    sv: np.ndarray,
    ssf: np.ndarray | None,
    method: Method,
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
    if np.all(w == 0):
        w = np.ones_like(w)
    w = w / w.sum()

    if method == "weighted_median":
        return weighted_median(v, w), w
    return float(np.sum(v * w)), w  # weighted_mean


def median_spatial_filter(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    sigmas: ArrayLike,
    ssf: np.ndarray | None = None,
    *,
    method: Method = "weighted_median",
    robust_network: bool = False,
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

    Returns
    -------
    np.ndarray
        Filtered values, same length/order as the input.

    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    values = np.asarray(values, float)
    sigmas = np.asarray(sigmas, float)

    lon2, lat2, v2, s2, groups = _collapse_duplicates(lon, lat, values, sigmas)
    tri = Delaunay(np.c_[lon2, lat2])
    neighborhoods = _tri_vertex_neighbors(tri)

    out2 = np.full(len(lon2), np.nan)
    for i, iw in enumerate(neighborhoods):
        if iw.size == 0:
            continue
        idx = iw
        if robust_network:
            dist_all = great_circle_degrees(lat2[i], lon2[i], lat2, lon2)
            md = np.median(dist_all[idx])
            idx = np.union1d(idx, np.flatnonzero(dist_all <= md))
        dist = great_circle_degrees(lat2[i], lon2[i], lat2[idx], lon2[idx])
        out2[i], _ = _neighborhood_estimate(dist, v2[idx], s2[idx], ssf, method)

    out = np.full(len(lon), np.nan)
    for i, g in enumerate(groups):
        out[g] = out2[i]
    return out


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

    Returns
    -------
    pd.DataFrame
        One row per evaluation point with columns ``value``,
        ``sigma_formal`` (propagated weights x sigmas), ``sigma_rms``
        (scatter of contributing stations about the estimate),
        ``sigma_robust`` (1.4826 x MAD of that scatter) and
        ``n_stations``.

    Notes
    -----
    ``sigma_robust`` uses the median absolute deviation; the original
    MATLAB code calls ``mad()`` whose default is the *mean* absolute
    deviation, but scales by 1.4826, which is the consistency constant
    of the median-based MAD - we follow the evident intent.

    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    values = np.asarray(values, float)
    sigmas = np.asarray(sigmas, float)
    lon_new = np.atleast_1d(np.asarray(lon_new, float))
    lat_new = np.atleast_1d(np.asarray(lat_new, float))

    lon2, lat2, v2, s2, _ = _collapse_duplicates(lon, lat, values, sigmas)
    tri0 = Delaunay(np.c_[lon2, lat2])
    nbr0 = _tri_vertex_neighbors(tri0)

    rows = []
    for lo, la in zip(lon_new, lat_new, strict=True):
        row = {
            "value": np.nan,
            "sigma_formal": np.nan,
            "sigma_rms": np.nan,
            "sigma_robust": np.nan,
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
        iw = _tri_vertex_neighbors(tri)[q]
        if robust_network:
            dista = great_circle_degrees(la, lo, bi, li)
            md = np.median(dista[iw])
            iw = np.union1d(iw, np.flatnonzero(dista <= md))
        if not point_has_data:
            iw = np.setdiff1d(iw, [q])
        if iw.size == 0:
            rows.append(row)
            continue

        dist = great_circle_degrees(la, lo, bi[iw], li[iw])
        est, w = _neighborhood_estimate(dist, vi[iw], si[iw], ssf, method)

        resid = vi[iw] - est
        row.update(
            value=est,
            sigma_formal=float(np.sqrt(np.sum((w * si[iw]) ** 2))),
            sigma_rms=float(np.sqrt(np.mean(resid**2))),
            sigma_robust=float(1.4826 * np.median(np.abs(resid - np.median(resid)))),
            n_stations=int(iw.size),
        )
        rows.append(row)

    return pd.DataFrame(rows)

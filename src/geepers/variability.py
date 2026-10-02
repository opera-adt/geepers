# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Spatial and temporal variability metrics for GNSS velocity fields.

Three related station-quality metrics:

- `spatial_structure_function` (SSF): how coherent a velocity field is
  as a function of station separation, following the GPS Imaging
  methodology (Hammond, W. C., Blewitt, G., & Kreemer, C., 2016, GPS
  Imaging of vertical land motion in California and Nevada,
  J. Geophys. Res. Solid Earth, 121, doi:10.1002/2016JB013458).
- `spatial_variability`: per-station RMS / MAD of the velocity
  differences with its Delaunay-network neighbors.
- `temporal_velocity_variability`: robust spread of MIDAS velocities
  estimated in sliding windows of increasing length around the
  full-series velocity — how stable a station's velocity is in time.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay

from geepers.gps_imaging import great_circle_degrees, make_ssf
from geepers.midas import midas

__all__ = [
    "TemporalVariability",
    "delaunay_neighbors",
    "spatial_structure_function",
    "spatial_variability",
    "ssf_per_station",
    "temporal_velocity_variability",
]

logger = logging.getLogger("geepers")

DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------------------
# Spatial structure function (GPS Imaging)
# ---------------------------------------------------------------------------
def spatial_structure_function(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    *,
    sigmas: ArrayLike | None = None,
    max_difference: float = 10.0,
    log_bin_edges: np.ndarray | None = None,
) -> np.ndarray:
    """Spatial structure function (SSF) of a scattered field.

    The GPS Imaging SSF (Hammond, W. C., Blewitt, G., & Kreemer, C., 2016,
    GPS Imaging of vertical land motion in California and Nevada,
    J. Geophys. Res. Solid Earth, 121, doi:10.1002/2016JB013458, sec. 3.1,
    eqs. 1-2), computed by `geepers.gps_imaging.make_ssf`: the MAD of the
    pairwise differences in log-distance bins of great-circle separation,
    forced to grow with distance, inverted and normalized to 1.

    Interpretation: each point is the coherence at that separation
    relative to the most coherent separation. A smooth field decays from
    1 toward 0 over the length scale on which it varies; a spatially
    uncorrelated field stays near 1 until the far-field taper.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    values : array-like
        Field values (e.g. vertical velocities). NaNs are ignored.
    sigmas : array-like, optional
        1-sigma value uncertainties; they set the zero-distance scatter.
        Without them the near field is flat at 1.
    max_difference : float
        Pairs with ``|dv|`` larger than this are treated as outliers and
        excluded (same units as `values`). Default 10.
    log_bin_edges : np.ndarray, optional
        log10-degree bin edges; default as in `make_ssf`.

    Returns
    -------
    np.ndarray
        (n_bins + 2, 2) array of (great-circle distance in degrees, SSF),
        anchored at (0, 1) and (180, 0) and non-increasing.

    """
    return make_ssf(
        lon,
        lat,
        values,
        sigmas,
        max_difference=max_difference,
        log_bin_edges=log_bin_edges,
    )


def delaunay_neighbors(lon: ArrayLike, lat: ArrayLike) -> dict[int, list[int]]:
    """Neighbor lists from the Delaunay triangulation of the stations.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.

    Returns
    -------
    dict[int, list[int]]
        For each station index, the indices of its Delaunay neighbors.

    """
    points = np.c_[np.asarray(lon, float), np.asarray(lat, float)]
    tri = Delaunay(points)
    neighbors: dict[int, set[int]] = defaultdict(set)
    for simplex in tri.simplices:
        for i, j in zip(simplex, np.roll(simplex, -1), strict=True):
            neighbors[i].add(int(j))
            neighbors[j].add(int(i))
    return {i: sorted(v) for i, v in sorted(neighbors.items())}


def ssf_per_station(
    lon: ArrayLike,
    lat: ArrayLike,
    components: dict[str, ArrayLike],
    *,
    sigmas: dict[str, ArrayLike] | None = None,
    max_difference: float = 10.0,
) -> pd.DataFrame:
    """Median SSF at the distances to each station's Delaunay neighbors.

    The resolution index of Hammond, W. C., Blewitt, G., Kreemer, C., &
    Nerem, R. S. (2021), GPS Imaging of global vertical land motion for
    studies of sea level rise, J. Geophys. Res. Solid Earth, 126,
    doi:10.1029/2021JB022355, eq. 4: one SSF per component is built from
    all stations (`spatial_structure_function`), and each station scores
    the median of that curve evaluated at the great-circle distances to
    its Delaunay neighbors.

    Interpretation: near 1, the neighbors are close compared with the
    length scale over which the field decorrelates, so the field is well
    resolved there; low values mark a network too sparse for the signal.
    It measures network geometry against the regional signal, not the
    station's own data quality (see `spatial_variability` for that).

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    components : dict[str, array-like]
        Mapping from component name (e.g. ``"east"``) to station values;
        one output column ``ssf_<name>`` per entry.
    sigmas : dict[str, array-like], optional
        Per-component 1-sigma uncertainties for the SSF zero-distance bin.
    max_difference : float
        Outlier cut on pairwise differences for the SSF. Default 10.

    Returns
    -------
    pd.DataFrame
        One row per station with columns ``ssf_n_neighbors`` and
        ``ssf_<component>``.

    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    neighbors = delaunay_neighbors(lon, lat)
    curves = {
        name: spatial_structure_function(
            lon,
            lat,
            np.asarray(vals, float),
            sigmas=None if sigmas is None else sigmas.get(name),
            max_difference=max_difference,
        )
        for name, vals in components.items()
    }

    rows = []
    for i, nbrs in neighbors.items():
        dist = great_circle_degrees(lat[i], lon[i], lat[nbrs], lon[nbrs])
        row: dict[str, float] = {"ssf_n_neighbors": len(nbrs)}
        for name, curve in curves.items():
            row[f"ssf_{name}"] = float(
                np.median(np.interp(dist, curve[:, 0], curve[:, 1]))
            )
        rows.append(row)
    return pd.DataFrame(rows, index=list(neighbors.keys()))


def spatial_variability(
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
) -> pd.DataFrame:
    """Per-station spread of the value differences with Delaunay neighbors.

    Parameters
    ----------
    lon, lat : array-like
        Station coordinates in degrees.
    values : array-like
        Field values (e.g. velocities) at the stations.

    Returns
    -------
    pd.DataFrame
        One row per station with columns ``rms`` (root-mean-square of
        neighbor differences) and ``mad`` (median absolute deviation of
        the differences about their median, unscaled).

    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    values = np.asarray(values, float)
    neighbors = delaunay_neighbors(lon, lat)

    out = np.full((len(values), 2), np.nan)
    for i, nbrs in neighbors.items():
        diff = values[np.array(nbrs)] - values[i]
        diff = diff[np.isfinite(diff)]
        if diff.size:
            out[i, 0] = np.sqrt(np.mean(diff**2))
            out[i, 1] = np.median(np.abs(diff - np.median(diff)))
    return pd.DataFrame(out, columns=["rms", "mad"])


# ---------------------------------------------------------------------------
# Temporal velocity variability
# ---------------------------------------------------------------------------
@dataclass
class TemporalVariability:
    """Windowed-velocity spread around the full-series velocity.

    Attributes
    ----------
    variability : dict[str, float]
        Per-component robust spread
        ``sqrt(median((v_window - v_full)**2))``, in input units/year.
    full_velocity : dict[str, float]
        MIDAS velocity of the full series per component.
    window_velocities : pd.DataFrame
        One row per (window start, window length) with the windowed
        MIDAS velocities; columns: ``start``, ``length_years`` and one
        column per component.

    """

    variability: dict[str, float]
    full_velocity: dict[str, float]
    window_velocities: pd.DataFrame = field(default_factory=pd.DataFrame)


def _window_starts(t_years: np.ndarray, window_years: float) -> list[float]:
    """Start times (years) of half-overlapping windows that fit the record."""
    starts = [t_years[0]]
    t_end = t_years[-1]
    current = t_years[0]
    while True:
        after = t_years[t_years > current + window_years / 2.0]
        if after.size == 0:
            break
        candidate = after[0]
        if t_end - candidate <= window_years:
            break
        starts.append(candidate)
        current = candidate
    return starts


def temporal_velocity_variability(
    dates: ArrayLike,
    values: ArrayLike | pd.DataFrame,
    *,
    components: tuple[str, ...] = ("east", "north", "up"),
    min_window_years: float = 3.0,
    max_window_fraction: float = 0.75,
    step_dates: ArrayLike | None = None,
) -> TemporalVariability:
    """Quantify how stable a station's velocity is in time.

    Estimates MIDAS velocities in half-overlapping sliding windows of
    every integer length from `min_window_years` up to
    ``max_window_fraction * record length``, and reduces the spread of
    those windowed velocities around the full-series velocity to one
    number per component: ``sqrt(median((v_window - v_full)**2))``.

    Parameters
    ----------
    dates : array-like of datetime64
        Observation epochs.
    values : array-like or pd.DataFrame
        Observations with one column per component (same order as
        `components`); a DataFrame with matching column names also works.
    components : tuple of str
        Component names for the outputs. Default ``("east", "north", "up")``.
    min_window_years : float
        Shortest window length. Default 3 years.
    max_window_fraction : float
        Longest window as a fraction of the record length. Default 0.75.
    step_dates : array-like of datetime64, optional
        Known offset epochs passed to MIDAS so windowed pairs never span
        them.

    Returns
    -------
    TemporalVariability

    """
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    if isinstance(values, pd.DataFrame):
        data = values[list(components)].to_numpy(float)
    else:
        data = np.atleast_2d(np.asarray(values, float))
        if data.shape[0] == len(components) and data.shape[1] == len(dates):
            data = data.T
    if data.shape != (len(dates), len(components)):
        msg = f"values must have shape ({len(dates)}, {len(components)})"
        raise ValueError(msg)

    order = np.argsort(dates.values)
    dates, data = dates[order], data[order]
    t_years = (dates - dates[0]).total_seconds().to_numpy() / 86400.0 / DAYS_PER_YEAR
    if step_dates is not None:
        steps_years = np.sort(
            (pd.DatetimeIndex(pd.to_datetime(np.asarray(step_dates))) - dates[0])
            .total_seconds()
            .to_numpy()
            / 86400.0
            / DAYS_PER_YEAR
        )
    else:
        steps_years = None

    duration = float(t_years[-1] - t_years[0])
    max_window = duration * max_window_fraction
    if max_window < min_window_years:
        msg = (
            f"Record of {duration:.1f} yr too short for "
            f"min_window_years={min_window_years}"
        )
        raise ValueError(msg)

    full = {
        comp: midas(t_years, data[:, k], steps_years).velocity
        for k, comp in enumerate(components)
    }

    rows = []
    for window in np.arange(min_window_years, np.floor(max_window) + 1):
        for start in _window_starts(t_years, float(window)):
            mask = (t_years >= start) & (t_years < start + window)
            if mask.sum() < 10:
                continue
            row: dict[str, float] = {
                "start": float(start),
                "length_years": float(window),
            }
            for k, comp in enumerate(components):
                res = midas(t_years[mask], data[mask, k], steps_years)
                row[comp] = res.velocity
            rows.append(row)

    windows_df = pd.DataFrame(rows)
    variability = {}
    for comp in components:
        if len(windows_df):
            dv = windows_df[comp].to_numpy() - full[comp]
            variability[comp] = float(np.sqrt(np.nanmedian(dv**2)))
        else:
            variability[comp] = np.nan

    return TemporalVariability(
        variability=variability,
        full_velocity=full,
        window_velocities=windows_df,
    )

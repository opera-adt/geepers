# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Rebuild UNR-style gridded GPS products from station positions.

Follows section 2.5 of the OPERA DISP-CAL ATBD, whose gridding solution is by
Dr. William C. Hammond (Nevada Geodetic Laboratory, UNR; see NOTICE). The time-constant
product images station MIDAS rates. The time-variable product fills each station's gaps
from its neighbours' displacements, then images displacement since the first day, day
by day. Positions are in mm, rates in mm/yr and times in years of 365.25 days.

Two departures from the ATBD text, because it appears to contain typos: Eq. (16) gives
the farther reference time the larger weight and is swapped here, and Eq. (15) is taken
to mean the displacement at the gap time.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from geepers.euler import local_plate_velocity
from geepers.gps_imaging import (
    Breaklines,
    great_circle_degrees,
    iterated_median_spatial_filter,
    msf_interpolate,
)
from geepers.midas import midas
from geepers.variability import rate_variability_index

__all__ = [
    "PUBLISHED_SELECTION",
    "SelectionCriteria",
    "atbd_ssf",
    "background_rates",
    "background_weight",
    "decluster",
    "filter_station_rates",
    "rate_grid",
    "read_breaklines",
    "select_time_variable",
    "station_parameters",
    "trajectory_design",
    "years_since",
]

COMPONENTS = ("east", "north", "up")
SIGMA_COLUMNS = ("sigma_east", "sigma_north", "sigma_up")
KM_PER_DEGREE = 111.195
# MIDAS can report almost no uncertainty on a clean series, and sigmas divide weights
MIN_SIGMA_RATE = 0.1
CLUSTER_RADIUS_KM = 0.01


@dataclass(frozen=True)
class SelectionCriteria:
    """Limits for stations in the time-variable product (ATBD section 2.5.2).

    Attributes
    ----------
    min_years : float
        Least duration within the product's time span.
    min_completeness : float
        Least percentage of days with a solution, first to last observation.
    max_speckle, max_misfit, max_variability : dict of str to float
        Per-component limits on the rate minus filtered rate (mm/yr), the robust
        misfit to `trajectory_design` (mm), and `rate_variability_index` (mm/yr).

    """

    min_years: float = 7.0
    min_completeness: float = 80.0
    max_speckle: Mapping[str, float] = field(
        default_factory=lambda: {"east": 5.0, "north": 5.0, "up": 15.0}
    )
    max_misfit: Mapping[str, float] = field(
        default_factory=lambda: {"east": 10.0, "north": 10.0, "up": 30.0}
    )
    max_variability: Mapping[str, float] = field(
        default_factory=lambda: {"east": 5.0, "north": 5.0, "up": 15.0}
    )


_NO_LIMIT = dict.fromkeys(COMPONENTS, np.inf)
# The stations that best reproduce the published Version 0.3 daily grid, inferred from
# it: the 4 year rule the ATBD gives for the time-constant product, and no other limit.
PUBLISHED_SELECTION = SelectionCriteria(
    min_years=4.0,
    min_completeness=0.0,
    max_speckle=_NO_LIMIT,
    max_misfit=_NO_LIMIT,
    max_variability=_NO_LIMIT,
)


def atbd_ssf() -> np.ndarray:
    """Return the spatial structure function read off ATBD Figure 2-23.

    The figure lists distance and weight for the 7 stations of one grid point. Weights
    are SSF / sigma, so this assumes those stations have similar sigmas. Beyond the last
    station the curve runs straight to zero at 180 degrees. Regional curves from
    `make_ssf` can be much steeper, and match the published rates less well.

    Returns
    -------
    np.ndarray
        (n, 2) of (distance in degrees, SSF), as `make_ssf` returns.

    """
    km = np.array([1.8, 27.8, 48.1, 60.0, 77.1, 79.7, 98.3])
    weight = np.array([0.21, 0.16, 0.14, 0.13, 0.12, 0.12, 0.11])
    return np.vstack(
        [[0.0, 1.0], np.c_[km / KM_PER_DEGREE, weight / weight[0]], [180.0, 0.0]]
    )


def years_since(dates: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Return years of 365.25 days from `origin`, as the UNR grid files count them."""
    return (dates - origin).days.to_numpy() / 365.25


def read_breaklines(
    path: str | Path, keep: Callable[[Mapping[str, Any]], bool] | None = None
) -> list[np.ndarray]:
    """Read fault or plate-boundary polylines from a GeoJSON file.

    Parameters
    ----------
    path : str or Path
        GeoJSON with LineString or MultiLineString features.
    keep : callable, optional
        Given a feature's properties, whether to use that feature.

    Returns
    -------
    list of np.ndarray
        Polylines, each (n_vertices, 2) of (lon, lat).

    """
    lines: list[np.ndarray] = []
    for feature in json.loads(Path(path).read_text())["features"]:
        if keep is not None and not keep(feature["properties"]):
            continue
        geometry = feature["geometry"]
        parts = (
            [geometry["coordinates"]]
            if geometry["type"] == "LineString"
            else geometry["coordinates"]
        )
        lines += [np.asarray(part, float)[:, :2] for part in parts]
    return lines


def trajectory_design(t: ArrayLike, step_times: ArrayLike = ()) -> np.ndarray:
    """Build the ATBD Eq. (3) design matrix, without its postseismic terms.

    Parameters
    ----------
    t : array-like
        Times in years.
    step_times : array-like
        Times of Heaviside steps.

    Returns
    -------
    np.ndarray
        Columns: intercept, rate, annual and semi-annual sine and cosine, then one per
        step.

    """
    t = np.asarray(t, float)
    w = 2 * np.pi * t
    columns = [np.ones_like(t), t, np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)]
    columns += [(t >= at).astype(float) for at in np.atleast_1d(step_times)]
    return np.column_stack(columns)


def _robust_scale(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def station_parameters(
    positions: pd.DataFrame,
    step_dates: Iterable[pd.Timestamp],
    start: pd.Timestamp,
    *,
    min_years: float = 4.0,
) -> dict[str, float] | None:
    """Compute the reconnaissance parameters of one station (ATBD section 2.5.1).

    Parameters
    ----------
    positions : pd.DataFrame
        Daily ``east``, ``north`` and ``up`` in mm, on a DatetimeIndex that ends where
        the product ends.
    step_dates : iterable of pd.Timestamp
        All catalogued steps. MIDAS pairs never span them.
    start : pd.Timestamp
        Start of the time-variable product.
    min_years : float
        Least record length. The misfit and variability, which are slow, are computed
        only for stations with this much data after `start`.

    Returns
    -------
    dict or None
        ``duration``, ``duration_since_start``, ``completeness``, per component ``v?``
        and ``s?`` (MIDAS rate and sigma), and where computed ``misfit_?`` and
        ``variability_?``, with ? the component's initial. None for too short a record.

    """
    if len(positions) < 2:
        return None
    t = years_since(positions.index, start)
    if t[-1] - t[0] < min_years:
        return None
    steps = years_since(pd.DatetimeIndex(sorted(step_dates)), start)
    steps = np.unique(steps[(steps > t[0]) & (steps < t[-1])])
    row: dict[str, float] = {
        "duration": float(t[-1] - t[0]),
        "duration_since_start": 0.0,
        "completeness": 0.0,
    }
    for c in COMPONENTS:
        fit = midas(t, positions[c].to_numpy(), steps if steps.size else None)
        row[f"v{c[0]}"] = fit.velocity
        row[f"s{c[0]}"] = max(fit.velocity_uncertainty, MIN_SIGMA_RATE)

    recent = positions[positions.index >= start]
    if len(recent) < 2:
        return row
    tr = years_since(recent.index, start)
    days = (recent.index[-1] - recent.index[0]).days + 1
    row["duration_since_start"] = float(tr[-1] - tr[0])
    row["completeness"] = 100.0 * len(recent) / days
    if row["duration_since_start"] < min_years:
        return row
    inside = steps[(steps > tr[0]) & (steps < tr[-1])]
    design = trajectory_design(tr, inside)
    for c in COMPONENTS:
        y = recent[c].to_numpy()
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        row[f"misfit_{c[0]}"] = _robust_scale(y - design @ coef)
        row[f"variability_{c[0]}"] = rate_variability_index(tr, y, step_times=inside)
    return row


def filter_station_rates(
    stats: pd.DataFrame,
    ssf: np.ndarray,
    *,
    breaklines: Breaklines | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Median-spatial-filter the station rates (ATBD section 2.5.3).

    Parameters
    ----------
    stats : pd.DataFrame
        ``lon``, ``lat`` and, per component, ``v?`` and ``s?``.
    ssf : np.ndarray
        Spatial structure function from `make_ssf`.
    breaklines : sequence of array-like, optional
        Links crossing these are not used.
    seed : int
        Seed for the SSF perturbations; each component gets its own stream.

    Returns
    -------
    pd.DataFrame
        ``vmsf?`` per component, on the index of `stats`.

    """
    out = pd.DataFrame(index=stats.index)
    for offset, c in enumerate("enu"):
        out[f"vmsf{c}"] = iterated_median_spatial_filter(
            stats.lon,
            stats.lat,
            stats[f"v{c}"],
            stats[f"s{c}"],
            ssf,
            seed=seed + offset,
            breaklines=breaklines,
        )
    return out


def select_time_variable(
    stats: pd.DataFrame, criteria: SelectionCriteria | None = None
) -> pd.Series:
    """Return which stations qualify for the time-variable product (section 2.5.2).

    Parameters
    ----------
    stats : pd.DataFrame
        Output of `station_parameters` for each station, plus `filter_station_rates`.
        A missing misfit, variability or filtered rate passes, since sparse records
        can have no estimate of them. The other limits say nothing about such a station.
    criteria : SelectionCriteria, optional
        Default: the limits as the ATBD states them.

    Returns
    -------
    pd.Series
        Boolean, on the index of `stats`.

    """
    limits = criteria or SelectionCriteria()
    keep = (stats.duration_since_start >= limits.min_years) & (
        stats.completeness >= limits.min_completeness
    )
    for c in COMPONENTS:
        k = c[0]
        keep &= ~((stats[f"v{k}"] - stats[f"vmsf{k}"]).abs() >= limits.max_speckle[c])
        keep &= ~(stats[f"misfit_{k}"] > limits.max_misfit[c])
        keep &= ~(stats[f"variability_{k}"] >= limits.max_variability[c])
    return keep


def decluster(
    stats: pd.DataFrame, radius_km: float = CLUSTER_RADIUS_KM
) -> pd.DataFrame:
    """Of stations within `radius_km`, keep the smallest vertical rate uncertainty.

    Delaunay triangulation works poorly on tight clusters (ATBD section 2.5.5.1).

    Parameters
    ----------
    stats : pd.DataFrame
        ``lon``, ``lat`` and ``su``.
    radius_km : float
        Default 10 m.

    Returns
    -------
    pd.DataFrame
        The kept rows, in their original order.

    """
    order = stats.sort_values("su", kind="stable")
    lon, lat = order.lon.to_numpy(), order.lat.to_numpy()
    kept: list[int] = []
    for i in range(len(order)):
        if kept:
            d = great_circle_degrees(lat[i], lon[i], lat[kept], lon[kept])
            if d.min() * KM_PER_DEGREE < radius_km:
                continue
        kept.append(i)
    return order.iloc[kept].sort_index()


def background_weight(
    station_lon: ArrayLike, station_lat: ArrayLike, lon: ArrayLike, lat: ArrayLike
) -> np.ndarray:
    """Return the weight of background models at each point (ATBD Eqs. 5 and 10).

    Parameters
    ----------
    station_lon, station_lat : array-like
        Station coordinates in degrees.
    lon, lat : array-like
        Grid points.

    Returns
    -------
    np.ndarray
        Zero where the ten nearest stations are within a median of 100 km, one where
        that median exceeds 500 km, and linear in log10 density between.

    """
    station_lon = np.asarray(station_lon, float)
    station_lat = np.asarray(station_lat, float)
    weights = []
    for lo, la in zip(np.atleast_1d(lon), np.atleast_1d(lat), strict=True):
        km = great_circle_degrees(la, lo, station_lat, station_lon) * KM_PER_DEGREE
        density = 1.0 / np.median(np.sort(km)[:10])
        weights.append(np.interp(np.log10(density), [-2.7, -2.0], [1.0, 0.0]))
    return np.array(weights)


def background_rates(
    stats: pd.DataFrame,
    lon: ArrayLike,
    lat: ArrayLike,
    *,
    breaklines: Breaklines | None = None,
) -> pd.DataFrame:
    """Return the tectonic background at each point (ATBD section 2.5.5.2).

    Parameters
    ----------
    stats : pd.DataFrame
        ``lon``, ``lat``, ``ve``, ``vn``, ``se`` and ``sn``.
    lon, lat : array-like
        Grid points.
    breaklines : sequence of array-like, optional
        Stations across these do not contribute.

    Returns
    -------
    pd.DataFrame
        ``ve`` and ``vn`` per point; there is no vertical tectonic background. A GIA
        model, which geepers does not include, should be added by the caller.

    """
    rows = [
        local_plate_velocity(
            stats.lon,
            stats.lat,
            stats.ve,
            stats.vn,
            stats.se,
            stats.sn,
            lo,
            la,
            breaklines=breaklines,
        )
        for lo, la in zip(np.atleast_1d(lon), np.atleast_1d(lat), strict=True)
    ]
    return pd.DataFrame(rows, columns=["ve", "vn"])


def rate_grid(
    stats: pd.DataFrame,
    lon: ArrayLike,
    lat: ArrayLike,
    ssf: np.ndarray,
    *,
    velocity: str = "vmsf",
    breaklines: Breaklines | None = None,
    background: pd.DataFrame | None = None,
    weight: ArrayLike | None = None,
) -> pd.DataFrame:
    """Image station rates onto grid points (ATBD section 2.5.5).

    Parameters
    ----------
    stats : pd.DataFrame
        ``lon``, ``lat`` and per component the rate and ``s?``.
    lon, lat : array-like
        Grid points.
    ssf : np.ndarray
        Spatial structure function from `make_ssf`.
    velocity : str
        Prefix of the rate columns: ``"vmsf"`` for filtered rates, ``"v"`` for raw.
    breaklines : sequence of array-like, optional
        Links crossing these are not used.
    background : pd.DataFrame, optional
        Model rates per point in ``v?`` columns; absent components have none.
    weight : array-like, optional
        The weight of `background` at each point.

    Returns
    -------
    pd.DataFrame
        Per component: ``v?``, ``s?`` (ATBD Eq. 11) and ``n?`` (contributing stations).

    """
    out = {}
    for c in "enu":
        kwargs = {}
        if background is not None and f"v{c}" in background:
            kwargs = {"background": background[f"v{c}"], "background_weight": weight}
        est = msf_interpolate(
            stats.lon,
            stats.lat,
            stats[f"{velocity}{c}"],
            stats[f"s{c}"],
            lon,
            lat,
            ssf,
            breaklines=breaklines,
            **kwargs,
        )
        out[f"v{c}"] = est["value"].to_numpy()
        out[f"s{c}"] = np.sqrt(
            (est["sigma_robust"] ** 2 + est["sigma_median"] ** 2) / est["n_stations"]
        ).to_numpy()
        out[f"n{c}"] = est["n_stations"].to_numpy()
    return pd.DataFrame(out)

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
import multiprocessing as mp
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay, QhullError

from geepers.euler import local_plate_velocity
from geepers.gps_imaging import (
    DEFAULT_BREAKLINE_WEIGHT,
    Breaklines,
    great_circle_degrees,
    iterated_median_spatial_filter,
    linked_stations,
    msf_interpolate,
    tri_vertex_neighbors,
    weighted_medians,
)
from geepers.midas import midas
from geepers.steps import remove_known_steps
from geepers.variability import rate_variability_index

__all__ = [
    "DEFAULT_BREAKLINE_WEIGHT",
    "PUBLISHED_SELECTION",
    "PositionCube",
    "SelectionCriteria",
    "atbd_ssf",
    "background_rates",
    "background_weight",
    "build_position_cube",
    "decluster",
    "fill_gaps",
    "filter_station_rates",
    "image_positions",
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
MAX_POSITION_SIGMA = 10.0
MAX_MEDIAN_DEPARTURE = 10.0
# "The nearest 1%" of the roughly 4500 stations in the ATBD
N_NEIGHBORS = 45
MIN_NEIGHBORS = 3


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
    breakline_weight: float = DEFAULT_BREAKLINE_WEIGHT,
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
        Links crossing these are penalized, see `breakline_weight`.
    breakline_weight : float
        Multiplies the weight of links crossing `breaklines`. 0 drops them and 1 ignores
        the breaklines.
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
            breakline_weight=breakline_weight,
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
    breakline_weight: float = DEFAULT_BREAKLINE_WEIGHT,
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
        Links crossing these are penalized, see `breakline_weight`.
    breakline_weight : float
        Multiplies the weight of links crossing `breaklines`. 0 drops them and 1 ignores
        the breaklines.
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
            breakline_weight=breakline_weight,
            **kwargs,
        )
        out[f"v{c}"] = est["value"].to_numpy()
        out[f"s{c}"] = np.sqrt(
            (est["sigma_robust"] ** 2 + est["sigma_median"] ** 2) / est["n_stations"]
        ).to_numpy()
        out[f"n{c}"] = est["n_stations"].to_numpy()
    return pd.DataFrame(out)


@dataclass
class PositionCube:
    """Selected stations on a common daily axis.

    Attributes
    ----------
    sites : list of str
        Station names.
    lon, lat : np.ndarray
        Station coordinates in degrees.
    days : pd.DatetimeIndex
        The daily axis.
    x, s : np.ndarray
        Positions and sigmas in mm, (station, day, component); NaN marks a gap.
    observed : np.ndarray
        True where (station, day) held a kept solution before filling.
    earthquakes : list of np.ndarray
        Each station's earthquake steps, as indices into `days`.

    """

    sites: list[str]
    lon: np.ndarray
    lat: np.ndarray
    days: pd.DatetimeIndex
    x: np.ndarray
    s: np.ndarray
    observed: np.ndarray
    earthquakes: list[np.ndarray]


def _median_of_adjacent(y: np.ndarray) -> np.ndarray:
    pad = np.r_[y[0], y[0], y, y[-1], y[-1]]
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, 5), axis=1)


def _steps_within(
    dates: Iterable[pd.Timestamp], origin: pd.Timestamp, t: np.ndarray
) -> np.ndarray:
    times = years_since(pd.DatetimeIndex(sorted(dates)), origin)
    return np.unique(times[(times > t[0]) & (times < t[-1])])


def build_position_cube(
    series: Mapping[str, pd.DataFrame],
    stations: pd.DataFrame,
    days: pd.DatetimeIndex,
    equipment: Mapping[str, Iterable[pd.Timestamp]],
    earthquakes: Mapping[str, Iterable[pd.Timestamp]],
) -> PositionCube:
    """Put stations on the daily axis without equipment steps and outliers.

    Earthquake steps stay in the series: they are signal. A day is dropped if any
    component's sigma exceeds 10 mm or it departs from the median of the 5 adjacent
    observations by more than 10 mm. The ATBD takes that median over a filled series;
    here it is over observations.

    Parameters
    ----------
    series : mapping of str to pd.DataFrame
        Per station: ``east``, ``north``, ``up`` and ``sigma_*`` in mm by date.
    stations : pd.DataFrame
        ``id``, ``lon`` and ``lat``.
    days : pd.DatetimeIndex
        The daily axis.
    equipment, earthquakes : mapping of str to iterable of pd.Timestamp
        Catalogued steps per station. A missing station has none.

    Returns
    -------
    PositionCube

    """
    x = np.full((len(stations), len(days), 3), np.nan)
    s = np.full_like(x, np.nan)
    day_times = years_since(days, days[0])
    quakes = []
    for i, site in enumerate(stations.id):
        df = series[site]
        df = df[(df.index >= days[0]) & (df.index <= days[-1])]
        where = days.get_indexer(df.index)
        t = day_times[where]

        kit = _steps_within(equipment.get(site, ()), days[0], t)
        eq = _steps_within(earthquakes.get(site, ()), days[0], t)
        quakes.append(np.searchsorted(day_times, eq))
        keep = np.ones(len(df), bool)
        cleaned = {}
        for c, sigma in zip(COMPONENTS, SIGMA_COLUMNS, strict=True):
            cleaned[c] = remove_known_steps(t, df[c].to_numpy(), kit, other_steps=eq)
            keep &= df[sigma].to_numpy() <= MAX_POSITION_SIGMA
            departure = np.abs(cleaned[c] - _median_of_adjacent(cleaned[c]))
            keep &= departure <= MAX_MEDIAN_DEPARTURE
        for k, (c, sigma) in enumerate(zip(COMPONENTS, SIGMA_COLUMNS, strict=True)):
            x[i, where[keep], k] = cleaned[c][keep]
            s[i, where[keep], k] = df[sigma].to_numpy()[keep]
    return PositionCube(
        list(stations.id),
        stations.lon.to_numpy(),
        stations.lat.to_numpy(),
        days,
        x,
        s,
        np.isfinite(x[:, :, 0]),
        quakes,
    )


# Worker state, inherited by forked workers instead of pickled for every task
_WORKER: dict[str, Any] = {}


def _set_worker(**state: Any) -> None:
    _WORKER.clear()
    _WORKER.update(state)


def _map(
    function: Callable[[int], Any],
    items: Sequence[int],
    n_workers: int,
    state: dict[str, Any],
    chunksize: int = 1,
) -> list[Any]:
    # Set before forking, for the workers to inherit without pickling the arrays
    _set_worker(**state)
    if n_workers <= 1:
        return [function(item) for item in items]
    with ProcessPoolExecutor(n_workers, mp_context=mp.get_context("fork")) as pool:
        return list(pool.map(function, items, chunksize=chunksize))


def _can_share_neighborhoods(
    lon: np.ndarray,
    lat: np.ndarray,
    breaklines: Breaklines | None,
    weight_by_sigma: bool,
) -> bool:
    """Tell whether weights stay the same for every value at the same stations."""
    if weight_by_sigma or breaklines is not None:
        return False
    return len(np.unique(np.c_[lon, lat], axis=0)) == len(lon)


def _distance_weights(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_point: float,
    lat_point: float,
    ssf: np.ndarray,
) -> np.ndarray:
    weight = np.interp(
        great_circle_degrees(lat_point, lon_point, lat, lon), ssf[:, 0], ssf[:, 1]
    )
    return np.ones_like(weight) if np.all(weight == 0) else weight


def _shared_neighborhood(
    j: int, usable: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    lon, lat = _WORKER["lon"], _WORKER["lat"]
    try:
        tri = Delaunay(np.c_[lon[usable], lat[usable]])
        linked = linked_stations(
            lon[usable], lat[usable], lon[j], lat[j], tri_vertex_neighbors(tri)
        )
    except QhullError:
        return None
    if linked.size == 0:
        return None
    stations = usable[linked]
    return stations, _distance_weights(
        lon[stations], lat[stations], lon[j], lat[j], _WORKER["ssf"]
    )


def _difference_at_target(
    j: int, near: np.ndarray, day: int, ref: int, known_links: dict[bytes, Any]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the target's displacement from `ref` to `day` (ATBD Eqs. 12-14, 18)."""
    x, s, have = _WORKER["x"], _WORKER["s"], _WORKER["have"]
    lon, lat = _WORKER["lon"], _WORKER["lat"]
    usable = near[have[near, day] & have[near, ref]]
    if usable.size < MIN_NEIGHBORS:
        return None
    if _WORKER["share_neighborhoods"]:
        key = usable.tobytes()
        if key not in known_links:
            known_links[key] = _shared_neighborhood(j, usable)
        if known_links[key] is None:
            return None
        stations, weight = known_links[key]
        change = x[stations, day] - x[stations, ref]
        value = weighted_medians(change, weight)
        scatter = change - value
        scatter = np.median(np.abs(scatter - np.median(scatter, axis=0)), axis=0)
        return value, np.hypot(np.median(s[usable, day], axis=0), 1.4826 * scatter)
    value, sigma = np.zeros(3), np.zeros(3)
    for k in range(3):
        try:
            est = msf_interpolate(
                lon[usable],
                lat[usable],
                x[usable, day, k] - x[usable, ref, k],
                (
                    np.hypot(s[usable, day, k], s[usable, ref, k])
                    if _WORKER["weight_by_sigma"]
                    else np.ones(usable.size)
                ),
                [lon[j]],
                [lat[j]],
                _WORKER["ssf"],
                breaklines=_WORKER["breaklines"],
                breakline_weight=_WORKER["breakline_weight"],
            ).iloc[0]
        except Exception:
            return None
        if not np.isfinite(est["value"]):
            return None
        value[k] = est["value"]
        robust = est["sigma_robust"] if np.isfinite(est["sigma_robust"]) else 0.0
        sigma[k] = np.hypot(np.median(s[usable, day, k]), robust)
    return value, sigma


def _fill_one_station(j: int) -> tuple[int, list[tuple[int, np.ndarray, np.ndarray]]]:
    x, have = _WORKER["x"], _WORKER["have"]
    lon, lat = _WORKER["lon"], _WORKER["lat"]
    quakes = _WORKER["earthquakes"][j]
    distance = great_circle_degrees(lat[j], lon[j], lat, lon)
    distance[j] = np.inf
    near = np.argsort(distance)[:N_NEIGHBORS]
    known = np.flatnonzero(have[j])
    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    if known.size == 0:
        return j, rows
    links: dict[bytes, Any] = {}
    for day in np.flatnonzero(~have[j]):
        pos = np.searchsorted(known, day)
        before = known[pos - 1] if pos > 0 else None
        after = known[pos] if pos < known.size else None
        # No earthquake may fall between a reference and the gap
        if before is not None and ((quakes > before) & (quakes <= day)).any():
            before = None
        if after is not None and ((quakes > day) & (quakes <= after)).any():
            after = None
        below = (
            None
            if before is None
            else _difference_at_target(j, near, day, before, links)
        )
        above = (
            None if after is None else _difference_at_target(j, near, day, after, links)
        )
        if before is not None and after is not None and below and above:
            weight = (after - day) / (after - before)
            value = weight * (x[j, before] + below[0]) + (1 - weight) * (
                x[j, after] + above[0]
            )
            sigma = weight * below[1] + (1 - weight) * above[1]
        elif below is not None:
            value, sigma = x[j, before] + below[0], below[1]
        elif above is not None:
            value, sigma = x[j, after] + above[0], above[1]
        else:
            continue
        rows.append((int(day), value, sigma))
    return j, rows


def _fill_from_model(cube: PositionCube) -> int:
    t = years_since(cube.days, cube.days[0])
    count = 0
    for j in range(len(cube.sites)):
        gaps = np.isnan(cube.x[j, :, 0])
        if not gaps.any():
            continue
        known = ~gaps
        design = trajectory_design(t, t[cube.earthquakes[j]])
        for k in range(3):
            coef = np.linalg.lstsq(design[known], cube.x[j, known, k], rcond=None)[0]
            cube.x[j, gaps, k] = design[gaps] @ coef
            cube.s[j, gaps, k] = np.sqrt(2) * np.nanmedian(cube.s[j, :, k])
        count += int(gaps.sum())
    return count


def fill_gaps(
    cube: PositionCube,
    ssf: np.ndarray,
    *,
    breaklines: Breaklines | None = None,
    breakline_weight: float = DEFAULT_BREAKLINE_WEIGHT,
    n_workers: int = 1,
    passes: int = 2,
    weight_by_sigma: bool = False,
) -> dict[str, int]:
    """Fill every gap in place (ATBD section 2.5.6.1).

    A gap takes the displacement of the stations around it since the nearest day with
    data on either side, unless an earthquake intervenes. A second pass treats filled
    days as data, and `trajectory_design` fills what remains.

    Parameters
    ----------
    cube : PositionCube
        Modified in place. Every station needs some data.
    ssf : np.ndarray
        Spatial structure function from `make_ssf`.
    breaklines : sequence of array-like, optional
        Links crossing these are penalized, see `breakline_weight`.
    breakline_weight : float
        Multiplies the weight of links crossing `breaklines`. 0 drops them and 1 ignores
        the breaklines.
    n_workers : int
        Processes.
    passes : int
        Spatial passes.
    weight_by_sigma : bool
        Divide weights by the position sigmas, as ATBD Eq. (6) reads. Default False,
        since UNR Version 0.3 weights by distance alone: see `image_positions`.

    Returns
    -------
    dict of str to int
        Station-days observed and filled by each method.

    """
    counts = {"observed": int(cube.observed.sum())}
    share = _can_share_neighborhoods(cube.lon, cube.lat, breaklines, weight_by_sigma)
    for attempt in range(passes):
        state = {
            "x": cube.x,
            "s": cube.s,
            "have": np.isfinite(cube.x[:, :, 0]),
            "lon": cube.lon,
            "lat": cube.lat,
            "ssf": ssf,
            "breaklines": breaklines,
            "breakline_weight": breakline_weight,
            "earthquakes": cube.earthquakes,
            "weight_by_sigma": weight_by_sigma,
            "share_neighborhoods": share,
        }
        results = _map(_fill_one_station, range(len(cube.sites)), n_workers, state)
        filled = 0
        for j, rows in results:
            for day, value, sigma in rows:
                cube.x[j, day] = value
                cube.s[j, day] = sigma
                filled += 1
        counts[f"spatial_pass_{attempt + 1}"] = filled
    counts["model"] = _fill_from_model(cube)
    return counts


def _image_one_day(day: int) -> np.ndarray:
    x, s = _WORKER["x"], _WORKER["s"]
    out = np.empty((len(_WORKER["node_lon"]), 3))
    for k in range(3):
        out[:, k] = msf_interpolate(
            _WORKER["lon"],
            _WORKER["lat"],
            x[:, day, k] - x[:, 0, k],
            s[:, day, k] if _WORKER["weight_by_sigma"] else np.ones(x.shape[0]),
            _WORKER["node_lon"],
            _WORKER["node_lat"],
            _WORKER["ssf"],
            breaklines=_WORKER["breaklines"],
            breakline_weight=_WORKER["breakline_weight"],
        )["value"].to_numpy()
    return out


def _image_one_point(point: int) -> np.ndarray:
    lon, lat = _WORKER["lon"], _WORKER["lat"]
    lon_point, lat_point = _WORKER["node_lon"][point], _WORKER["node_lat"][point]
    try:
        stations = linked_stations(lon, lat, lon_point, lat_point, _WORKER["neighbors"])
    except QhullError:
        stations = np.empty(0, int)
    if stations.size == 0:
        return np.full((_WORKER["displacement"].shape[1], 3), np.nan)
    weight = _distance_weights(
        lon[stations], lat[stations], lon_point, lat_point, _WORKER["ssf"]
    )
    return weighted_medians(_WORKER["displacement"][stations], weight)


def image_positions(
    cube: PositionCube,
    lon: ArrayLike,
    lat: ArrayLike,
    ssf: np.ndarray,
    *,
    breaklines: Breaklines | None = None,
    breakline_weight: float = DEFAULT_BREAKLINE_WEIGHT,
    n_workers: int = 1,
    weight_by_sigma: bool = False,
) -> np.ndarray:
    """Image displacement since the first day onto grid points (ATBD section 2.5.6.3).

    Weights come from distance alone and do not change through time. Replaying UNR's
    Version 0.3 medians with its own station values, this predicts the winning station
    far better than dividing by daily sigmas. Rates, in `rate_grid`, do use sigmas.

    Parameters
    ----------
    cube : PositionCube
        Gap-free, from `fill_gaps`.
    lon, lat : array-like
        Grid points.
    ssf : np.ndarray
        Spatial structure function from `make_ssf`.
    breaklines : sequence of array-like, optional
        Links crossing these are penalized, see `breakline_weight`.
    breakline_weight : float
        Multiplies the weight of links crossing `breaklines`. 0 drops them and 1 ignores
        the breaklines.
    n_workers : int
        Processes.
    weight_by_sigma : bool
        Divide weights by each day's position sigmas. Default False.

    Returns
    -------
    np.ndarray
        Displacement in mm, (day, point, component).

    """
    if np.isnan(cube.x).any():
        msg = "the cube has gaps: run fill_gaps first"
        raise ValueError(msg)
    node_lon = np.atleast_1d(np.asarray(lon, float))
    node_lat = np.atleast_1d(np.asarray(lat, float))
    if _can_share_neighborhoods(cube.lon, cube.lat, breaklines, weight_by_sigma):
        # Weights never change, so each point needs its neighborhood only once
        shared = {
            "displacement": cube.x - cube.x[:, :1],
            "lon": cube.lon,
            "lat": cube.lat,
            "ssf": ssf,
            "neighbors": tri_vertex_neighbors(Delaunay(np.c_[cube.lon, cube.lat])),
            "node_lon": node_lon,
            "node_lat": node_lat,
        }
        results = _map(
            _image_one_point, range(len(node_lon)), n_workers, shared, chunksize=16
        )
        return np.stack(results, axis=1)
    state = {
        "x": cube.x,
        "s": cube.s,
        "lon": cube.lon,
        "lat": cube.lat,
        "ssf": ssf,
        "breaklines": breaklines,
        "breakline_weight": breakline_weight,
        "weight_by_sigma": weight_by_sigma,
        "node_lon": node_lon,
        "node_lat": node_lat,
    }
    results = _map(
        _image_one_day, range(len(cube.days)), n_workers, state, chunksize=16
    )
    return np.stack(results)

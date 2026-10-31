# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Use neighboring stations to judge what one record cannot.

On a single record a smooth signal, a bad monument and a slow random walk
look alike, which is why `geepers.smooth.fit_smooth_trend` needs a limit on
the random walk it allows. Differencing close stations removes what they
share, so it separates two questions:

- Do the neighbors see a station's smooth signal? If it disappears from the
  differences it is shared: real regional motion or common-mode noise, but
  not this monument. If it stays, it is local to the station: a bad monument
  or deformation smaller than the station spacing, which only a site visit,
  InSAR or a closer station can tell apart.
- How much random walk belongs to each station? A pair's difference holds
  the random-walk variance of both stations, so with enough pairs each
  station's share can be solved for, as in a three-cornered hat (Gray, J. E.,
  & Allan, D. W., 1974, A method for estimating the frequency stability of an
  individual oscillator, Proc. 28th Annual Symposium on Frequency Control,
  243-246, https://doi.org/10.1109/FREQ.1974.200027).

Example:
-------
>>> check = check_neighbors(series, coordinates, units="m")  # doctest: +SKIP
>>> check.stations[["shape", "signal", "local_random_walk_mm"]]  # doctest: +SKIP

"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import optimize

from geepers.collocation import distance_matrix
from geepers.smooth import _UNITS_PER_MM, SmoothResult, _unique_steps, fit_smooth_trend
from geepers.trend import DAYS_PER_YEAR, estimate_trend

__all__ = ["NeighborCheck", "check_neighbors"]

_T = TypeVar("_T")
_R = TypeVar("_R")
_ORIGIN = pd.Timestamp("1980-01-06")


@dataclass
class NeighborCheck:
    """Output of `check_neighbors`.

    Attributes
    ----------
    stations : pd.DataFrame
        One row per station, indexed by id:

        - ``shape``, ``evidence``: `geepers.smooth.fit_smooth_trend` on the
          station alone.
        - ``n_pairs``: neighbors with enough common data.
        - ``pairs_smooth``: how many of those differences are smooth.
        - ``signal``: "none" if the station alone is not smooth. Otherwise
          "shared" if most differences are not smooth, "local" if most
          are, and "unresolved" if there is no pair or an even split.
        - ``random_walk_mm``: fitted on the station alone, mm/sqrt(yr),
          including whatever deformation looks like one.
        - ``local_random_walk_mm``: the station's share of the differences'
          random walk, NaN where the pairs cannot determine it.
        - ``noisy``: `local_random_walk_mm` exceeds the limit.
    pairs : pd.DataFrame
        One row per pair: ``a``, ``b``, ``km``, ``n_epochs``, ``shape``,
        ``evidence`` and ``random_walk_mm`` of the difference.

    """

    stations: pd.DataFrame
    pairs: pd.DataFrame


def _on_grid(series: pd.Series, sampling_days: float) -> pd.Series:
    """Take medians on a grid shared by all stations, dated at the bin centers.

    A bin's median changes once a step passes its center, so that is the date
    that puts a known step at the first epoch showing it.
    """
    clean = series.dropna()
    days = np.asarray((clean.index - _ORIGIN).total_seconds()) / 86400.0
    bins = np.floor(days / sampling_days).astype(int)
    medians = clean.groupby(bins).median()
    dates = _ORIGIN + pd.to_timedelta((medians.index + 0.5) * sampling_days, unit="D")
    return pd.Series(medians.to_numpy(), index=dates)


def _select_pairs(
    ids: list[str], km: np.ndarray, max_distance_km: float, max_neighbors: int
) -> list[tuple[str, str, float]]:
    """Keep each station's closest neighbors, in either station's list."""
    chosen: set[tuple[int, int]] = set()
    for i in range(len(ids)):
        close = np.flatnonzero((km[i] <= max_distance_km) & (np.arange(len(ids)) != i))
        for j in close[np.argsort(km[i, close], kind="stable")][:max_neighbors]:
            chosen.add((min(i, int(j)), max(i, int(j))))
    return [(ids[i], ids[j], float(km[i, j])) for i, j in sorted(chosen)]


def _random_walk_mm(
    fit: SmoothResult, series: pd.Series, sampling_days: float, units: str
) -> float:
    """Fit a line, steps and random-walk noise; the amplitude in mm/sqrt(yr)."""
    epochs = pd.DatetimeIndex(series.index)
    trend = estimate_trend(
        epochs,
        series.to_numpy(),
        sampling_days=sampling_days,
        step_dates=fit.step_dates or None,
        noise_model="RWFNWN",
    )
    per_root_year = trend.sigma_randomwalk / np.sqrt(sampling_days / DAYS_PER_YEAR)
    return float(per_root_year / _UNITS_PER_MM[units])


def _station_shares(ids: list[str], pairs: pd.DataFrame) -> dict[str, float]:
    """Split the pairs' random-walk variance among the stations.

    Each difference gives ``a_i + a_j``. That fixes the shares only for a
    connected group holding an odd cycle (three stations all within reach
    of each other, say), since a chain can move variance along itself.
    """
    usable = pairs[pairs["random_walk_mm"].notna()]
    position = {station: k for k, station in enumerate(ids)}
    parent = list(range(len(ids)))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a, b in zip(usable["a"], usable["b"], strict=True):
        parent[find(position[a])] = find(position[b])

    shares = dict.fromkeys(ids, float("nan"))
    groups: dict[int, list[str]] = {}
    for station in ids:
        groups.setdefault(find(position[station]), []).append(station)
    for members in groups.values():
        inside = usable[usable["a"].isin(members)]
        if len(members) < 3 or len(inside) < len(members):
            continue
        index = {station: k for k, station in enumerate(members)}
        design = np.zeros((len(inside), len(members)))
        rows = np.arange(len(inside))
        design[rows, inside["a"].map(index).to_numpy()] = 1.0
        design[rows, inside["b"].map(index).to_numpy()] = 1.0
        if np.linalg.matrix_rank(design) < len(members):
            continue
        solution, _ = optimize.nnls(design, inside["random_walk_mm"].to_numpy() ** 2)
        for station, variance in zip(members, solution, strict=True):
            shares[station] = float(np.sqrt(variance))
    return shares


def _classify(shape: str, n_pairs: int, n_smooth: int) -> str:
    if shape != "smooth":
        return "none"
    if n_pairs == 0 or 2 * n_smooth == n_pairs:
        return "unresolved"
    return "local" if 2 * n_smooth > n_pairs else "shared"


def check_neighbors(
    series: Mapping[str, pd.Series],
    coordinates: pd.DataFrame,
    *,
    units: Literal["m", "mm"],
    sampling_days: float = 30.0,
    max_distance_km: float = 25.0,
    max_neighbors: int = 5,
    min_overlap_years: float = 5.0,
    step_dates: Mapping[str, ArrayLike] | None = None,
    max_random_walk_mm: float = 3.0,
    max_workers: int = 1,
    **fit_kwargs: Any,
) -> NeighborCheck:
    """Compare each station's smooth signal and noise with its neighbors.

    Parameters
    ----------
    series : mapping of station id to pd.Series
        One component of each station, indexed by date, at any cadence:
        medians on a shared grid of `sampling_days` are taken first. Remove
        any known steps that are not given in `step_dates`.
    coordinates : pd.DataFrame
        Indexed by station id, with ``lon`` and ``lat`` in degrees. Every
        station in `series` needs a row.
    units : {"m", "mm"}
        Unit of the series, as in `geepers.smooth.fit_smooth_trend`.
    sampling_days : float
        Spacing of the shared grid. Default 30.
    max_distance_km : float
        Farthest neighbor used. Default 25: farther away a real signal
        stops being shared, and a difference then says little.
    max_neighbors : int
        Closest neighbors used per station. Default 5. Each pair costs a
        fit, so the time grows with this and the number of stations.
    min_overlap_years : float
        Shortest common record for a pair. Default 5.
    step_dates : mapping of station id to array-like of datetime64, optional
        Known offsets of each station. A difference gets both stations'.
    max_random_walk_mm : float
        Passed to `fit_smooth_trend`, and the limit behind
        ``NeighborCheck.stations["noisy"]``. Default 3.
    max_workers : int
        Threads fitting at once. Default 1. Limit the threads of the linear
        algebra library when raising this.
    **fit_kwargs
        Passed to `fit_smooth_trend` for stations and differences.

    Returns
    -------
    NeighborCheck

    Raises
    ------
    ValueError
        If a station has no coordinates.

    """
    missing = sorted(set(series) - set(coordinates.index))
    if missing:
        msg = f"No coordinates for stations: {missing}"
        raise ValueError(msg)
    ids = list(series)
    grid = {name: _on_grid(series[name], sampling_days) for name in ids}
    known = dict(step_dates or {})
    where = coordinates.loc[ids]
    lon, lat = where["lon"].to_numpy(float), where["lat"].to_numpy(float)
    km = distance_matrix(lon, lat, lon, lat)
    candidates = _select_pairs(ids, km, max_distance_km, max_neighbors)
    min_epochs = int(min_overlap_years * DAYS_PER_YEAR / sampling_days)

    def analyze(values: pd.Series, names: list[str]) -> tuple[str, float, float]:
        steps = [d for name in names for d in np.atleast_1d(known.get(name, []))]
        epochs = pd.DatetimeIndex(values.index)
        fit = fit_smooth_trend(
            epochs,
            values.to_numpy(),
            units=units,
            sampling_days=sampling_days,
            step_dates=_unique_steps(steps, epochs) or None,
            max_random_walk_mm=max_random_walk_mm,
            **fit_kwargs,
        )
        amplitude = _random_walk_mm(fit, values, sampling_days, units)
        return fit.shape, fit.evidence, amplitude

    def station_job(name: str) -> tuple[str, float, float]:
        return analyze(grid[name], [name])

    def pair_job(pair: tuple[str, str, float]) -> dict[str, Any] | None:
        a, b, distance = pair
        common = grid[a].index.intersection(grid[b].index)
        if len(common) < min_epochs:
            return None
        shape, evidence, amplitude = analyze(grid[a][common] - grid[b][common], [a, b])
        return {
            "a": a,
            "b": b,
            "km": distance,
            "n_epochs": len(common),
            "shape": shape,
            "evidence": evidence,
            "random_walk_mm": amplitude,
        }

    single = _map(station_job, ids, max_workers)
    found = [p for p in _map(pair_job, candidates, max_workers) if p is not None]
    pairs = pd.DataFrame(
        found,
        columns=["a", "b", "km", "n_epochs", "shape", "evidence", "random_walk_mm"],
    )
    shares = _station_shares(ids, pairs)

    rows = []
    for name, (shape, evidence, amplitude) in zip(ids, single, strict=True):
        mine = pairs[(pairs["a"] == name) | (pairs["b"] == name)]
        n_smooth = int((mine["shape"] == "smooth").sum())
        rows.append(
            {
                "shape": shape,
                "evidence": evidence,
                "n_pairs": len(mine),
                "pairs_smooth": n_smooth,
                "signal": _classify(shape, len(mine), n_smooth),
                "random_walk_mm": amplitude,
                "local_random_walk_mm": shares[name],
                # NaN compares False: no verdict is not a verdict of noisy
                "noisy": bool(shares[name] > max_random_walk_mm),
            }
        )
    return NeighborCheck(
        stations=pd.DataFrame(rows, index=pd.Index(ids, name="id")), pairs=pairs
    )


def _map(job: Callable[[_T], _R], items: list[_T], max_workers: int) -> list[_R]:
    if max_workers <= 1:
        return [job(item) for item in items]
    with ThreadPoolExecutor(max_workers) as pool:
        return list(pool.map(job, items))

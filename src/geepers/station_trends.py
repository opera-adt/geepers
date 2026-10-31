# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Fit trends to GNSS stations from the command line."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import requests
import tyro

from geepers.changepoints import detect_changepoints, fit_sparse_changepoints
from geepers.forecast import forecast_trend
from geepers.gps_sources import SideshowSource, UnrSource
from geepers.neighbors import _on_grid, check_neighbors
from geepers.quality import drop_relocations
from geepers.smooth import fit_smooth_trend, forecast_smooth
from geepers.steps import clean_step_dates

__all__ = ["main"]

logger = logging.getLogger("geepers")

Method = Literal["smooth", "piecewise", "sparse"]
_M_TO_MM = 1000.0


def _fit_one(
    series: pd.Series,
    steps: np.ndarray,
    method: Method,
    sampling_days: float,
    horizon_years: float | None,
) -> tuple[dict, pd.DataFrame | None]:
    """Fit one component (meters); report in mm, with its forecast if asked."""
    epochs = pd.DatetimeIndex(series.index)
    values = series.to_numpy()
    known = clean_step_dates(steps, epochs[0], epochs[-1], epochs=epochs)
    table = None
    if method == "smooth":
        fit = fit_smooth_trend(
            epochs,
            values,
            units="m",
            sampling_days=sampling_days,
            step_dates=known or None,
        )
        row = {
            "shape": fit.shape,
            "evidence": fit.evidence,
            "rate_mm_yr": fit.rate * _M_TO_MM,
            "rate_sigma_mm_yr": fit.rate_sigma * _M_TO_MM,
            "n_steps": len(fit.step_dates),
            "n_rate_changes": len(fit.rate_change_dates),
        }
        if horizon_years is not None:
            table = forecast_smooth(
                epochs, values, horizon_years, sampling_days=sampling_days, fit=fit
            ).table
    else:
        detect = (
            detect_changepoints if method == "piecewise" else fit_sparse_changepoints
        )
        found = detect(
            epochs, values, sampling_days=sampling_days, step_dates=known or None
        )
        changes = found.changepoints
        row = {
            "shape": "piecewise" if len(changes) else "linear",
            "evidence": found.bic_none - found.bic,
            "rate_mm_yr": found.rate * _M_TO_MM,
            "rate_sigma_mm_yr": found.rate_sigma * _M_TO_MM,
            "n_steps": len(known) + int(changes["offset"].notna().sum()),
            "n_rate_changes": int(changes["rate_change"].notna().sum()),
        }
        if horizon_years is not None:
            table = forecast_trend(
                epochs,
                values,
                horizon_years,
                sampling_days=sampling_days,
                step_dates=known or None,
                changepoints=found,
            ).table
    if table is not None:
        columns = ["mean", "lower", "upper"]
        table = (table[["lead_years", *columns]]).copy()
        table[columns] *= _M_TO_MM
    return row, table


def main(
    *,
    stations: Sequence[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    output_dir: Annotated[Path, tyro.conf.arg(aliases=["-o"])] = Path("./trends"),
    gps_source: Literal["unr", "jpl"] = "unr",
    components: Sequence[Literal["east", "north", "up"]] = ("up",),
    method: Method = "smooth",
    sampling_days: float = 30.0,
    start_date: str | None = None,
    end_date: str | None = None,
    min_years: float = 3.0,
    horizon_years: float | None = None,
    neighbors: bool = False,
    max_distance_km: float = 25.0,
    max_gps_step_meters: float | None = 10.0,
) -> None:
    """Fit trends, and optionally forecasts, to GNSS stations.

    Parameters
    ----------
    stations
        Station ids. Give this or `bbox`.
    bbox
        Stations inside (west, south, east, north), degrees.
    output_dir
        Where to write the CSV outputs. Created if needed.
    gps_source
        GNSS data source, "unr" or "jpl". See `geepers.gps_sources`.
    components
        Components to fit. Default up.
    method
        "smooth" (`geepers.smooth.fit_smooth_trend`), "piecewise"
        (`geepers.changepoints.detect_changepoints`) or "sparse"
        (`geepers.changepoints.fit_sparse_changepoints`).
    sampling_days
        Medians are taken on a grid of this spacing first. Default 30: each
        fit factorizes an n x n matrix many times, and a few bad daily
        solutions move a mean by meters.
    start_date, end_date
        Limit the record (ISO format).
    min_years
        Skip a component with less than this record. Default 3.
    horizon_years
        Also forecast this many years ahead.
    neighbors
        Also compare stations with their neighbors
        (`geepers.neighbors.check_neighbors`). Each pair costs a fit, and
        a bounding box is the natural input.
    max_distance_km
        Farthest neighbor used. Default 25.
    max_gps_step_meters
        A position step above this between consecutive days is a monument
        moved under the same id, and only the longest segment is used
        (`geepers.quality.drop_relocations`). Default 10 m; None disables it.

    Notes
    -----
    The script writes into `output_dir`:

    trends.csv
        One row per station and component. Rates are in mm/yr. ``evidence``
        is the gain of the smooth trend for "smooth" and the BIC gain of the
        change points otherwise.
    forecasts.csv
        Mean and interval in mm, with `horizon_years`.
    neighbors.csv, neighbor_pairs.csv
        `NeighborCheck.stations` and `.pairs` per component, with `neighbors`.

    """
    if (stations is None) == (bbox is None):
        msg = "Give exactly one of stations and bbox"
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = UnrSource() if gps_source == "unr" else SideshowSource()
    metadata = source.stations(bbox=bbox)
    metadata = metadata.set_index("id")
    if stations is not None:
        wanted = [s.upper() for s in stations]
        unknown = sorted(set(wanted) - set(metadata.index))
        if unknown:
            msg = f"Unknown stations: {unknown}"
            raise ValueError(msg)
        metadata = metadata.loc[wanted]
    ids = list(metadata.index)
    logger.info("Fitting %d stations", len(ids))

    steps = pd.DataFrame({"id": [], "date": []})
    if hasattr(source, "steps"):
        steps = source.steps(ids)
    step_dates = {
        name: pd.to_datetime(steps.loc[steps["id"] == name, "date"]).to_numpy()
        for name in ids
    }

    records: dict[str, pd.DataFrame] = {}
    for name in ids:
        try:
            df = source.timeseries(name, start_date=start_date, end_date=end_date)
        except requests.HTTPError as e:
            logger.warning("%s: could not download (%s); skipping", name, e)
            continue
        if max_gps_step_meters is not None:
            df, _ = drop_relocations(df, max_gps_step_meters)
        records[name] = df.set_index(pd.to_datetime(df["date"]))

    rows, forecasts = [], []
    grids: dict[str, dict[str, pd.Series]] = {c: {} for c in components}
    for name, df in records.items():
        for component in components:
            series = _on_grid(df[component], sampling_days)
            span = (
                (series.index[-1] - series.index[0]).days / 365.25 if len(series) else 0
            )
            if span < min_years:
                logger.info("%s %s: %.1f years; skipping", name, component, span)
                continue
            grids[component][name] = series
            try:
                row, table = _fit_one(
                    series, step_dates[name], method, sampling_days, horizon_years
                )
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning("%s %s: fit failed (%s)", name, component, e)
                continue
            rows.append({"station": name, "component": component, **row})
            if table is not None:
                forecasts.append(
                    table.reset_index().assign(station=name, component=component)
                )
    columns = [
        "station",
        "component",
        "shape",
        "evidence",
        "rate_mm_yr",
        "rate_sigma_mm_yr",
        "n_steps",
        "n_rate_changes",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output_dir / "trends.csv", index=False)
    if forecasts:
        pd.concat(forecasts, ignore_index=True).to_csv(
            output_dir / "forecasts.csv", index=False
        )

    if neighbors:
        checks, pairs = [], []
        for component in components:
            check = check_neighbors(
                grids[component],
                metadata[["lon", "lat"]],
                units="m",
                sampling_days=sampling_days,
                max_distance_km=max_distance_km,
                step_dates=step_dates,
            )
            checks.append(check.stations.reset_index().assign(component=component))
            pairs.append(check.pairs.assign(component=component))
        pd.concat(checks, ignore_index=True).to_csv(
            output_dir / "neighbors.csv", index=False
        )
        pd.concat(pairs, ignore_index=True).to_csv(
            output_dir / "neighbor_pairs.csv", index=False
        )
    logger.info("Finished - results written to %s", output_dir)

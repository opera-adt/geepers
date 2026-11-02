# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
r"""Rebuild the UNR gridded GPS product in a box, from station positions.

    python scripts/recreate-unr-grid.py --bbox -119.5 38.5 -117.5 40.0 -o nevada
    python scripts/recreate-unr-grid.py --bbox -118.5 33.8 -116.5 35.2 -o socal \\
        --workers 64 --cache-dir unr_cache

Writes `stations.csv`, `rates.csv` and `daily.npz` to the output directory. Unless
`--no-compare` is given, it also scores both against the published Version 0.3 files
and writes `scores_rates.csv` and `scores_daily.csv`.

The defaults reproduce the published grid best in the regions tested: the curve of ATBD
Figure 2-23, and for daily stations only the 4 year rule the ATBD gives for rates, since
published nodes follow stations its stricter limits reject. See "Recreating the UNR
grid" in docs/notebooks/gps_imaging_unr_grid.ipynb for what is documented and inferred.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from geepers import gridding
from geepers.gps_imaging import make_ssf
from geepers.gps_sources import UnrGridSource, UnrSource

COMPONENTS = ("east", "north", "up")
SIGMAS = tuple(f"sigma_{c}" for c in COMPONENTS)
CODE_EQUIPMENT, CODE_EARTHQUAKE = 1, 2
_WORKER: dict[str, Any] = {}


def read_positions(source: Any, site: str, end: pd.Timestamp) -> pd.DataFrame | None:
    """Return a station's daily ENU in mm up to `end`, or None if UNR has no file."""
    try:
        df = source.timeseries(site, frame="ENU", zero_by="none")
    except Exception:
        return None
    df = df.set_index(pd.DatetimeIndex(df["date"]).floor("D"))
    df = df[[*COMPONENTS, *SIGMAS]] * 1000.0
    df = df[~df.index.duplicated()]
    return df[df.index < end]


def _station_row(job: tuple[str, float, float]) -> dict[str, Any] | None:
    site, lon, lat = job
    df = read_positions(_WORKER["source"], site, _WORKER["end"])
    if df is None:
        return None
    row = gridding.station_parameters(
        df, _WORKER["steps"].get(site, []), _WORKER["start"]
    )
    return None if row is None else {"id": site, "lon": lon, "lat": lat, **row}


def reconnaissance(
    source: Any,
    sites: pd.DataFrame,
    steps: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    workers: int,
) -> pd.DataFrame:
    """Return the ATBD section 2.5.1 parameters of every station with enough data."""
    every = {s: pd.DatetimeIndex(g.date.unique()) for s, g in steps.groupby("id")}
    _WORKER.update(source=source, steps=every, start=start, end=end)
    jobs = [(s.id, s.lon, s.lat) for s in sites.itertuples()]
    if workers <= 1:
        rows = [_station_row(job) for job in jobs]
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
            rows = list(pool.map(_station_row, jobs))
    return pd.DataFrame([r for r in rows if r is not None])


def score_rates(
    rates: pd.DataFrame, published: Callable[[int], pd.DataFrame]
) -> pd.DataFrame:
    """Compare rates with the slopes of the time-constant files, which are straight."""
    rows = []
    for node in rates.id:
        df = published(int(node))
        rows.append(
            {f"v{c[0]}": np.polyfit(df.decimal_year, df[c], 1)[0] for c in COMPONENTS}
        )
    theirs = pd.DataFrame(rows)
    out = []
    for c in COMPONENTS:
        key = f"v{c[0]}"
        difference = rates[key].to_numpy() - theirs[key].to_numpy()
        out.append(
            {
                "component": c,
                "rms_mm_yr": np.sqrt(np.mean(difference**2)),
                "median_abs_mm_yr": np.median(np.abs(difference)),
                "bias_mm_yr": difference.mean(),
            }
        )
    return pd.DataFrame(out)


def score_daily(
    images: np.ndarray,
    nodes: pd.DataFrame,
    days: pd.DatetimeIndex,
    published: Callable[[int], pd.DataFrame],
) -> pd.DataFrame:
    """Compare daily images with the time-variable files, per node and component."""
    rows = []
    for j, node in enumerate(nodes.id):
        theirs = published(int(node)).reindex(days)
        for k, c in enumerate(COMPONENTS):
            error = images[:, j, k] - theirs[c].to_numpy()
            error = error - np.nanmedian(error)
            monthly = pd.Series(error, index=days).resample("30D").median()
            rows.append(
                {
                    "id": int(node),
                    "component": c,
                    "rms_monthly_mm": np.sqrt(np.nanmean(monthly**2)),
                    "published_sigma_mm": theirs[f"sigma_{c}"].median(),
                }
            )
    return pd.DataFrame(rows)


def run(
    args: argparse.Namespace,
    stations: Any | None = None,
    grid: Callable[[str], Any] | None = None,
) -> dict[str, Path]:
    """Rebuild one box and return the files written."""
    cache = args.cache_dir
    stations = stations or UnrSource(
        cache_dir=None if cache is None else cache / "stations"
    )

    def published_source(kind: Literal["constant", "variable"]) -> UnrGridSource:
        where = None if cache is None else cache / f"grid_{kind}"
        return UnrGridSource(gridded_type=kind, cache_dir=where)

    open_grid = grid or published_source
    args.output.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    west, south, east, north = args.bbox

    nodes = pd.DataFrame(open_grid("variable").stations(bbox=tuple(args.bbox)))[
        ["id", "lon", "lat"]
    ].reset_index(drop=True)
    box = (
        west - args.buffer,
        south - args.buffer,
        east + args.buffer,
        north + args.buffer,
    )
    sites = pd.DataFrame(stations.stations(bbox=box))[["id", "lon", "lat"]]
    sites = sites.drop_duplicates("id").reset_index(drop=True)
    steps = stations.steps(list(sites.id))
    print(
        f"{len(nodes)} nodes, {len(sites)} stations in the box and its buffer",
        flush=True,
    )

    table = reconnaissance(stations, sites, steps, start, end, args.workers)
    if table.empty:
        msg = "no station in the box has enough data"
        raise SystemExit(msg)
    ssf = (
        gridding.atbd_ssf()
        if args.ssf == "fig2-23"
        else make_ssf(table.lon, table.lat, table.vu, table.su)
    )
    table = table.join(gridding.filter_station_rates(table, ssf))
    table.to_csv(args.output / "stations.csv", index=False)
    written = {"stations": args.output / "stations.csv"}

    rates = gridding.rate_grid(gridding.decluster(table), nodes.lon, nodes.lat, ssf)
    rates.insert(0, "id", nodes.id.to_numpy())
    rates.to_csv(args.output / "rates.csv", index=False)
    written["rates"] = args.output / "rates.csv"

    limits = {
        "as-published": gridding.PUBLISHED_SELECTION,
        "no-completeness": gridding.SelectionCriteria(min_completeness=0.0),
        "atbd": gridding.SelectionCriteria(),
    }[args.selection]
    chosen = gridding.decluster(table[gridding.select_time_variable(table, limits)])
    print(
        f"{len(table)} stations long enough, {len(chosen)} selected for daily",
        flush=True,
    )
    series = {s: read_positions(stations, s, end) for s in chosen.id}
    by_code = {
        code: {
            s: pd.DatetimeIndex(g.date.unique())
            for s, g in steps[steps.code == code].groupby("id")
        }
        for code in (CODE_EQUIPMENT, CODE_EARTHQUAKE)
    }
    cube = gridding.build_position_cube(
        series, chosen, days, by_code[CODE_EQUIPMENT], by_code[CODE_EARTHQUAKE]
    )
    print("filled", gridding.fill_gaps(cube, ssf, n_workers=args.workers), flush=True)
    images = gridding.image_positions(
        cube, nodes.lon, nodes.lat, ssf, n_workers=args.workers
    )
    np.savez_compressed(
        args.output / "daily.npz",
        displacement_mm=images.astype("f4"),
        days=days.to_numpy(),
        node_id=nodes.id.to_numpy(),
        component=np.array(COMPONENTS),
    )
    written["daily"] = args.output / "daily.npz"

    if not args.no_compare:
        constant, variable = open_grid("constant"), open_grid("variable")
        scores = score_rates(rates, constant.node_series)
        scores.to_csv(args.output / "scores_rates.csv", index=False)
        print(scores.round(3).to_string(index=False))
        daily = score_daily(images, nodes, days, variable.node_series)
        daily.to_csv(args.output / "scores_daily.csv", index=False)
        print(
            daily.groupby("component")[["rms_monthly_mm", "published_sigma_mm"]]
            .median()
            .round(2)
        )
        written["scores_rates"] = args.output / "scores_rates.csv"
        written["scores_daily"] = args.output / "scores_daily.csv"
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="grid points to rebuild, in degrees",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("unr_grid_rebuild"))
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="keep downloads here"
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=1.5,
        help="stations this far outside, in degrees",
    )
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2026-03-06", help="first day not to use")
    parser.add_argument(
        "--selection",
        choices=("as-published", "no-completeness", "atbd"),
        default="as-published",
        help="daily stations: atbd applies section 2.5.2 as written",
    )
    parser.add_argument("--ssf", choices=("fig2-23", "regional"), default="fig2-23")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--no-compare", action="store_true", help="do not read the published files"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Rebuild one box."""
    run(parse_args(argv))


if __name__ == "__main__":
    main()

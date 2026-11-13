# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
r"""Regenerate the UNR time-constant grid as a smooth surface with uncertainties.

    python scripts/regenerate-unr-grid.py --stations rates.parquet -o grid \\
        --areas deforming_areas.geojson --breaklines breaklines.geojson
    python scripts/regenerate-unr-grid.py --rates-from-unr --cache-dir unr_cache -o grid

Station rates (mm/yr) are screened with the GPS Imaging median filter, the stations
inside `--areas` are left out of the vertical (remove-restore: the grid carries the
regional field only), and the rest are kriged onto the nodes of UNR Version 0.3 with
`geepers.structure_kriging`: horizontals about a local plane where the stations show
one, the vertical about a local mean, both with weights kept non-negative. Breaklines
cut the horizontals only.

Writes to the output directory:

    rates.parquet                   per node: rates, sigmas and the kriging diagnostics
    unrgrid/<id>_IGS20.tenv8        one straight-line series per node, in the layout of
                                    UNR's time-constant files

so that ``UnrGridSource(version="0.3", gridded_type="constant", cache_dir=OUT)`` opens
it like the published grid. With ``--viewer``, also ``viewer.parquet`` for
``scripts/browse_unr_grid.html`` (monthly, from 2016).

`--stations` takes a table with ``id``, ``lon``, ``lat`` and per component ``v?`` and
``s?`` (mm/yr), such as the station table of ``recreate-unr-grid.py``.
``--rates-from-unr`` computes it instead: MIDAS rates from UNR positions, avoiding
catalogued steps, for stations with 4 years of data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geepers import gridding
from geepers import structure_kriging as sk
from geepers._optional import require
from geepers.gps_sources import UnrGridSource, UnrSource

COMPONENTS = "enu"
# A raw rate is an outlier beyond both of these from its median-filtered rate
OUTLIER_SIGMAS, OUTLIER_FLOOR = 3.0, 3.0
# Epochs of the published time-constant files: daily, 2014.0014 to 2026.1766
FIRST_EPOCH, N_EPOCHS = 2014.0014, 4449
LAST_MONTH = "2026-03-01"
FRAME = "IGS20"
_SHARED: dict[str, Any] = {}


def unwrap(lon: Any) -> np.ndarray:
    """Return longitudes east of 0 as negative numbers, so nothing splits at 180."""
    lon = np.asarray(lon, float)
    return np.where(lon > 0, lon - 360.0, lon)


def read_areas(paths: Sequence[Path]) -> Any:
    """Return the union of the polygons in GeoJSON files, or None."""
    if not paths:
        return None
    shapely = require("shapely")

    shapes = [
        shapely.geometry.shape(f["geometry"])
        for path in paths
        for f in json.loads(Path(path).read_text())["features"]
    ]
    return shapely.union_all(shapes)


def inside(areas: Any, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Tell which points lie in the areas, wherever their longitude is written."""
    if areas is None:
        return np.zeros(len(lon), bool)
    shapely = require("shapely")

    wrapped = ((np.asarray(lon, float) + 180.0) % 360.0) - 180.0
    return np.asarray(shapely.contains_xy(areas, wrapped, np.asarray(lat, float)))


class Component:
    """What `image` needs for one component: screened stations and their structure."""

    def __init__(
        self, table: pd.DataFrame, c: str, lines: list[np.ndarray] | None, trend: bool
    ) -> None:
        """Screen component `c` of `table` and measure its structure."""
        sigma = table[f"s{c}"].to_numpy()
        value = table[f"v{c}"].to_numpy()
        limit = np.maximum(OUTLIER_SIGMAS * sigma, OUTLIER_FLOOR)
        keep = np.abs(value - table[f"vmsf{c}"].to_numpy()) <= limit
        self.lon = table.lon.to_numpy()[keep]
        self.lat = table.lat.to_numpy()[keep]
        self.value, self.sigma = value[keep], sigma[keep]
        self.trend, self.lines = trend, lines
        left = sk.detrend(self.lon, self.lat, self.value) if trend else self.value
        self.g = sk.structure(self.lon, self.lat, left)
        self.g_plain = sk.structure(self.lon, self.lat, self.value) if trend else None

    def krige(self, lon: np.ndarray, lat: np.ndarray) -> pd.DataFrame:
        est = sk.krige(
            self.lon,
            self.lat,
            self.value,
            self.sigma,
            lon,
            lat,
            self.g,
            breaklines=self.lines,
            trend=self.trend,
            g_plain=self.g_plain,
            positive=True,
        )
        # About a plane what is left is unmodelled signal, so the sigma keeps it
        planar = est.planar.to_numpy() == 1
        est["sigma"] = np.where(planar, est.sigma_total, est.sigma)
        return est


def prepare(
    stations: pd.DataFrame, areas: Any, lines: list[np.ndarray] | None
) -> dict[str, Any]:
    """Screen the stations and measure the structure of each component."""
    ssf = gridding.atbd_ssf()
    stations = stations.assign(lon=unwrap(stations.lon)).reset_index(drop=True)

    def screened(subset: pd.DataFrame) -> pd.DataFrame:
        table = subset.drop(columns=[c for c in subset if c.startswith("vmsf")])
        table = table.join(gridding.filter_station_rates(table, ssf, breaklines=lines))
        return gridding.decluster(table).reset_index(drop=True)

    everything = screened(stations)
    left_in = ~inside(areas, stations.lon.to_numpy(), stations.lat.to_numpy())
    regional = screened(stations[left_in].reset_index(drop=True))
    return {
        "ssf": ssf,
        "lines": lines,
        "tables": {"h": everything, "u": regional},
        "e": Component(everything, "e", lines, trend=True),
        "n": Component(everything, "n", lines, trend=True),
        # A plate boundary or a creeping fault offsets horizontal rates, not vertical
        "u": Component(regional, "u", None, trend=False),
        "n_left_out": int((~left_in).sum()),
    }


def image(prepared: dict[str, Any], lon: np.ndarray, lat: np.ndarray) -> pd.DataFrame:
    """Estimate every component at the points; the median grid where kriging cannot."""
    out = {}
    for c in COMPONENTS:
        est = prepared[c].krige(lon, lat)
        missing = est.value.isna().to_numpy()
        if missing.any():
            table = prepared["tables"]["u" if c == "u" else "h"]
            lines = None if c == "u" else prepared["lines"]
            fallback = gridding.rate_grid(
                table, lon[missing], lat[missing], prepared["ssf"], breaklines=lines
            )
            est.loc[missing, ["value", "sigma", "n"]] = fallback[
                [f"v{c}", f"s{c}", f"n{c}"]
            ].to_numpy()
        out[f"v{c}"] = est.value.to_numpy()
        out[f"s{c}"] = est.sigma.to_numpy()
        out[f"st{c}"] = est.sigma_total.to_numpy()
        out[f"n{c}"] = est.n.to_numpy()
        out[f"a{c}"] = est.amplitude.to_numpy()
        out[f"km{c}"] = est.km.to_numpy()
        out[f"p{c}"] = est.planar.to_numpy()
    return pd.DataFrame(out)


def _image_chunk(index: np.ndarray) -> pd.DataFrame:
    nodes = _SHARED["nodes"]
    est = image(
        _SHARED["prepared"], nodes.lon.to_numpy()[index], nodes.lat.to_numpy()[index]
    )
    return est.assign(position=index)


def image_nodes(
    prepared: dict[str, Any], nodes: pd.DataFrame, workers: int = 8
) -> pd.DataFrame:
    """Image all nodes, in parallel chunks."""
    nodes = nodes.assign(lon=unwrap(nodes.lon)).reset_index(drop=True)
    chunks = np.array_split(np.arange(len(nodes)), max(1, len(nodes) // 120))
    _SHARED.update(nodes=nodes, prepared=prepared)
    if workers <= 1:
        parts = [_image_chunk(c) for c in chunks]
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
            parts = list(pool.map(_image_chunk, chunks))
    est = pd.concat(parts).sort_values("position").drop(columns="position")
    return pd.concat([nodes[["id", "lon", "lat"]], est.reset_index(drop=True)], axis=1)


def write_unr_files(rates: pd.DataFrame, out: Path) -> Path:
    """Write one straight-line .tenv8 per node where `UnrGridSource` looks for it."""
    folder = out / "unrgrid"
    folder.mkdir(parents=True, exist_ok=True)
    years = FIRST_EPOCH + np.arange(N_EPOCHS) / 365.25
    dt = years - years[0]
    for row in rates.itertuples():
        # sigmas are printed to 0.001 mm/yr, and a zero is not a valid sigma
        se, sn, su = (max(float(s), 0.001) for s in (row.se, row.sn, row.su))
        lines = [
            f" {y:9.4f} {row.ve * t:11.3f} {row.vn * t:11.3f} {row.vu * t:11.3f}"
            f" {se:6.3f} {sn:6.3f} {su:6.3f} 0"
            for y, t in zip(years, dt, strict=True)
        ]
        name = f"{int(row.id):06d}_{FRAME}.tenv8"
        (folder / name).write_text("\n".join(lines) + "\n")
    return folder


def write_viewer(rates: pd.DataFrame, out: Path, start: str = "2016-01-01") -> Path:
    """Write monthly displacement for browse_unr_grid.html, through its own exporter."""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "create_geoparquet", here / "create-geoparquet.py"
    )
    assert spec is not None
    assert spec.loader is not None
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    months = pd.date_range(start, LAST_MONTH, freq="MS")
    years = ((months - months[0]).days / 365.25).to_numpy()
    n, m = len(rates), len(months)
    frame = pd.DataFrame(
        {
            "id": np.repeat(rates.id.map(lambda i: f"{int(i):06d}").to_numpy(), m),
            "date": np.tile(months, n),
            "lon": np.repeat(((rates.lon + 180) % 360 - 180).to_numpy(), m),
            "lat": np.repeat(rates.lat.to_numpy(), m),
        }
    )
    for c, name in zip(COMPONENTS, ("east", "north", "up"), strict=True):
        # the viewer reads meters
        frame[name] = (np.outer(rates[f"v{c}"].to_numpy(), years) / 1000).ravel()
        frame[f"sigma_{name}"] = np.repeat(rates[f"s{c}"].to_numpy() / 1000, m)
    return exporter.export_gdf_to_parquet(frame, out / "viewer.parquet")


def rates_from_unr(
    cache_dir: Path | None, bbox: Sequence[float], workers: int
) -> pd.DataFrame:
    """Compute MIDAS station rates from UNR positions, as recreate-unr-grid.py does."""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "recreate_unr_grid", here / "recreate-unr-grid.py"
    )
    assert spec is not None
    assert spec.loader is not None
    recreate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = recreate
    spec.loader.exec_module(recreate)
    source = UnrSource(cache_dir=None if cache_dir is None else cache_dir / "stations")
    sites = pd.DataFrame(source.stations(bbox=tuple(bbox)))[["id", "lon", "lat"]]
    sites = sites.drop_duplicates("id").reset_index(drop=True)
    steps = source.steps(list(sites.id))
    table = recreate.reconnaissance(
        source,
        sites,
        steps,
        pd.Timestamp("2012-01-01"),
        pd.Timestamp("2026-03-06"),
        workers,
    )
    good = np.isfinite(table[["ve", "vn", "vu", "se", "sn", "su"]]).all(axis=1)
    return table[good].reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    given = parser.add_mutually_exclusive_group(required=True)
    given.add_argument(
        "--stations", type=Path, help="station rate table (.parquet or .csv)"
    )
    given.add_argument(
        "--rates-from-unr", action="store_true", help="compute MIDAS rates"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("unr_grid_regenerated")
    )
    parser.add_argument(
        "--areas",
        type=Path,
        nargs="*",
        default=[],
        help="GeoJSON polygons whose stations are left out of the vertical",
    )
    parser.add_argument(
        "--breaklines",
        type=Path,
        nargs="*",
        default=[],
        help="GeoJSON lines across which horizontal rates are discontinuous",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-180, -90, 180, 90),
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="nodes to image",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--viewer", action="store_true", help="also write viewer.parquet"
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, nodes: pd.DataFrame | None = None) -> dict[str, Path]:
    """Regenerate the grid and return the files written."""
    args.output.mkdir(parents=True, exist_ok=True)
    if args.rates_from_unr:
        stations = rates_from_unr(args.cache_dir, (-180, -90, 180, 90), args.workers)
    elif args.stations.suffix == ".csv":
        stations = pd.read_csv(args.stations)
    else:
        stations = pd.read_parquet(args.stations)
    lines = [
        ln for path in args.breaklines for ln in gridding.read_breaklines(path)
    ] or None
    prepared = prepare(stations, read_areas(args.areas), lines)
    print(
        f"{len(stations)} stations, {prepared['n_left_out']} left out of the vertical",
        flush=True,
    )
    if nodes is None:
        nodes = pd.DataFrame(
            UnrGridSource(version="0.3").stations(bbox=tuple(args.bbox))
        )
    rates = image_nodes(prepared, nodes[["id", "lon", "lat"]], args.workers)
    rates.to_parquet(args.output / "rates.parquet", index=False)
    written = {
        "rates": args.output / "rates.parquet",
        "unrgrid": write_unr_files(rates, args.output),
    }
    if args.viewer:
        written["viewer"] = write_viewer(rates, args.output)
    print(f"{len(rates)} nodes written to {args.output}", flush=True)
    return written


def main(argv: list[str] | None = None) -> None:
    """Regenerate the grid."""
    run(parse_args(argv))


if __name__ == "__main__":
    main()

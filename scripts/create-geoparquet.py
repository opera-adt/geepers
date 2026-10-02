# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Export UNR gridded time series to a browser-optimized Parquet file.

The output file is consumed by `browse_unr_grid.html` (MapLibre GL viewer),
and is equally usable from pandas / GeoPandas / DuckDB:

    duckdb -c "SELECT * FROM 'unr_grid.parquet' LIMIT 5"

Layout notes
------------
- Long format: one row per (grid point, date).
- Sorted by (date, point) and written with snappy compression, which
  hyparquet can decompress natively in the browser (no extra codecs).
- `date_idx` / `point_idx` integer columns let the viewer scatter values
  into dense [n_dates x n_points] matrices without parsing dates or ids.
- The full date list and point coordinates are embedded as JSON in the
  Parquet file-level metadata (key ``unr_grid_meta``), so the viewer can
  build the map without scanning string columns.
"""

import datetime
import json
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

from geepers.gps_sources import BaseGpsSource, UnrGridSource, UnrSource

META_KEY = "unr_grid_meta"
MIDAS_URL = "https://geodesy.unr.edu/gps_timeseries/IGS20/midas/midas.IGS.txt"


def export_gdf_to_parquet(gdf, output_file="unr_grid.parquet") -> Path:
    """Export a long-format GeoDataFrame from `timeseries_many` to Parquet.

    Parameters
    ----------
    gdf : GeoDataFrame
        Result from `timeseries_many` (one row per point per date).
    output_file : str | Path
        Output .parquet path.

    Returns
    -------
    Path
        Path to the written file.

    """
    output_path = Path(output_file).with_suffix(".parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["date", "id"], kind="mergesort", ignore_index=True)

    # Integer indices for fast dense-matrix assembly in the browser
    dates = df["date"].dt.normalize()
    unique_dates = dates.drop_duplicates().reset_index(drop=True)
    df["date_idx"] = dates.map(
        pd.Series(np.arange(len(unique_dates), dtype=np.int32), index=unique_dates)
    )
    point_codes, unique_ids = pd.factorize(df["id"], sort=True)
    df["point_idx"] = point_codes.astype(np.int32)

    points = (
        df.drop_duplicates("point_idx")
        .sort_values("point_idx")[["id", "lon", "lat"]]
        .reset_index(drop=True)
    )

    all_value_cols = ["east", "north", "up", "sigma_east", "sigma_north", "sigma_up"]
    value_cols = [c for c in all_value_cols if c in df.columns]
    out = pd.DataFrame(
        {
            "id": df["id"].astype("string"),
            "date": df["date"].dt.date,  # date32: 4 bytes, no tz ambiguity
            "date_idx": df["date_idx"],
            "point_idx": df["point_idx"],
            "lon": df["lon"].astype(np.float32),
            "lat": df["lat"].astype(np.float32),
            **{c: df[c].astype(np.float32) for c in value_cols},
        }
    )

    meta = {
        "dates": [d.strftime("%Y-%m-%d") for d in unique_dates],
        "points": {
            "id": points["id"].tolist(),
            "lon": [round(float(v), 6) for v in points["lon"]],
            "lat": [round(float(v), 6) for v in points["lat"]],
        },
        "value_columns": value_cols,
        "units": "meters",
    }

    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), META_KEY.encode(): json.dumps(meta).encode()}
    )
    # Row groups aligned to whole dates keep per-date reads contiguous
    n_points = len(points)
    rows_per_group = max(n_points * max(1, 262_144 // max(n_points, 1)), n_points)
    pq.write_table(
        table,
        output_path,
        compression="snappy",  # hyparquet decodes snappy without extra codecs
        row_group_size=rows_per_group,
        use_dictionary=["id"],
    )

    size_mb = output_path.stat().st_size / 2**20
    print(
        f"Wrote {output_path} ({size_mb:.1f} MB): "
        f"{len(out):,} rows, {n_points} points, {len(unique_dates)} dates"
    )
    return output_path


def midas_timeseries(
    bbox: tuple[float, float, float, float], start_date: datetime.datetime
) -> pd.DataFrame:
    """Build monthly straight-line series from UNR MIDAS station velocities.

    A lightweight stand-in for real station positions, for when downloading
    every station's series is impractical (e.g. a global demo of the viewer):
    each station's east/north/up is its MIDAS velocity times the time from
    the middle of its record, on the 1st of each month it was observing.
    Seasonal signals, offsets, noise and uncertainties are all absent.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box (west, south, east, north) in degrees.
    start_date : datetime
        First month to keep.

    Returns
    -------
    pd.DataFrame
        Long format (id, date, lon, lat, east, north, up), in meters.

    References
    ----------
    Blewitt, G., et al. (2016), MIDAS robust trend estimator for accurate GPS
    station velocities without step detection, JGR Solid Earth,
    https://doi.org/10.1002/2015JB012552

    """
    cols = {
        0: "id",
        2: "t_first",
        3: "t_last",
        8: "ve",
        9: "vn",
        10: "vu",
        24: "lat",
        25: "lon",
    }
    midas = pd.read_csv(MIDAS_URL, sep=r"\s+", header=None, usecols=list(cols))
    midas = midas.rename(columns=cols)
    # MIDAS longitudes run over [-360, 0]
    midas["lon"] = (midas["lon"] + 180) % 360 - 180
    west, south, east, north = bbox
    midas = midas[midas["lon"].between(west, east) & midas["lat"].between(south, north)]

    months = pd.date_range(start_date.replace(day=1), datetime.date.today(), freq="MS")
    years = (months.year + (months.dayofyear - 1) / 365.25).to_numpy()
    t_first, t_last = midas["t_first"].to_numpy(), midas["t_last"].to_numpy()
    observing = (years[:, None] >= t_first) & (years[:, None] <= t_last)
    d_idx, p_idx = np.nonzero(observing)
    dt = years[d_idx] - (t_first + t_last)[p_idx] / 2
    return pd.DataFrame(
        {
            "id": midas["id"].to_numpy()[p_idx],
            "date": months[d_idx],
            "lon": midas["lon"].to_numpy()[p_idx],
            "lat": midas["lat"].to_numpy()[p_idx],
            **{
                c: midas[v].to_numpy()[p_idx] * dt
                for c, v in [("east", "ve"), ("north", "vn"), ("up", "vu")]
            },
        }
    )


def main(
    bbox: tuple[float, float, float, float],
    source: Literal["grid", "stations", "midas"] = "grid",
    start_date: datetime.datetime = datetime.datetime(2016, 1, 1),
    output_file: Path = Path("unr_grid.parquet"),
    version: Literal["0.1", "0.3"] = "0.3",
    gridded_type: Literal["constant", "variable"] = "variable",
    cache_dir: Path | None = None,
    max_workers: int = 8,
    clear_cache: bool = False,
    zero_by: Literal["mean", "start", "none"] = "mean",
):
    """Download UNR time series and export a viewer-ready Parquet file.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Bounding box (west, south, east, north) in degrees.
    source : {"grid", "stations", "midas"}
        "grid" downloads the UNR gridded (interpolated) product;
        "stations" downloads real UNR GPS station positions (.tenv3);
        "midas" builds monthly straight lines from UNR MIDAS station
        velocities (a light demo stand-in; see `midas_timeseries`).
        Default is "grid".
    start_date : datetime
        First date to keep. Default is 2016-01-01.
    output_file : Path
        Output .parquet path. Default is unr_grid.parquet.
    version : {"0.1", "0.3"}
        UNR grid data version (grid source only).
    gridded_type : {"constant", "variable"}
        Time-constant or time-variable gridded product (0.3 only; grid source only).
    cache_dir : Path, optional
        Where downloaded .tenv8/.tenv3 files are cached.
        Default is ~/.cache/geepers.
    max_workers : int
        Parallel download threads. Default is 8.
    clear_cache : bool
        Delete this source's download cache before fetching, forcing
        fresh downloads. Default is False.
    zero_by : {"mean", "start", "none"}
        How each point's time series is zeroed: subtract its mean, its
        first ~10 epochs, or "none" to keep values exactly as published.
        Default is "mean".

    """
    if source == "midas":
        export_gdf_to_parquet(
            gdf=midas_timeseries(bbox, start_date), output_file=output_file
        )
        return

    src: BaseGpsSource
    if source == "grid":
        src = UnrGridSource(
            version=version, gridded_type=gridded_type, cache_dir=cache_dir
        )
    else:
        src = UnrSource(cache_dir=cache_dir)

    if clear_cache:
        print(f"Clearing download cache: {src._cache_dir}")
        shutil.rmtree(src._cache_dir, ignore_errors=True)
        src._cache_dir.mkdir(parents=True, exist_ok=True)

    gdf = src.timeseries_many(
        bbox=bbox,
        start_date=start_date.replace(tzinfo=datetime.UTC).isoformat(),
        zero_by=zero_by,
        max_workers=max_workers,
    )
    export_gdf_to_parquet(gdf=gdf, output_file=output_file)


if __name__ == "__main__":
    tyro.cli(main)

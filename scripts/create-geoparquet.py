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

from geepers.gps_sources import UnrGridSource, UnrSource

META_KEY = "unr_grid_meta"


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

    value_cols = ["east", "north", "up", "sigma_east", "sigma_north", "sigma_up"]
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


def main(
    bbox: tuple[float, float, float, float],
    source: Literal["grid", "stations"] = "grid",
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
    source : {"grid", "stations"}
        "grid" downloads the UNR gridded (interpolated) product;
        "stations" downloads real UNR GPS station positions (.tenv3).
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

    gdf = src.timeseries_many(bbox=bbox, start_date=start_date,
                              zero_by=zero_by, max_workers=max_workers)
    export_gdf_to_parquet(gdf=gdf, output_file=output_file)


if __name__ == "__main__":
    tyro.cli(main)

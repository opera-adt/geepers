import datetime
import json
from pathlib import Path
from typing import Literal

import pandas as pd
import tyro

from geepers.gps_sources import UnrGridSource


def export_gdf_to_geoparquet(gdf, output_file="unr_grid.parquet"):
    """Export a GeoDataFrame to multiple GeoJSON files organized by date.

    Parameters
    ----------
    gdf : GeoDataFrame
        Result from `timeseries_many`
    output_dir : str
        Directory to save GeoJSON files

    Returns
    -------
    dict
        Mapping of source names to file paths

    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(gdf["date"]):
        gdf["date"] = pd.to_datetime(gdf["date"])

    # Create output directory
    output_path = Path(output_file)
    if output_path.suffix.lower() != ".parquet":
        output_path = output_path.with_suffix(".parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_parquet(
        output_file, compression="snappy", row_group_size=None, geometry_encoding="WKB"
    )

    # Print file size statistics
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nOutput file size: {file_size_mb:.2f} MB")
    print(f"Successfully created: {output_file}")


def main(
    bbox: tuple[float, float, float, float],
    start_date: datetime.datetime = datetime.datetime(2016, 1, 1),
    output_dir=Path("geojson_sources"),
    version: Literal["0.1", "0.2"] = "0.2",
):
    """Export a GeoDataFrame to multiple GeoJSON files organized by date.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Result from `timeseries_many`
    output_dir : str
        Directory to save GeoJSON files
    start_date : datetime
        First date to download from UNR.
        Default is 2016-01-01

    """
    unrg = UnrGridSource(version=version)
    gdf = unrg.timeseries_many(bbox=bbox, start_date=start_date)
    export_gdf_to_geoparquet(gdf=gdf, output_dir=output_dir)


if __name__ == "__main__":
    tyro.cli(main)

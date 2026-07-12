import datetime
import json
from pathlib import Path
from typing import Literal

import pandas as pd
import tyro

from geepers.gps_sources import UnrGridSource


def export_gdf_to_geojson_sources(gdf, output_dir="geojson_sources"):
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
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Group by date
    grouped = gdf.groupby(gdf["date"].dt.date)

    sources_map = {}

    # Create one source per date with all variables
    for date, date_group in grouped:
        date_str = date.strftime("%Y-%m-%d")

        features = []

        for idx, row in date_group.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["lon"], row["lat"]],
                },
                "properties": {
                    "id": row.get("id", idx),
                    "date": row["date"].isoformat(),
                    "east": float(row["east"]),
                    "north": float(row["north"]),
                    "up": float(row["up"]),
                    "sigma_east": float(row["sigma_east"]),
                    "sigma_north": float(row["sigma_north"]),
                    "sigma_up": float(row["sigma_up"]),
                },
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}

        # Save to file
        filename = f"{date_str}.geojson"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            json.dump(geojson, f, indent=2)

        # Store in sources map
        source_name = date_str
        sources_map[source_name] = filepath

    print(f"Exported {len(sources_map)} GeoJSON files to '{output_dir}/'")
    return sources_map


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
    export_gdf_to_geojson_sources(gdf=gdf, output_dir=output_dir)


if __name__ == "__main__":
    tyro.cli(main)

"""Tests for scripts/regenerate-unr-grid.py, on a small synthetic network."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geepers.gps_sources import UnrGridSource

SCRIPT = Path(__file__).parents[1] / "scripts" / "regenerate-unr-grid.py"
spec = importlib.util.spec_from_file_location("regenerate_unr_grid", SCRIPT)
assert spec is not None
assert spec.loader is not None
regen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = regen
spec.loader.exec_module(regen)

pytest.importorskip("shapely")


@pytest.fixture
def stations() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 250
    lon = rng.uniform(-118.0, -112.0, n)
    lat = rng.uniform(34.0, 39.0, n)
    table = pd.DataFrame(
        {
            "id": [f"S{i:03d}" for i in range(n)],
            "lon": lon,
            "lat": lat,
            # a tectonic gradient east, flat north, small noise up
            "ve": 2.0 * (lon + 115.0) + rng.normal(0, 0.2, n),
            "vn": rng.normal(0, 0.2, n),
            "vu": rng.normal(0, 0.3, n),
        }
    )
    for c in "enu":
        table[f"s{c}"] = 0.3
    return table


@pytest.fixture
def nodes() -> pd.DataFrame:
    lon, lat = np.meshgrid(np.arange(-117.0, -113.0, 0.5), np.arange(35.0, 38.0, 0.5))
    return pd.DataFrame(
        {"id": np.arange(1, lon.size + 1), "lon": lon.ravel(), "lat": lat.ravel()}
    )


def subsidence_area(tmp_path: Path) -> Path:
    square = [
        [-115.6, 36.0],
        [-114.6, 36.0],
        [-114.6, 37.0],
        [-115.6, 37.0],
        [-115.6, 36.0],
    ]
    path = tmp_path / "areas.geojson"
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [square]},
    }
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    return path


def test_basin_stations_leave_the_vertical_only(stations, nodes, tmp_path):
    basin = stations.lon.between(-115.6, -114.6) & stations.lat.between(36.0, 37.0)
    stations.loc[basin, "vu"] = -20.0
    centre = nodes[(nodes.lon == -115.0) & (nodes.lat == 36.5)]

    with_basin = regen.image(
        regen.prepare(stations, None, None),
        centre.lon.to_numpy(),
        centre.lat.to_numpy(),
    )
    areas = regen.read_areas([subsidence_area(tmp_path)])
    without = regen.image(
        regen.prepare(stations, areas, None),
        centre.lon.to_numpy(),
        centre.lat.to_numpy(),
    )

    assert with_basin.vu[0] < -10
    assert abs(without.vu[0]) < 1.0
    # horizontals still use the basin stations
    assert without.ve[0] == pytest.approx(with_basin.ve[0])


def test_gradient_is_followed_with_a_plane(stations, nodes):
    est = regen.image_nodes(regen.prepare(stations, None, None), nodes, workers=1)

    truth = 2.0 * (est.lon + 115.0)
    assert np.abs(est.ve - truth).max() < 0.5
    assert (est.pe == 1).mean() > 0.8
    assert (est[["se", "sn", "su"]] > 0).all().all()


def test_outlier_station_is_screened(stations, nodes):
    nearest = np.argmin(np.hypot(stations.lon + 115.0, stations.lat - 36.5))
    stations.loc[nearest, "vn"] = 40.0
    centre = nodes[(nodes.lon == -115.0) & (nodes.lat == 36.5)]

    est = regen.image(
        regen.prepare(stations, None, None),
        centre.lon.to_numpy(),
        centre.lat.to_numpy(),
    )

    assert abs(est.vn[0]) < 1.0


def test_output_opens_with_unr_grid_source(stations, nodes, tmp_path):
    args = regen.parse_args(
        ["--stations", "unused", "-o", str(tmp_path / "out"), "--workers", "1"]
    )
    args.stations = tmp_path / "stations.parquet"
    stations.to_parquet(args.stations)

    written = regen.run(args, nodes=nodes)
    rates = pd.read_parquet(written["rates"])
    node = int(rates.id.iloc[7])
    source = UnrGridSource(
        version="0.3", gridded_type="constant", cache_dir=tmp_path / "out"
    )
    series = source.node_series(node)

    slope = np.polyfit(series.decimal_year, series.east, 1)[0]
    assert len(series) == regen.N_EPOCHS
    assert slope == pytest.approx(rates.ve.iloc[7], abs=2e-3)
    assert series.sigma_east.iloc[0] == pytest.approx(
        max(rates.se.iloc[7], 0.001), abs=5e-4
    )

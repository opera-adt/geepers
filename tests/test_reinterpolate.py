# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""GNSS-side remove-restore: re-interpolating grid nodes inside exclusion areas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geepers.gps_imaging import reinterpolate_nodes

shapely = pytest.importorskip("shapely")


@pytest.fixture
def grid_with_bowl():
    """A regular 0.25-degree grid, a planar field, and a subsidence bowl in a box."""
    lons, lats = np.meshgrid(np.arange(-96.5, -94.4, 0.25), np.arange(28.5, 31.1, 0.25))
    lon = lons.ravel()
    lat = lats.ravel()
    plane = 1.5 * (lon + 95.5) - 0.8 * (lat - 29.8) - 2.0  # mm/yr
    box = shapely.box(-95.6, 29.4, -95.0, 30.0)
    inside = shapely.contains_xy(box, lon, lat)
    # A 20 mm/yr bowl where the basin is; the grid has smeared nothing here
    # on purpose, so the plane is the truth we want back
    vu = plane - 20.0 * inside
    nodes = pd.DataFrame(
        {
            "id": np.arange(lon.size),
            "lon": lon,
            "lat": lat,
            "vu": vu,
            "sigma_vu": np.full(lon.size, 0.4),
            "ve": 0.1 * plane,
            "sigma_ve": np.full(lon.size, 0.3),
        }
    )
    return nodes, box, plane, inside


def test_inside_nodes_recover_the_regional_field(grid_with_bowl):
    nodes, box, plane, inside = grid_with_bowl
    out = reinterpolate_nodes(
        nodes, box, columns=(("vu", "sigma_vu"), ("ve", "sigma_ve"))
    )

    assert out["reinterpolated"].to_numpy().tolist() == inside.tolist()
    # Inside: the bowl is gone, the plane is back (median filter of a plane
    # is not exact, hence the tolerance)
    err = out.loc[inside, "vu"].to_numpy() - plane[inside]
    assert np.max(np.abs(err)) < 1.0, err
    # Outside: untouched, bit for bit
    np.testing.assert_array_equal(
        out.loc[~inside, "vu"].to_numpy(), nodes.loc[~inside, "vu"].to_numpy()
    )
    np.testing.assert_array_equal(
        out.loc[~inside, "sigma_vu"], nodes.loc[~inside, "sigma_vu"]
    )
    # Sigmas inside are finite and positive
    assert np.all(np.isfinite(out.loc[inside, "sigma_vu"]))
    assert np.all(out.loc[inside, "sigma_vu"] > 0)
    # Columns and ids preserved; input not mutated
    assert list(out["id"]) == list(nodes["id"])
    assert nodes["vu"].min() < -15.0


def test_no_node_inside_returns_copy_with_flag(grid_with_bowl):
    nodes, _, _, _ = grid_with_bowl
    far = shapely.box(10, 10, 11, 11)
    out = reinterpolate_nodes(nodes, far)
    assert not out["reinterpolated"].any()
    pd.testing.assert_frame_equal(out.drop(columns="reinterpolated"), nodes)


def test_everything_inside_raises(grid_with_bowl):
    nodes, _, _, _ = grid_with_bowl
    everything = shapely.box(-100, 20, -90, 40)
    with pytest.raises(ValueError, match="every node"):
        reinterpolate_nodes(nodes, everything)


def test_accepts_iterable_and_geoseries(grid_with_bowl):
    nodes, box, _, inside = grid_with_bowl
    out_iter = reinterpolate_nodes(nodes, [box], columns=(("vu", "sigma_vu"),))
    assert out_iter["reinterpolated"].sum() == inside.sum()
    gpd = pytest.importorskip("geopandas")
    out_gs = reinterpolate_nodes(
        nodes, gpd.GeoSeries([box], crs="EPSG:4326"), columns=(("vu", "sigma_vu"),)
    )
    pd.testing.assert_frame_equal(out_iter, out_gs)


def test_missing_columns_are_skipped(grid_with_bowl):
    nodes, box, _, inside = grid_with_bowl
    out = reinterpolate_nodes(
        nodes, box, columns=(("vn", "sigma_vn"), ("vu", "sigma_vu"))
    )
    assert "vn" not in out.columns
    assert out["reinterpolated"].sum() == inside.sum()


def test_requires_shapely_with_hint(grid_with_bowl, monkeypatch):
    from geepers import _optional

    nodes, box, _, _ = grid_with_bowl
    monkeypatch.setattr(
        _optional.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    with pytest.raises(ImportError, match=r"geepers\[grid\]"):
        reinterpolate_nodes(nodes, box)

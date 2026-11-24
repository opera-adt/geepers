# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""CustomGridSource reads a local UNR-format grid and never downloads."""

from __future__ import annotations

import json

import pytest

from geepers.gps_sources import CustomGridSource, UnrGridSource

ROWS = "".join(
    f"{2020 + i / 365.25:.6f} {0.001 * i:.6f} {0.002 * i:.6f} 0.0 0.001 0.001 0.002 0\n"
    for i in range(10)
)


@pytest.fixture
def snapshot(tmp_path):
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    (tmp_path / "grid_latlon_lookup.txt").write_text(
        "000001 237.500000  36.000000\n"
        "043447 238.900000  37.600000\n"  # an id past UNR's last
    )
    for i in (1, 43447):
        (nodes / f"{i:06d}_IGS20_constant.tenv8").write_text(ROWS)
    (tmp_path / "snapshot.json").write_text(
        json.dumps({"reference_frame": "IGS20", "grid_types": ["constant"]})
    )
    return tmp_path


def test_nodes_come_from_the_local_lookup(snapshot):
    grid = CustomGridSource.from_snapshot(snapshot)
    nodes = grid.stations(bbox=(-123.0, 35.0, -120.0, 38.0))
    assert sorted(int(i) for i in nodes.id) == [1, 43447]
    row = nodes[nodes.id.astype(int) == 43447].iloc[0]
    assert row.lon == pytest.approx(-121.1)  # 0-360 longitudes wrapped like UNR's
    assert row.lat == pytest.approx(37.6)


def test_series_read_from_local_files(snapshot):
    grid = CustomGridSource.from_snapshot(snapshot)
    series = grid.node_series(43447)
    assert len(series) == 10
    assert series.east.iloc[-1] == pytest.approx(0.009)
    ts = grid.timeseries("043447")  # meters, mean removed, as UnrGridSource
    assert ts.north.iloc[-1] == pytest.approx(0.009e-3)


def test_missing_node_is_an_error_not_a_download(snapshot, monkeypatch):
    def no_network(*_, **__):
        msg = "CustomGridSource must not download"
        raise AssertionError(msg)

    monkeypatch.setattr("geepers.gps_sources.unr_grid.download_file", no_network)
    grid = CustomGridSource.from_snapshot(snapshot)
    with pytest.raises(FileNotFoundError, match="000002"):
        grid.timeseries("000002")
    with pytest.raises(NotImplementedError):
        grid.download_data_files()


def test_unr_grid_source_is_unchanged():
    assert "lookup_file" not in UnrGridSource.__init__.__code__.co_varnames

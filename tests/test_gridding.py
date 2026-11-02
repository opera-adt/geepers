# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Rebuilding UNR-style grids from station positions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from geepers import gridding
from geepers.gps_imaging import make_ssf

DAYS = pd.date_range("2012-01-01", periods=400, freq="D")
SSF = np.c_[[0.0, 1.0, 10.0, 180.0], [1.0, 0.7, 0.2, 0.0]]
FAULT = [np.array([(0.0, -10.0), (0.0, 10.0)])]


def network(n: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": [f"S{i:03d}" for i in range(n)],
            "lon": rng.uniform(-2, 2, n),
            "lat": rng.uniform(-2, 2, n),
        }
    )


def series_for(motion: float, gaps: slice | None = None) -> pd.DataFrame:
    """A shared signal, so a filled day has a right answer."""
    signal = 5.0 * np.sin(2 * np.pi * np.arange(len(DAYS)) / 365.25)
    df = pd.DataFrame(
        {
            "east": signal + motion,
            "north": 2 * signal,
            "up": -signal,
            "sigma_east": 1.0,
            "sigma_north": 1.0,
            "sigma_up": 3.0,
        },
        index=DAYS,
    )
    return df if gaps is None else df.drop(DAYS[gaps])


def test_years_since_counts_365_25_day_years():
    dates = pd.date_range("2012-01-01", periods=2, freq="D")
    assert gridding.years_since(dates, dates[0]).tolist() == [
        0.0,
        pytest.approx(1 / 365.25),
    ]


def test_atbd_ssf_is_a_valid_structure_function():
    ssf = gridding.atbd_ssf()
    assert (np.diff(ssf[:, 0]) > 0).all()
    assert (np.diff(ssf[:, 1]) <= 0).all()
    assert (ssf[0, 1], ssf[-1, 1]) == (1.0, 0.0)
    assert np.interp(98.3 / 111.195, ssf[:, 0], ssf[:, 1]) == pytest.approx(0.11 / 0.21)


def test_trajectory_design_has_one_column_per_step():
    assert gridding.trajectory_design(np.linspace(0, 5, 100), [1.0, 2.0]).shape == (
        100,
        8,
    )


class TestReadBreaklines:
    def test_reads_both_line_types_and_filters(self, tmp_path):
        path = tmp_path / "faults.geojson"
        features = [
            {
                "properties": {"name": "San Andreas"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0, 0, 5], [1, 1, 5]],
                },
            },
            {
                "properties": {"name": "Garlock"},
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [[[2, 2], [3, 3]], [[4, 4], [5, 5]]],
                },
            },
        ]
        path.write_text(json.dumps({"features": features}))
        assert len(gridding.read_breaklines(path)) == 3
        kept = gridding.read_breaklines(path, keep=lambda p: p["name"] == "San Andreas")
        assert len(kept) == 1
        assert kept[0].tolist() == [[0.0, 0.0], [1.0, 1.0]]


class TestStationParameters:
    start = pd.Timestamp("2012-01-01")

    def positions(self, days: int, every: int = 1) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        index = pd.date_range("2010-01-01", periods=days, freq="D")[::every]
        t = gridding.years_since(index, self.start)
        return pd.DataFrame(
            {c: 3.0 * t + rng.normal(0, 1, len(t)) for c in ("east", "north", "up")},
            index=index,
        )

    def test_short_record_is_rejected(self):
        assert gridding.station_parameters(self.positions(300), [], self.start) is None

    def test_rate_and_duration_since_the_start(self):
        row = gridding.station_parameters(self.positions(3000), [], self.start)
        assert row["ve"] == pytest.approx(3.0, abs=0.2)
        assert row["duration"] == pytest.approx(2999 / 365.25)
        assert row["duration_since_start"] == pytest.approx((2999 - 730) / 365.25)
        assert row["completeness"] == pytest.approx(100.0)
        assert row["misfit_u"] == pytest.approx(1.0, abs=0.2)
        assert row["variability_u"] < 1.5

    def test_completeness_counts_missing_days(self):
        row = gridding.station_parameters(self.positions(3000, every=2), [], self.start)
        assert row["completeness"] == pytest.approx(50.0, abs=0.1)

    def test_slow_parameters_skip_a_record_mostly_before_the_start(self):
        row = gridding.station_parameters(self.positions(1600), [], self.start)
        assert row is not None
        assert "misfit_e" not in row


class TestSelectTimeVariable:
    def row(self, **changes) -> dict:
        row = {
            "duration_since_start": 10.0,
            "completeness": 90.0,
            **{f"v{c}": 1.0 for c in "enu"},
            **{f"vmsf{c}": 1.0 for c in "enu"},
            **{f"misfit_{c}": 1.0 for c in "enu"},
            **{f"variability_{c}": 1.0 for c in "enu"},
        }
        return {**row, **changes}

    @pytest.mark.parametrize(
        ("change", "expected"),
        [
            ({}, True),
            ({"duration_since_start": 6.9}, False),
            ({"completeness": 79.0}, False),
            ({"ve": 7.0}, False),
            ({"vu": 7.0}, True),
            ({"vu": 17.0}, False),
            ({"misfit_n": 11.0}, False),
            ({"variability_u": 16.0}, False),
            ({"misfit_e": np.nan}, True),
            ({"variability_n": np.nan}, True),
        ],
    )
    def test_atbd_limits(self, change, expected):
        stats = pd.DataFrame([self.row(**change)])
        assert bool(gridding.select_time_variable(stats).iloc[0]) is expected

    @pytest.mark.parametrize(
        ("change", "expected"),
        [
            ({"duration_since_start": 4.0}, True),
            ({"duration_since_start": 3.9}, False),
            ({"completeness": 5.0}, True),
            ({"ve": 50.0}, True),
            ({"misfit_u": 500.0}, True),
            ({"variability_e": 50.0}, True),
        ],
    )
    def test_published_selection_keeps_only_the_duration_rule(self, change, expected):
        stats = pd.DataFrame([self.row(**change)])
        kept = gridding.select_time_variable(stats, gridding.PUBLISHED_SELECTION)
        assert bool(kept.iloc[0]) is expected

    def test_a_criterion_can_be_relaxed(self):
        stats = pd.DataFrame([self.row(completeness=10.0)])
        limits = gridding.SelectionCriteria(min_completeness=0.0)
        assert gridding.select_time_variable(stats, limits).iloc[0]


def test_decluster_keeps_the_best_of_a_tight_group():
    stats = pd.DataFrame(
        {"lon": [0.0, 0.00001, 1.0], "lat": [0.0, 0.0, 1.0], "su": [0.5, 0.2, 0.9]},
        index=[10, 11, 12],
    )
    assert gridding.decluster(stats).index.tolist() == [11, 12]


class TestBackgroundWeight:
    def weight_at_spacing(self, degrees: float) -> float:
        grid = np.arange(-3, 4) * degrees
        lon, lat = (a.ravel() for a in np.meshgrid(grid, grid))
        return float(gridding.background_weight(lon, lat, [0.1], [0.1])[0])

    def test_dense_network_gets_none(self):
        assert self.weight_at_spacing(0.3) == 0.0

    def test_sparse_network_gets_all(self):
        assert self.weight_at_spacing(8.0) == 1.0

    def test_between_is_partial(self):
        assert 0.0 < self.weight_at_spacing(2.0) < 1.0


class TestRateGrid:
    def stats(self) -> pd.DataFrame:
        stats = network(40)
        for c in "enu":
            stats[f"v{c}"] = 3.0
            stats[f"vmsf{c}"] = 3.0
            stats[f"s{c}"] = 0.5
        return stats

    def test_uniform_field_comes_back_with_eq_11_uncertainty(self):
        out = gridding.rate_grid(self.stats(), [0.1], [0.2], SSF)
        assert out["vu"].iloc[0] == 3.0
        n = out["nu"].iloc[0]
        assert n >= 3
        assert out["su"].iloc[0] == pytest.approx(0.5 / np.sqrt(n))

    def test_background_reaches_only_the_components_it_has(self):
        background = pd.DataFrame({"ve": [9.0]})
        out = gridding.rate_grid(
            self.stats(), [0.1], [0.2], SSF, background=background, weight=[1.0]
        )
        assert out["ve"].iloc[0] == 9.0
        assert out["vn"].iloc[0] == 3.0

    def test_breakline_keeps_the_far_plate_out(self):
        stats = self.stats()
        stats.loc[stats.lon < 0, ["ve", "vmsfe"]] = -40.0
        west = gridding.rate_grid(stats, [-0.1], [0.0], SSF, breaklines=FAULT)
        east = gridding.rate_grid(stats, [0.1], [0.0], SSF, breaklines=FAULT)
        assert west["ve"].iloc[0] == -40.0
        assert east["ve"].iloc[0] == 3.0


def test_filter_station_rates_removes_an_outlier():
    stats = network(30)
    for c in "enu":
        stats[f"v{c}"] = 1.0
        stats[f"s{c}"] = 0.5
    stats.loc[3, "vu"] = 50.0
    ssf = make_ssf(stats.lon, stats.lat, stats.vu, stats.su)
    out = gridding.filter_station_rates(stats, ssf)
    assert out.loc[3, "vmsfu"] == 1.0
    assert out.index.equals(stats.index)


def test_background_rates_returns_a_row_per_point():
    stats = network(10)
    stats["ve"], stats["vn"], stats["se"], stats["sn"] = 1.0, 2.0, 0.5, 0.5
    out = gridding.background_rates(stats, [0.0, 80.0], [0.0, 0.0])
    assert out["ve"].iloc[0] == pytest.approx(1.0, abs=0.01)
    assert out["ve"].isna().tolist() == [False, True]

# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Rebuilding UNR-style grids from station positions."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from geepers import gridding
from geepers.gps_imaging import make_ssf

DAYS = pd.date_range("2012-01-01", periods=400, freq="D")
SSF = np.c_[[0.0, 1.0, 10.0, 180.0], [1.0, 0.7, 0.2, 0.0]]
FAULT = [np.array([(0.0, -10.0), (0.0, 10.0)])]
# Crosses nothing, but takes the path that recomputes every neighborhood
FAR_AWAY = [np.array([(100.0, 60.0), (101.0, 61.0)])]


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


def filled_network(
    stations: pd.DataFrame, gappy: dict[str, slice]
) -> gridding.PositionCube:
    series = {s: series_for(0.0, gappy.get(s)) for s in stations.id}
    return gridding.build_position_cube(series, stations, DAYS, {}, {})


def noisy_network(
    rng: np.random.Generator, gappy: dict[str, slice]
) -> gridding.PositionCube:
    """Stations that disagree, so the median has to choose between them."""
    stations = network(14, seed=2)
    series = {}
    for site in stations.id:
        df = series_for(rng.normal(0, 3))
        for c in ("east", "north", "up"):
            df[c] += rng.normal(0, 1.5, len(df)) + rng.normal(0, 0.02) * np.arange(
                len(df)
            )
        series[site] = df if site not in gappy else df.drop(DAYS[gappy[site]])
    return gridding.build_position_cube(series, stations, DAYS, {}, {})


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
        kwargs = {"breaklines": FAULT, "breakline_weight": 0.0}
        west = gridding.rate_grid(stats, [-0.1], [0.0], SSF, **kwargs)
        east = gridding.rate_grid(stats, [0.1], [0.0], SSF, **kwargs)
        assert west["ve"].iloc[0] == -40.0
        assert east["ve"].iloc[0] == 3.0

    def test_breakline_weight_reaches_the_rates(self):
        stats = self.stats()
        stats.loc[stats.lon < 0, ["ve", "vmsfe"]] = -40.0
        at = ([0.1], [0.0])
        ignored = gridding.rate_grid(
            stats, *at, SSF, breaklines=FAULT, breakline_weight=1.0
        )
        plain = gridding.rate_grid(stats, *at, SSF)
        assert ignored["ve"].iloc[0] == plain["ve"].iloc[0]
        small = gridding.rate_grid(
            stats, *at, SSF, breaklines=FAULT, breakline_weight=1e-6
        )
        assert small["ve"].iloc[0] == 3.0


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


class TestBuildPositionCube:
    def test_gaps_are_nan_and_observed_is_recorded(self):
        cube = filled_network(network(4), {"S000": slice(10, 20)})
        assert np.isnan(cube.x[0, 10:20]).all()
        assert not cube.observed[0, 10:20].any()
        assert cube.observed[1].all()

    def test_equipment_step_is_removed_and_earthquake_kept(self):
        stations = network(1)
        df = series_for(0.0)
        df.loc[DAYS[200] :, "east"] += 30.0
        df.loc[DAYS[300] :, "north"] += 30.0
        cube = gridding.build_position_cube(
            {"S000": df}, stations, DAYS, {"S000": [DAYS[200]]}, {"S000": [DAYS[300]]}
        )
        assert abs(cube.x[0, 205, 0] - cube.x[0, 195, 0]) < 3.0
        assert cube.x[0, 305, 1] - cube.x[0, 295, 1] > 20.0
        assert cube.earthquakes[0].tolist() == [300]

    def test_outlier_and_noisy_days_become_gaps(self):
        stations = network(1)
        df = series_for(0.0)
        df.loc[DAYS[50], "up"] += 100.0
        df.loc[DAYS[60], "sigma_up"] = 50.0
        cube = gridding.build_position_cube({"S000": df}, stations, DAYS, {}, {})
        assert not cube.observed[0, 50]
        assert not cube.observed[0, 60]
        assert cube.observed[0, 70]


class TestFillGaps:
    def test_two_sided_gap_recovers_the_shared_signal(self):
        cube = filled_network(network(12), {"S000": slice(100, 140)})
        counts = gridding.fill_gaps(cube, SSF)
        assert counts["spatial_pass_1"] == 40
        assert counts["model"] == 0
        truth = series_for(0.0)["east"].to_numpy()[100:140]
        assert cube.x[0, 100:140, 0] == pytest.approx(truth, abs=1e-6)

    def test_gap_before_the_first_observation_is_filled_from_after(self):
        cube = filled_network(network(12), {"S000": slice(0, 60)})
        gridding.fill_gaps(cube, SSF)
        truth = series_for(0.0)["east"].to_numpy()[:60]
        assert cube.x[0, :60, 0] == pytest.approx(truth, abs=1e-6)

    def test_no_neighbours_falls_back_to_the_model(self):
        cube = filled_network(network(1), {"S000": slice(100, 110)})
        counts = gridding.fill_gaps(cube, SSF)
        assert counts["model"] == 10
        assert np.isfinite(cube.x).all()

    def test_earthquake_blocks_a_reference_across_it(self):
        stations = network(12)
        series = {s: series_for(0.0) for s in stations.id}
        series["S000"] = series["S000"].drop(DAYS[100:140])
        series["S000"].loc[DAYS[140] :, "east"] += 30.0
        cube = gridding.build_position_cube(
            series, stations, DAYS, {}, {"S000": [DAYS[120]]}
        )
        gridding.fill_gaps(cube, SSF)
        # Days after the earthquake tie to later data, which carries the step
        assert cube.x[0, 130, 0] - series_for(0.0)["east"].iloc[130] == pytest.approx(
            30.0, abs=1e-6
        )
        assert cube.x[0, 110, 0] == pytest.approx(
            series_for(0.0)["east"].iloc[110], abs=1e-6
        )

    def test_parallel_matches_serial(self):
        gappy = {"S000": slice(100, 110), "S001": slice(5, 9)}
        serial = filled_network(network(12), gappy)
        parallel = filled_network(network(12), gappy)
        gridding.fill_gaps(serial, SSF)
        gridding.fill_gaps(parallel, SSF, n_workers=2)
        assert np.array_equal(serial.x, parallel.x)

    def test_shared_neighborhoods_match_the_general_path(self):
        rng = np.random.default_rng(3)
        gappy = {"S000": slice(100, 140), "S003": slice(0, 30), "S007": slice(250, 300)}
        fast = noisy_network(rng, gappy)
        general = deepcopy(fast)
        gridding.fill_gaps(fast, SSF)
        gridding.fill_gaps(general, SSF, breaklines=FAR_AWAY)
        assert fast.x == pytest.approx(general.x, abs=1e-9)
        assert fast.s == pytest.approx(general.s, abs=1e-9)

    def test_breakline_weight_of_one_matches_no_breakline(self):
        rng = np.random.default_rng(6)
        gappy = {"S000": slice(100, 140), "S003": slice(0, 30)}
        plain = noisy_network(rng, gappy)
        soft = deepcopy(plain)
        gridding.fill_gaps(plain, SSF)
        gridding.fill_gaps(soft, SSF, breaklines=FAULT, breakline_weight=1.0)
        assert soft.x == pytest.approx(plain.x, abs=1e-9)

    def test_coincident_stations_still_fill(self):
        stations = network(12)
        stations.loc[1, ["lon", "lat"]] = stations.loc[2, ["lon", "lat"]].to_numpy()
        series = {s: series_for(0.0) for s in stations.id}
        series["S000"] = series["S000"].drop(DAYS[100:110])
        cube = gridding.build_position_cube(series, stations, DAYS, {}, {})
        gridding.fill_gaps(cube, SSF)
        assert np.isfinite(cube.x).all()

    @pytest.mark.parametrize(("weight_by_sigma", "shift"), [(False, 0.0), (True, 50.0)])
    def test_a_confident_outlier_only_wins_when_weighting_by_sigma(
        self, weight_by_sigma, shift
    ):
        cube = filled_network(network(12), {"S000": slice(105, 106)})
        km = np.hypot(cube.lon - cube.lon[0], cube.lat - cube.lat[0])
        km[0] = np.inf
        nearest = int(np.argmin(km))
        cube.x[nearest, 105] += 50.0
        # The fill weights by the sigma of a difference, so the references count too
        cube.s[nearest, 104:107] = 1e-6
        gridding.fill_gaps(cube, SSF, weight_by_sigma=weight_by_sigma)
        truth = series_for(0.0)["east"].iloc[105]
        assert cube.x[0, 105, 0] == pytest.approx(truth + shift, abs=1e-6)


class TestImagePositions:
    def test_returns_displacement_since_the_first_day(self):
        cube = filled_network(network(12), {})
        out = gridding.image_positions(cube, [0.1, -0.1], [0.0, 0.2], SSF)
        assert out.shape == (len(DAYS), 2, 3)
        assert (out[0] == 0).all()
        truth = series_for(0.0)["east"].to_numpy()
        assert out[:, 0, 0] == pytest.approx(truth - truth[0], abs=1e-6)

    def test_shared_neighborhoods_match_the_general_path(self):
        cube = noisy_network(np.random.default_rng(4), {})
        # Includes a point exactly on a station, and one beyond the network
        lon = [0.1, -0.1, cube.lon[2], 5.0]
        lat = [0.0, 0.2, cube.lat[2], 5.0]
        fast = gridding.image_positions(cube, lon, lat, SSF)
        general = gridding.image_positions(cube, lon, lat, SSF, breaklines=FAR_AWAY)
        assert fast == pytest.approx(general, abs=1e-12)

    def test_parallel_matches_serial(self):
        cube = noisy_network(np.random.default_rng(5), {})
        lon, lat = [0.1, -0.1, 0.05], [0.0, 0.2, -0.1]
        serial = gridding.image_positions(cube, lon, lat, SSF)
        parallel = gridding.image_positions(cube, lon, lat, SSF, n_workers=2)
        assert np.array_equal(serial, parallel)

    def test_gaps_are_refused(self):
        cube = filled_network(network(4), {"S000": slice(1, 3)})
        with pytest.raises(ValueError, match="run fill_gaps first"):
            gridding.image_positions(cube, [0.0], [0.0], SSF)

    def test_breakline_keeps_each_side_to_itself(self):
        stations = network(30)
        series = {s: series_for(0.0) for s in stations.id}
        ramp = np.arange(len(DAYS)) * 0.1
        for row in stations.itertuples():
            if row.lon < 0:
                series[row.id]["east"] = ramp
            else:
                series[row.id]["east"] = -ramp
        cube = gridding.build_position_cube(series, stations, DAYS, {}, {})
        out = gridding.image_positions(
            cube, [-0.05, 0.05], [0.0, 0.0], SSF, breaklines=FAULT, breakline_weight=0.0
        )
        assert out[-1, 0, 0] == pytest.approx(ramp[-1])
        assert out[-1, 1, 0] == pytest.approx(-ramp[-1])

    def test_breakline_weight_reaches_the_imaging(self):
        cube = noisy_network(np.random.default_rng(7), {})
        lon, lat = [0.1, -0.1], [0.0, 0.2]
        plain = gridding.image_positions(cube, lon, lat, SSF)
        ignored = gridding.image_positions(
            cube, lon, lat, SSF, breaklines=FAULT, breakline_weight=1.0
        )
        assert ignored == pytest.approx(plain, abs=1e-12)

    @pytest.mark.parametrize(
        ("weight_by_sigma", "expected"), [(False, 0.0), (True, 100.0)]
    )
    def test_a_confident_outlier_only_wins_when_weighting_by_sigma(
        self, weight_by_sigma, expected
    ):
        cube = filled_network(network(12), {})
        cube.x[:, -1, 0] = cube.x[:, 0, 0]
        nearest = int(np.argmin(np.hypot(cube.lon - 0.1, cube.lat)))
        cube.x[nearest, -1, 0] += 100.0
        cube.s[nearest, -1, 0] = 1e-6
        out = gridding.image_positions(
            cube, [0.1], [0.0], SSF, weight_by_sigma=weight_by_sigma
        )
        assert out[-1, 0, 0] == pytest.approx(expected)

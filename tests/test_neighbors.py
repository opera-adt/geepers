"""Tests for the neighbor comparison (geepers.neighbors)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geepers.neighbors import _on_grid, _station_shares, check_neighbors
from geepers.smooth import fit_smooth_trend
from geepers.synthetic import power_law_noise

N = 156  # thirteen years of 30-day medians
DAYS = 30.0
ROOT_YEAR = np.sqrt(DAYS / 365.25)
DATES = pd.Timestamp("2010-01-01") + pd.to_timedelta(np.arange(N) * DAYS, unit="D")
T = np.arange(N) * DAYS / 365.25
DECAY = -60.0 * (1.0 - np.exp(-np.maximum(0.0, T - 5.0) / 1.5))


def _noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 500)
    return power_law_noise(N, -1.0, 0.7, seed=seed) + rng.normal(0, 1.0, N)


def _station(seed: int, signal: np.ndarray | None = None, rw: float = 0.0) -> pd.Series:
    values = -2.0 * T + _noise(seed)
    if signal is not None:
        values = values + signal
    if rw:
        rng = np.random.default_rng(seed + 900)
        values = values + rw * ROOT_YEAR * np.cumsum(rng.normal(0, 1.0, N))
    return pd.Series(values, index=DATES)


def _coords(names: list[str]) -> pd.DataFrame:
    """Stations about 5 km apart, every pair within 25 km."""
    return pd.DataFrame(
        {
            "lon": [-117.0 + 0.05 * k for k in range(len(names))],
            "lat": [34.0 + 0.01 * (k % 2) for k in range(len(names))],
        },
        index=names,
    )


@pytest.fixture(scope="module")
def shared():
    series = {f"S{k}": _station(k, DECAY) for k in range(4)}
    return check_neighbors(series, _coords(list(series)), units="mm")


@pytest.fixture(scope="module")
def one_local():
    series = {f"S{k}": _station(k) for k in range(4)}
    series["S0"] = _station(0, DECAY)
    return check_neighbors(series, _coords(list(series)), units="mm")


class TestSignal:
    def test_a_signal_all_neighbors_see_is_shared(self, shared):
        assert (shared.stations["shape"] == "smooth").all()
        assert (shared.stations["signal"] == "shared").all()

    def test_a_signal_only_one_station_sees_is_local(self, one_local):
        assert one_local.stations.loc["S0", "signal"] == "local"
        assert one_local.stations.loc["S0", "pairs_smooth"] == 3

    def test_quiet_stations_have_no_signal(self, one_local):
        others = one_local.stations.drop("S0")
        assert (others["signal"] == "none").all()

    def test_no_neighbor_leaves_a_signal_unresolved(self):
        series = {"S0": _station(0, DECAY), "FAR": _station(1)}
        coords = pd.DataFrame(
            {"lon": [-117.0, 10.0], "lat": [34.0, 50.0]}, index=["S0", "FAR"]
        )
        check = check_neighbors(series, coords, units="mm")
        assert len(check.pairs) == 0
        assert check.stations.loc["S0", "signal"] == "unresolved"
        assert check.stations.loc["S0", "n_pairs"] == 0

    def test_meters_and_millimeters_agree(self, one_local):
        series = {f"S{k}": _station(k) / 1000.0 for k in range(4)}
        series["S0"] = _station(0, DECAY) / 1000.0
        check = check_neighbors(series, _coords(list(series)), units="m")
        assert list(check.stations["signal"]) == list(one_local.stations["signal"])
        np.testing.assert_allclose(
            check.pairs["random_walk_mm"], one_local.pairs["random_walk_mm"], atol=0.05
        )


class TestRandomWalk:
    def test_the_noisy_station_is_identified(self):
        series = {f"S{k}": _station(k) for k in range(5)}
        series["S2"] = _station(2, rw=10.0)
        check = check_neighbors(series, _coords(list(series)), units="mm")
        table = check.stations
        assert table["local_random_walk_mm"].idxmax() == "S2"
        assert table.loc["S2", "noisy"]
        assert not table.drop("S2")["noisy"].any()
        assert table.loc["S2", "local_random_walk_mm"] == pytest.approx(10.0, rel=0.5)

    def test_shared_random_walk_is_not_blamed_on_the_stations(self):
        rng = np.random.default_rng(77)
        common = 10.0 * ROOT_YEAR * np.cumsum(rng.normal(0, 1.0, N))
        series = {f"S{k}": _station(k) + common for k in range(4)}
        check = check_neighbors(series, _coords(list(series)), units="mm")
        assert check.stations["random_walk_mm"].median() > 3.0
        assert not check.stations["noisy"].any()

    def test_two_stations_cannot_split_their_noise(self):
        series = {"A": _station(0), "B": _station(1)}
        check = check_neighbors(series, _coords(["A", "B"]), units="mm")
        assert check.stations["local_random_walk_mm"].isna().all()
        assert not check.stations["noisy"].any()

    def test_a_chain_cannot_split_its_noise(self):
        series = {name: _station(k) for k, name in enumerate("ABC")}
        coords = pd.DataFrame(
            {"lon": [-117.0, -116.85, -116.7], "lat": [34.0, 34.0, 34.0]},
            index=list("ABC"),
        )
        check = check_neighbors(series, coords, units="mm", max_distance_km=20.0)
        assert len(check.pairs) == 2
        assert check.stations["local_random_walk_mm"].isna().all()


class TestShares:
    @staticmethod
    def _pairs(rows):
        return pd.DataFrame(rows, columns=["a", "b", "random_walk_mm"])

    def test_a_consistent_triangle_is_solved_exactly(self):
        pairs = self._pairs(
            [
                ("A", "B", np.sqrt(5.0)),
                ("A", "C", np.sqrt(10.0)),
                ("B", "C", np.sqrt(13.0)),
            ]
        )
        shares = _station_shares(list("ABC"), pairs, dict.fromkeys("ABC", 100.0))
        assert shares["A"] == pytest.approx(1.0, abs=1e-3)
        assert shares["B"] == pytest.approx(2.0, abs=1e-3)
        assert shares["C"] == pytest.approx(3.0, abs=1e-3)

    def test_a_station_holds_no_more_than_its_own_record(self):
        """Regression: pairs not adding up put 165 on a station with 6 of its own."""
        pairs = self._pairs([("A", "B", 200.0), ("A", "C", 3.0), ("B", "C", 190.0)])
        shares = _station_shares(list("ABC"), pairs, {"A": 3.0, "B": 195.0, "C": 2.0})
        assert shares["A"] <= 3.0 + 1e-6
        assert shares["C"] <= 2.0 + 1e-6
        assert shares["B"] > 100.0

    def test_a_record_without_random_walk_holds_none(self):
        pairs = self._pairs([("A", "B", 5.0), ("A", "C", 5.0), ("B", "C", 7.0)])
        shares = _station_shares(list("ABC"), pairs, {"A": 0.0, "B": 7.0, "C": 7.0})
        assert shares["A"] == pytest.approx(0.0, abs=1e-3)


class TestInputs:
    def test_missing_coordinates_raise(self):
        with pytest.raises(ValueError, match="No coordinates for stations.*'B'"):
            check_neighbors(
                {"A": _station(0), "B": _station(1)}, _coords(["A"]), units="mm"
            )

    def test_unknown_units_raise(self):
        with pytest.raises(ValueError, match="units must be 'm' or 'mm'"):
            check_neighbors({"A": _station(0)}, _coords(["A"]), units="cm")

    def test_a_short_overlap_drops_the_pair(self):
        series = {"A": _station(0), "B": _station(1).iloc[:30]}
        check = check_neighbors(series, _coords(["A", "B"]), units="mm")
        assert len(check.pairs) == 0

    def test_max_neighbors_limits_the_pairs(self):
        series = {f"S{k}": _station(k) for k in range(5)}
        coords = _coords(list(series))
        check = check_neighbors(
            series, coords, units="mm", max_neighbors=1, detect_changes=False
        )
        assert len(check.pairs) < 10

    def test_known_steps_reach_the_differences(self):
        jump = np.where(pd.Timestamp("2015-01-01") <= DATES, 40.0, 0.0)
        series = {"A": _station(0) + jump, "B": _station(1), "C": _station(2)}
        check = check_neighbors(
            series,
            _coords(list(series)),
            units="mm",
            detect_changes=False,
            step_dates={"A": ["2015-01-01"]},
        )
        assert check.stations.loc["A", "shape"] == "linear"
        assert check.pairs["random_walk_mm"].max() < 3.0

    def test_stations_on_different_grids_are_aligned(self):
        """Daily records starting on different days must still share epochs."""
        days = pd.date_range("2010-01-01", periods=N * 30, freq="D")
        rng = np.random.default_rng(1)
        a = pd.Series(rng.normal(0, 1.0, len(days)), index=days)
        b = pd.Series(rng.normal(0, 1.0, len(days) - 11), index=days[11:])
        check = check_neighbors(
            {"A": a, "B": b}, _coords(["A", "B"]), units="mm", detect_changes=False
        )
        assert check.pairs.loc[0, "n_epochs"] >= N - 2

    @pytest.mark.parametrize("offset_days", [2, 10, 20, 28])
    def test_a_known_step_inside_a_bin_is_one_step(self, offset_days):
        """Regression: bins dated at their start put the step an epoch late."""
        days = pd.date_range("2012-01-01", periods=30 * 120, freq="D")
        rng = np.random.default_rng(3)
        when = (
            _on_grid(pd.Series(0.0, index=days), DAYS).index[60]
            - pd.Timedelta(days=15)
            + pd.Timedelta(days=offset_days)
        )
        values = rng.normal(0, 1.0, len(days)) + 40.0 * (days >= when)
        grid = _on_grid(pd.Series(values, index=days), DAYS)
        fit = fit_smooth_trend(
            grid.index,
            grid.to_numpy(),
            units="mm",
            sampling_days=DAYS,
            step_dates=[when],
        )
        assert fit.step_dates[0] == when
        assert fit.parameters["step_0"][0] == pytest.approx(40.0, abs=1.5)

    def test_a_pair_that_cannot_be_fitted_does_not_stop_the_others(self):
        """Regression: co-located stations can have identical pixels, so the
        difference is exactly zero and its noise cannot be estimated."""
        series = {"A": _station(0), "B": _station(0), "C": _station(1)}
        check = check_neighbors(series, _coords(list(series)), units="mm")
        assert len(check.stations) == 3
        assert {"A", "B"} not in [
            set(p) for p in zip(check.pairs["a"], check.pairs["b"], strict=False)
        ]
        assert len(check.pairs) == 2

    def test_threads_give_the_same_answer(self):
        series = {f"S{k}": _station(k) for k in range(3)}
        coords = _coords(list(series))
        serial = check_neighbors(series, coords, units="mm", detect_changes=False)
        threaded = check_neighbors(
            series, coords, units="mm", detect_changes=False, max_workers=2
        )
        pd.testing.assert_frame_equal(serial.stations, threaded.stations)
        pd.testing.assert_frame_equal(serial.pairs, threaded.pairs)

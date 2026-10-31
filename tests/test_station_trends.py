"""Tests for the station trend command (geepers.station_trends)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests
import tyro

import geepers.station_trends as module
from geepers.station_trends import main
from geepers.synthetic import power_law_noise

DAYS = pd.date_range("2012-01-01", periods=365 * 12, freq="D")
T = np.arange(len(DAYS)) / 365.25
RATE = -3.0  # mm/yr


class FakeSource:
    """Only what a real source offers; series in meters, as the schema says."""

    def stations(self, bbox=None):
        table = pd.DataFrame(
            {
                "id": ["AAAA", "BBBB", "CCCC", "FAIL"],
                "lon": [-117.0, -116.95, -116.9, -116.85],
                "lat": [34.0, 34.0, 34.01, 34.0],
            }
        )
        if bbox is not None:
            table = table[table["lon"].between(bbox[0], bbox[2])]
        return table

    def steps(self, ids):  # noqa: ARG002
        return pd.DataFrame({"id": ["AAAA"], "date": [pd.Timestamp("2016-03-03")]})

    def timeseries(self, name, start_date=None, end_date=None):  # noqa: ARG002
        if name == "FAIL":
            msg = "404"
            raise requests.HTTPError(msg)
        seed = sum(map(ord, name))
        rng = np.random.default_rng(seed)
        up = RATE * T + power_law_noise(len(T), -1.0, 1.0, seed=seed)
        up = up + rng.normal(0, 3.0, len(T))
        if name == "AAAA":
            up = up + 40.0 * (pd.Timestamp("2016-03-03") <= DAYS)
        return pd.DataFrame(
            {"date": DAYS, "east": up / 2000.0, "north": up / 3000.0, "up": up / 1000.0}
        )


@pytest.fixture(autouse=True)
def fake_source(monkeypatch):
    monkeypatch.setattr(module, "UnrSource", FakeSource)


def _trends(path):
    return pd.read_csv(path / "trends.csv").set_index("station")


class TestMain:
    @pytest.mark.parametrize("method", ["smooth", "piecewise", "sparse"])
    def test_rates_come_out_in_mm_per_year(self, tmp_path, method):
        main(stations=["bbbb"], output_dir=tmp_path, method=method)
        table = _trends(tmp_path)
        assert table.loc["BBBB", "rate_mm_yr"] == pytest.approx(RATE, abs=1.0)
        assert table.loc["BBBB", "shape"] == "linear"

    def test_known_steps_are_used(self, tmp_path):
        main(stations=["AAAA"], output_dir=tmp_path)
        table = _trends(tmp_path)
        assert table.loc["AAAA", "n_steps"] == 1
        assert table.loc["AAAA", "rate_mm_yr"] == pytest.approx(RATE, abs=1.5)

    def test_a_failed_download_is_skipped(self, tmp_path):
        main(stations=["BBBB", "FAIL"], output_dir=tmp_path, method="piecewise")
        assert list(_trends(tmp_path).index) == ["BBBB"]

    def test_short_records_are_skipped(self, tmp_path):
        main(stations=["BBBB"], output_dir=tmp_path, min_years=50.0)
        assert _trends(tmp_path).empty

    def test_components_are_each_fitted(self, tmp_path):
        main(
            stations=["BBBB"],
            output_dir=tmp_path,
            components=["up", "east"],
            method="piecewise",
        )
        table = pd.read_csv(tmp_path / "trends.csv")
        assert sorted(table["component"]) == ["east", "up"]

    @pytest.mark.parametrize("method", ["smooth", "piecewise"])
    def test_forecast_is_written_in_mm(self, tmp_path, method):
        main(stations=["BBBB"], output_dir=tmp_path, method=method, horizon_years=2.0)
        table = pd.read_csv(tmp_path / "forecasts.csv")
        assert set(table["station"]) == {"BBBB"}
        assert len(table) == 24
        assert table["lead_years"].max() <= 2.0
        assert (table["lower"] < table["upper"]).all()
        # In mm: about 12 years at -3 mm/yr, not 0.03 m
        assert abs(table["mean"].iloc[-1]) > 10.0

    def test_no_forecast_file_without_a_horizon(self, tmp_path):
        main(stations=["BBBB"], output_dir=tmp_path, method="piecewise")
        assert not (tmp_path / "forecasts.csv").exists()

    def test_bbox_selects_stations(self, tmp_path):
        main(
            bbox=(-117.01, 33.0, -116.97, 35.0), output_dir=tmp_path, method="piecewise"
        )
        assert list(_trends(tmp_path).index) == ["AAAA"]

    def test_neighbors_are_written(self, tmp_path):
        main(
            stations=["AAAA", "BBBB", "CCCC"],
            output_dir=tmp_path,
            method="piecewise",
            neighbors=True,
        )
        checks = pd.read_csv(tmp_path / "neighbors.csv")
        pairs = pd.read_csv(tmp_path / "neighbor_pairs.csv")
        assert sorted(checks["id"]) == ["AAAA", "BBBB", "CCCC"]
        assert len(pairs) == 3

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "exactly one of stations and bbox"),
            (
                {"stations": ["AAAA"], "bbox": (0.0, 0.0, 1.0, 1.0)},
                "exactly one of stations and bbox",
            ),
            ({"stations": ["NOPE"]}, "Unknown stations.*NOPE"),
        ],
    )
    def test_bad_selection_raises(self, tmp_path, kwargs, match):
        with pytest.raises(ValueError, match=match):
            main(output_dir=tmp_path, **kwargs)


class TestCli:
    def test_arguments_reach_main(self, tmp_path):
        tyro.cli(
            main,
            args=[
                "--stations",
                "BBBB",
                "--output-dir",
                str(tmp_path),
                "--method",
                "piecewise",
                "--components",
                "up",
            ],
        )
        assert _trends(tmp_path).loc["BBBB", "n_rate_changes"] == 0

    def test_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as raised:
            tyro.cli(main, args=["--help"])
        assert raised.value.code == 0

    def test_entry_point_exists(self):
        from geepers.cli import trends_cli

        assert callable(trends_cli)

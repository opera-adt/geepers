"""Tests for scripts/recreate-unr-grid.py, on a small synthetic network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "recreate-unr-grid.py"
spec = importlib.util.spec_from_file_location("recreate_unr_grid", SCRIPT)
assert spec is not None
assert spec.loader is not None
recreate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recreate
spec.loader.exec_module(recreate)

START, END = "2012-01-01", "2020-01-01"
DAYS = pd.date_range(START, pd.Timestamp(END) - pd.Timedelta(days=1), freq="D")
RATE = 10.0  # mm/yr, east, at every station


def signal(rng: np.random.Generator, noise: float = 1.0) -> dict[str, np.ndarray]:
    years = np.arange(len(DAYS)) / 365.25
    east = RATE * years + 3 * np.sin(2 * np.pi * years)
    return {
        "east": east + rng.normal(0, noise, len(DAYS)),
        "north": rng.normal(0, noise, len(DAYS)),
        "up": rng.normal(0, noise, len(DAYS)),
    }


class FakeStations:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.table = pd.DataFrame(
            {
                "id": [f"S{i:03d}" for i in range(10)] + ["NONE"],
                "lon": np.r_[rng.uniform(-1, 1, 10), 0.3],
                "lat": np.r_[rng.uniform(-1, 1, 10), 0.3],
            }
        )

    def stations(self, **_):
        return self.table

    def steps(self, _ids=None):
        return pd.DataFrame({"id": ["S000"], "date": [DAYS[900]], "code": [1]})

    def timeseries(self, site, **_):
        if site == "NONE":
            raise FileNotFoundError(site)
        rng = np.random.default_rng(int(site[1:]))
        values = signal(rng)
        df = pd.DataFrame({"date": DAYS, **{c: v / 1000 for c, v in values.items()}})
        for c in ("east", "north", "up"):
            df[f"sigma_{c}"] = 0.002
        return df


class FakeGrid:
    def __init__(self, kind):
        self.kind = kind

    def stations(self, **_):
        return pd.DataFrame({"id": [1, 2], "lon": [0.1, -0.2], "lat": [0.1, 0.2]})

    def node_series(self, _grid_id):
        years = np.arange(len(DAYS)) / 365.25
        seasonal = 0 if self.kind == "constant" else 3 * np.sin(2 * np.pi * years)
        df = pd.DataFrame(
            {
                "decimal_year": 2012 + years,
                "east": RATE * years + seasonal,
                "north": 0.0,
                "up": 0.0,
            },
            index=DAYS,
        )
        for c in ("east", "north", "up"):
            df[f"sigma_{c}"] = 1.0
        return df


def arguments(tmp_path, *extra):
    return recreate.parse_args(
        [
            "--bbox",
            "-1",
            "-1",
            "1",
            "1",
            "-o",
            str(tmp_path),
            "--workers",
            "1",
            "--start",
            START,
            "--end",
            END,
            *extra,
        ]
    )


def test_bbox_is_required(capsys):
    with pytest.raises(SystemExit):
        recreate.parse_args([])
    assert "--bbox" in capsys.readouterr().err


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as stop:
        recreate.parse_args(["--help"])
    assert stop.value.code == 0


def test_rebuild_writes_and_recovers_a_shared_signal(tmp_path):
    written = recreate.run(arguments(tmp_path), FakeStations(), FakeGrid)
    assert set(written) == {
        "stations",
        "rates",
        "daily",
        "scores_rates",
        "scores_daily",
    }
    assert all(path.exists() for path in written.values())

    table = pd.read_csv(written["stations"])
    assert "NONE" not in set(table.id)
    rates = pd.read_csv(written["rates"])
    assert rates.id.tolist() == [1, 2]
    assert rates.ve.to_numpy() == pytest.approx(RATE, abs=0.3)

    saved = np.load(written["daily"])
    assert saved["displacement_mm"].shape == (len(DAYS), 2, 3)
    assert saved["component"].tolist() == ["east", "north", "up"]
    assert (saved["displacement_mm"][0] == 0).all()

    scores = pd.read_csv(written["scores_rates"]).set_index("component")
    assert scores.loc["east", "rms_mm_yr"] < 0.3
    daily = pd.read_csv(written["scores_daily"])
    assert daily[daily.component == "east"].rms_monthly_mm.max() < 2.0


def test_selection_defaults_to_the_published_one(tmp_path):
    assert arguments(tmp_path).selection == "as-published"
    assert arguments(tmp_path, "--selection", "atbd").selection == "atbd"


def test_no_compare_reads_no_published_files(tmp_path):
    class NoFiles(FakeGrid):
        def node_series(self, _grid_id):
            msg = "read a published file"
            raise AssertionError(msg)

    written = recreate.run(arguments(tmp_path, "--no-compare"), FakeStations(), NoFiles)
    assert set(written) == {"stations", "rates", "daily"}


def test_a_box_with_no_usable_station_stops(tmp_path):
    class Empty(FakeStations):
        def timeseries(self, site, **_):
            raise FileNotFoundError(site)

    with pytest.raises(SystemExit, match="no station"):
        recreate.run(arguments(tmp_path), Empty(), FakeGrid)

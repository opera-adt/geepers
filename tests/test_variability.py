"""Tests for spatial/temporal variability metrics (geepers.variability)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geepers.quality import gap_percentage
from geepers.variability import (
    delaunay_neighbors,
    spatial_structure_function,
    spatial_variability,
    ssf_per_station,
    temporal_velocity_variability,
)


@pytest.fixture
def network():
    rng = np.random.default_rng(4)
    n = 80
    lon = rng.uniform(140, 150, n)
    lat = rng.uniform(-35, -25, n)
    return lon, lat


class TestSSF:
    def test_coherent_field_scores_high(self, network):
        lon, lat = network
        # Perfectly smooth field: differences ~0 -> uniform inverse -> high
        values = 0.1 * lon + 0.05 * lat
        ssf = spatial_structure_function(lon, lat, values)
        assert ssf[0].tolist() == [0.0, 1.0]
        assert ssf[-1].tolist() == [180.0, 0.0]
        # Coherent field: nearby pairs differ less -> SSF decays with distance
        vals = ssf[1:-1, 1]
        finite = vals[np.isfinite(vals)]
        assert finite[0] >= finite[-1]

    def test_uncorrelated_field_stays_flat(self, network):
        # The normalized SSF median is ~1 for spatially uncorrelated
        # fields (differences independent of distance) and low for
        # smooth fields whose coherence decays with separation.
        lon, lat = network
        rng = np.random.default_rng(0)
        smooth = 0.1 * lon
        noisy = rng.normal(0, 5, len(lon))
        s_smooth = np.nanmedian(spatial_structure_function(lon, lat, smooth)[:, 1])
        s_noisy = np.nanmedian(spatial_structure_function(lon, lat, noisy)[:, 1])
        assert s_noisy > s_smooth
        assert s_noisy > 0.5
        assert s_smooth < 0.3

    def test_outlier_cut(self, network):
        lon, lat = network
        values = np.zeros(len(lon))
        values[0] = 1e6  # extreme outlier: pairs with it are excluded
        ssf = spatial_structure_function(lon, lat, values, max_difference=10)
        assert np.isfinite(ssf[:, 0]).all()

    def test_zero_differences_no_crash(self, network):
        lon, lat = network
        ssf = spatial_structure_function(lon, lat, np.zeros(len(lon)))
        assert not np.isinf(ssf[:, 1]).any()


class TestDelaunay:
    def test_neighbors_symmetric(self, network):
        lon, lat = network
        nbrs = delaunay_neighbors(lon, lat)
        assert len(nbrs) == len(lon)
        for i, ns in nbrs.items():
            for j in ns:
                assert i in nbrs[j]

    def test_ssf_per_station_shape(self, network):
        lon, lat = network
        rng = np.random.default_rng(1)
        df = ssf_per_station(
            lon, lat,
            {"east": rng.normal(size=len(lon)), "up": rng.normal(size=len(lon))},
        )
        assert len(df) == len(lon)
        assert {"ssf_east", "ssf_up", "ssf_n_neighbors"} <= set(df.columns)
        assert (df["ssf_n_neighbors"] >= 2).all()


class TestSpatialVariability:
    def test_smooth_vs_noisy(self, network):
        lon, lat = network
        rng = np.random.default_rng(2)
        smooth = spatial_variability(lon, lat, 0.01 * lon)
        noisy = spatial_variability(lon, lat, rng.normal(0, 5, len(lon)))
        assert smooth["rms"].median() < noisy["rms"].median()
        assert smooth["mad"].median() < noisy["mad"].median()

    def test_nan_neighbor_handled(self, network):
        lon, lat = network
        values = np.ones(len(lon))
        values[3] = np.nan
        out = spatial_variability(lon, lat, values)
        assert np.isfinite(out["rms"].drop(index=3)).all()


class TestTemporalVariability:
    def test_stable_station_low_variability(self):
        rng = np.random.default_rng(8)
        n = 3000  # ~8 years daily
        dates = pd.date_range("2015-01-01", periods=n, freq="D")
        t = np.arange(n) / 365.25
        enu = np.c_[
            0.005 * t + rng.normal(0, 0.001, n),
            -0.003 * t + rng.normal(0, 0.001, n),
            0.001 * t + rng.normal(0, 0.003, n),
        ]
        res = temporal_velocity_variability(dates, enu)
        assert res.full_velocity["east"] == pytest.approx(0.005, abs=0.001)
        # windowed velocities should scatter tightly around the full one
        assert res.variability["east"] < 0.002
        assert len(res.window_velocities) > 3

    def test_velocity_change_increases_variability(self):
        rng = np.random.default_rng(9)
        n = 3000
        dates = pd.date_range("2015-01-01", periods=n, freq="D")
        t = np.arange(n) / 365.25
        stable = 0.005 * t + rng.normal(0, 0.001, n)
        # rate change halfway through
        changing = np.where(t < t[n // 2], 0.005 * t, 0.02 * t - 0.015 * t[n // 2])
        changing = changing + rng.normal(0, 0.001, n)
        r_stable = temporal_velocity_variability(
            dates, np.c_[stable, stable, stable]
        )
        r_changing = temporal_velocity_variability(
            dates, np.c_[changing, changing, changing]
        )
        assert r_changing.variability["east"] > 2 * r_stable.variability["east"]

    def test_short_record_raises(self):
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        with pytest.raises(ValueError, match="too short"):
            temporal_velocity_variability(dates, np.zeros((300, 3)))


class TestGapPercentage:
    def test_no_gaps(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        assert gap_percentage(dates) == pytest.approx(0.0)

    def test_half_missing(self):
        dates = pd.date_range("2020-01-01", periods=101, freq="2D")
        # 101 present on a 201-slot daily grid
        assert gap_percentage(dates) == pytest.approx(100 * (1 - 101 / 201))

    def test_window_extension_counts_missing_tails(self):
        dates = pd.date_range("2020-06-01", periods=30, freq="D")
        pct = gap_percentage(dates, start="2020-01-01", end="2020-12-31")
        assert 90 < pct < 95

    def test_empty_window(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        assert np.isnan(gap_percentage(dates, start="2021-01-01", end="2021-02-01"))

"""Tests for GNSS-InSAR validation utilities (geepers.validation)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from geepers.validation import (
    comparison_stats,
    fit_velocities,
    misfit_semivariogram,
    pairwise_differential_rmse,
    plot_semivariogram,
    plot_velocity_scatter,
)


@pytest.fixture
def merged_two_years():
    """Two years of 12-day sampling: enough for MIDAS one-year pairs."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=61, freq="12D")
    t = np.arange(61) * 12 / 365.25
    merged, truth = {}, {}
    for k, v_true in enumerate([0.005, -0.002, 0.010, 0.0]):
        name = f"ST{k}"
        gps = v_true * t + rng.normal(0, 0.0005, len(t))
        insar = v_true * t + rng.normal(0, 0.001, len(t))
        merged[name] = pd.DataFrame({"los_gps": gps, "los_insar": insar}, index=dates)
        truth[name] = v_true
    return merged, truth


class TestFitVelocities:
    @pytest.mark.parametrize("method", ["midas", "lsq"])
    def test_recovers_both_velocities(self, merged_two_years, method):
        merged, truth = merged_two_years
        out = fit_velocities(merged, method=method)
        assert set(out.index) == set(merged)
        for sid, v_true in truth.items():
            assert out.loc[sid, "gps_velocity"] == pytest.approx(v_true, abs=0.002)
            assert out.loc[sid, "insar_velocity"] == pytest.approx(v_true, abs=0.003)
        assert (out["n_dates"] == 61).all()

    def test_short_series_gives_nan_midas(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"los_gps": np.zeros(10), "los_insar": np.zeros(10)}, index=dates
        )
        out = fit_velocities({"A": df}, method="midas")
        assert np.isnan(out.loc["A", "gps_velocity"])

    def test_same_estimator_both_series(self, merged_two_years):
        # identical inputs -> identical outputs, whatever the method
        merged, _ = merged_two_years
        df = merged["ST0"]
        both_same = pd.DataFrame(
            {"los_gps": df["los_gps"], "los_insar": df["los_gps"]},
            index=df.index,
        )
        out = fit_velocities({"A": both_same}, method="midas")
        assert out.loc["A", "gps_velocity"] == out.loc["A", "insar_velocity"]


class TestComparisonStats:
    def test_perfect_agreement(self):
        x = np.arange(10.0)
        s = comparison_stats(x, x)
        assert s["bias"] == 0
        assert s["rmse"] == 0
        assert s["mad"] == 0
        assert s["r2"] == pytest.approx(1.0)
        assert s["n"] == 10

    def test_constant_offset(self):
        x = np.arange(10.0)
        s = comparison_stats(x, x + 2.0)
        assert s["bias"] == pytest.approx(2.0)
        assert s["rmse"] == pytest.approx(2.0)
        assert s["mad"] == pytest.approx(0.0)  # offset, no scatter
        assert s["r2"] == pytest.approx(1.0)

    def test_nan_pairs_dropped(self):
        x = np.array([1.0, 2.0, np.nan, 4.0])
        y = np.array([1.0, np.nan, 3.0, 4.0])
        assert comparison_stats(x, y)["n"] == 2


class TestSemivariogram:
    @pytest.fixture
    def rmse_df(self, merged_two_years):
        merged, _ = merged_two_years
        coords = {s: (-117.0 + 0.5 * i, 35.0) for i, s in enumerate(merged)}
        return pairwise_differential_rmse(merged, coords)

    def test_gamma_is_half_rmse_squared(self, rmse_df):
        out = misfit_semivariogram(rmse_df, n_bins=3)
        assert len(out) >= 1
        # total gamma mass conserved: weighted mean of bins == overall mean
        overall = np.mean(0.5 * rmse_df["rmse"] ** 2)
        weighted = np.average(out["gamma_mean"], weights=out["n_pairs"])
        assert weighted == pytest.approx(overall)

    def test_log_bins(self, rmse_df):
        out = misfit_semivariogram(rmse_df, n_bins=4, log_bins=True)
        assert (np.diff(out["distance_km"]) > 0).all()
        assert out["n_pairs"].sum() == len(rmse_df)

    def test_plots_run(self, rmse_df, merged_two_years):
        merged, _ = merged_two_years
        binned = misfit_semivariogram(rmse_df, n_bins=3)
        for scale in ("loglog", "logx", "linear"):
            ax = plot_semivariogram(rmse_df, binned, scale=scale)
            assert ax is not None
        vel = fit_velocities(merged, method="lsq")
        ax = plot_velocity_scatter(vel, units="mm/yr", scale=1000)
        assert ax is not None

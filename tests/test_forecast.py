"""Tests for forecasting and backtesting (geepers.forecast)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geepers.changepoints import detect_changepoints, fit_sparse_changepoints
from geepers.forecast import backtest, forecast_trend
from geepers.synthetic import power_law_noise

MONTH = 365.25 / 12


def _monthly(rate_change: float = 0.0, flicker: float = 0.7, seed: int = 4, n=120):
    """Ten years of monthly data (mm): -3 mm/yr, flicker + white, annual."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 12
    values = (
        -3.0 * t
        + flicker * power_law_noise(n, -1.0, 1.0, seed=seed)
        + rng.normal(0, 1.0, n)
        + 3.0 * np.sin(2 * np.pi * t)
        + rate_change * np.maximum(0.0, t - 4.0)
    )
    dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(
        np.round(np.arange(n) * MONTH), unit="D"
    )
    return dates, values


@pytest.fixture(scope="module")
def linear():
    dates, values = _monthly()
    found = detect_changepoints(dates, values, sampling_days=MONTH)
    return dates, values, found


class TestForecastTrend:
    def test_extends_the_fitted_model(self, linear):
        dates, values, found = linear
        fc = forecast_trend(dates, values, 3, sampling_days=MONTH, changepoints=found)
        table = fc.table
        assert len(table) == 36
        assert table.index[0] > dates[-1]
        np.testing.assert_allclose(table["lead_years"].iloc[[0, -1]], [1 / 12, 3.0])
        # Twelve months apart the annual term repeats: the model moves by the rate
        assert table["model"].iloc[23] - table["model"].iloc[11] == pytest.approx(
            found.rate, rel=1e-6
        )
        assert (table["lower"] < table["mean"]).all()
        assert (table["mean"] < table["upper"]).all()
        assert np.all(np.diff(table["sigma_model"]) > 0)

    def test_white_noise_is_not_predicted(self):
        dates, values = _monthly(flicker=0.0)
        fc = forecast_trend(dates, values, 1, sampling_days=MONTH, noise_model="WN")
        table = fc.table
        np.testing.assert_allclose(table["mean"], table["model"])
        sigma_white = fc.changepoints.trend.sigma_white
        np.testing.assert_allclose(table["sigma_noise"], sigma_white, rtol=1e-6)

    def test_flicker_noise_persists(self, linear):
        dates, values, found = linear
        table = forecast_trend(
            dates, values, 3, sampling_days=MONTH, changepoints=found
        ).table
        assert not np.allclose(table["mean"], table["model"])
        assert table["sigma_noise"].iloc[-1] > table["sigma_noise"].iloc[0]

    def test_noise_free_forecast(self, linear):
        dates, values, found = linear
        table = forecast_trend(
            dates,
            values,
            2,
            sampling_days=MONTH,
            changepoints=found,
            include_noise=False,
        ).table
        np.testing.assert_allclose(table["mean"], table["model"])
        assert (table["sigma_noise"] == 0).all()

    def test_samples_match_the_gaussian_parts(self, linear):
        dates, values, found = linear
        fc = forecast_trend(
            dates,
            values,
            3,
            sampling_days=MONTH,
            changepoints=found,
            simulate_changes=(),
            n_simulations=4000,
        )
        expected = np.hypot(fc.table["sigma_model"], fc.table["sigma_noise"])
        np.testing.assert_allclose(fc.samples.std(axis=0), expected, rtol=0.08)

    def test_past_rate_changes_widen_the_interval(self):
        dates, values = _monthly(rate_change=-4.0)
        found = detect_changepoints(dates, values, sampling_days=MONTH)
        assert found.changepoints["rate_change"].notna().any()

        def width(**kwargs):
            fc = forecast_trend(
                dates, values, 5, sampling_days=MONTH, changepoints=found, **kwargs
            )
            return fc, (fc.table["upper"] - fc.table["lower"]).iloc[-1]

        with_changes, wide = width()
        without, narrow = width(simulate_changes=())
        assert with_changes.change_rates["rate"] > 0
        assert with_changes.change_rates["offset"] == 0  # not simulated by default
        assert without.change_rates == {"offset": 0.0, "rate": 0.0}
        assert wide > 1.2 * narrow

    def test_accepts_the_sparse_fit(self):
        dates, values = _monthly(rate_change=-4.0)
        found = fit_sparse_changepoints(dates, values, sampling_days=MONTH)
        fc = forecast_trend(dates, values, 2, sampling_days=MONTH, changepoints=found)
        assert fc.changepoints is found
        assert len(fc.table) == 24

    def test_future_dates(self, linear):
        dates, values, found = linear
        future = pd.DatetimeIndex(["2026-06-01", "2027-06-01"])
        fc = forecast_trend(
            dates, values, future_dates=future, sampling_days=MONTH, changepoints=found
        )
        assert list(fc.table.index) == list(future)

    def test_reproducible(self, linear):
        dates, values, found = linear
        runs = [
            forecast_trend(
                dates, values, 2, sampling_days=MONTH, changepoints=found, seed=3
            )
            for _ in range(2)
        ]
        np.testing.assert_array_equal(runs[0].samples, runs[1].samples)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "exactly one of horizon_years and future_dates"),
            (
                {"horizon_years": 1, "future_dates": ["2030-01-01"]},
                "exactly one of horizon_years and future_dates",
            ),
            ({"future_dates": ["2020-01-01"]}, "after the last observation"),
        ],
    )
    def test_bad_epochs_raise(self, linear, kwargs, match):
        dates, values, found = linear
        with pytest.raises(ValueError, match=match):
            forecast_trend(
                dates, values, sampling_days=MONTH, changepoints=found, **kwargs
            )


@pytest.fixture(scope="module")
def result():
    dates, values = _monthly(n=144)
    return backtest(dates, values, 3, sampling_days=MONTH, n_cutoffs=3)


class TestBacktest:
    def test_predictions(self, result):
        pred = result.predictions
        assert pred["cutoff"].nunique() == 3
        assert (pred["date"] > pred["cutoff"]).all()
        assert pred["lead_years"].between(0, 3 + 1e-9).all()
        np.testing.assert_allclose(pred["error"], pred["observed"] - pred["mean"])
        assert result.coverage == pytest.approx(pred["covered"].mean())

    def test_metrics(self, result):
        metrics = result.metrics
        assert metrics["n"].sum() == len(result.predictions)
        assert list(metrics.columns) == [
            "lead_years",
            "n",
            "rmse",
            "mae",
            "coverage",
            "width",
        ]
        assert (metrics["rmse"] >= metrics["mae"]).all()

    def test_intervals_are_calibrated_without_change_points(self, result):
        # The model is right here, so the 95 % intervals should cover about
        # 95 % of the later observations
        assert 0.85 <= result.coverage <= 1.0

    def test_too_short_raises(self):
        dates, values = _monthly(n=60)
        with pytest.raises(ValueError, match="do not fit in the record"):
            backtest(dates, values, 3, sampling_days=MONTH)


def _rows(table: pd.DataFrame, positions: list[int]) -> list[dict]:
    picked = table.iloc[positions]
    return [
        {"date": str(date.date()), **row}
        for date, row in zip(
            picked.index, picked.to_dict(orient="records"), strict=True
        )
    ]


class TestForecastRegression:
    """Golden-value regression for `forecast_trend` and `backtest`."""

    def test_noise_free_extrapolates_exactly(self):
        n = 96
        t = np.arange(n) / 12
        truth = 2.0 - 3.0 * t + 3.0 * np.sin(2 * np.pi * t)
        values = truth + np.random.default_rng(0).normal(0, 0.01, n)
        dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(
            np.round(np.arange(n) * MONTH), unit="D"
        )
        fc = forecast_trend(dates, values, 2, sampling_days=MONTH, noise_model="WN")
        t_future = np.arange(n, n + 24) / 12  # the next 24 monthly epochs
        expected = 2.0 - 3.0 * t_future + 3.0 * np.sin(2 * np.pi * t_future)
        np.testing.assert_allclose(fc.table["model"], expected, atol=0.02)
        assert fc.change_rates == {"offset": 0.0, "rate": 0.0}

    def test_forecast_golden(self, golden):
        dates, values = _monthly(rate_change=-4.0)
        fc = forecast_trend(dates, values, 3, sampling_days=MONTH, n_simulations=500)
        golden(
            "forecast.json",
            {
                "table": _rows(fc.table, [0, 11, 23, 35]),
                "change_rates": fc.change_rates,
                "change_scales": fc.change_scales,
            },
        )

    def test_backtest_golden(self, result, golden):
        metrics = result.metrics.astype({"n": int})
        golden(
            "backtest.json",
            {
                "metrics": metrics.to_dict(orient="records"),
                "coverage": result.coverage,
            },
        )

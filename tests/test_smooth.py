"""Tests for the smooth trend and its forecast (geepers.smooth)."""

from __future__ import annotations

import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geepers.changepoints import detect_changepoints
from geepers.forecast import backtest
from geepers.smooth import fit_smooth_trend as _fit_smooth_trend
from geepers.smooth import forecast_smooth
from geepers.synthetic import power_law_noise
from geepers.trend import estimate_trend

MONTH = 365.25 / 12
# Every series here is in millimeters
fit_smooth_trend = partial(_fit_smooth_trend, units="mm")


def _series(signal=None, *, n: int = 144, flicker: float = 0.7, seed: int = 4):
    """Twelve years of monthly data (mm): -3 mm/yr, flicker + white, annual."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 12
    truth = -3.0 * t + 3.0 * np.sin(2 * np.pi * t)
    if signal is not None:
        truth = truth + signal(t)
    values = (
        truth
        + flicker * power_law_noise(n, -1.0, 1.0, seed=seed)
        + rng.normal(0, 1.0, n)
    )
    dates = pd.Timestamp("2012-01-01") + pd.to_timedelta(
        np.round(np.arange(n) * MONTH), unit="D"
    )
    return dates, values, truth


def _decay(t):
    return -60.0 * (1.0 - np.exp(-np.maximum(0.0, t - 4.0) / 1.5))


def _cycle(t):
    return 6.0 * np.sin(2 * np.pi * t / 5.0)


@pytest.fixture(scope="module")
def linear():
    dates, values, _ = _series()
    return dates, values, fit_smooth_trend(dates, values, sampling_days=MONTH)


@pytest.fixture(scope="module")
def decay():
    dates, values, truth = _series(_decay)
    return dates, values, truth, fit_smooth_trend(dates, values, sampling_days=MONTH)


class TestFitSmoothTrend:
    def test_a_linear_series_stays_linear(self, linear):
        dates, values, fit = linear
        assert fit.shape == "linear"
        assert fit.sigma_slope_change == 0.0
        trend = estimate_trend(dates, values, sampling_days=MONTH, noise_model="FNWN")
        assert fit.rate == pytest.approx(trend.velocity, abs=trend.velocity_uncertainty)
        assert fit.rate_sigma == pytest.approx(trend.velocity_uncertainty, rel=0.3)
        # One rate throughout
        np.testing.assert_allclose(fit.rates, fit.rate)

    def test_a_decay_goes_smooth_and_beats_the_line(self, decay):
        dates, values, truth, fit = decay
        assert fit.shape == "smooth"
        assert fit.evidence >= 6.0
        assert fit.sigma_slope_change > 0.0
        line = estimate_trend(dates, values, sampling_days=MONTH, noise_model="FNWN")
        error = np.sqrt(np.mean((fit.model - truth) ** 2))
        assert error < 0.5 * np.sqrt(np.mean((line.model - truth) ** 2))

    def test_rate_follows_the_decay(self, decay):
        dates, _, _, fit = decay
        assert fit.rates.index.equals(dates)
        t = np.arange(len(dates)) / 12
        true_rate = -3.0 - 40.0 * np.exp(-np.maximum(0.0, t - 4.0) / 1.5) * (t > 4.0)
        # Rates are noisy but must find the fast phase and the return to -3
        assert fit.rates.iloc[56] < -8.0
        assert abs(fit.rates.iloc[-1] - true_rate[-1]) < 3 * fit.rates_sigma.iloc[-1]
        assert (fit.rates_sigma > 0).all()

    def test_evidence_gate_controls_the_shape(self):
        dates, values, _ = _series(_cycle)
        fit = fit_smooth_trend(dates, values, sampling_days=MONTH)
        assert fit.shape == "smooth"
        kept = fit_smooth_trend(
            dates, values, sampling_days=MONTH, min_evidence=fit.evidence + 1.0
        )
        assert kept.shape != "smooth"
        assert kept.sigma_slope_change == 0.0

    def test_random_walk_noise_is_not_taken_for_a_signal(self):
        # A straight line with a 5 mm/sqrt(yr) random walk: the flicker-only
        # comparison often reads it as a smooth signal, the limited one rarely
        month_years = MONTH / 365.25

        def shapes(limit):
            count = 0
            for seed in range(12):
                # Its own stream: the series' white noise uses `seed`
                rng = np.random.default_rng(seed + 1000)
                dates, values, _ = _series(seed=seed)
                values = values + 5.0 * np.sqrt(month_years) * np.cumsum(
                    rng.normal(0, 1.0, len(values))
                )
                fit = fit_smooth_trend(
                    dates, values, sampling_days=MONTH, max_random_walk_mm=limit
                )
                count += fit.shape == "smooth"
            return count

        free, limited = shapes(None), shapes(3.0)
        assert free >= 3
        assert limited < free

    def test_large_signals_pass_the_random_walk_limit(self, decay):
        *_, fit = decay
        assert fit.shape == "smooth"
        dates, values, _ = _series(lambda t: -80.0 / (1 + np.exp(-(t - 6.0) / 0.7)))
        sigmoid = fit_smooth_trend(dates, values, sampling_days=MONTH)
        assert sigmoid.shape == "smooth"

    def test_meters_and_millimeters_agree(self):
        """Regression: a 3 m limit on a series in meters had no effect."""
        dates, values, _ = _series(_decay)
        mm = fit_smooth_trend(dates, values, sampling_days=MONTH)
        m = _fit_smooth_trend(dates, values / 1000.0, units="m", sampling_days=MONTH)
        assert m.shape == mm.shape
        assert m.evidence == pytest.approx(mm.evidence, abs=0.5)
        assert 1000.0 * m.rate == pytest.approx(mm.rate, rel=1e-3)

    def test_units_are_required(self):
        dates, values, _ = _series()
        with pytest.raises(TypeError, match="units"):
            _fit_smooth_trend(dates, values, sampling_days=MONTH)

    def test_unknown_units_raise(self):
        dates, values, _ = _series()
        with pytest.raises(ValueError, match="units must be 'm' or 'mm'"):
            _fit_smooth_trend(dates, values, units="cm", sampling_days=MONTH)

    def test_the_limit_lowers_the_evidence(self):
        dates, values, _ = _series(_decay)
        free = fit_smooth_trend(
            dates, values, sampling_days=MONTH, max_random_walk_mm=None
        )
        limited = fit_smooth_trend(dates, values, sampling_days=MONTH)
        assert limited.evidence < free.evidence

    def test_rejected_smoothness_keeps_detected_rate_changes(self):
        # Regression: falling back to a plain line dropped a detected rate
        # change and got the present-day rate wrong (COSO, early cutoffs)
        dates, values, _ = _series(lambda t: -4.0 * np.maximum(0.0, t - 6.5))
        fit = fit_smooth_trend(dates, values, sampling_days=MONTH, min_evidence=1e9)
        assert fit.shape == "piecewise"
        assert len(fit.rate_change_dates) >= 1
        stepwise = detect_changepoints(dates, values, sampling_days=MONTH)
        assert fit.rate == pytest.approx(stepwise.rate, abs=2 * stepwise.rate_sigma)
        assert fit.rates.iloc[0] != pytest.approx(fit.rates.iloc[-1], abs=1.0)

    def test_default_knots_do_not_follow_the_seasons(self):
        # Regression: at 4 knots per year a seasonal cycle of varying amplitude
        # leaked into the rate, so the present-day rate depended on the month
        # the record ended (P056, LEMA)
        rng = np.random.default_rng(3)
        t = np.arange(144) / 12
        amplitude = 10.0 + 8.0 * np.sin(2 * np.pi * t / 7.0)
        values = -20.0 * t + amplitude * np.sin(2 * np.pi * t) + rng.normal(0, 1.0, 144)
        dates = pd.Timestamp("2012-01-01") + pd.to_timedelta(
            np.round(np.arange(144) * MONTH), unit="D"
        )

        def wiggle(knots_per_year):
            fit = fit_smooth_trend(
                dates,
                values,
                sampling_days=MONTH,
                detect_changes=False,
                knots_per_year=knots_per_year,
                min_evidence=0.0,
            )
            return float(fit.rates.std())

        assert wiggle(1.0) < 0.5 * wiggle(4.0)

    def test_detected_offset_is_not_smoothed(self):
        dates, values, _ = _series(lambda t: 25.0 * (t >= 5.0))
        fit = fit_smooth_trend(dates, values, sampling_days=MONTH)
        assert len(fit.step_dates) == 1
        assert abs((fit.step_dates[0] - dates[60]).days) <= 62
        jump = fit.model[66] - fit.model[54]
        assert jump == pytest.approx(25.0 - 3.0, abs=6.0)

    def test_without_detection_nothing_is_added(self):
        dates, values, _ = _series(lambda t: 25.0 * (t >= 5.0))
        fit = fit_smooth_trend(dates, values, sampling_days=MONTH, detect_changes=False)
        assert fit.step_dates == []
        assert fit.rate_change_dates == []

    def test_steps_in_one_gap_are_one_step(self):
        # Two catalogued steps inside a data gap are the same column, and steps
        # outside the record have no data on one side; none of them may raise
        dates, values, _ = _series()
        keep = np.r_[0:40, 50:144]
        steps = [
            dates[42],
            dates[44],
            dates[0] - pd.Timedelta(days=30),
            dates[-1] + pd.Timedelta(days=30),
        ]
        fit = fit_smooth_trend(
            dates[keep],
            values[keep],
            sampling_days=MONTH,
            step_dates=steps,
            detect_changes=False,
        )
        assert fit.step_dates == [dates[42]]

    def test_nans_are_dropped(self):
        dates, values, _ = _series()
        values = values.copy()
        values[[3, 70]] = np.nan
        fit = fit_smooth_trend(dates, values, sampling_days=MONTH)
        assert len(fit.model) == 142
        assert np.isfinite(fit.rate)

    def test_too_few_values_raise(self):
        dates, values, _ = _series(n=8)
        with pytest.raises(ValueError, match="at least 10 finite"):
            fit_smooth_trend(dates, values, sampling_days=MONTH)


class TestForecastSmooth:
    def test_the_last_rate_continues(self, decay):
        dates, values, _, fit = decay
        table = forecast_smooth(
            dates, values, 3, sampling_days=MONTH, fit=fit, include_noise=False
        ).table
        assert len(table) == 36
        assert table.index[0] > dates[-1]
        # Twelve months apart the periodic terms repeat
        assert table["model"].iloc[23] - table["model"].iloc[11] == pytest.approx(
            fit.rate, rel=1e-6
        )
        np.testing.assert_allclose(table["mean"], table["model"])
        assert (table["sigma_noise"] == 0).all()

    def test_a_smooth_interval_grows_faster_than_the_rate_uncertainty(self, decay):
        dates, values, _, fit = decay
        table = forecast_smooth(dates, values, 6, sampling_days=MONTH, fit=fit).table
        assert np.all(np.diff(table["sigma_model"]) > 0)
        # More than linearly with lead time: slope changes are still to come
        early, late = table["sigma_model"].iloc[11], table["sigma_model"].iloc[-1]
        assert late > 6.0 * early
        assert (table["lower"] < table["mean"]).all()
        assert (table["mean"] < table["upper"]).all()

    def test_a_linear_interval_grows_through_the_rate_only(self, linear):
        dates, values, fit = linear
        table = forecast_smooth(dates, values, 6, sampling_days=MONTH, fit=fit).table
        lead = table["lead_years"].to_numpy()
        sigma = table["sigma_model"].to_numpy()
        slope = (sigma[-1] - sigma[-13]) / (lead[-1] - lead[-13])
        assert slope == pytest.approx(fit.rate_sigma, rel=0.2)

    def test_a_piecewise_fit_expects_further_changes(self):
        dates, values, _ = _series(lambda t: -4.0 * np.maximum(0.0, t - 6.5))
        fit = fit_smooth_trend(dates, values, sampling_days=MONTH, min_evidence=1e9)
        table = forecast_smooth(dates, values, 5, sampling_days=MONTH, fit=fit).table
        # A rate uncertainty alone could not reach this far
        assert table["sigma_model"].iloc[-1] > 5.0 * fit.rate_sigma * 1.5

    def test_future_dates(self, linear):
        dates, values, fit = linear
        future = pd.DatetimeIndex(["2024-06-01", "2025-06-01"])
        fc = forecast_smooth(
            dates, values, future_dates=future, sampling_days=MONTH, fit=fit
        )
        assert list(fc.table.index) == list(future)
        assert fc.fit is fit
        assert fc.level == 0.95

    def test_level_scales_the_interval(self, linear):
        dates, values, fit = linear
        wide = forecast_smooth(dates, values, 2, sampling_days=MONTH, fit=fit)
        narrow = forecast_smooth(
            dates, values, 2, sampling_days=MONTH, fit=fit, level=0.5
        )
        ratio = (wide.table["upper"] - wide.table["lower"]) / (
            narrow.table["upper"] - narrow.table["lower"]
        )
        np.testing.assert_allclose(ratio, 1.96 / 0.6745, rtol=1e-3)

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
        dates, values, fit = linear
        with pytest.raises(ValueError, match=match):
            forecast_smooth(dates, values, sampling_days=MONTH, fit=fit, **kwargs)

    def test_backtest_scores_it(self):
        dates, values, _ = _series(_cycle)
        result = backtest(
            dates,
            values,
            3,
            n_cutoffs=3,
            forecaster=forecast_smooth,
            sampling_days=MONTH,
            units="mm",
        )
        assert result.predictions["cutoff"].nunique() == 3
        assert 0.7 <= result.coverage <= 1.0


class TestSmoothRegressors:
    @staticmethod
    def _driven():
        t = np.arange(144) / 12
        driver = 4.0 * np.sin(2 * np.pi * t / 7.0) + 2.0 * np.sin(2 * np.pi * t / 2.7)
        dates, values, _ = _series(
            lambda t: 8.0 * np.interp(t, np.arange(144) / 12, driver)
        )
        return dates, values, driver

    def test_a_signal_the_driver_explains_is_not_smooth(self):
        dates, values, driver = self._driven()
        assert fit_smooth_trend(dates, values, sampling_days=MONTH).shape == "smooth"
        fit = fit_smooth_trend(
            dates, values, sampling_days=MONTH, regressors={"well": driver}
        )
        assert fit.shape == "linear"
        amplitude, sigma = fit.parameters["reg_well"]
        assert abs(amplitude - 8.0) < max(3 * sigma, 0.5)
        assert fit.rate == pytest.approx(-3.0, abs=0.5)

    def test_a_smooth_signal_beside_the_driver_is_kept(self):
        dates, values, driver = self._driven()
        fit = fit_smooth_trend(
            dates,
            values + _decay(np.arange(144) / 12),
            sampling_days=MONTH,
            regressors={"well": driver},
        )
        assert fit.shape == "smooth"
        assert "reg_well" in fit.parameters

    def test_forecast_uses_the_future_driver(self):
        dates, values, driver = self._driven()
        past = np.arange(144) < 108
        fc = forecast_smooth(
            dates[past],
            values[past],
            future_dates=dates[~past],
            sampling_days=MONTH,
            units="mm",
            regressors={"well": driver[past]},
            future_regressors={"well": driver[~past]},
        )
        plain = forecast_smooth(
            dates[past],
            values[past],
            future_dates=dates[~past],
            sampling_days=MONTH,
            units="mm",
        )

        def rms(f):
            return np.sqrt(np.mean((f.table["mean"] - values[~past]) ** 2))

        assert rms(fc) < rms(plain) / 2

    def test_missing_future_driver_raises(self):
        dates, values, driver = self._driven()
        fit = fit_smooth_trend(
            dates, values, sampling_days=MONTH, regressors={"well": driver}
        )
        with pytest.raises(ValueError, match="must match the regressors"):
            forecast_smooth(dates, values, 1, sampling_days=MONTH, fit=fit)

    def test_piecewise_forecast_scales_on_rate_changes_not_on_the_driver(self):
        """The interval's change sizes must not take the regressor's amplitude."""
        dates, values, _ = _series(lambda t: 2.0 * np.maximum(0.0, t - 6.0))
        driver = np.cos(np.arange(144) / 11.0)
        kwargs = {"sampling_days": MONTH, "min_evidence": 1e9}
        huge = 1000.0 * driver
        fit = fit_smooth_trend(dates, values + huge, regressors={"d": driver}, **kwargs)
        base = fit_smooth_trend(dates, values, **kwargs)
        assert fit.shape == base.shape == "piecewise"
        future = dates[-1] + pd.to_timedelta(np.arange(1, 25) * MONTH, unit="D")
        with_driver = forecast_smooth(
            dates,
            values + huge,
            future_dates=future,
            sampling_days=MONTH,
            fit=fit,
            include_noise=False,
            future_regressors={"d": np.zeros(24)},
        ).table
        without = forecast_smooth(
            dates,
            values,
            future_dates=future,
            sampling_days=MONTH,
            fit=base,
            include_noise=False,
        ).table
        assert with_driver["sigma_model"].iloc[-1] == pytest.approx(
            without["sigma_model"].iloc[-1], rel=0.3
        )


class TestSmoothRegression:
    """Exact recovery, a real station, and pinned numbers."""

    def test_noise_free_cycle_is_recovered(self):
        t = np.arange(144) / 12
        truth = -3.0 * t + 6.0 * np.sin(2 * np.pi * t / 5.0)
        values = truth + np.random.default_rng(0).normal(0, 0.05, 144)
        dates = pd.Timestamp("2012-01-01") + pd.to_timedelta(
            np.round(np.arange(144) * MONTH), unit="D"
        )
        fit = fit_smooth_trend(
            dates,
            values,
            sampling_days=MONTH,
            periods_years=(),
            detect_changes=False,
            knots_per_year=4.0,
        )
        assert fit.shape == "smooth"
        assert np.sqrt(np.mean((fit.model - truth) ** 2)) < 0.05
        true_rate = -3.0 + 6.0 * 2 * np.pi / 5.0 * np.cos(2 * np.pi * t[-1] / 5.0)
        assert fit.rate == pytest.approx(true_rate, abs=0.5)

    def test_real_transient_fits_better_than_piecewise(self):
        # AHUP's 2006-08 inflation is not piecewise linear
        with (
            zipfile.ZipFile(Path(__file__).parent / "data/unr.zip") as archive,
            archive.open("tests/data/unr/AHUP.tenv3") as f,
        ):
            df = pd.read_csv(f, sep=r"\s+")
        mm = (df["u0(m)"] + df["____up(m)"]).to_numpy() * 1000.0
        s = pd.Series(mm - mm[0], index=pd.to_datetime(df["YYMMMDD"], format="%y%b%d"))
        edges = pd.date_range(s.index[0], s.index[-1], freq=pd.Timedelta(days=MONTH))
        medians = s.groupby(pd.cut(s.index, edges, right=False), observed=True).median()
        series = pd.Series(
            medians.to_numpy(), index=pd.DatetimeIndex([b.left for b in medians.index])
        ).dropna()
        fit = fit_smooth_trend(series.index, series.to_numpy(), sampling_days=MONTH)
        stepwise = detect_changepoints(
            series.index, series.to_numpy(), sampling_days=MONTH
        )
        assert fit.shape == "smooth"
        residual = series.to_numpy() - fit.model
        assert np.std(residual) < 0.7 * np.std(series.to_numpy() - stepwise.model)

    def test_golden(self, decay, golden):
        dates, values, _, fit = decay
        table = forecast_smooth(dates, values, 3, sampling_days=MONTH, fit=fit).table
        golden(
            "smooth.json",
            {
                "shape": fit.shape,
                "evidence": fit.evidence,
                "rate": fit.rate,
                "rate_sigma": fit.rate_sigma,
                "sigma_slope_change": fit.sigma_slope_change,
                "sigma_white": fit.sigma_white,
                "sigma_flicker": fit.sigma_flicker,
                "steps": [str(d.date()) for d in fit.step_dates],
                "rates": fit.rates.iloc[[0, 40, 56, 80, 143]].tolist(),
                "table": (
                    table.iloc[[0, 11, 23, 35]].reset_index(drop=True).to_dict("list")
                ),
            },
            rtol=1e-3,
        )

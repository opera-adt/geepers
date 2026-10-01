"""Tests for the linearity test and validity horizon (geepers.linearity)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geepers.linearity import (
    LinearityResult,
    _noise_covariance,
    linearity_test,
    validity_horizon,
)
from geepers.synthetic import power_law_noise
from geepers.trend import TrendResult, _powerlaw_covariance, estimate_trend

MONTH = 365.25 / 12


def _monthly(shape: str = "linear", seed: int = 3, n: int = 108):
    """Nine years of monthly data (mm): -3 mm/yr, flicker + white, annual."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 12
    values = (
        -3.0 * (t - t.mean())
        + 1.5 * power_law_noise(n, -1.0, 1.0, seed=seed)
        + rng.normal(0, 1.5, n)
        + 3.0 * np.sin(2 * np.pi * t)
    )
    if shape == "hinge":
        values -= 4.0 * np.maximum(0.0, t - 5.0)
    elif shape == "quadratic":
        values -= 0.35 * (t - t.mean()) ** 2
    dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(
        np.round(np.arange(n) * MONTH), unit="D"
    )
    return dates, values


class TestNoiseCovariance:
    def test_matches_the_fitted_components(self):
        trend = TrendResult(
            velocity=0.0,
            velocity_uncertainty=1.0,
            kappa=-1.0,
            sigma_powerlaw=2.0,
            sigma_white=0.5,
        )
        obs_idx = np.array([0, 1, 4, 5])  # a gap of two epochs
        C = _noise_covariance(trend, obs_idx, 6)
        expected = 4.0 * _powerlaw_covariance(-1.0, 6)[
            np.ix_(obs_idx, obs_idx)
        ] + 0.25 * np.eye(4)
        np.testing.assert_allclose(C, expected)

    def test_mixture_components(self):
        trend = TrendResult(
            velocity=0.0,
            velocity_uncertainty=1.0,
            sigma_white=1.0,
            sigma_flicker=2.0,
            sigma_randomwalk=0.5,
        )
        idx = np.arange(5)
        C = _noise_covariance(trend, idx, 5)
        expected = (
            np.eye(5)
            + 4.0 * _powerlaw_covariance(-1.0, 5)
            + 0.25 * _powerlaw_covariance(-2.0, 5)
        )
        np.testing.assert_allclose(C, expected)


class TestLinearityTest:
    def test_noise_comes_from_the_quadratic_fit(self):
        # Regression for the design: with the noise estimated around a
        # straight line, curvature was absorbed into a steeper spectrum and
        # no curved series was ever detected
        dates, values = _monthly("quadratic")
        linear_noise = estimate_trend(dates, values, sampling_days=MONTH)
        curved_noise = estimate_trend(dates, values, sampling_days=MONTH, poly_deg=2)
        assert linear_noise.kappa < curved_noise.kappa - 0.5
        result = linearity_test(dates, values, sampling_days=MONTH)
        assert result.model == "quadratic"
        assert result.trend.kappa == pytest.approx(linear_noise.kappa)

    def test_linear_series_is_called_linear(self):
        dates, values = _monthly("linear")
        result = linearity_test(dates, values, sampling_days=MONTH)
        assert result.model == "linear"
        assert result.departure == 0.0
        assert result.p_values["quadratic"] > 0.05
        assert result.p_values["piecewise"] > 0.05

    def test_linear_fit_is_the_trend_estimate(self):
        dates, values = _monthly("linear")
        result = linearity_test(dates, values, sampling_days=MONTH)
        trend = estimate_trend(dates, values, sampling_days=MONTH)
        assert result.trend.velocity == pytest.approx(trend.velocity)
        assert result.trend.velocity == pytest.approx(
            -3.0, abs=3 * trend.velocity_uncertainty
        )

    def test_detects_a_rate_change(self):
        dates, values = _monthly("hinge")
        result = linearity_test(dates, values, sampling_days=MONTH)
        # A mid-record hinge is close to a parabola; either verdict rejects
        # the single rate, and the hinge parameters are reported regardless
        assert result.model != "linear"
        assert result.p_values["piecewise"] < 0.01
        # True break five years in; rate -3 before and -7 after
        years_in = (result.breakpoint - dates[0]).days / 365.25
        assert years_in == pytest.approx(5.0, abs=1.0)
        assert result.rate_before == pytest.approx(-3.0, abs=1.0)
        assert result.rate_after == pytest.approx(-7.0, abs=1.5)
        assert result.departure > 2.0

    def test_detects_curvature(self):
        dates, values = _monthly("quadratic")
        result = linearity_test(dates, values, sampling_days=MONTH)
        assert result.model == "quadratic"
        assert result.p_values["quadratic"] < 0.01
        assert result.acceleration == pytest.approx(
            -0.7, abs=3 * result.acceleration_sigma
        )

    def test_reproducible_p_value(self):
        dates, values = _monthly("linear")
        a = linearity_test(dates, values, sampling_days=MONTH, n_simulations=300)
        b = linearity_test(dates, values, sampling_days=MONTH, n_simulations=300)
        assert a.p_values == b.p_values

    def test_short_series_has_no_piecewise_model(self):
        dates, values = _monthly("linear", n=30)  # 2.5 yr < 2 x 1.5 yr segments
        result = linearity_test(dates, values, sampling_days=MONTH)
        assert "piecewise" not in result.bic
        assert result.breakpoint is None
        assert np.isnan(result.rate_after)

    def test_handles_gaps_and_nans(self):
        dates, values = _monthly("hinge")
        values = values.copy()
        values[30:42] = np.nan  # a one-year gap
        result = linearity_test(dates, values, sampling_days=MONTH)
        assert result.model != "linear"
        assert result.p_values["piecewise"] < 0.05

    def test_piecewise_p_value_is_calibrated(self):
        # The breakpoint is searched for, so a chi-square p-value would be
        # far too small; the simulated one must be roughly uniform under
        # the null. White noise keeps this fast and exactly specified.
        rng = np.random.default_rng(11)
        dates = pd.date_range("2016-01-01", periods=120, freq="30D")
        p = [
            linearity_test(
                dates,
                rng.normal(0, 1, 120),
                sampling_days=30,
                noise_model="WN",
                periods_years=(),
                n_simulations=400,
                seed=k,
            ).p_values["piecewise"]
            for k in range(120)
        ]
        assert 0.01 <= np.mean(np.array(p) < 0.05) <= 0.11
        assert 0.4 < np.median(p) < 0.6


def _result(model, sigma=0.2, **kwargs) -> LinearityResult:
    trend = TrendResult(velocity=-3.0, velocity_uncertainty=sigma)
    return LinearityResult(model=model, trend=trend, **kwargs)


class TestValidityHorizon:
    def test_linear_is_set_by_rate_uncertainty(self):
        horizon = validity_horizon(_result("linear", sigma=0.2), tolerance=10.0)
        assert horizon.years == pytest.approx(50.0)
        assert horizon.driver == "rate uncertainty"

    def test_acceleration_shortens_it(self):
        result = _result("quadratic", acceleration=-0.5, acceleration_sigma=0.1)
        horizon = validity_horizon(result, tolerance=10.0)
        # Error after T years: 0.2 T + 0.25 T^2 = 10
        assert 0.2 * horizon.years + 0.25 * horizon.years**2 == pytest.approx(10.0)
        assert horizon.years < horizon.from_rate
        assert horizon.driver == "non-linearity"

    def test_unresolved_acceleration_is_ignored(self):
        result = _result("quadratic", acceleration=-0.5, acceleration_sigma=0.4)
        horizon = validity_horizon(result, tolerance=10.0)
        assert horizon.years == pytest.approx(horizon.from_rate)

    def test_rate_change_shortens_it(self):
        result = _result("piecewise", rate_before=-3.0, rate_after=-7.0)
        horizon = validity_horizon(result, tolerance=10.0)
        assert horizon.years == pytest.approx(10.0 / (0.2 + 4.0))
        assert horizon.driver == "non-linearity"

    def test_cap(self):
        horizon = validity_horizon(
            _result("linear", sigma=0.01), tolerance=10.0, cap_years=130.0
        )
        assert horizon.years == 130.0
        assert horizon.driver == "capped"

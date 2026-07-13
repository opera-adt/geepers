"""Tests for maximum-likelihood trend estimation (geepers.trend)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geepers.trend import (
    _fractional_diff_coeffs,
    _powerlaw_covariance,
    estimate_trend,
    estimate_trend_many,
)


def _synthetic(
    n: int = 600,
    velocity: float = 5.0,
    sigma_white: float = 1.0,
    sigma_pl: float = 0.0,
    kappa: float = -1.0,
    seed: int = 1234,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    t_yr = np.arange(n) / 365.25
    y = velocity * t_yr + rng.normal(0, sigma_white, n)
    if sigma_pl > 0:
        d = -kappa / 2.0
        psi = _fractional_diff_coeffs(d, n)
        y += np.convolve(rng.normal(0, sigma_pl, n), psi)[:n]
    return dates, y


class TestFractionalDiffCoeffs:
    def test_white(self):
        # d=0 -> identity: psi = [1, 0, 0, ...]
        psi = _fractional_diff_coeffs(0.0, 5)
        np.testing.assert_allclose(psi, [1, 0, 0, 0, 0])

    def test_random_walk(self):
        # d=1 -> integration: psi = all ones
        psi = _fractional_diff_coeffs(1.0, 5)
        np.testing.assert_allclose(psi, np.ones(5))

    def test_flicker_recursion(self):
        # psi_i = psi_{i-1} * (d + i - 1) / i with d = 0.5
        psi = _fractional_diff_coeffs(0.5, 4)
        np.testing.assert_allclose(psi, [1.0, 0.5, 0.375, 0.3125])


class TestPowerlawCovariance:
    def test_white_is_identity(self):
        C = _powerlaw_covariance(0.0, 20)
        np.testing.assert_allclose(C, np.eye(20), atol=1e-12)

    def test_random_walk_is_min_ij(self):
        # RW covariance on a regular grid is min(i, j) + 1 (1-indexed epochs)
        n = 15
        C = _powerlaw_covariance(-2.0, n)
        i, j = np.indices((n, n))
        np.testing.assert_allclose(C, np.minimum(i, j) + 1.0)

    def test_matches_uut_construction(self):
        # Cross-check the O(n^2) diagonal build against the direct U U^T
        from scipy.linalg import toeplitz

        n, kappa = 40, -0.7
        psi = _fractional_diff_coeffs(-kappa / 2, n)
        U = np.tril(toeplitz(psi))
        np.testing.assert_allclose(_powerlaw_covariance(kappa, n), U @ U.T)


class TestEstimateTrend:
    def test_white_noise_recovers_trend(self):
        dates, y = _synthetic(velocity=5.0, sigma_white=1.0)
        res = estimate_trend(dates, y, noise_model="WN", periods_years=())
        assert res.velocity == pytest.approx(5.0, abs=3 * res.velocity_uncertainty)
        # OLS-equivalent uncertainty for white noise should be small
        assert res.velocity_uncertainty < 0.5
        assert res.sigma_white == pytest.approx(1.0, rel=0.15)

    def test_plwn_recovers_noise_parameters(self):
        dates, y = _synthetic(
            n=800, velocity=3.0, sigma_white=1.0, sigma_pl=3.0, kappa=-1.0
        )
        res = estimate_trend(dates, y, periods_years=())
        # kappa within plausible range of the flicker truth
        assert -1.4 < res.kappa < -0.6
        # colored-noise uncertainty must be much larger than the OLS value
        res_wn = estimate_trend(dates, y, noise_model="WN", periods_years=())
        assert res.velocity_uncertainty > 3 * res_wn.velocity_uncertainty
        assert res.velocity == pytest.approx(3.0, abs=3 * res.velocity_uncertainty)

    def test_gaps_are_handled(self):
        dates, y = _synthetic(velocity=5.0, sigma_white=1.0)
        rng = np.random.default_rng(7)
        keep = rng.random(len(y)) > 0.3
        res = estimate_trend(dates[keep], y[keep], noise_model="WN", periods_years=())
        assert res.velocity == pytest.approx(5.0, abs=3 * res.velocity_uncertainty)

    def test_nan_values_dropped(self):
        dates, y = _synthetic(velocity=5.0)
        y[::7] = np.nan
        res = estimate_trend(dates, y, noise_model="WN", periods_years=())
        assert np.isfinite(res.velocity)

    def test_step_estimated(self):
        dates, y = _synthetic(n=500, velocity=0.0, sigma_white=0.5)
        step_date = dates[250]
        y[250:] += 20.0
        res = estimate_trend(
            dates, y, noise_model="WN", periods_years=(), step_dates=[step_date]
        )
        value, sigma = res.parameters["step_0"]
        assert value == pytest.approx(20.0, abs=3 * sigma)
        # trend must not absorb the step
        assert res.velocity == pytest.approx(0.0, abs=3 * res.velocity_uncertainty)

    def test_annual_amplitude(self):
        rng = np.random.default_rng(3)
        n = 730
        dates = pd.date_range("2018-01-01", periods=n, freq="D")
        t_yr = np.arange(n) / 365.25
        y = 4.0 * np.cos(2 * np.pi * t_yr) + rng.normal(0, 0.5, n)
        res = estimate_trend(dates, y, noise_model="WN", periods_years=(1.0,))
        value, sigma = res.parameters["cos_1yr"]
        assert value == pytest.approx(4.0, abs=3 * sigma)

    def test_too_few_observations_raises(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        with pytest.raises(ValueError, match="at least 10"):
            estimate_trend(dates, np.zeros(5))

    def test_result_metadata(self):
        dates, y = _synthetic(n=300)
        res = estimate_trend(dates, y, noise_model="WN", periods_years=())
        assert np.isfinite(res.log_likelihood)
        assert res.aic > 0 or res.bic > 0  # defined
        assert len(res.residuals) == len(y)
        assert "trend" in res.parameters
        assert "intercept" in res.parameters


class TestWhittleMethod:
    def test_matches_exact_on_plwn(self):
        dates, y = _synthetic(n=600, velocity=5.0, sigma_white=1.0, sigma_pl=2.0)
        exact = estimate_trend(dates, y, periods_years=())
        fast = estimate_trend(dates, y, periods_years=(), method="whittle")
        assert abs(fast.kappa - exact.kappa) < 0.35
        assert abs(fast.velocity - exact.velocity) < 3 * exact.velocity_uncertainty
        # Uncertainties should agree to well within a factor of two
        ratio = fast.velocity_uncertainty / exact.velocity_uncertainty
        assert 0.5 < ratio < 2.0

    def test_long_series_kappa_recovery(self):
        # Long series are impractical for the exact method but cheap here
        dates, y = _synthetic(
            n=3000, velocity=3.0, sigma_white=1.0, sigma_pl=2.0, kappa=-1.0
        )
        res = estimate_trend(dates, y, periods_years=(), method="whittle")
        assert abs(res.kappa - (-1.0)) < 0.25
        assert abs(res.velocity - 3.0) < 4 * res.velocity_uncertainty

    def test_white_noise_series(self):
        dates, y = _synthetic(n=600, velocity=5.0, sigma_white=1.0, sigma_pl=0.0)
        res = estimate_trend(dates, y, periods_years=(), method="whittle")
        # The white/power-law split is not identifiable as kappa -> 0
        # (power-law noise with kappa=0 IS white), so check the spectral
        # index and total variance instead of the split.
        assert res.kappa > -0.5
        total_sigma = np.hypot(res.sigma_white, res.sigma_powerlaw)
        assert 0.85 < total_sigma < 1.15

    def test_with_gaps_and_steps(self):
        dates, y = _synthetic(n=800, velocity=5.0, sigma_white=1.0, sigma_pl=1.0)
        y = y.copy()
        step_date = dates[400]
        y[400:] += 30.0
        keep = np.ones(len(y), dtype=bool)
        keep[100:150] = False  # a 50-day gap
        res = estimate_trend(
            dates[keep],
            y[keep],
            periods_years=(),
            step_dates=[step_date],
            method="whittle",
        )
        step_value = res.parameters["step_0"][0]
        assert abs(step_value - 30.0) < 1.0
        assert abs(res.velocity - 5.0) < 4 * res.velocity_uncertainty


def _synthetic_rw(n, sigma_rw, seed):
    """Add a random-walk component to a flicker+white series."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, sigma_rw, n))


class TestMixtureNoiseModels:
    def test_fnwn_recovers_amplitudes(self):
        # Flicker (kappa=-1) + white, the standard GNSS model
        dates, y = _synthetic(
            n=1500, velocity=3.0, sigma_white=1.0, sigma_pl=2.0, kappa=-1.0
        )
        res = estimate_trend(dates, y, periods_years=(), noise_model="FNWN")
        assert res.kappa == -1.0
        assert res.sigma_flicker == pytest.approx(2.0, rel=0.25)
        assert res.sigma_white == pytest.approx(1.0, rel=0.25)
        assert np.isnan(res.sigma_powerlaw)
        assert np.isnan(res.sigma_randomwalk)
        assert res.velocity == pytest.approx(3.0, abs=3 * res.velocity_uncertainty)

    def test_fnwn_exact_matches_whittle(self):
        dates, y = _synthetic(
            n=1200, velocity=2.0, sigma_white=1.0, sigma_pl=2.0, kappa=-1.0
        )
        exact = estimate_trend(dates, y, periods_years=(), noise_model="FNWN")
        fast = estimate_trend(
            dates, y, periods_years=(), noise_model="FNWN", method="whittle"
        )
        assert fast.sigma_flicker == pytest.approx(exact.sigma_flicker, rel=0.1)
        assert fast.velocity_uncertainty == pytest.approx(
            exact.velocity_uncertainty, rel=0.15
        )

    def test_rwfnwn_recovers_three_components(self):
        dates, y = _synthetic(
            n=1500, velocity=3.0, sigma_white=1.0, sigma_pl=1.5, kappa=-1.0, seed=2
        )
        y = y + _synthetic_rw(len(y), sigma_rw=0.3, seed=99)
        res = estimate_trend(dates, y, periods_years=(), noise_model="RWFNWN")
        assert res.sigma_randomwalk == pytest.approx(0.3, rel=0.5)
        assert res.sigma_flicker == pytest.approx(1.5, rel=0.4)
        assert res.sigma_white == pytest.approx(1.0, rel=0.3)

    def test_flicker_uncertainty_between_wn_and_plwn(self):
        # A flicker+white series: FNWN uncertainty should greatly exceed
        # the (wrong) white-only value
        dates, y = _synthetic(
            n=1000, velocity=3.0, sigma_white=1.0, sigma_pl=2.0, kappa=-1.0
        )
        wn = estimate_trend(dates, y, periods_years=(), noise_model="WN")
        fn = estimate_trend(dates, y, periods_years=(), noise_model="FNWN")
        assert fn.velocity_uncertainty > 3 * wn.velocity_uncertainty


class TestEstimateTrendMany:
    def test_matches_single_fits(self):
        dates, y0 = _synthetic(n=400, velocity=5.0, seed=1)
        _, y1 = _synthetic(n=400, velocity=-3.0, seed=2)
        df = estimate_trend_many(dates, np.vstack([y0, y1]), n_jobs=1, periods_years=())
        assert len(df) == 2
        single = estimate_trend(dates, y0, periods_years=(), method="whittle")
        np.testing.assert_allclose(df.velocity.iloc[0], single.velocity)
        np.testing.assert_allclose(
            df.velocity_uncertainty.iloc[0], single.velocity_uncertainty
        )

    def test_parallel_processes(self):
        dates, y0 = _synthetic(n=300, velocity=5.0, seed=3)
        _, y1 = _synthetic(n=300, velocity=1.0, seed=4)
        serial = estimate_trend_many(
            dates, np.vstack([y0, y1]), n_jobs=1, periods_years=()
        )
        parallel = estimate_trend_many(
            dates, np.vstack([y0, y1]), n_jobs=2, periods_years=()
        )
        pd.testing.assert_frame_equal(serial, parallel)

    def test_dataframe_input_and_failures(self):
        dates, y0 = _synthetic(n=300, velocity=5.0, seed=5)
        bad = np.full(300, np.nan)
        frame = pd.DataFrame({"good": y0, "bad": bad}, index=dates)
        df = estimate_trend_many(dates, frame, n_jobs=1, periods_years=())
        assert list(df.index) == ["good", "bad"]
        assert np.isfinite(df.loc["good", "velocity"])
        assert np.isnan(df.loc["bad", "velocity"])

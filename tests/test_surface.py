"""Tests for polynomial trend surfaces (geepers.surface)."""

from __future__ import annotations

import numpy as np
import pytest

from geepers.collocation import collocate, noise_covariance, predict, signal_covariance
from geepers.surface import _local_xy, _powers, fit_polynomial_surface


@pytest.fixture
def network():
    rng = np.random.default_rng(15)
    n = 120
    return rng.uniform(-120, -116, n), rng.uniform(33, 37, n)


def _km(lon, lat, lon0, lat0):
    return _local_xy(np.asarray(lon), np.asarray(lat), lon0, lat0, 1.0)


class TestPowers:
    def test_term_counts(self):
        assert _powers(0) == [(0, 0)]
        assert _powers(1) == [(0, 0), (1, 0), (0, 1)]
        assert len(_powers(2)) == 6
        assert len(_powers(3)) == 10


class TestFitPolynomialSurface:
    def test_recovers_a_plane(self, network):
        lon, lat = network
        x, y = _km(lon, lat, lon.mean(), lat.mean())
        values = 2.0 + 0.01 * x - 0.02 * y  # mm/yr, with mm/yr per km slopes
        surface = fit_polynomial_surface(lon, lat, values)

        np.testing.assert_allclose(surface.residuals, 0, atol=1e-10)
        # Coefficients are per scaled unit: divide by the scale for per-km
        assert surface.coefficients[0] == pytest.approx(2.0)
        assert surface.coefficients[1] / surface.scale_km == pytest.approx(0.01)
        assert surface.coefficients[2] / surface.scale_km == pytest.approx(-0.02)

    def test_predicts_at_new_points(self, network):
        lon, lat = network
        lon0, lat0 = lon.mean(), lat.mean()

        def truth(lo, la):
            x, y = _km(lo, la, lon0, lat0)
            return 1.0 + 0.01 * x - 0.02 * y + 1e-5 * x * y + 2e-5 * x**2

        surface = fit_polynomial_surface(lon, lat, truth(lon, lat), degree=2)
        lon_g, lat_g = np.meshgrid(np.linspace(-119, -117, 6), np.linspace(34, 36, 5))
        predicted = surface.predict(lon_g, lat_g)
        assert predicted.shape == (5, 6)
        np.testing.assert_allclose(predicted, truth(lon_g, lat_g), atol=1e-8)

    def test_degree_zero_is_weighted_mean(self, network):
        lon, lat = network
        rng = np.random.default_rng(1)
        values = rng.normal(3, 1, lon.size)
        sigmas = rng.uniform(0.2, 2, lon.size)
        surface = fit_polynomial_surface(lon, lat, values, sigmas, degree=0)
        expected = np.sum(values / sigmas**2) / np.sum(1 / sigmas**2)
        assert surface.coefficients[0] == pytest.approx(expected)

    def test_weights_downweight_outlier(self, network):
        lon, lat = network
        x, y = _km(lon, lat, lon.mean(), lat.mean())
        values = 0.01 * x
        values[3] += 100.0
        sigmas = np.ones(lon.size)
        sigmas[3] = 1e4
        plain = fit_polynomial_surface(lon, lat, values)
        weighted = fit_polynomial_surface(lon, lat, values, sigmas)
        clean = np.arange(lon.size) != 3
        assert np.abs(weighted.residuals[clean]).max() < 1e-3
        assert np.abs(plain.residuals[clean]).max() > 0.5

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="terms"):
            fit_polynomial_surface([0.0, 1.0], [0.0, 1.0], [1.0, 2.0], degree=1)

    def test_negative_degree_raises(self, network):
        lon, lat = network
        with pytest.raises(ValueError, match="degree"):
            fit_polynomial_surface(lon, lat, np.zeros(lon.size), degree=-1)


class TestRemoveRestore:
    def test_improves_collocation_of_a_tilted_field(self, network):
        # Collocation assumes a zero-mean signal: with an offset and tilt
        # left in, predictions sag toward zero between and beyond stations
        lon, lat = network
        rng = np.random.default_rng(2)
        lon_new, lat_new = rng.uniform(-120, -116, 60), rng.uniform(33, 37, 60)
        lon_all, lat_all = np.r_[lon, lon_new], np.r_[lat, lat_new]
        n = lon.size

        parameters = np.array([1.0, 60.0])
        C = signal_covariance(
            lon_all, lat_all, lon_all, lat_all, parameters, components=("up",)
        )
        signal = np.linalg.cholesky(C + 1e-9 * np.eye(len(lon_all))) @ rng.normal(
            size=len(lon_all)
        )
        x, y = _km(lon_all, lat_all, lon.mean(), lat.mean())
        truth = signal + 8.0 + 0.02 * x - 0.015 * y
        sigmas = np.full(n, 0.2)
        observed = truth[:n] + rng.normal(0, 0.2, n)

        def collocation(values):
            Css = signal_covariance(lon, lat, lon, lat, parameters, components=("up",))
            _, Czz_inv = collocate(values, Css, noise_covariance(sigmas))
            Cps = signal_covariance(
                lon, lat, lon_new, lat_new, parameters, components=("up",)
            )
            Cpp = signal_covariance(
                lon_new, lat_new, lon_new, lat_new, parameters, components=("up",)
            )
            return predict(values, Cps, Cpp, Czz_inv).signal[:, 0]

        direct = collocation(observed)
        surface = fit_polynomial_surface(lon, lat, observed, sigmas, degree=1)
        restored = collocation(surface.residuals) + surface.predict(lon_new, lat_new)

        def rmse(prediction):
            return np.sqrt(np.mean((prediction - truth[n:]) ** 2))

        assert rmse(restored) < 0.5 * rmse(direct)

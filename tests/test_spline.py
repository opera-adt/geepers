"""Tests for the elastically coupled vector spline (geepers.spline)."""

from __future__ import annotations

import numpy as np
import pytest

from geepers.spline import (
    _greens_functions,
    _grid_to_true,
    _jacobian,
    _project,
    _true_to_grid,
    fit_vector_spline,
)


@pytest.fixture
def network():
    rng = np.random.default_rng(8)
    n = 80
    return rng.uniform(-120, -116, n), rng.uniform(33, 37, n)


def _elastic_field(lon, lat, lon0=-118.0, lat0=35.0):
    """Smooth field: the elastic response to three forces outside the network."""
    x, y, gamma = _project(np.asarray(lon), np.asarray(lat), lon0, lat0)
    fx = np.array([-600.0, 500.0, 100.0])
    fy = np.array([-300.0, 450.0, -700.0])
    f_east = np.array([4.0, -2.0, 1.0])
    f_north = np.array([-1.0, 3.0, 2.0])
    g_ee, g_nn, g_en = _greens_functions(x[:, None] - fx, y[:, None] - fy, 10.0, 0.5)
    vx = g_ee @ f_east + g_en @ f_north
    vy = g_en @ f_east + g_nn @ f_north
    vx, vy = vx - vx.mean(), vy - vy.mean()
    return _grid_to_true(vx, vy, gamma)


class TestGreensFunctions:
    def test_reference_values(self):
        # Sandwell & Wessel (2016), eq. 5, for a force 3 km west, 4 km south
        g_ee, g_nn, g_en = _greens_functions(
            np.array([3.0]), np.array([4.0]), 0.0, 0.25
        )
        assert g_ee[0] == pytest.approx(2.75 * np.log(5.0) + 1.25 * 16 / 25)
        assert g_nn[0] == pytest.approx(2.75 * np.log(5.0) + 1.25 * 9 / 25)
        assert g_en[0] == pytest.approx(-1.25 * 12 / 25)

    def test_jacobian_symmetric_blocks(self):
        rng = np.random.default_rng(0)
        x, y = rng.uniform(0, 100, 6), rng.uniform(0, 100, 6)
        jac = _jacobian(x, y, x, y, 10.0, 0.5)
        assert jac.shape == (12, 12)
        np.testing.assert_allclose(jac, jac.T)


class TestProjection:
    def test_component_rotation_roundtrip(self, network):
        lon, lat = network
        _, _, gamma = _project(lon, lat, lon.mean(), lat.mean())
        rng = np.random.default_rng(1)
        east, north = rng.normal(size=lon.size), rng.normal(size=lon.size)
        back = _grid_to_true(*_true_to_grid(east, north, gamma), gamma)
        np.testing.assert_allclose(back[0], east)
        np.testing.assert_allclose(back[1], north)

    def test_true_north_follows_projected_meridian(self):
        # Far from the projection center grid north is not true north
        lon, lat = np.array([-110.0]), np.array([40.0])
        lon0, lat0 = -118.0, 35.0
        x, y, gamma = _project(lon, lat, lon0, lat0)
        x2, y2, _ = _project(lon, lat + 0.01, lon0, lat0)
        meridian = np.array([x2[0] - x[0], y2[0] - y[0]])
        meridian /= np.linalg.norm(meridian)
        vx, vy = _true_to_grid(np.array([0.0]), np.array([1.0]), gamma)
        assert abs(np.degrees(gamma[0])) > 3
        np.testing.assert_allclose([vx[0], vy[0]], meridian, atol=1e-4)


class TestFitVectorSpline:
    def test_exact_at_stations_without_damping(self, network):
        lon, lat = network
        rng = np.random.default_rng(2)
        east, north = rng.normal(0, 3, lon.size), rng.normal(0, 3, lon.size)
        spline = fit_vector_spline(lon, lat, east, north)
        e, n = spline.predict(lon, lat)
        np.testing.assert_allclose(e, east, atol=1e-8)
        np.testing.assert_allclose(n, north, atol=1e-8)
        np.testing.assert_allclose(spline.residuals, 0, atol=1e-8)

    def test_recovers_smooth_field_at_new_points(self, network):
        lon, lat = network
        east, north = _elastic_field(lon, lat)
        rng = np.random.default_rng(3)
        lon_new, lat_new = rng.uniform(-119.5, -116.5, 50), rng.uniform(33.5, 36.5, 50)
        true_e, true_n = _elastic_field(np.r_[lon, lon_new], np.r_[lat, lat_new])
        east, north = true_e[: lon.size], true_n[: lon.size]

        spline = fit_vector_spline(lon, lat, east, north)
        e, n = spline.predict(lon_new, lat_new)
        signal = np.hypot(true_e, true_n).std()
        misfit = np.sqrt(np.mean((e - true_e[lon.size :]) ** 2))
        misfit_n = np.sqrt(np.mean((n - true_n[lon.size :]) ** 2))
        assert misfit < 0.05 * signal
        assert misfit_n < 0.05 * signal

    def test_damping_filters_noise(self, network):
        lon, lat = network
        true_e, true_n = _elastic_field(lon, lat)
        rng = np.random.default_rng(4)
        sigma = 0.3 * np.hypot(true_e, true_n).std()
        east = true_e + rng.normal(0, sigma, lon.size)
        north = true_n + rng.normal(0, sigma, lon.size)

        exact = fit_vector_spline(lon, lat, east, north)
        damped = fit_vector_spline(lon, lat, east, north, damping=1.0)

        def error(spline):
            e, n = spline.predict(lon, lat)
            return np.sqrt(np.mean((e - true_e) ** 2 + (n - true_n) ** 2))

        assert np.abs(damped.residuals).max() > 0
        assert error(damped) < 0.75 * error(exact)

    def test_poisson_minus_one_decouples_components(self, network):
        lon, lat = network
        rng = np.random.default_rng(5)
        east = rng.normal(0, 3, lon.size)
        north_a, north_b = rng.normal(0, 3, lon.size), rng.normal(0, 3, lon.size)
        # Near the projection center, where grid and true axes coincide
        lon_new, lat_new = np.array([lon.mean()]), np.array([lat.mean()])

        def east_at_center(north, poisson):
            spline = fit_vector_spline(lon, lat, east, north, poisson=poisson)
            return spline.predict(lon_new, lat_new)[0][0]

        decoupled = east_at_center(north_a, -1.0) - east_at_center(north_b, -1.0)
        coupled = east_at_center(north_a, 0.5) - east_at_center(north_b, 0.5)
        assert abs(decoupled) < 0.02
        assert abs(coupled) > 10 * abs(decoupled)

    def test_weights_downweight_noisy_station(self, network):
        lon, lat = network
        true_e, true_n = _elastic_field(lon, lat)
        east = true_e.copy()
        east[10] += 50.0
        sigma = np.ones(lon.size)
        noisy = sigma.copy()
        noisy[10] = 1e3

        plain = fit_vector_spline(lon, lat, east, true_n, sigma, sigma, damping=1e-3)
        weighted = fit_vector_spline(lon, lat, east, true_n, noisy, sigma, damping=1e-3)
        err_plain = abs(plain.predict(lon[10], lat[10])[0] - true_e[10])
        err_weighted = abs(weighted.predict(lon[10], lat[10])[0] - true_e[10])
        assert err_weighted < 0.2 * err_plain

    def test_predict_keeps_shape(self, network):
        lon, lat = network
        east, north = _elastic_field(lon, lat)
        spline = fit_vector_spline(lon, lat, east, north)
        lon_g, lat_g = np.meshgrid(np.linspace(-119, -117, 7), np.linspace(34, 36, 5))
        e, n = spline.predict(lon_g, lat_g)
        assert e.shape == n.shape == (5, 7)

    def test_one_sigma_array_raises(self, network):
        lon, lat = network
        v = np.zeros(lon.size)
        with pytest.raises(ValueError, match="both"):
            fit_vector_spline(lon, lat, v, v, sigma_east=np.ones(lon.size))

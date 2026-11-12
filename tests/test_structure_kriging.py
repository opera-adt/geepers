"""Tests for kriging guided by the structure function (geepers.structure_kriging)."""

from __future__ import annotations

import numpy as np
import pytest

from geepers import structure_kriging as sk


@pytest.fixture
def network():
    rng = np.random.default_rng(3)
    n = 400
    lon = rng.uniform(-120.0, -112.0, n)
    lat = rng.uniform(34.0, 41.0, n)
    return lon, lat, rng


def flat_curve(scatter2: float = 0.25) -> np.ndarray:
    """A structure with the same scatter at every separation: noise only."""
    return np.array([[0.0, scatter2], [100.0, scatter2], [500.0, scatter2]])


def rising_curve() -> np.ndarray:
    return np.array([[0.0, 0.04], [50.0, 0.3], [200.0, 1.0], [500.0, 2.5]])


class TestStructure:
    def test_scatter_grows_with_separation_for_a_gradient(self, network):
        lon, lat, rng = network
        g = sk.structure(lon, lat, 2.0 * lon + rng.normal(0, 0.2, lon.size))

        assert g[0, 0] == 0.0
        assert np.all(np.diff(g[:, 1]) >= 0)
        assert g[-1, 1] > 20 * g[0, 1]

    def test_zero_lag_is_the_noise_of_one_station(self, network):
        lon, lat, rng = network
        g = sk.structure(lon, lat, rng.normal(0, 0.5, lon.size))

        # white noise of 0.5: every separation shows about the same scatter, 0.5 squared
        # (a little above it, because the curve is forced not to decrease)
        assert g[:, 1].max() < 1.3 * g[:, 1].min()
        assert g[0, 1] == pytest.approx(0.25, rel=0.3)

    def test_implied_ssf_starts_at_one_and_decreases(self, network):
        lon, lat, rng = network
        ssf = sk.implied_ssf(sk.structure(lon, lat, lon + rng.normal(0, 0.2, lon.size)))

        assert ssf[0, 1] == 1.0
        assert np.all(np.diff(ssf[:, 1]) <= 0)


class TestDetrend:
    def test_a_plane_leaves_nothing(self, network):
        lon, lat, _ = network
        residual = sk.detrend(lon, lat, 3.0 + 2.0 * lon - 1.5 * lat)

        assert np.abs(residual).max() < 1e-6

    def test_gradient_no_longer_shows_in_the_structure(self, network):
        lon, lat, rng = network
        values = 2.0 * lon + rng.normal(0, 0.2, lon.size)
        raw = sk.structure(lon, lat, values)
        left = sk.structure(lon, lat, sk.detrend(lon, lat, values))

        assert left[-1, 1] < 0.02 * raw[-1, 1]


class TestKrige:
    def test_trend_reproduces_a_plane_exactly(self, network):
        lon, lat, _ = network
        values = 3.0 + 2.0 * lon - 1.5 * lat
        at_lon, at_lat = np.array([-116.3, -114.9]), np.array([37.2, 39.4])
        est = sk.krige(
            lon,
            lat,
            values,
            np.full(lon.size, 0.2),
            at_lon,
            at_lat,
            flat_curve(),
            trend=True,
        )

        assert est.value.to_numpy() == pytest.approx(
            3.0 + 2.0 * at_lon - 1.5 * at_lat, abs=1e-6
        )
        assert (est.planar == 1).all()

    def test_constant_mean_smooths_a_gradient_that_a_plane_follows(self, network):
        lon, lat, _ = network
        values = 2.0 * lon
        # at the edge of the network a mean of the neighbours cannot reach the true value
        est = {
            trend: (
                sk.krige(
                    lon,
                    lat,
                    values,
                    np.full(lon.size, 0.2),
                    [-119.95],
                    [37.5],
                    flat_curve(),
                    trend=trend,
                ).value[0]
            )
            for trend in (False, True)
        }

        assert abs(est[True] - 2.0 * -119.95) < 1e-6
        assert abs(est[False] - 2.0 * -119.95) > 0.2

    def test_sigma_grows_away_from_stations(self, network):
        lon, lat, rng = network
        values = rng.normal(0, 0.3, lon.size)
        inside = sk.krige(
            lon, lat, values, np.full(lon.size, 0.2), [-116.0], [37.5], rising_curve()
        )
        outside = sk.krige(
            lon, lat, values, np.full(lon.size, 0.2), [-108.0], [37.5], rising_curve()
        )

        assert outside.km[0] > 3 * inside.km[0]
        assert outside.sigma[0] > 2 * inside.sigma[0]
        assert outside.sigma_total[0] >= outside.sigma[0]

    def test_positive_keeps_the_value_within_the_stations(self):
        # stations on a line with a steady gradient; the point is beyond their end
        lon = np.linspace(-116.0, -114.0, 12)
        lat = np.full(12, 36.0) + np.tile([0.0, 0.05], 6)
        values = 10.0 * (lon + 116.0)
        # a very smooth field (scatter growing with the square of the separation) is what
        # makes kriging continue a gradient past the last station
        d = np.array([0.0, 25.0, 50.0, 100.0, 200.0, 300.0, 500.0])
        smooth = np.c_[d, np.maximum((d / 100.0) ** 2, 1e-3)]
        sigmas = np.full(12, 0.02)
        free = sk.krige(lon, lat, values, sigmas, [-113.6], [36.0], smooth)
        kept = sk.krige(
            lon, lat, values, sigmas, [-113.6], [36.0], smooth, positive=True
        )

        assert free.value[0] > values.max()
        assert values.min() <= kept.value[0] <= values.max()

    def test_breakline_keeps_the_two_sides_apart(self, network):
        lon, lat, _ = network
        values = np.where(lon < -116.0, -40.0, -12.0)
        fault = [np.array([[-116.0, 30.0], [-116.0, 45.0]])]
        est = sk.krige(
            lon,
            lat,
            values,
            np.full(lon.size, 0.2),
            [-116.05, -115.95],
            [37.5, 37.5],
            flat_curve(),
            breaklines=fault,
        )
        mixed = sk.krige(
            lon, lat, values, np.full(lon.size, 0.2), [-116.05], [37.5], flat_curve()
        )

        assert est.value.to_numpy() == pytest.approx([-40.0, -12.0])
        assert -40.0 < mixed.value[0] < -12.0

    def test_noisy_station_weighs_less(self, network):
        lon, lat, _ = network
        values = np.zeros(lon.size)
        sigmas = np.full(lon.size, 0.2)
        nearest = np.argmin(np.hypot(lon + 116.0, lat - 37.5))
        values[nearest] = 30.0
        trusted = sk.krige(lon, lat, values, sigmas, [-116.0], [37.5], rising_curve())
        sigmas[nearest] = 30.0
        doubted = sk.krige(lon, lat, values, sigmas, [-116.0], [37.5], rising_curve())

        assert abs(doubted.value[0]) < 0.2 * abs(trusted.value[0])

    def test_plane_is_used_only_where_the_stations_show_one(self, network):
        lon, lat, rng = network
        noise = rng.normal(0, 0.3, lon.size)
        point = {"lon_new": [-116.0], "lat_new": [37.5]}
        sigmas = np.full(lon.size, 0.2)
        flat = sk.krige(
            lon,
            lat,
            noise,
            sigmas,
            g=flat_curve(),
            trend=True,
            g_plain=flat_curve(),
            **point,
        )
        tilted = sk.krige(
            lon,
            lat,
            5.0 * lon + noise,
            sigmas,
            g=flat_curve(),
            trend=True,
            g_plain=rising_curve(),
            **point,
        )

        assert flat.planar[0] == 0
        assert tilted.planar[0] == 1

    def test_too_few_stations_on_the_side_gives_nan(self):
        lon = np.array([-116.5, -116.4, -115.5, -115.4, -115.3, -115.2])
        lat = np.array([37.0, 37.2, 37.0, 37.1, 37.2, 37.3])
        fault = [np.array([[-116.0, 30.0], [-116.0, 45.0]])]
        est = sk.krige(
            lon,
            lat,
            np.ones(6),
            np.full(6, 0.2),
            [-116.45],
            [37.1],
            flat_curve(),
            breaklines=fault,
        )

        assert np.isnan(est.value[0])
        assert est.shape == (1, 7)

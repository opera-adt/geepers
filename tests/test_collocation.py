"""Tests for least-squares collocation (geepers.collocation)."""

from __future__ import annotations

import numpy as np
import pytest

from geepers.collocation import (
    COVARIANCE_MODELS,
    collocate,
    create_regular_grid,
    distance_matrix,
    empirical_covariance,
    interpolate_velocities,
    noise_covariance,
    predict,
    signal_covariance,
)


@pytest.fixture
def stations():
    """A small synthetic network with a smooth GM1 signal."""
    rng = np.random.default_rng(11)
    n = 60
    lon = rng.uniform(14, 18, n)
    lat = rng.uniform(44, 47, n)
    # Draw a smooth correlated field from the GM1 covariance itself
    C = signal_covariance(
        lon, lat, lon, lat, np.array([4.0, 150.0]), components=("up",)
    )
    L = np.linalg.cholesky(C + 1e-9 * np.eye(n))
    ve = L @ rng.normal(size=n)
    vn = L @ rng.normal(size=n)
    return lon, lat, ve, vn


class TestDistanceMatrix:
    def test_diagonal_zero_and_symmetric(self, stations):
        lon, lat, *_ = stations
        D = distance_matrix(lon, lat, lon, lat)
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-9)
        np.testing.assert_allclose(D, D.T, atol=1e-6)

    def test_one_degree_latitude(self):
        D = distance_matrix(np.r_[0.0], np.r_[0.0], np.r_[0.0], np.r_[1.0])
        assert D[0, 0] == pytest.approx(110.57, abs=0.5)  # km

    def test_meters_unit(self):
        Dk = distance_matrix(np.r_[0.0], np.r_[0.0], np.r_[1.0], np.r_[0.0], "km")
        Dm = distance_matrix(np.r_[0.0], np.r_[0.0], np.r_[1.0], np.r_[0.0], "m")
        assert Dm[0, 0] == pytest.approx(1000 * Dk[0, 0])


class TestCovarianceModels:
    def test_all_models_at_zero_distance(self):
        for name, func in COVARIANCE_MODELS.items():
            assert func(0.0, 3.0, 100.0) == pytest.approx(3.0), name

    def test_gm1_decay(self):
        f = COVARIANCE_MODELS["gm1"]
        assert f(100.0, 1.0, 100.0) == pytest.approx(np.exp(-1))


class TestSignalCovariance:
    def test_scalar_field_symmetric_posdef(self, stations):
        lon, lat, *_ = stations
        C = signal_covariance(
            lon, lat, lon, lat, np.array([2.0, 200.0]), components=("up",)
        )
        np.testing.assert_allclose(C, C.T, atol=1e-9)
        assert np.all(np.linalg.eigvalsh(C) > -1e-9)

    def test_block_shape(self, stations):
        lon, lat, *_ = stations
        n = len(lon)
        C = signal_covariance(lon, lat, lon, lat, np.array([2.0, 200.0]))
        assert C.shape == (2 * n, 2 * n)

    def test_no_cross_correlation_blocks_zero(self, stations):
        lon, lat, *_ = stations
        n = len(lon)
        C = signal_covariance(
            lon, lat, lon, lat, np.array([2.0, 200.0]), cross_correlation=False
        )
        np.testing.assert_allclose(C[:n, n:], 0)
        np.testing.assert_allclose(C[n:, :n], 0)


class TestCollocation:
    def test_zero_noise_reproduces_observations(self, stations):
        lon, lat, ve, vn = stations
        n = len(lon)
        Css = signal_covariance(lon, lat, lon, lat, np.array([4.0, 150.0]))
        Cnn = noise_covariance(np.full((n, 2), 1e-6))
        res, _ = collocate(np.c_[ve, vn], Css, Cnn)
        np.testing.assert_allclose(res.signal[:, 0], ve, atol=1e-3)
        np.testing.assert_allclose(res.signal[:, 1], vn, atol=1e-3)
        # noise estimate must be negligible
        assert np.abs(res.noise).max() < 1e-3

    def test_noise_shrinks_signal(self, stations):
        lon, lat, ve, vn = stations
        n = len(lon)
        Css = signal_covariance(lon, lat, lon, lat, np.array([4.0, 150.0]))
        res_lo, _ = collocate(
            np.c_[ve, vn], Css, noise_covariance(np.full((n, 2), 1e-6))
        )
        res_hi, _ = collocate(
            np.c_[ve, vn], Css, noise_covariance(np.full((n, 2), 5.0))
        )
        # heavier noise -> stronger smoothing toward zero
        assert np.abs(res_hi.signal).sum() < np.abs(res_lo.signal).sum()
        # and larger posterior uncertainty
        assert res_hi.signal_sigma.mean() > res_lo.signal_sigma.mean()

    def test_predict_at_observation_points_matches(self, stations):
        lon, lat, ve, vn = stations
        n = len(lon)
        params = np.array([4.0, 150.0])
        Css = signal_covariance(lon, lat, lon, lat, params)
        Cnn = noise_covariance(np.full((n, 2), 0.5))
        obs = np.c_[ve, vn]
        at_obs, Czz_inv = collocate(obs, Css, Cnn)
        pred = predict(obs, Css, Css, Czz_inv)
        np.testing.assert_allclose(pred.signal, at_obs.signal, atol=1e-8)

    def test_interpolate_velocities_end_to_end(self, stations):
        lon, lat, ve, vn = stations
        n = len(lon)
        rng = np.random.default_rng(5)
        lon_new = rng.uniform(14.5, 17.5, 25)
        lat_new = rng.uniform(44.5, 46.5, 25)
        _at_sta, at_new = interpolate_velocities(
            lon,
            lat,
            ve,
            vn,
            np.full(n, 0.3),
            np.full(n, 0.3),
            lon_new,
            lat_new,
            covariance_parameters=np.array([4.0, 150.0]),
        )
        assert at_new.signal.shape == (25, 2)
        assert at_new.signal_sigma.shape == (25, 2)
        assert np.all(np.isfinite(at_new.signal))
        # interpolated values must stay within the observed range
        assert at_new.signal[:, 0].max() <= ve.max() + 1
        assert at_new.signal[:, 0].min() >= ve.min() - 1


class TestEmpiricalCovariance:
    def test_recovers_parameters_roughly(self):
        rng = np.random.default_rng(21)
        n = 300
        # Domain spans ~12 correlation lengths so a single realization
        # carries enough information to recover the parameters
        lon = rng.uniform(8, 22, n)
        lat = rng.uniform(38, 50, n)
        c0_true, d0_true = 9.0, 100.0
        C = signal_covariance(
            lon, lat, lon, lat, np.array([c0_true, d0_true]), components=("up",)
        )
        L = np.linalg.cholesky(C + 1e-9 * np.eye(n))
        d1 = L @ rng.normal(size=n)
        noise = np.full(n, 0.1)
        # Scalar field: pass the same component twice (original convention)
        emp = empirical_covariance(lon, lat, d1, d1, noise, noise, bin_spacing_km=40)
        c0, d0 = emp.parameters
        # Single-realization covariograms are noisy: accept loose recovery
        assert c0 == pytest.approx(c0_true, rel=0.5)
        assert 0.3 * d0_true < d0 < 3 * d0_true
        assert emp.pearson > 0.8


class TestGrid:
    def test_create_regular_grid(self):
        lon, lat = create_regular_grid(16.0, 45.0, 400, 300, dx_km=50, dy_km=50)
        assert len(lon) == len(lat)
        assert len(lon) > 20
        # center should be inside the grid
        assert lon.min() < 16 < lon.max()
        assert lat.min() < 45 < lat.max()
        # ~400 km wide -> about +/- 2.5 deg longitude at 45N
        assert 12 < lon.min() < 16
        assert 16 < lon.max() < 20


class TestOrdinaryKriging:
    def test_recovers_smooth_field_with_offset(self, stations):
        # OK is mean-invariant: add a large constant, no demeaning needed
        from geepers.collocation import ordinary_kriging

        lon, lat, ve, _ = stations
        n = len(lon)
        v = ve + 100.0
        rng = np.random.default_rng(2)
        lon_i = rng.uniform(14.5, 17.5, 15)
        lat_i = rng.uniform(44.5, 46.5, 15)
        out = ordinary_kriging(
            lon, lat, v, np.full(n, 0.1), lon_i, lat_i, np.array([4.0, 150.0])
        )
        assert out.signal.shape == (15, 1)
        assert np.all(out.signal > 90)  # offset preserved
        assert np.all(out.signal_sigma >= 0)

    def test_sigma_grows_away_from_data(self, stations):
        from geepers.collocation import ordinary_kriging

        lon, lat, ve, _ = stations
        n = len(lon)
        near = ordinary_kriging(
            lon,
            lat,
            ve,
            np.full(n, 0.1),
            [lon.mean()],
            [lat.mean()],
            np.array([4.0, 150.0]),
        )
        far = ordinary_kriging(
            lon,
            lat,
            ve,
            np.full(n, 0.1),
            [lon.mean() + 15],
            [lat.mean()],
            np.array([4.0, 150.0]),
        )
        assert far.signal_sigma[0, 0] > near.signal_sigma[0, 0]

    def test_matches_collocation_on_demeaned_data(self, stations):
        # With the mean removed, OK and simple collocation should agree
        # closely away from the constraint's influence
        from geepers.collocation import (
            collocate,
            noise_covariance,
            ordinary_kriging,
            predict,
            signal_covariance,
        )

        lon, lat, ve, _ = stations
        n = len(lon)
        v = ve - ve.mean()
        sig = np.full(n, 0.3)
        params = np.array([4.0, 150.0])
        rng = np.random.default_rng(3)
        lon_i = rng.uniform(15, 17, 10)
        lat_i = rng.uniform(45, 46.5, 10)

        ok = ordinary_kriging(lon, lat, v, sig, lon_i, lat_i, params)
        Css = signal_covariance(lon, lat, lon, lat, params, components=("v",))
        _, Cinv = collocate(v, Css, noise_covariance(sig))
        Cps = signal_covariance(lon, lat, lon_i, lat_i, params, components=("v",))
        Cpp = signal_covariance(lon_i, lat_i, lon_i, lat_i, params, components=("v",))
        lsc = predict(v, Cps, Cpp, Cinv)
        np.testing.assert_allclose(ok.signal.ravel(), lsc.signal.ravel(), atol=0.3)


class TestSeparatePlates:
    @pytest.fixture
    def two_plates(self):
        import geopandas as gpd
        from shapely.geometry import box as sbox

        plates = gpd.GeoDataFrame(
            {"Code": ["A", "B"]},
            geometry=[sbox(0, 0, 5, 10), sbox(5, 0, 10, 10)],
            crs="EPSG:4326",
        )
        rng = np.random.default_rng(0)
        lon = np.r_[rng.uniform(1, 4, 30), rng.uniform(6, 9, 30)]
        lat = rng.uniform(1, 9, 60)
        return plates, lon, lat

    def test_plates_pushed_apart(self, two_plates):
        from geepers.collocation import distance_matrix, separate_plates

        plates, lon, lat = two_plates
        lon_m, lat_m, idx = separate_plates(lon, lat, plates, min_separation_km=1500)
        assert set(idx) == {0, 1}
        a, b = idx == 0, idx == 1
        gap = distance_matrix(lon_m[a], lat_m[a], lon_m[b], lat_m[b]).min()
        assert gap >= 1500

    def test_central_plate_unmoved(self, two_plates):
        from geepers.collocation import separate_plates

        plates, lon, lat = two_plates
        lon_m, lat_m, idx = separate_plates(
            lon, lat, plates, min_separation_km=1000, central_plate=0
        )
        a = idx == 0
        np.testing.assert_allclose(lon_m[a], lon[a])
        np.testing.assert_allclose(lat_m[a], lat[a])

    def test_relative_geometry_preserved_within_plate(self, two_plates):
        from geepers.collocation import distance_matrix, separate_plates

        plates, lon, lat = two_plates
        lon_m, lat_m, idx = separate_plates(lon, lat, plates, central_plate=0)
        b = idx == 1
        d_before = distance_matrix(lon[b], lat[b], lon[b], lat[b])
        d_after = distance_matrix(lon_m[b], lat_m[b], lon_m[b], lat_m[b])
        # rigid-ish translation: pairwise distances change by < 5%
        scale = d_after[d_before > 10] / d_before[d_before > 10]
        assert 0.95 < np.median(scale) < 1.05


class TestInterpolateVelocitiesRegression:
    """Golden-value regression for the collocation interpolation entry point.

    With covariance parameters supplied explicitly the whole path is
    deterministic linear algebra (no empirical estimation / optimization), so
    the interpolated signal and its uncertainty are pinned exactly. Guards
    against silent changes to the covariance assembly or the collocation solve.
    """

    # Fixed 8-station network (deg) with a smooth horizontal velocity field.
    LON = np.array([-118.0, -117.6, -117.2, -118.0, -117.2, -117.6, -117.8, -117.4])
    LAT = np.array([34.0, 34.0, 34.0, 34.6, 34.6, 34.6, 34.3, 34.3])
    EAST = np.array([2.0, 2.4, 2.8, 2.3, 3.1, 2.7, 2.45, 2.65])
    NORTH = np.array([-1.0, -0.88, -0.76, -1.12, -0.88, -1.0, -1.0, -0.94])
    PARAMS = np.array([[4.0, 40.0]])  # (C0 mm^2, d0 km)

    def _run(self):
        se = np.full(self.LON.size, 0.2)
        sn = np.full(self.LON.size, 0.2)
        lon_n = np.array([-117.6, -117.8])
        lat_n = np.array([34.3, 34.15])
        return interpolate_velocities(
            self.LON,
            self.LAT,
            self.EAST,
            self.NORTH,
            se,
            sn,
            lon_n,
            lat_n,
            covariance_parameters=self.PARAMS,
        )

    def test_interpolated_signal_and_sigma(self):
        _, at_new = self._run()
        np.testing.assert_allclose(
            np.asarray(at_new.signal),
            [[2.55293065, -0.96402871], [2.26822667, -0.94776713]],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(at_new.signal_sigma),
            [[1.28685762, 1.28685906], [1.29859423, 1.29859321]],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_deterministic(self):
        _, a = self._run()
        _, b = self._run()
        np.testing.assert_array_equal(np.asarray(a.signal), np.asarray(b.signal))

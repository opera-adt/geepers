"""Tests for Euler pole estimation (geepers.euler)."""

from __future__ import annotations

import numpy as np
import pytest

from geepers.euler import (
    EulerPole,
    estimate_euler_pole,
    local_plate_velocity,
    pole_to_rotation_vector,
    predict_plate_motion,
    rotation_vector_to_pole,
)

# ITRF2014 plate motion model (Altamimi et al., 2017), Australia:
# pole at (lon, lat) ~ (38.0, 32.5), rate ~ 0.63 deg/Myr
AUS_POLE = EulerPole(lon=38.0, lat=32.5, rate=0.63)


class TestConversions:
    def test_pole_vector_roundtrip(self):
        w = pole_to_rotation_vector(38.0, 32.5, 0.63)
        lon, lat, rate = rotation_vector_to_pole(w)
        assert lon == pytest.approx(38.0, abs=1e-9)
        assert lat == pytest.approx(32.5, abs=1e-9)
        assert rate == pytest.approx(0.63, abs=1e-12)

    def test_rate_magnitude(self):
        # 0.63 deg/Myr in rad/yr
        w = pole_to_rotation_vector(38.0, 32.5, 0.63)
        assert np.linalg.norm(w) == pytest.approx(np.radians(0.63) * 1e-6)


class TestPredict:
    def test_australia_magnitude(self):
        # Australian plate moves ~5-7 cm/yr NE in ITRF
        ve, vn = predict_plate_motion(AUS_POLE, [133.0], [-25.0])
        speed = np.hypot(ve[0], vn[0])
        assert 50 < speed < 75  # mm/yr
        assert ve[0] > 0
        assert vn[0] > 0

    def test_near_zero_at_pole(self):
        # On the ellipsoid the geodetic pole location is ~0.2 deg off the
        # rotation axis (geodetic vs geocentric latitude), so a small
        # residual velocity remains - but it must be far below plate speed
        ve, vn = predict_plate_motion(AUS_POLE, [AUS_POLE.lon], [AUS_POLE.lat])
        assert np.hypot(ve[0], vn[0]) < 0.5  # mm/yr vs ~65 plate speed


class TestEstimate:
    @pytest.fixture
    def synthetic_network(self):
        rng = np.random.default_rng(3)
        n = 40
        lon = rng.uniform(115, 150, n)
        lat = rng.uniform(-38, -12, n)
        ve, vn = predict_plate_motion(AUS_POLE, lon, lat)
        sig = np.full(n, 0.5)
        ve_obs = ve + rng.normal(0, 0.5, n)
        vn_obs = vn + rng.normal(0, 0.5, n)
        return lon, lat, ve_obs, vn_obs, sig

    def test_recovers_pole(self, synthetic_network):
        lon, lat, ve, vn, sig = synthetic_network
        pole = estimate_euler_pole(lon, lat, ve, vn, sig, sig)
        assert pole.lon == pytest.approx(AUS_POLE.lon, abs=1.0)
        assert pole.lat == pytest.approx(AUS_POLE.lat, abs=1.0)
        assert pole.rate == pytest.approx(AUS_POLE.rate, rel=0.02)

    def test_residual_stats_consistent(self, synthetic_network):
        lon, lat, ve, vn, sig = synthetic_network
        pole = estimate_euler_pole(lon, lat, ve, vn, sig, sig)
        # noise sigma == weights sigma -> reduced chi2 ~ 1, rms ~ 0.5
        assert 0.7 < pole.reduced_chi2 < 1.3
        assert 0.3 < pole.rms < 0.7
        assert pole.dof == 2 * len(lon) - 3

    def test_noise_free_exact(self):
        rng = np.random.default_rng(1)
        lon = rng.uniform(0, 40, 20)
        lat = rng.uniform(30, 60, 20)
        ve, vn = predict_plate_motion(AUS_POLE, lon, lat)
        pole = estimate_euler_pole(lon, lat, ve, vn, np.full(20, 0.1), np.full(20, 0.1))
        assert pole.lon == pytest.approx(AUS_POLE.lon, abs=1e-6)
        assert pole.lat == pytest.approx(AUS_POLE.lat, abs=1e-6)
        assert pole.rate == pytest.approx(AUS_POLE.rate, rel=1e-8)
        assert pole.rms < 1e-9

    def test_remove_restore_workflow(self, synthetic_network):
        # The plate-boundary use case: residuals after removal are just noise
        lon, lat, ve, vn, sig = synthetic_network
        pole = estimate_euler_pole(lon, lat, ve, vn, sig, sig)
        ve_plate, vn_plate = predict_plate_motion(pole, lon, lat)
        resid_e = ve - ve_plate
        resid_n = vn - vn_plate
        assert np.abs(resid_e).max() < 2.0  # mm/yr, pure noise level
        assert np.abs(np.mean(resid_e)) < 0.2
        assert np.abs(np.mean(resid_n)) < 0.2

    def test_uncertainty_finite(self, synthetic_network):
        lon, lat, ve, vn, sig = synthetic_network
        pole = estimate_euler_pole(lon, lat, ve, vn, sig, sig)
        unc = pole.uncertainty()
        assert 0 < unc["semi_major_deg"] < 10
        assert 0 < unc["semi_minor_deg"] <= unc["semi_major_deg"]
        assert 0 < unc["sigma_rate"] < 0.1

    def test_too_few_stations_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            estimate_euler_pole([1.0], [1.0], [1.0], [1.0], [0.1], [0.1])


class TestLocalPlateVelocity:
    def stations(self):
        rng = np.random.default_rng(4)
        lon, lat = rng.uniform(-2, 2, 40), rng.uniform(28, 32, 40)
        ve = 40.0 * np.cos(np.radians(lat)) / np.cos(np.radians(30.0))
        return lon, lat, ve, np.zeros(40), np.full(40, 0.5), np.full(40, 0.5)

    def test_recovers_a_rigid_rotation(self):
        ve, vn = local_plate_velocity(*self.stations(), 0.0, 30.0)
        assert ve == pytest.approx(40.0, abs=0.01)
        assert vn == pytest.approx(0.0, abs=0.01)

    def test_strain_does_not_bias_the_point(self):
        lon, lat, ve, vn, se, sn = self.stations()
        ve = ve + 3.0 * lon
        out, _ = local_plate_velocity(lon, lat, ve, vn, se, sn, 0.0, 30.0)
        assert out == pytest.approx(40.0, abs=0.1)

    def test_too_few_stations_gives_nan(self):
        assert np.isnan(local_plate_velocity(*self.stations(), 80.0, 30.0)[0])

    def test_stations_with_no_usable_rate_are_left_out(self):
        lon, lat, ve, vn, se, sn = self.stations()
        clean = local_plate_velocity(lon, lat, ve, vn, se, sn, 0.0, 30.0)
        lon, lat = np.r_[lon, 0.1], np.r_[lat, 30.1]
        ve, vn = np.r_[ve, np.nan], np.r_[vn, 1.0]
        se, sn = np.r_[se, 1.0], np.r_[sn, np.inf]
        assert local_plate_velocity(lon, lat, ve, vn, se, sn, 0.0, 30.0) == clean

    def test_breakline_excludes_the_far_plate(self):
        lon, lat, ve, vn, se, sn = self.stations()
        ve = np.where(lon < 0, -40.0, 40.0)
        fault = [np.array([(0.0, 20.0), (0.0, 40.0)])]
        out, _ = local_plate_velocity(
            lon, lat, ve, vn, se, sn, 1.0, 30.0, breaklines=fault
        )
        assert out == pytest.approx(40.0, abs=1.0)

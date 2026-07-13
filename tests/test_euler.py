"""Tests for Euler pole estimation (geepers.euler)."""

from __future__ import annotations

import numpy as np
import pytest

from geepers.euler import (
    EulerPole,
    estimate_euler_pole,
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
        assert ve[0] > 0 and vn[0] > 0  # toward NE

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
        pole = estimate_euler_pole(lon, lat, ve, vn,
                                   np.full(20, 0.1), np.full(20, 0.1))
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

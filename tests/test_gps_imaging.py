"""Tests for GPS Imaging median spatial filter (geepers.gps_imaging)."""

from __future__ import annotations

import numpy as np
import pytest

from geepers.gps_imaging import (
    great_circle_degrees,
    make_ssf,
    median_spatial_filter,
    msf_interpolate,
    weighted_median,
)


@pytest.fixture
def network():
    rng = np.random.default_rng(6)
    n = 120
    lon = rng.uniform(-120, -114, n)
    lat = rng.uniform(36, 41, n)
    # Smooth field: gentle east-west gradient (mm/yr) + small noise
    v = 2.0 * (lon + 117.0) + rng.normal(0, 0.3, n)
    sv = np.full(n, 0.5)
    return lon, lat, v, sv


class TestWeightedMedian:
    def test_equal_weights_matches_median(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=101)
        assert weighted_median(v, np.ones(101)) == pytest.approx(np.median(v))

    def test_dominant_weight_wins(self):
        v = np.array([1.0, 2.0, 100.0])
        w = np.array([0.01, 0.01, 10.0])
        assert weighted_median(v, w) == 100.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            weighted_median([1, 2], [1])


class TestGreatCircle:
    def test_one_degree(self):
        d = great_circle_degrees(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(1.0, abs=1e-9)

    def test_zero_distance(self):
        assert great_circle_degrees(45.0, 10.0, 45.0, 10.0) == 0.0


class TestMakeSSF:
    def test_shape_and_anchors(self, network):
        lon, lat, v, sv = network
        ssf = make_ssf(lon, lat, v, sv)
        assert ssf.shape[1] == 2
        assert ssf[0].tolist() == [0.0, 1.0]
        assert ssf[-1].tolist() == [180.0, 0.0]

    def test_monotonic_non_increasing(self, network):
        lon, lat, v, sv = network
        ssf = make_ssf(lon, lat, v, sv)
        vals = ssf[:, 1]
        assert np.all(np.diff(vals[np.isfinite(vals)]) <= 1e-12)

    def test_uncertainty_anchor(self, network):
        # Larger measurement noise floor -> flatter SSF near zero
        lon, lat, v, _ = network
        ssf_lo = make_ssf(lon, lat, v, np.full(len(v), 0.1))
        ssf_hi = make_ssf(lon, lat, v, np.full(len(v), 5.0))
        # with a high noise floor, short-distance bins can't beat it
        assert np.nansum(ssf_hi[1:-1, 1]) >= np.nansum(ssf_lo[1:-1, 1]) - 1e-9


class TestMedianSpatialFilter:
    def test_outlier_suppressed(self, network):
        lon, lat, v, sv = network
        v_bad = v.copy()
        v_bad[10] += 50.0  # gross outlier
        ssf = make_ssf(lon, lat, v, sv)
        filtered = median_spatial_filter(lon, lat, v_bad, sv, ssf)
        # the outlier station is pulled back toward its neighborhood
        assert abs(filtered[10] - v[10]) < 5.0
        # smooth stations barely move
        others = np.delete(np.arange(len(v)), 10)
        assert np.median(np.abs(filtered[others] - v[others])) < 1.0

    def test_plain_median_method(self, network):
        lon, lat, v, sv = network
        filtered = median_spatial_filter(lon, lat, v, sv, None, method="median")
        assert np.isfinite(filtered).all()

    def test_duplicate_coordinates_handled(self, network):
        lon, lat, v, sv = network
        # duplicate the first station with a different value
        lon2 = np.r_[lon, lon[0]]
        lat2 = np.r_[lat, lat[0]]
        v2 = np.r_[v, v[0] + 1.0]
        sv2 = np.r_[sv, sv[0]]
        filtered = median_spatial_filter(lon2, lat2, v2, sv2, None)
        assert np.isfinite(filtered).all()
        # duplicates receive identical filtered values
        assert filtered[0] == filtered[-1]

    def test_robust_network_option(self, network):
        lon, lat, v, sv = network
        ssf = make_ssf(lon, lat, v, sv)
        filtered = median_spatial_filter(lon, lat, v, sv, ssf, robust_network=True)
        assert np.isfinite(filtered).all()


class TestMSFInterpolate:
    def test_recovers_smooth_field(self, network):
        lon, lat, v, sv = network
        ssf = make_ssf(lon, lat, v, sv)
        rng = np.random.default_rng(1)
        lon_i = rng.uniform(-119, -115, 20)
        lat_i = rng.uniform(37, 40, 20)
        out = msf_interpolate(lon, lat, v, sv, lon_i, lat_i, ssf)
        assert len(out) == 20
        expected = 2.0 * (lon_i + 117.0)
        err = np.abs(out["value"].to_numpy() - expected)
        assert np.nanmedian(err) < 1.0
        assert (out["n_stations"].to_numpy() >= 3).all()
        for col in ("sigma_formal", "sigma_rms", "sigma_robust"):
            assert (out[col].dropna() >= 0).all()

    def test_at_station_location(self, network):
        lon, lat, v, sv = network
        ssf = make_ssf(lon, lat, v, sv)
        out = msf_interpolate(lon, lat, v, sv, [lon[5]], [lat[5]], ssf)
        # the coincident station contributes, estimate close to its value
        assert out["value"].iloc[0] == pytest.approx(v[5], abs=1.5)

    def test_outlier_resistant_interpolation(self, network):
        lon, lat, v, sv = network
        v_bad = v.copy()
        # find the nearest station to the eval point and corrupt it
        lon_i, lat_i = -117.0, 38.5
        d = great_circle_degrees(lat_i, lon_i, lat, lon)
        v_bad[np.argmin(d)] += 100.0
        ssf = make_ssf(lon, lat, v, sv)
        out_med = msf_interpolate(lon, lat, v_bad, sv, [lon_i], [lat_i], ssf)
        expected = 2.0 * (lon_i + 117.0)
        # weighted median shrugs off the corrupted nearest station
        assert abs(out_med["value"].iloc[0] - expected) < 3.0

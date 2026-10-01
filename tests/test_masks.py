"""Tests for interpolation-grid masks (geepers.masks)."""

from __future__ import annotations

import numpy as np
import pytest
from pyproj import Geod

from geepers.masks import convex_hull_mask, distance_mask

GEOD = Geod(ellps="WGS84")


@pytest.fixture
def stations():
    rng = np.random.default_rng(6)
    return rng.uniform(-119, -117, 40), rng.uniform(34, 36, 40)


class TestDistanceMask:
    def test_matches_brute_force_distances(self, stations):
        lon, lat = stations
        rng = np.random.default_rng(0)
        lon_new, lat_new = rng.uniform(-121, -115, 500), rng.uniform(32, 38, 500)
        mask = distance_mask(lon, lat, lon_new, lat_new, max_distance_km=60)

        nearest = (
            np.array(
                [
                    GEOD.inv(np.full(lon.size, lo), np.full(lon.size, la), lon, lat)[
                        2
                    ].min()
                    for lo, la in zip(lon_new, lat_new, strict=True)
                ]
            )
            / 1e3
        )
        # Spherical vs ellipsoidal distances differ by < 0.5%: skip the edge
        clear = np.abs(nearest - 60) > 0.5
        np.testing.assert_array_equal(mask[clear], (nearest < 60)[clear])
        assert mask.any()
        assert not mask.all()

    def test_stations_are_inside(self, stations):
        lon, lat = stations
        assert distance_mask(lon, lat, lon, lat, max_distance_km=1e-3).all()

    def test_min_stations(self):
        # Two stations 0.1 deg apart, and a lone one far away
        lon, lat = np.array([0.0, 0.1, 5.0]), np.zeros(3)
        one = distance_mask(lon, lat, lon, lat, max_distance_km=30)
        two = distance_mask(lon, lat, lon, lat, max_distance_km=30, min_stations=2)
        assert one.tolist() == [True, True, True]
        assert two.tolist() == [True, True, False]

    def test_across_the_antimeridian(self):
        mask = distance_mask([179.9], [0.0], [-179.9, 170.0], [0.0, 0.0], 50)
        assert mask.tolist() == [True, False]

    def test_keeps_grid_shape(self, stations):
        lon, lat = stations
        lon_g, lat_g = np.meshgrid(np.linspace(-120, -116, 9), np.linspace(33, 37, 7))
        assert distance_mask(lon, lat, lon_g, lat_g, 50).shape == (7, 9)


class TestConvexHullMask:
    def test_square_hull(self):
        lon = np.array([-1.0, 1.0, 1.0, -1.0])
        lat = np.array([-1.0, -1.0, 1.0, 1.0])
        lon_new = np.array([0.0, 0.9, 1.5, 0.0])
        lat_new = np.array([0.0, 0.9, 0.0, 2.0])
        mask = convex_hull_mask(lon, lat, lon_new, lat_new)
        assert mask.tolist() == [True, True, False, False]

    def test_stations_are_inside(self, stations):
        lon, lat = stations
        assert convex_hull_mask(lon, lat, lon, lat).all()

    def test_buffer_grows_and_shrinks(self):
        lon = np.array([-1.0, 1.0, 1.0, -1.0])
        lat = np.array([-1.0, -1.0, 1.0, 1.0])
        # ~56 km east of the eastern edge, and ~11 km inside it
        outside = ([1.5], [0.0])
        inside = ([0.9], [0.0])
        assert not convex_hull_mask(lon, lat, *outside)[0]
        assert convex_hull_mask(lon, lat, *outside, buffer_km=80)[0]
        assert convex_hull_mask(lon, lat, *inside)[0]
        assert not convex_hull_mask(lon, lat, *inside, buffer_km=-30)[0]

    def test_keeps_grid_shape(self, stations):
        lon, lat = stations
        lon_g, lat_g = np.meshgrid(np.linspace(-120, -116, 9), np.linspace(33, 37, 7))
        mask = convex_hull_mask(lon, lat, lon_g, lat_g)
        assert mask.shape == (7, 9)
        assert mask.any()
        assert not mask.all()

import numpy as np
import pytest

from geepers.strain import EARTH_RADIUS, strain_rate_field

DEG2M = np.deg2rad(1.0) * EARTH_RADIUS


@pytest.fixture
def grid():
    lon = np.linspace(-120, -118, 21)
    lat = np.linspace(34, 36, 21)
    return lon, lat


class TestStrainRateField:
    def test_uniform_field_zero_strain(self, grid):
        lon, lat = grid
        v = np.full((lat.size, lon.size), 0.01)
        ds = strain_rate_field(lon, lat, v, v)
        np.testing.assert_allclose(ds.second_invariant, 0, atol=1e-15)
        np.testing.assert_allclose(ds.rotation, 0, atol=1e-15)

    def test_uniaxial_extension(self, grid):
        lon, lat = grid
        # ve increasing eastward at 1e-7 /yr true strain rate
        exx_true = 1e-7
        lon2d = np.broadcast_to(lon, (lat.size, lon.size))
        cos_lat = np.cos(np.deg2rad(lat))[:, None]
        ve = exx_true * (lon2d * DEG2M * cos_lat)
        vn = np.zeros_like(ve)
        ds = strain_rate_field(lon, lat, ve, vn)
        # Interior points should recover exx (edges use one-sided diffs)
        interior = ds.exx.values[5:-5, 5:-5]
        np.testing.assert_allclose(interior, exx_true, rtol=0.05)
        assert np.abs(ds.eyy.values[5:-5, 5:-5]).max() < 0.1 * exx_true

    def test_rigid_rotation_no_strain(self, grid):
        lon, lat = grid
        omega = 1e-8  # rad/yr
        lat0, lon0 = lat.mean(), lon.mean()
        cos_lat0 = np.cos(np.deg2rad(lat0))
        x = (np.broadcast_to(lon, (lat.size, lon.size)) - lon0) * DEG2M * cos_lat0
        y = (np.broadcast_to(lat[:, None], (lat.size, lon.size)) - lat0) * DEG2M
        ve = -omega * y
        vn = omega * x
        ds = strain_rate_field(lon, lat, ve, vn)
        interior = slice(5, -5)
        np.testing.assert_allclose(
            ds.rotation.values[interior, interior], omega, rtol=0.1
        )
        assert ds.max_shear.values[interior, interior].max() < 0.1 * omega

    def test_shape_mismatch_raises(self, grid):
        lon, lat = grid
        v = np.zeros((3, 4))
        with pytest.raises(ValueError, match="shape"):
            strain_rate_field(lon, lat, v, v)

    def test_output_variables(self, grid):
        lon, lat = grid
        v = np.zeros((lat.size, lon.size))
        ds = strain_rate_field(lon, lat, v, v)
        for var in [
            "exx",
            "eyy",
            "exy",
            "rotation",
            "dilatation",
            "max_shear",
            "second_invariant",
        ]:
            assert var in ds
        assert ds.exx.dims == ("lat", "lon")

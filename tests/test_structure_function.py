"""Tests for the structure-function validation (geepers.analysis additions)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  (registers the .rio accessor)

from geepers.analysis import (
    binned_rmse_profile,
    epoch_rmse,
    pairwise_differential_rmse,
)


@pytest.fixture
def merged_network():
    """Synthetic merged GPS/InSAR tables with distance-growing misfit."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("2020-01-01", periods=60, freq="12D")
    lon0, lat0 = -117.0, 35.0
    coords, merged = {}, {}
    for k in range(8):
        lon, lat = lon0 + 0.3 * k, lat0
        gps = rng.normal(0, 0.001, len(dates))
        # InSAR = GPS + noise that grows with distance from network origin
        insar = gps + rng.normal(0, 0.001 + 0.002 * k, len(dates))
        name = f"ST{k:02d}"
        coords[name] = (lon, lat)
        merged[name] = pd.DataFrame(
            {"los_gps": gps, "los_insar": insar}, index=dates
        )
    return merged, coords


class TestPairwiseRMSE:
    def test_all_pairs_present(self, merged_network):
        merged, coords = merged_network
        df = pairwise_differential_rmse(merged, coords)
        n = len(merged)
        assert len(df) == n * (n - 1) // 2
        assert (df["n_dates"] == 60).all()
        assert (df["rmse"] > 0).all()

    def test_distances_correct(self, merged_network):
        merged, coords = merged_network
        df = pairwise_differential_rmse(merged, coords)
        # ST00 and ST01 are 0.3 deg apart at 35N: ~27.3 km
        row = df[(df.station1 == "ST00") & (df.station2 == "ST01")].iloc[0]
        assert row.distance_km == pytest.approx(27.3, abs=0.5)

    def test_reference_datum_cancels(self, merged_network):
        # Adding a common time-dependent shift to every InSAR series
        # (a reference-frame wobble) must not change the pairwise RMSE
        merged, coords = merged_network
        base = pairwise_differential_rmse(merged, coords)
        wobble = np.sin(np.arange(60))
        shifted = {
            k: df.assign(los_insar=df["los_insar"] + wobble * 0.01)
            for k, df in merged.items()
        }
        after = pairwise_differential_rmse(shifted, coords)
        np.testing.assert_allclose(base["rmse"], after["rmse"], atol=1e-12)

    def test_min_common_dates(self, merged_network):
        merged, coords = merged_network
        # Cripple one station to 2 valid epochs
        merged["ST00"] = merged["ST00"].iloc[:2]
        df = pairwise_differential_rmse(merged, coords, min_common_dates=3)
        assert not (df[["station1", "station2"]] == "ST00").any().any()

    def test_empty_when_no_overlap(self):
        d1 = pd.DataFrame(
            {"los_gps": [0.0] * 5, "los_insar": [0.0] * 5},
            index=pd.date_range("2020-01-01", periods=5),
        )
        d2 = pd.DataFrame(
            {"los_gps": [0.0] * 5, "los_insar": [0.0] * 5},
            index=pd.date_range("2021-01-01", periods=5),
        )
        out = pairwise_differential_rmse(
            {"A": d1, "B": d2}, {"A": (0, 0), "B": (1, 0)}
        )
        assert out.empty


class TestBinnedProfile:
    def test_bins_and_requirement(self, merged_network):
        merged, coords = merged_network
        df = pairwise_differential_rmse(merged, coords)
        req = lambda d: (3 + 0.5 * np.sqrt(d)) / 1000  # noqa: E731
        prof = binned_rmse_profile(df, n_bins=5, requirement=req)
        assert 1 <= len(prof) <= 5
        assert (prof["n_pairs"] >= 1).all()
        assert prof["n_pairs"].sum() == len(df)
        assert {"requirement", "fraction_passing"} <= set(prof.columns)
        assert prof["fraction_passing"].between(0, 1).all()

    def test_empty_input(self):
        assert binned_rmse_profile(pd.DataFrame()).empty


class TestEpochRMSE:
    def test_flags_bad_epoch(self, merged_network):
        merged, _ = merged_network
        # corrupt one acquisition across all stations *differently*
        bad_date = next(iter(merged.values())).index[30]
        for k, df in merged.items():
            df.loc[bad_date, "los_insar"] += np.random.default_rng(
                hash(k) % 2**32
            ).normal(0, 0.05)
        out = epoch_rmse(merged)
        assert out.loc[bad_date, "rmse"] > 3 * out["rmse"].median()

    def test_datum_shift_invariant(self, merged_network):
        merged, _ = merged_network
        base = epoch_rmse(merged)
        shifted = {
            k: df.assign(los_insar=df["los_insar"] + 5.0)
            for k, df in merged.items()
        }
        after = epoch_rmse(shifted)
        np.testing.assert_allclose(base["rmse"], after["rmse"], atol=1e-12)

    def test_min_stations(self, merged_network):
        merged, _ = merged_network
        two = {k: merged[k] for k in list(merged)[:2]}
        assert epoch_rmse(two, min_stations=3).empty


class TestBufferMeters:
    """Metric circular buffering in XarrayReader.read_window."""

    @pytest.fixture
    def utm_reader(self):
        import xarray as xr

        from geepers.io import XarrayReader

        # 100x100 grid at 30 m spacing in a UTM CRS, value = column index
        n = 100
        x = 500_000 + 30.0 * np.arange(n)
        y = 4_000_000 - 30.0 * np.arange(n)
        data = np.tile(np.arange(n, dtype=float), (n, 1))
        da = xr.DataArray(
            data,
            coords={"y": y, "x": x},
            dims=["y", "x"],
            attrs={"units": "meters"},
        )
        da.rio.write_crs("EPSG:32611", inplace=True)
        return XarrayReader(da)

    def _lonlat_of_center(self, reader):
        from pyproj import Transformer

        t = Transformer.from_crs(reader.crs, "EPSG:4326", always_xy=True)
        cx = float(reader.da.x[50])
        cy = float(reader.da.y[50])
        lon, lat = t.transform(cx, cy)
        return lon, lat

    def test_circular_footprint_pixel_count(self, utm_reader):
        lon, lat = self._lonlat_of_center(utm_reader)
        (win,) = utm_reader.read_window([lon], [lat], buffer_meters=100.0)
        n_valid = int(np.isfinite(win.values).sum())
        # 100 m radius on a 30 m grid: circle of ~pi*(100/30)^2 = 35 cells;
        # the enclosing 7x7 box would be 49 - the mask must cut corners
        assert 25 <= n_valid <= 40
        assert n_valid < win.size

    def test_median_matches_local_value(self, utm_reader):
        lon, lat = self._lonlat_of_center(utm_reader)
        (win,) = utm_reader.read_window([lon], [lat], buffer_meters=100.0)
        med = float(np.nanmedian(win.values))
        # data = column index; center is column 50
        assert med == pytest.approx(50.0, abs=1.0)

    def test_zero_buffer_still_single_pixel(self, utm_reader):
        lon, lat = self._lonlat_of_center(utm_reader)
        (win,) = utm_reader.read_window([lon], [lat], buffer_pixels=0)
        assert win.size == 1

    def test_geographic_crs_cos_lat_correction(self):
        import xarray as xr

        from geepers.io import XarrayReader

        # 0.001 deg grid at 60N: 1 deg lon ~ 55.8 km, 1 deg lat ~ 111.3 km
        n = 60
        lon = -120.0 + 0.001 * np.arange(n)
        lat = 60.03 - 0.001 * np.arange(n)
        da = xr.DataArray(
            np.random.default_rng(0).normal(size=(n, n)),
            coords={"y": lat, "x": lon},
            dims=["y", "x"],
            attrs={"units": "meters"},
        )
        da.rio.write_crs("EPSG:4326", inplace=True)
        reader = XarrayReader(da)
        (win,) = reader.read_window(
            [float(lon[30])], [float(lat[30])], buffer_meters=300.0
        )
        valid = np.isfinite(win.values)
        # at 60N the lon radius spans ~2x more cells than the lat radius
        rows_covered = valid.any(axis=1).sum()
        cols_covered = valid.any(axis=0).sum()
        assert cols_covered > rows_covered

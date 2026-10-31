"""The GNSS-retrieval core imports and works without the heavy extras.

The operational DISP-CAL image installs geepers with no extras. These tests
pin that contract: importing the package and `gps_sources` must not load
dask, zarr, pandera, rasterio, geopandas, xarray or matplotlib (checked in a
subprocess so this test file's own imports cannot mask a leak), and the
optional-dependency seams must behave the same with and without the extras.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from geepers import _optional
from geepers.gps_sources.base import BaseGpsSource

HEAVY = {"dask", "zarr", "pandera", "rasterio", "geopandas", "xarray", "matplotlib"}


def test_core_import_does_not_load_heavy_modules():
    code = (
        "import sys, geepers, geepers.gps_sources, geepers.constants, geepers.utils\n"
        "from geepers.gps_sources import UnrGridSource, UnrSource, SideshowSource\n"
        f"print(sorted({HEAVY!r} & set(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", f"core import loaded: {out.stdout}"


def test_all_lists_exactly_the_lazy_exports():
    import geepers

    assert set(geepers.__all__) == {"__version__", *geepers._LAZY_EXPORTS}


def test_lazy_export_resolves_and_caches():
    pytest.importorskip("xarray")  # geepers.io is an [analysis] module
    import geepers

    assert "XarrayReader" in dir(geepers)
    reader = geepers.XarrayReader  # triggers geepers.io
    assert reader.__name__ == "XarrayReader"
    assert geepers.__dict__["XarrayReader"] is reader  # cached
    with pytest.raises(AttributeError, match="no attribute"):
        _ = geepers.not_a_real_name


def test_lazy_export_without_extra_raises_import_error(monkeypatch):
    """Without the extra, touching an analysis export fails at that point, not at import."""
    import geepers

    monkeypatch.setattr(geepers.importlib, "import_module", _raise_import_error)
    geepers.__dict__.pop(
        "XarrayReader", None
    )  # drop the cache if an earlier test filled it
    with pytest.raises(ImportError):
        _ = geepers.XarrayReader


class TestOptionalSeams:
    def test_validate_without_pandera_returns_frame_with_units(self, monkeypatch):
        monkeypatch.setattr(_optional, "has_module", lambda _name: False)
        df = pd.DataFrame({"east": [0.0]})
        out = _optional.validate(df, "StationObservationSchema", lazy=True)
        assert out is df
        assert out.attrs["units"] == "meters"

    def test_validate_with_pandera_validates(self):
        pytest.importorskip("pandera")
        import pandera.errors

        bad = pd.DataFrame({"lat": [95.0], "lon": [0.0], "alt": [0.0]})
        with pytest.raises(pandera.errors.SchemaErrors):
            _optional.validate(bad, "PointSchema", lazy=True)

    def test_to_point_frame_plain_without_geopandas(self, monkeypatch):
        monkeypatch.setattr(_optional, "has_module", lambda _name: False)
        df = pd.DataFrame({"lon": [1.0], "lat": [2.0]})
        out = _optional.to_point_frame(df)
        assert type(out) is pd.DataFrame
        assert list(out.columns) == ["lon", "lat"]

    def test_to_point_frame_geo_with_geopandas(self):
        gpd = pytest.importorskip("geopandas")
        df = pd.DataFrame({"lon": [1.0], "lat": [2.0]})
        out = _optional.to_point_frame(df)
        assert isinstance(out, gpd.GeoDataFrame)
        assert out.crs.to_epsg() == 4326
        assert out.geometry.iloc[0].x == 1.0

    def test_require_names_the_extra(self, monkeypatch):
        monkeypatch.setattr(_optional.importlib, "import_module", _raise_import_error)
        with pytest.raises(ImportError, match=r"geepers\[analysis\]"):
            _optional.require("geopandas")
        with pytest.raises(ImportError, match=r"geepers\[grid\]"):
            _optional.require("shapely")
        # unknown optional names point at the everything extra
        with pytest.raises(ImportError, match=r"geepers\[all\]"):
            _optional.require("some_other_package")


def _raise_import_error(name):
    raise ImportError(name)


class _FakeSource(BaseGpsSource):
    """Three stations as a plain DataFrame: what the core sees without geopandas."""

    def timeseries(self, station_id, /, **kwargs):  # pragma: no cover - not used
        raise NotImplementedError

    def _read_station_data(self):
        return pd.DataFrame(
            {
                "id": ["AAAA", "BBBB", "CCCC"],
                "lon": [-120.0, -118.0, 10.0],
                "lat": [34.0, 36.0, 50.0],
                "alt": [0.0, 1.0, 2.0],
            }
        )


class TestBboxFilterOnPlainFrame:
    def test_bbox_uses_lon_lat_columns(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_optional, "has_module", lambda _name: False)
        src = _FakeSource(cache_dir=tmp_path)
        out = src.stations(bbox=(-121, 33, -117, 37))
        assert list(out["id"]) == ["AAAA", "BBBB"]
        assert list(out.index) == [0, 1]  # reset after filtering

    def test_mask_without_geopandas_raises_with_hint(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_optional, "has_module", lambda _name: False)
        monkeypatch.setattr(_optional.importlib, "import_module", _raise_import_error)
        src = _FakeSource(cache_dir=tmp_path)
        with pytest.raises(ImportError, match=r"geepers\[analysis\]"):
            src.stations(mask=object())

    def test_coordinates_work_on_plain_frame(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_optional, "has_module", lambda _name: False)
        src = _FakeSource(cache_dir=tmp_path)
        assert src.coordinates("bbbb") == (-118.0, 36.0, 1.0)
        assert np.isclose(src.coordinates("AAAA")[1], 34.0)

"""Tests for UNR Grid GPS data source."""

from __future__ import annotations

import numpy as np

from geepers.gps_sources.unr_grid import UnrGridSource
from geepers.schemas import EPS


class TestUnrGridSource:
    """Tests for UnrGridSource class."""

    def test_init(self):
        """Test UnrGridSource initialization."""
        source = UnrGridSource()
        assert isinstance(source, UnrGridSource)


class TestParseDataFile:
    """Regression tests for the .tenv8 parser.

    Locks the contract that later refactors must preserve: unit conversion
    (UNR grid files are in millimeters), placeholder correlations, datetime
    derivation, and the zero-sigma clamp that keeps the schema happy for the
    time-constant gridded products.
    """

    # decimal_year east north up sig_e sig_n sig_u rapid_flag  (mm)
    TENV8 = (
        "2020.0000  12.0  -8.0   3.0   1.0  1.5  2.0  0\n"
        "2020.5000  15.0  -6.0   5.0   1.2  1.4  2.1  0\n"
        "2021.0000  18.0  -4.0   7.0   0.0  0.0  0.0  1\n"
    )

    def _write(self, tmp_path):
        p = tmp_path / "000007_NA.tenv8"
        p.write_text(self.TENV8)
        return p

    def test_columns_and_units(self, tmp_path):
        df = UnrGridSource().parse_data_file(self._write(tmp_path))
        assert list(df.columns) == [
            "date",
            "east",
            "north",
            "up",
            "sigma_east",
            "sigma_north",
            "sigma_up",
            "corr_en",
            "corr_eu",
            "corr_nu",
        ]
        # millimeters -> meters for values and sigmas
        np.testing.assert_allclose(df["east"].to_numpy(), [0.012, 0.015, 0.018])
        np.testing.assert_allclose(df["north"].to_numpy(), [-0.008, -0.006, -0.004])
        np.testing.assert_allclose(df["up"].to_numpy(), [0.003, 0.005, 0.007])
        np.testing.assert_allclose(df["sigma_north"].to_numpy(), [0.0015, 0.0014, EPS])

    def test_zero_sigma_clamped_to_eps(self, tmp_path):
        # The last row has exactly-zero sigmas (time-constant products);
        # they must be clamped to the schema minimum, not left at 0.
        df = UnrGridSource().parse_data_file(self._write(tmp_path))
        last = df.iloc[-1]
        assert last["sigma_east"] == EPS
        assert last["sigma_north"] == EPS
        assert last["sigma_up"] == EPS

    def test_placeholder_correlations_are_zero(self, tmp_path):
        df = UnrGridSource().parse_data_file(self._write(tmp_path))
        assert (df[["corr_en", "corr_eu", "corr_nu"]] == 0.0).all().all()

    def test_dates_from_decimal_year(self, tmp_path):
        df = UnrGridSource().parse_data_file(self._write(tmp_path))
        got = [d.date().isoformat() for d in df["date"]]
        assert got == ["2020-01-01", "2020-07-02", "2020-12-31"]

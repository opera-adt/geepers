import numpy as np
import pandas as pd
import pytest

from geepers.synthetic import (
    SyntheticStep,
    power_law_noise,
    synthetic_network_timeseries,
    synthetic_stations,
    synthetic_timeseries,
)


@pytest.fixture
def dates():
    return pd.date_range("2020-01-01", periods=730, freq="D")


class TestPowerLawNoise:
    def test_shape_and_reproducibility(self):
        n1 = power_law_noise(500, spectral_index=-1, seed=42)
        n2 = power_law_noise(500, spectral_index=-1, seed=42)
        assert n1.shape == (500,)
        np.testing.assert_array_equal(n1, n2)

    def test_white_noise_case(self):
        n = power_law_noise(10_000, spectral_index=0, scale=2.0, seed=1)
        assert abs(n.std() - 2.0) < 0.1

    def test_colored_noise_grows(self):
        # Random walk variance grows with time; white does not
        rw = power_law_noise(2000, spectral_index=-2, seed=1)
        white = power_law_noise(2000, spectral_index=0, seed=1)
        assert rw[1000:].var() > 5 * white[1000:].var()


class TestSyntheticTimeseries:
    def test_schema_columns(self, dates):
        df = synthetic_timeseries(dates, seed=0)
        expected = {
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
        }
        assert expected <= set(df.columns)
        assert len(df) == len(dates)

    def test_velocity_recovered(self, dates):
        ve = 0.02  # 2 cm/yr
        df = synthetic_timeseries(
            dates, velocity_enu=(ve, 0, 0), white_sigma=0.001, seed=1
        )
        t_years = (df.date - df.date.iloc[0]).dt.days / 365.25
        slope = np.polyfit(t_years, df.east, 1)[0]
        assert abs(slope - ve) < 0.002

    def test_step_applied(self, dates):
        step_date = dates[365]
        df = synthetic_timeseries(
            dates,
            steps=[SyntheticStep(date=step_date, up=0.05)],
            white_sigma=0.0001,
            seed=2,
        )
        before = df[df.date < step_date].up.mean()
        after = df[df.date >= step_date].up.mean()
        assert 0.04 < after - before < 0.06


class TestSyntheticStations:
    def test_within_bbox(self):
        bbox = (-120.0, 34.0, -118.0, 36.0)
        gdf = synthetic_stations(20, bbox=bbox, seed=3)
        assert len(gdf) == 20
        assert gdf.lon.between(bbox[0], bbox[2]).all()
        assert gdf.lat.between(bbox[1], bbox[3]).all()
        assert gdf.id.is_unique


class TestSyntheticNetwork:
    def test_network_common_mode(self, dates):
        net = synthetic_network_timeseries(
            n_stations=5,
            dates=dates,
            common_mode_sigma=0.005,
            white_sigma=0.001,
            seed=4,
        )
        assert len(net.observations) == 5
        # Every station contains the same injected common mode: the
        # correlation of two stations' east series should be high
        ids = list(net.observations)
        east0 = net.observations[ids[0]].east
        east1 = net.observations[ids[1]].east
        assert np.corrcoef(east0, east1)[0, 1] > 0.8

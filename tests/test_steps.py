import numpy as np
import pandas as pd
import pytest

from geepers.steps import detect_steps, detect_steps_enu
from geepers.synthetic import SyntheticStep, synthetic_timeseries


@pytest.fixture
def dates():
    return pd.date_range("2020-01-01", periods=730, freq="D")


def _series(dates, values):
    return pd.Series(values, index=dates)


class TestDetectSteps:
    def test_finds_known_step(self, dates):
        rng = np.random.default_rng(1)
        step_date = dates[400]
        values = rng.normal(scale=0.001, size=len(dates))
        values[400:] += 0.02
        found = detect_steps(_series(dates, values))
        assert len(found) == 1
        assert abs((found.date.iloc[0] - step_date).days) <= 2
        assert 0.015 < found.step_size.iloc[0] < 0.025

    def test_no_false_positives_on_clean_trend(self, dates):
        rng = np.random.default_rng(2)
        t = np.arange(len(dates))
        values = 1e-5 * t + rng.normal(scale=0.001, size=len(dates))
        found = detect_steps(_series(dates, values))
        assert found.empty

    def test_short_series_returns_empty(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        found = detect_steps(_series(dates, np.zeros(5)))
        assert found.empty
        assert list(found.columns) == ["date", "step_size", "delta_aic"]

    def test_two_separated_steps(self, dates):
        rng = np.random.default_rng(3)
        values = rng.normal(scale=0.001, size=len(dates))
        values[200:] += 0.03
        values[500:] -= 0.04
        found = detect_steps(_series(dates, values))
        assert len(found) == 2
        assert found.step_size.iloc[0] > 0
        assert found.step_size.iloc[1] < 0

    def test_handles_nans(self, dates):
        rng = np.random.default_rng(4)
        values = rng.normal(scale=0.001, size=len(dates))
        values[300:] += 0.02
        values[::7] = np.nan
        found = detect_steps(_series(dates, values))
        assert len(found) == 1


class TestDetectStepsEnu:
    def test_component_column(self, dates):
        df = synthetic_timeseries(
            dates,
            steps=[SyntheticStep(date=dates[365], up=0.05)],
            white_sigma=0.001,
            seed=5,
        )
        found = detect_steps_enu(df)
        assert "component" in found.columns
        up_steps = found[found.component == "up"]
        assert len(up_steps) == 1
        assert abs((up_steps.date.iloc[0] - dates[365]).days) <= 2

import numpy as np
import pandas as pd
import pytest

from geepers.steps import detect_steps, detect_steps_enu, white_noise_sigma
from geepers.synthetic import SyntheticStep, power_law_noise, synthetic_timeseries


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


class TestMinimumStepSize:
    @pytest.fixture
    def colored(self):
        """Ten years of daily white + flicker noise (mm) with no steps."""
        dates = pd.date_range("2014-01-01", periods=3650, freq="D")
        series = []
        for seed in range(4):
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, 2, 3650) + power_law_noise(3650, -1, 1.5, seed=seed)
            series.append(_series(dates, noise))
        return series

    def test_white_noise_sigma(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0, 2.0, 5000) + 0.01 * np.arange(5000)
        values[2500:] += 40.0  # a step must not inflate the estimate
        assert white_noise_sigma(values) == pytest.approx(2.0, rel=0.05)

    def test_colored_noise_is_not_reported_as_steps(self, colored):
        # Regression: the white-noise AIC test alone reports the wander of
        # flicker noise as steps, increasingly so for longer windows
        for window in (60, 300):
            aic_only = sum(
                len(detect_steps(s, window_days=window, min_step_sigma=0))
                for s in colored
            )
            sized = sum(len(detect_steps(s, window_days=window)) for s in colored)
            assert aic_only >= 4
            assert sized <= 0.25 * aic_only

    def test_real_step_survives_in_colored_noise(self, colored):
        for series in colored:
            with_step = series.copy()
            with_step.iloc[1800:] += 15.0
            found = detect_steps(with_step)
            near = found[(found.date - series.index[1800]).abs().dt.days <= 10]
            assert len(near) == 1
            assert near.step_size.iloc[0] == pytest.approx(15.0, abs=5.0)

    def test_step_below_the_size_limit_is_dropped(self, dates):
        rng = np.random.default_rng(6)
        values = rng.normal(scale=1.0, size=len(dates))
        values[400:] += 2.5  # significant by AIC in a long window, below 3 sigma
        series = _series(dates, values)
        assert len(detect_steps(series, window_days=200, min_step_sigma=0)) == 1
        assert detect_steps(series, window_days=200).empty


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

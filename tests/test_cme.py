import numpy as np
import pandas as pd
import pytest

from geepers.cme import estimate_cme, remove_cme
from geepers.synthetic import synthetic_network_timeseries


@pytest.fixture
def network_residuals():
    """Residual matrix (date x station) with a strong injected common mode."""
    net = synthetic_network_timeseries(
        n_stations=8,
        common_mode_sigma=0.005,
        white_sigma=0.001,
        seed=10,
    )
    east = {sid: df.set_index("date").east for sid, df in net.observations.items()}
    return pd.DataFrame(east), net.common_mode["east"]


class TestEstimateCme:
    def test_recovers_injected_signal(self, network_residuals):
        residuals, injected = network_residuals
        result = estimate_cme(residuals)
        corr = np.corrcoef(result.common_mode.cme_0, injected)[0, 1]
        assert abs(corr) > 0.95
        assert result.explained_variance[0] > 0.5

    def test_cleaning_reduces_scatter(self, network_residuals):
        residuals, _ = network_residuals
        cleaned = remove_cme(residuals)
        assert cleaned.std().mean() < 0.5 * residuals.std().mean()
        assert cleaned.shape == residuals.shape

    def test_preserves_nans(self, network_residuals):
        residuals, _ = network_residuals
        residuals.iloc[10:20, 0] = np.nan
        cleaned = remove_cme(residuals)
        assert cleaned.iloc[10:20, 0].isna().all()

    def test_coverage_filter(self, network_residuals):
        residuals, _ = network_residuals
        # Station 0 mostly missing: excluded from the decomposition but
        # still present (untouched) in the cleaned output
        residuals.iloc[: int(0.8 * len(residuals)), 0] = np.nan
        result = estimate_cme(residuals, min_coverage=0.5)
        assert residuals.columns[0] not in result.spatial_response.index
        assert residuals.columns[0] in result.cleaned.columns

    def test_too_few_stations_raises(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        residuals = pd.DataFrame(
            np.random.default_rng(0).normal(size=(100, 2)),
            index=dates,
            columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="at least 3 stations"):
            estimate_cme(residuals)

    def test_unknown_method_raises(self, network_residuals):
        residuals, _ = network_residuals
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_cme(residuals, method="magic")

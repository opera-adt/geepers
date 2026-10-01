"""End-to-end checks of the workflows the tutorials and how-to guides teach.

Each test chains several modules the way the documentation does, so a change
that breaks the hand-off between them fails here even if every module's own
tests still pass.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geepers.collocation import collocate, noise_covariance, predict, signal_covariance
from geepers.cross_validation import cross_validate
from geepers.linearity import linearity_test, validity_horizon
from geepers.masks import convex_hull_mask, distance_mask
from geepers.spline import fit_vector_spline
from geepers.steps import clean_step_dates, detect_steps
from geepers.strain import strain_rate_field
from geepers.surface import fit_polynomial_surface
from geepers.synthetic import power_law_noise

NOTEBOOKS = Path(__file__).parents[1] / "docs" / "notebooks"


@pytest.fixture(scope="module")
def station():
    """Ten years of daily positions (mm): -3 mm/yr, annual, flicker + white."""
    n = 3650
    dates = pd.date_range("2014-01-01", periods=n, freq="D")
    t = np.arange(n) / 365.25
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 2.0, n) + 1.5 * power_law_noise(n, -1, seed=7)
    return dates, t, -3.0 * t + 3.0 * np.sin(2 * np.pi * t) + noise


def _monthly(series: pd.Series) -> pd.Series:
    return series.resample("30D").mean().dropna()


def _network(n: int = 150):
    """Scattered stations over southern California, and the generator used."""
    rng = np.random.default_rng(31)
    return rng.uniform(-120, -116, n), rng.uniform(33, 37, n), rng


class TestStepsThenLinearity:
    """The steps-and-linearity tutorial: find offsets, then test linearity."""

    def test_detected_step_restores_a_linear_verdict(self, station):
        dates, _, steady = station
        step_date = pd.Timestamp("2016-06-10")
        series = pd.Series(steady + 20.0 * (dates >= step_date), index=dates)

        found = detect_steps(series)
        assert len(found) == 1
        assert abs((found["date"].iloc[0] - step_date).days) <= 2

        monthly = _monthly(series)
        undeclared = linearity_test(monthly.index, monthly.to_numpy(), sampling_days=30)
        assert undeclared.model != "linear"
        assert abs(undeclared.trend.velocity + 3.0) > 1.5  # badly biased rate

        # A catalog would add entries the fit cannot use as they come
        catalog = [pd.Timestamp("2011-03-11"), step_date + pd.Timedelta(days=2)]
        steps = clean_step_dates(
            found["date"].tolist() + catalog,
            start=monthly.index[0],
            end=monthly.index[-1],
            min_separation_days=30,
        )
        assert len(steps) == 1
        declared = linearity_test(
            monthly.index, monthly.to_numpy(), sampling_days=30, step_dates=steps
        )
        assert declared.model == "linear"
        sigma = declared.trend.velocity_uncertainty
        assert declared.trend.velocity == pytest.approx(-3.0, abs=3 * sigma)
        offset, offset_sigma = declared.trend.parameters["step_0"]
        assert offset == pytest.approx(20.0, abs=3 * offset_sigma)

    def test_mid_record_step_is_not_flagged_but_biases_the_rate(self, station):
        # The blind spot the tutorial warns about: this is why steps are
        # detected first rather than left to the linearity test
        dates, _, steady = station
        series = pd.Series(steady + 12.0 * (dates >= "2019-03-10"), index=dates)
        monthly = _monthly(series)
        result = linearity_test(monthly.index, monthly.to_numpy(), sampling_days=30)
        assert result.model == "linear"
        bias = abs(result.trend.velocity + 3.0)
        assert bias > 3 * result.trend.velocity_uncertainty
        # ... while the detector does find it
        assert len(detect_steps(series)) == 1

    def test_non_linear_series_get_short_horizons(self, station):
        dates, t, steady = station
        horizons = {}
        for name, values in (
            ("steady", steady),
            ("rate change", steady - 3.0 * np.maximum(0, t - 5.2)),
            ("accelerating", steady - 0.3 * (t - t.mean()) ** 2),
        ):
            monthly = _monthly(pd.Series(values, index=dates))
            result = linearity_test(
                monthly.index, monthly.to_numpy(), sampling_days=30, n_simulations=300
            )
            assert (result.model == "linear") == (name == "steady")
            horizons[name] = validity_horizon(result, tolerance=10.0)
        assert horizons["steady"].driver == "rate uncertainty"
        assert horizons["steady"].years > 20
        for name in ("rate change", "accelerating"):
            assert horizons[name].driver == "non-linearity"
            assert horizons[name].years < 0.5 * horizons["steady"].years


class TestGriddingWorkflow:
    """The interpolation how-to: remove-restore, interpolate, mask, score."""

    def test_remove_restore_collocation_with_mask(self):
        lon, lat, rng = _network()
        parameters = np.array([1.0, 60.0])
        lon_g, lat_g = np.meshgrid(np.linspace(-122, -114, 17), np.linspace(31, 39, 17))
        lon_all = np.r_[lon, lon_g.ravel()]
        lat_all = np.r_[lat, lat_g.ravel()]
        n = lon.size

        C = signal_covariance(
            lon_all, lat_all, lon_all, lat_all, parameters, components=("up",)
        )
        signal = np.linalg.cholesky(C + 1e-9 * np.eye(len(lon_all))) @ rng.normal(
            size=len(lon_all)
        )
        truth = signal + 6.0 + 1.5 * (lon_all + 118) - 1.0 * (lat_all - 35)
        sigma = np.full(n, 0.2)
        observed = truth[:n] + rng.normal(0, 0.2, n)

        surface = fit_polynomial_surface(lon, lat, observed, sigma, degree=1)
        Css = signal_covariance(lon, lat, lon, lat, parameters, components=("up",))
        _, Czz_inv = collocate(surface.residuals, Css, noise_covariance(sigma))
        Cps = signal_covariance(
            lon, lat, lon_g.ravel(), lat_g.ravel(), parameters, components=("up",)
        )
        Cpp = signal_covariance(
            lon_g.ravel(),
            lat_g.ravel(),
            lon_g.ravel(),
            lat_g.ravel(),
            parameters,
            components=("up",),
        )
        grid = predict(surface.residuals, Cps, Cpp, Czz_inv).signal[:, 0].reshape(
            lon_g.shape
        ) + surface.predict(lon_g, lat_g)

        keep = distance_mask(lon, lat, lon_g, lat_g, 60) & convex_hull_mask(
            lon, lat, lon_g, lat_g
        )
        error = np.abs(grid - truth[n:].reshape(lon_g.shape))
        assert 0.1 < keep.mean() < 0.6
        # The mask keeps the part of the grid the stations constrain
        assert error[keep].mean() < 0.7 * error[~keep].mean()
        assert np.sqrt(np.mean(error[keep] ** 2)) < 1.0

    def test_spline_scored_by_blocked_cross_validation_then_strain(self):
        lon, lat, _ = _network()
        # A smooth horizontal field with a known uniform strain rate
        exx = 2e-8  # 20 nanostrain/yr of east-west extension
        east = exx * np.radians(lon + 118) * 6_371_000.0 * np.cos(np.radians(35.0))
        north = np.zeros_like(east)
        values = np.c_[east, north] * 1000  # mm/yr

        def predict_fold(train, test):
            fit = fit_vector_spline(
                lon[train], lat[train], values[train, 0], values[train, 1]
            )
            return np.c_[fit.predict(lon[test], lat[test])]

        cv = cross_validate(predict_fold, lon, lat, values, block_km=80)
        assert cv.rmse.shape == (2,)
        assert cv.rmse.max() < 0.1 * np.ptp(values[:, 0])

        spline = fit_vector_spline(lon, lat, values[:, 0], values[:, 1])
        lon_1d, lat_1d = np.linspace(-119, -117, 21), np.linspace(34, 36, 21)
        ve, vn = spline.predict(*np.meshgrid(lon_1d, lat_1d))
        strain = strain_rate_field(lon_1d, lat_1d, ve / 1000, vn / 1000)
        interior = strain.exx.values[5:-5, 5:-5]
        assert np.median(interior) == pytest.approx(exx, rel=0.15)


class TestTutorialNotebooks:
    """The rendered notebooks are stored with outputs: they must be clean."""

    @pytest.mark.parametrize(
        "path", sorted(NOTEBOOKS.glob("*.ipynb")), ids=lambda p: p.stem
    )
    def test_notebook_is_executed_and_error_free(self, path):
        notebook = json.loads(path.read_text())
        code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        assert any(c.get("outputs") for c in code), "notebook has no stored outputs"
        for cell in code:
            assert not any(
                o.get("output_type") == "error" for o in cell.get("outputs", [])
            )
            source = "".join(cell["source"])
            # IPython magics are not Python; everything else must parse
            ast.parse(
                "\n".join(ln for ln in source.splitlines() if not ln.startswith("%"))
            )

    def test_steps_notebook_uses_the_public_api(self):
        source = (NOTEBOOKS / "steps_and_linearity.ipynb").read_text()
        for name in (
            "detect_steps",
            "clean_step_dates",
            "linearity_test",
            "validity_horizon",
        ):
            assert name in source

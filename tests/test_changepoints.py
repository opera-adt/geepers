"""Tests for change-point detection and sampling (geepers.changepoints)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from geepers.changepoints import (
    ChangepointResult,
    detect_changepoints,
    fit_sparse_changepoints,
    fit_sparse_changepoints_many,
    sample_changepoints,
)
from geepers.linearity import _noise_covariance
from geepers.synthetic import power_law_noise
from geepers.trend import DAYS_PER_YEAR, TrendResult, estimate_trend

MONTH = 365.25 / 12
OFFSET_AT, RATE_AT = 3.0, 6.5  # years after the first epoch


def _monthly(
    offset: float = 8.0, rate_change: float = -4.0, seed: int = 4, n: int = 120
):
    """Ten years of monthly data (mm): -3 mm/yr, flicker + white, annual."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 12
    values = (
        -3.0 * t
        + 0.7 * power_law_noise(n, -1.0, 1.0, seed=seed)
        + rng.normal(0, 1.0, n)
        + 3.0 * np.sin(2 * np.pi * t)
        + offset * (t >= OFFSET_AT)
        + rate_change * np.maximum(0.0, t - RATE_AT)
    )
    dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(
        np.round(np.arange(n) * MONTH), unit="D"
    )
    return dates, values


def _years(dates, date) -> float:
    return (pd.Timestamp(date) - dates[0]).days / DAYS_PER_YEAR


@pytest.fixture(scope="module")
def offset_and_rate_change() -> tuple[pd.DatetimeIndex, np.ndarray, ChangepointResult]:
    dates, values = _monthly()
    return dates, values, detect_changepoints(dates, values, sampling_days=MONTH)


class TestDetectChangepoints:
    def test_finds_the_offset_and_the_rate_change(self, offset_and_rate_change):
        dates, _, found = offset_and_rate_change
        cps = found.changepoints
        assert list(cps["kind"]) == ["offset", "rate"]
        offset, rate = cps.iloc[0], cps.iloc[1]
        assert _years(dates, offset["date"]) == pytest.approx(OFFSET_AT, abs=0.2)
        assert _years(dates, rate["date"]) == pytest.approx(RATE_AT, abs=0.5)
        assert offset["offset"] == pytest.approx(8.0, abs=3 * offset["offset_sigma"])
        assert rate["rate_change"] == pytest.approx(
            -4.0, abs=3 * rate["rate_change_sigma"]
        )
        assert found.rate == pytest.approx(-7.0, abs=3 * found.rate_sigma)
        assert found.bic < found.bic_none

    def test_segments_chain_the_rate_changes(self, offset_and_rate_change):
        _, _, found = offset_and_rate_change
        seg, cps = found.segments, found.changepoints
        assert len(seg) == len(cps) + 1
        assert list(seg["start"].iloc[1:]) == list(cps["date"])
        steps = np.diff(seg["rate"].to_numpy())
        np.testing.assert_allclose(steps, cps["rate_change"].fillna(0.0), atol=1e-9)
        assert found.rate == seg["rate"].iloc[-1]

    def test_trend_fit_includes_the_changepoints(self, offset_and_rate_change):
        _, values, found = offset_and_rate_change
        assert "step_0" in found.trend.parameters
        assert "rate_change_0" in found.trend.parameters
        assert len(found.model) == len(values)
        # The model explains the series down to the noise level
        assert np.std(values - found.model) < 2.0

    def test_no_changepoints_in_a_linear_series(self):
        dates, values = _monthly(offset=0.0, rate_change=0.0)
        found = detect_changepoints(dates, values, sampling_days=MONTH)
        assert found.changepoints.empty
        assert len(found.segments) == 1
        # Without change points the rate is the `estimate_trend` velocity
        trend = estimate_trend(dates, values, sampling_days=MONTH, noise_model="FNWN")
        assert found.rate == pytest.approx(trend.velocity, rel=1e-6)
        assert found.rate_sigma == pytest.approx(trend.velocity_uncertainty, rel=1e-6)

    def test_kinds_restricts_the_search(self):
        dates, values = _monthly()
        found = detect_changepoints(
            dates, values, sampling_days=MONTH, kinds=("offset",)
        )
        assert set(found.changepoints["kind"]) == {"offset"}
        assert found.changepoints["rate_change"].isna().all()

    def test_max_changepoints(self):
        dates, values = _monthly()
        found = detect_changepoints(
            dates, values, sampling_days=MONTH, max_changepoints=1
        )
        assert len(found.changepoints) == 1

    def test_min_segment_years(self):
        dates, values = _monthly()
        found = detect_changepoints(
            dates, values, sampling_days=MONTH, min_segment_years=4.0
        )
        t = [_years(dates, d) for d in found.changepoints["date"]]
        assert all(4.0 - 1e-9 <= ti <= _years(dates, dates[-1]) - 4.0 for ti in t)
        assert np.all(np.diff(t) >= 4.0)

    def test_known_steps_are_not_changepoints(self):
        dates, values = _monthly(rate_change=0.0)
        found = detect_changepoints(
            dates, values, sampling_days=MONTH, step_dates=[dates[36]]
        )
        assert found.changepoints.empty

    def test_penalty_trades_detections(self):
        dates, values = _monthly(offset=0.0, rate_change=0.0)
        lenient = detect_changepoints(dates, values, sampling_days=MONTH, penalty=0.0)
        strict = detect_changepoints(dates, values, sampling_days=MONTH, penalty=1e6)
        # Without a cost every addition lowers the criterion
        assert len(lenient.changepoints) == 5
        assert strict.changepoints.empty

    def test_unknown_kind_raises(self):
        dates, values = _monthly()
        with pytest.raises(ValueError, match="kinds must be a non-empty subset"):
            detect_changepoints(dates, values, kinds=("jump",))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def posterior(offset_and_rate_change):
    dates, values, found = offset_and_rate_change
    return sample_changepoints(
        dates, values, sampling_days=MONTH, detection=found, n_samples=3000
    )


class TestSampleChangepoints:
    def test_posterior_number_and_epochs(self, posterior, offset_and_rate_change):
        dates, *_ = offset_and_rate_change
        assert posterior.n_probability.idxmax() == 2
        assert posterior.n_probability[0] < 0.01
        assert posterior.n_probability.sum() == pytest.approx(1.0)
        cps = posterior.changepoints
        assert list(cps["kind"]) == ["offset", "rate"]
        assert _years(dates, cps["date"].iloc[0]) == pytest.approx(OFFSET_AT, abs=0.2)
        assert _years(dates, cps["date"].iloc[1]) == pytest.approx(RATE_AT, abs=0.5)
        assert (cps["date_low"] <= cps["date"]).all()
        assert (cps["date"] <= cps["date_high"]).all()
        assert (cps["probability"] > 0.5).all()

    def test_present_day_rate(self, posterior):
        assert posterior.rate == pytest.approx(-7.0, abs=3 * posterior.rate_sigma)
        assert np.mean(posterior.rate_samples) == pytest.approx(posterior.rate, abs=0.1)
        assert posterior.rhat["n"] < 1.05
        assert posterior.rhat["rate"] < 1.05
        assert 0.0 < posterior.acceptance_rate < 1.0

    def test_probability_table(self, posterior):
        prob = posterior.probability
        assert list(prob.columns) == ["any", "offset", "rate"]
        assert (prob["any"] >= prob[["offset", "rate"]].max(axis=1) - 1e-12).all()
        # The expected number of change points is the sum over epochs
        expected_n = float(
            np.dot(posterior.n_probability.index, posterior.n_probability)
        )
        assert prob["any"].sum() == pytest.approx(expected_n)

    def test_linear_series_favors_no_changepoint(self):
        dates, values = _monthly(offset=0.0, rate_change=0.0)
        posterior = sample_changepoints(
            dates, values, sampling_days=MONTH, n_samples=2000, n_chains=2
        )
        assert posterior.n_probability[0] > 0.5

    def test_reproducible(self, offset_and_rate_change):
        dates, values, found = offset_and_rate_change
        runs = [
            sample_changepoints(
                dates,
                values,
                sampling_days=MONTH,
                detection=found,
                n_samples=300,
                n_burn=50,
                n_chains=2,
                seed=7,
            )
            for _ in range(2)
        ]
        np.testing.assert_array_equal(runs[0].rate_samples, runs[1].rate_samples)

    def test_matches_the_enumerated_posterior(self):
        # One offset slot on a few candidate epochs: the exact posterior can
        # be enumerated with the Gaussian marginal likelihood written out in
        # full (a broad prior stands in for the flat one on the base terms)
        dates, values = _monthly(rate_change=0.0, n=60)
        p_change, scale = 0.5, 5.0
        posterior = sample_changepoints(
            dates,
            values,
            sampling_days=MONTH,
            kinds=("offset",),
            max_changepoints=1,
            n_candidates=8,
            p_change=p_change,
            offset_scale=scale,
            n_samples=20000,
            n_chains=2,
        )
        candidates = posterior.probability.index

        t = np.round((dates - dates[0]).days.to_numpy() / MONTH) * MONTH / DAYS_PER_YEAR
        w1, w2 = 2 * np.pi, 4 * np.pi
        base = np.column_stack(
            [
                np.ones_like(t),
                t,
                np.cos(w1 * t),
                np.sin(w1 * t),
                np.cos(w2 * t),
                np.sin(w2 * t),
            ]
        )
        obs_idx = np.round((dates - dates[0]).days.to_numpy() / MONTH).astype(int)
        C = _noise_covariance(posterior.detection.trend, obs_idx, obs_idx[-1] + 1)
        cov0 = C + 1e4 * base @ base.T

        def log_ml(column=None):
            cov = cov0 if column is None else cov0 + scale**2 * np.outer(column, column)
            return stats.multivariate_normal(np.zeros(len(t)), cov).logpdf(values)

        logs = [np.log1p(-p_change) + log_ml()] + [
            np.log(p_change / len(candidates)) + log_ml((dates >= date).astype(float))
            for date in candidates
        ]
        exact = np.exp(np.array(logs) - np.logaddexp.reduce(logs))
        assert posterior.n_probability[0] == pytest.approx(exact[0], abs=0.02)
        np.testing.assert_allclose(
            posterior.probability["offset"], exact[1:], atol=0.02
        )

    def test_invalid_p_change_raises(self):
        dates, values = _monthly()
        with pytest.raises(ValueError, match="p_change must be in"):
            sample_changepoints(dates, values, p_change=1.0)


def _records(frame: pd.DataFrame) -> list[dict]:
    """Rows as JSON-able dicts, dates as ISO strings."""
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
    return [
        {k: None if isinstance(v, float) and np.isnan(v) else v for k, v in row.items()}
        for row in out.to_dict(orient="records")
    ]


class TestDetectChangepointsRegression:
    """Golden-value regression for `detect_changepoints`.

    The near noise-free series must give back the inputs; the noisy fixture
    pins the selection, sizes, segment rates and noise estimate, so a change
    in the search, the BIC or the noise refit shows up here.
    """

    def test_recovers_exact_parameters(self):
        n = 120
        t = np.arange(n) / 12
        values = (
            -3.0 * t
            + 3.0 * np.sin(2 * np.pi * t)
            + 8.0 * (t >= OFFSET_AT)
            - 4.0 * np.maximum(0.0, t - RATE_AT)
            + np.random.default_rng(0).normal(0, 0.05, n)
        )
        dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(
            np.round(np.arange(n) * MONTH), unit="D"
        )
        found = detect_changepoints(
            dates, values, sampling_days=MONTH, noise_model="WN"
        )
        cps = found.changepoints
        assert list(cps["kind"]) == ["offset", "rate"]
        assert list(cps["date"]) == [dates[36], dates[78]]
        assert cps["offset"].iloc[0] == pytest.approx(8.0, abs=0.05)
        assert cps["rate_change"].iloc[1] == pytest.approx(-4.0, abs=0.02)
        assert found.rate == pytest.approx(-7.0, abs=0.02)
        assert list(found.segments["rate"].round(1)) == [-3.0, -3.0, -7.0]

    def test_golden(self, offset_and_rate_change, golden):
        _, _, found = offset_and_rate_change
        golden(
            "changepoints_detect.json",
            {
                "changepoints": _records(found.changepoints),
                "segments": _records(found.segments),
                "rate": found.rate,
                "rate_sigma": found.rate_sigma,
                "bic": found.bic,
                "bic_none": found.bic_none,
                "sigma_white": found.trend.sigma_white,
                "sigma_flicker": found.trend.sigma_flicker,
            },
        )


class TestSampleChangepointsRegression:
    """Golden-value regression for `sample_changepoints` with a fixed seed."""

    def test_golden(self, offset_and_rate_change, golden):
        dates, values, found = offset_and_rate_change
        post = sample_changepoints(
            dates,
            values,
            sampling_days=MONTH,
            detection=found,
            n_samples=1000,
            n_burn=200,
            n_chains=2,
            seed=11,
        )
        golden(
            "changepoints_sample.json",
            {
                "changepoints": _records(post.changepoints),
                "n_probability": post.n_probability.tolist(),
                "rate": post.rate,
                "rate_sigma": post.rate_sigma,
                "acceptance_rate": post.acceptance_rate,
                "rate_samples_head": post.rate_samples[:5].tolist(),
            },
        )


@pytest.fixture(scope="module")
def sparse():
    dates, values = _monthly()
    return dates, fit_sparse_changepoints(dates, values, sampling_days=MONTH)


class TestFitSparseChangepoints:
    def test_finds_the_offset_and_the_rate_change(self, sparse):
        dates, found = sparse
        cps = found.changepoints
        assert list(cps["kind"]) == ["offset", "rate"]
        assert _years(dates, cps["date"].iloc[0]) == pytest.approx(OFFSET_AT, abs=0.2)
        assert _years(dates, cps["date"].iloc[1]) == pytest.approx(RATE_AT, abs=0.5)
        assert found.rate == pytest.approx(-7.0, abs=3 * found.rate_sigma)

    def test_agrees_with_the_stepwise_search(self, sparse, offset_and_rate_change):
        _, found = sparse
        _, _, stepwise = offset_and_rate_change
        pd.testing.assert_frame_equal(found.changepoints, stepwise.changepoints)

    def test_refinement_recovers_a_smooth_signal(self):
        # Regression: the lasso selection alone keeps too few change points on
        # a sigmoid and misses its flattening, which ruins the present-day rate
        n = 108
        t = np.arange(n) / 12
        rng = np.random.default_rng(1)
        values = -40.0 / (1 + np.exp(-(t - 6.0) / 0.7)) + rng.normal(0, 1.0, n)
        dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(
            np.round(np.arange(n) * MONTH), unit="D"
        )
        found = fit_sparse_changepoints(dates, values, sampling_days=MONTH)
        assert abs(found.rate) < 5.0

    def test_no_changepoints_in_a_linear_series(self):
        dates, values = _monthly(offset=0.0, rate_change=0.0)
        found = fit_sparse_changepoints(dates, values, sampling_days=MONTH)
        assert found.changepoints.empty

    def test_kinds_restricts_the_search(self):
        dates, values = _monthly()
        found = fit_sparse_changepoints(
            dates, values, sampling_days=MONTH, kinds=("offset",)
        )
        assert set(found.changepoints["kind"]) <= {"offset"}

    def test_max_changepoints(self):
        dates, values = _monthly()
        found = fit_sparse_changepoints(
            dates, values, sampling_days=MONTH, max_changepoints=1
        )
        assert len(found.changepoints) <= 1

    def test_min_segment_years(self):
        dates, values = _monthly()
        found = fit_sparse_changepoints(
            dates, values, sampling_days=MONTH, min_segment_years=2.0
        )
        t = [_years(dates, d) for d in found.changepoints["date"]]
        assert np.all(np.diff(t) >= 2.0)

    def test_unknown_kind_raises(self):
        dates, values = _monthly()
        with pytest.raises(ValueError, match="kinds must be a non-empty subset"):
            fit_sparse_changepoints(dates, values, kinds=("jump",))  # type: ignore[arg-type]


def _station_monthly(site: str) -> pd.Series:
    """Vertical (mm) of a station in tests/data/unr.zip, as 30-day means."""
    import zipfile
    from pathlib import Path

    with (
        zipfile.ZipFile(Path(__file__).parent / "data/unr.zip") as archive,
        archive.open(f"tests/data/unr/{site}.tenv3") as f,
    ):
        df = pd.read_csv(f, sep=r"\s+")
    mm = (df["u0(m)"] + df["____up(m)"]).to_numpy() * 1000.0
    s = pd.Series(mm - mm[0], index=pd.to_datetime(df["YYMMMDD"], format="%y%b%d"))
    edges = pd.date_range(s.index[0], s.index[-1], freq=pd.Timedelta(days=MONTH))
    means = s.groupby(pd.cut(s.index, edges, right=False), observed=True).mean()
    return pd.Series(
        means.to_numpy(), index=pd.DatetimeIndex([b.left for b in means.index])
    ).dropna()


class TestFitSparseChangepointsRegression:
    """Golden-value regression for `fit_sparse_changepoints`."""

    def test_recovers_exact_parameters(self):
        n = 120
        t = np.arange(n) / 12
        values = (
            -3.0 * t
            + 3.0 * np.sin(2 * np.pi * t)
            + 8.0 * (t >= OFFSET_AT)
            - 4.0 * np.maximum(0.0, t - RATE_AT)
            + np.random.default_rng(0).normal(0, 0.05, n)
        )
        dates = pd.Timestamp("2016-01-01") + pd.to_timedelta(
            np.round(np.arange(n) * MONTH), unit="D"
        )
        found = fit_sparse_changepoints(
            dates, values, sampling_days=MONTH, noise_model="WN"
        )
        cps = found.changepoints
        assert list(cps["kind"]) == ["offset", "rate"]
        assert list(cps["date"]) == [dates[36], dates[78]]
        assert found.rate == pytest.approx(-7.0, abs=0.02)

    def test_busy_station_keeps_its_early_transient(self):
        """Regression: the lasso path stopped once it held more clusters than
        `max_changepoints`, and on a station with many changes (AHUP, Kilauea)
        nothing was found near the 2006-08 inflation."""
        s = _station_monthly("AHUP")
        found = fit_sparse_changepoints(s.index, s.to_numpy(), sampling_days=MONTH)
        gap = np.abs((found.changepoints["date"] - pd.Timestamp("2006-02-27")).dt.days)
        assert (gap < 366).any()

    def test_golden(self, golden):
        dates, values = _monthly(seed=6)
        found = fit_sparse_changepoints(dates, values, sampling_days=MONTH)
        golden(
            "changepoints_sparse.json",
            {
                "changepoints": _records(found.changepoints),
                "segments": _records(found.segments),
                "rate": found.rate,
                "rate_sigma": found.rate_sigma,
                "bic": found.bic,
                "bic_none": found.bic_none,
            },
        )


def _batch(n_each: int = 4):
    """Series of three kinds: linear, offset at year 3, rate change at 6.5."""
    rng = np.random.default_rng(1)
    rows, kinds = [], []
    for i in range(3 * n_each):
        kind = i % 3
        _, v = _monthly(
            offset=8.0 if kind == 1 else 0.0,
            rate_change=-4.0 if kind == 2 else 0.0,
            seed=100 + i,
        )
        rows.append(v * rng.uniform(0.8, 1.2))
        kinds.append(kind)
    dates, _ = _monthly()
    return dates, np.array(rows), np.array(kinds)


FNWN = TrendResult(
    velocity=np.nan, velocity_uncertainty=np.nan, sigma_white=1.0, sigma_flicker=0.7
)


@pytest.fixture(scope="module")
def batch():
    dates, rows, kinds = _batch()
    return (
        dates,
        rows,
        kinds,
        fit_sparse_changepoints_many(dates, rows, sampling_days=MONTH),
    )


class TestFitSparseChangepointsMany:
    def test_finds_the_change_points_of_each_series(self, batch):
        _, _, kinds, result = batch
        n = result.summary["n_changepoints"].to_numpy()
        assert (n[kinds == 0] == 0).all()
        assert (n[kinds != 0] >= 1).all()
        cps = result.changepoints
        found = cps.groupby("series")["kind"].apply(set)
        for name, kind in zip(result.summary.index, kinds, strict=True):
            if kind == 1:
                assert "offset" in found[name]
            elif kind == 2:
                assert found[name] & {"rate", "both"}

    def test_a_series_alone_matches_the_batch(self):
        # With the noise given, nothing couples the series: each must come
        # out as it does on its own
        dates, rows, _ = _batch(n_each=2)
        together = fit_sparse_changepoints_many(
            dates, rows, noise=FNWN, sampling_days=MONTH
        )
        for k in (0, 2, 4):
            alone = fit_sparse_changepoints_many(
                dates, rows[[k]], noise=FNWN, sampling_days=MONTH
            )
            a = alone.summary.iloc[0]
            b = together.summary.iloc[k]
            assert a["n_changepoints"] == b["n_changepoints"]
            assert a["rate"] == pytest.approx(b["rate"], rel=1e-9)
            mine = together.changepoints[together.changepoints["series"] == str(k)]
            assert list(alone.changepoints["date"]) == list(mine["date"])
            assert list(alone.changepoints["kind"]) == list(mine["kind"])
            np.testing.assert_allclose(
                alone.changepoints[["offset", "rate_change"]].to_numpy(float),
                mine[["offset", "rate_change"]].to_numpy(float),
                rtol=1e-9,
            )

    def test_noise_scale_tracks_the_amplitude(self):
        dates, rows, _ = _batch(n_each=1)
        doubled = np.vstack([rows[0], 2.0 * rows[0]])
        result = fit_sparse_changepoints_many(
            dates, doubled, noise=FNWN, sampling_days=MONTH
        )
        scale = result.summary["noise_scale"].to_numpy()
        assert scale[1] == pytest.approx(2.0 * scale[0], rel=1e-9)
        # Scaling a series scales its rate and leaves the selection alone
        assert result.summary["rate"].iloc[1] == pytest.approx(
            2.0 * result.summary["rate"].iloc[0], rel=1e-6
        )

    def test_dataframe_input_and_incomplete_series(self):
        dates, rows, _ = _batch(n_each=1)
        frame = pd.DataFrame(rows.T, index=dates, columns=["a", "b", "c"])
        frame.iloc[5, 1] = np.nan
        result = fit_sparse_changepoints_many(None, frame, sampling_days=MONTH)
        assert list(result.summary.index) == ["a", "b", "c"]
        assert result.summary.loc["b"].isna().all()
        assert "b" not in set(result.changepoints["series"])

    def test_given_noise_is_used(self):
        dates, rows, _ = _batch(n_each=1)
        result = fit_sparse_changepoints_many(
            dates, rows, noise=FNWN, sampling_days=MONTH
        )
        assert result.noise is FNWN

    def test_no_complete_series_raises(self):
        dates, rows, _ = _batch(n_each=1)
        rows[:, 3] = np.nan
        with pytest.raises(ValueError, match="No series without NaNs"):
            fit_sparse_changepoints_many(dates, rows, sampling_days=MONTH)


class TestFitSparseChangepointsManyRegression:
    """Golden-value regression for `fit_sparse_changepoints_many`."""

    def test_golden(self, batch, golden):
        *_, result = batch
        golden(
            "changepoints_many.json",
            {
                "summary": _records(result.summary.reset_index()),
                "changepoints": _records(result.changepoints),
                "noise": {
                    "sigma_white": result.noise.sigma_white,
                    "sigma_flicker": result.noise.sigma_flicker,
                },
            },
        )

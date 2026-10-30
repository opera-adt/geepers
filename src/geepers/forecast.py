# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Extrapolate a time series with honest intervals, and check them by backtest.

`forecast_trend` continues the model of `geepers.changepoints.detect_changepoints`
(present-day rate, periodic terms) past the last epoch. Its interval has
three parts:

- the uncertainty of the fitted parameters, which grows linearly with
  lead time through the rate;
- the colored noise, predicted from the past residuals: flicker and
  random-walk noise persist, so their conditional mean is not zero and
  their variance grows with lead time;
- change points that have not happened yet. Future offsets and rate
  changes are simulated at the rate this series has had them, with
  Laplace-distributed sizes of the same mean magnitude. A series whose
  rate has changed twice in ten years is assumed to keep doing so. This
  is the trend uncertainty of Prophet (Taylor, S. J., & Letham, B., 2018,
  Forecasting at scale, Am. Stat., 72(1), 37-45,
  https://doi.org/10.1080/00031305.2017.1380080), applied to the change
  points found here.

`backtest` fits on the data up to a series of cutoffs and scores the
forecasts against what followed (simulated historical forecasts, as in
Prophet's ``cross_validation``): the coverage of the intervals shows
whether they can be trusted at each lead time.

Example:
-------
>>> fc = forecast_trend(dates, up_mm, 5, sampling_days=30)   # doctest: +SKIP
>>> fc.table[["mean", "lower", "upper"]]                     # doctest: +SKIP
>>> bt = backtest(dates, up_mm, 3, sampling_days=30)         # doctest: +SKIP
>>> bt.metrics                                               # doctest: +SKIP

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg

from geepers.changepoints import ChangepointResult, detect_changepoints
from geepers.linearity import _noise_covariance
from geepers.trend import DAYS_PER_YEAR, _design_matrix

__all__ = ["Backtest", "Forecast", "backtest", "forecast_trend"]


@dataclass
class Forecast:
    """Output of `forecast_trend`.

    Attributes
    ----------
    table : pd.DataFrame
        Indexed by forecast epoch. Columns ``lead_years``; ``model`` (the
        deterministic extrapolation); ``mean`` (plus the predicted noise);
        ``lower`` and ``upper`` (the `level` interval, from the samples);
        ``sigma_model`` and ``sigma_noise`` (1-sigma of the parameter and
        noise parts).
    samples : np.ndarray
        Simulated future paths, shape ``(n_simulations, n_epochs)``.
    level : float
        Coverage of the interval.
    change_rates : dict[str, float]
        Simulated future events per year, ``offset`` and ``rate``.
    change_scales : dict[str, float]
        Mean magnitude of those events: input units, and input units
        per year for rate changes.
    changepoints : ChangepointResult
        The fit that was extrapolated.

    """

    table: pd.DataFrame
    samples: np.ndarray
    level: float
    change_rates: dict[str, float]
    change_scales: dict[str, float]
    changepoints: ChangepointResult


@dataclass
class Backtest:
    """Output of `backtest`.

    Attributes
    ----------
    predictions : pd.DataFrame
        One row per (cutoff, later observation): ``cutoff``, ``date``,
        ``lead_years``, ``observed``, ``mean``, ``lower``, ``upper``,
        ``error`` (observed - mean) and ``covered``.
    metrics : pd.DataFrame
        By lead-time bin (``lead_years`` is the bin's upper edge): ``n``,
        ``rmse``, ``mae``, ``coverage`` and ``width`` (mean interval width).
    coverage : float
        Fraction of all later observations inside their interval.

    """

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    coverage: float


def _grid_index(days: np.ndarray, sampling_days: float) -> np.ndarray:
    """Epoch index on the `estimate_trend` sampling grid."""
    return np.round(days / sampling_days).astype(int)


def forecast_trend(
    dates: ArrayLike,
    values: ArrayLike,
    horizon_years: float | None = None,
    *,
    future_dates: ArrayLike | None = None,
    sampling_days: float = 1.0,
    periods_years: tuple[float, ...] = (1.0, 0.5),
    step_dates: ArrayLike | None = None,
    changepoints: ChangepointResult | None = None,
    include_noise: bool = True,
    simulate_changes: tuple[str, ...] = ("rate",),
    level: float = 0.95,
    n_simulations: int = 2000,
    seed: int | np.random.Generator | None = 0,
    **detect_kwargs: Any,
) -> Forecast:
    """Extrapolate the change-point model with a forecast interval.

    Parameters
    ----------
    dates : array-like of datetime64
        Observation epochs; gaps are allowed.
    values : array-like of float
        Observations (any unit); NaNs are dropped.
    horizon_years : float, optional
        Forecast every `sampling_days` up to this many years after the
        last epoch. Give this or `future_dates`.
    future_dates : array-like of datetime64, optional
        Epochs to forecast, all after the last observation.
    sampling_days : float
        Nominal sampling interval in days. Default 1.
    periods_years : tuple of float
        Periodic terms. Must match those of `changepoints` if given.
    step_dates : array-like of datetime64, optional
        Known offsets, passed to `detect_changepoints`.
    changepoints : ChangepointResult, optional
        A `detect_changepoints` result for this series; run with
        `detect_kwargs` if not given.
    include_noise : bool
        Forecast an observation (True, default) or the noise-free signal.
    simulate_changes : tuple of {"offset", "rate"}
        Kinds of future change point added to the interval. Default rate
        changes only, as in Prophet: an offset in the record is usually an
        event (an earthquake, an antenna change) whose recurrence the
        record cannot predict, and simulating it widens the interval far
        beyond what backtests support. A kind the record never shows is
        not simulated; ``()`` turns the simulation off.
    level : float
        Coverage of the interval. Default 0.95.
    n_simulations : int
        Number of simulated paths. Default 2000.
    seed : int or np.random.Generator, optional
        Seed of the simulation. Default 0 (reproducible).
    **detect_kwargs
        Passed to `detect_changepoints` (e.g. ``noise_model``, ``kinds``).

    Returns
    -------
    Forecast

    Raises
    ------
    ValueError
        If neither or both of `horizon_years` and `future_dates` are
        given, or a forecast epoch is not after the last observation.

    """
    if (horizon_years is None) == (future_dates is None):
        msg = "Give exactly one of horizon_years and future_dates"
        raise ValueError(msg)
    epochs = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    y_all = np.asarray(values, dtype=float)
    good = np.isfinite(y_all)
    epochs, y = epochs[good], y_all[good]
    if changepoints is None:
        changepoints = detect_changepoints(
            epochs,
            y,
            sampling_days=sampling_days,
            periods_years=periods_years,
            step_dates=step_dates,
            **detect_kwargs,
        )
    trend = changepoints.trend

    obs_idx = _grid_index(
        np.asarray((epochs - epochs[0]).total_seconds()) / 86400.0, sampling_days
    )
    if future_dates is None:
        assert horizon_years is not None
        n_ahead = int(np.floor(horizon_years * DAYS_PER_YEAR / sampling_days))
        fut_idx = obs_idx[-1] + np.arange(1, n_ahead + 1)
        future = epochs[0] + pd.to_timedelta(fut_idx * sampling_days, unit="D")
    else:
        future = pd.DatetimeIndex(pd.to_datetime(np.asarray(future_dates)))
        fut_idx = _grid_index(
            np.asarray((future - epochs[0]).total_seconds()) / 86400.0, sampling_days
        )
    if len(fut_idx) == 0 or fut_idx.min() <= obs_idx[-1]:
        msg = "Forecast epochs must all come after the last observation"
        raise ValueError(msg)
    to_years = sampling_days / DAYS_PER_YEAR
    t_future = fut_idx * to_years

    # The fit's design at the future epochs: every step has happened, and
    # the rate changes sit on the grid time of their epoch
    cps = changepoints.changepoints
    # An empty table's date column is not datetime-typed
    rate_dates = pd.DatetimeIndex(
        pd.to_datetime(cps.loc[cps["kind"].isin(["rate", "both"]), "date"])
    )
    rate_years = (
        _grid_index(
            np.asarray((rate_dates - epochs[0]).total_seconds()) / 86400.0,
            sampling_days,
        )
        * to_years
    )
    n_steps = sum(name.startswith("step_") for name in trend.parameters)
    design, names = _design_matrix(
        t_future, 1, periods_years, np.zeros(n_steps), [], [], rate_years
    )
    if names != list(trend.parameters):
        msg = (
            "The change-point fit has terms the forecast cannot extend:"
            f" {sorted(set(trend.parameters) - set(names))}; pass the same"
            " periods_years and no postseismic terms"
        )
        raise ValueError(msg)
    beta = np.array([value for value, _ in trend.parameters.values()])
    model = design @ beta
    cov_model = design @ trend.covariance @ design.T

    n_obs = len(obs_idx)
    noise_mean = np.zeros(len(t_future))
    cov_noise = np.zeros((len(t_future),) * 2)
    if include_noise:
        # Gaussian conditioning of the future noise on the past residuals
        idx = np.r_[obs_idx, fut_idx]
        C = _noise_covariance(trend, idx, int(idx.max()) + 1)
        cho = linalg.cho_factor(C[:n_obs, :n_obs], lower=True, check_finite=False)
        gain = linalg.cho_solve(cho, C[:n_obs, n_obs:], check_finite=False).T
        noise_mean = gain @ trend.residuals
        cov_noise = C[n_obs:, n_obs:] - gain @ C[:n_obs, n_obs:]

    rng = np.random.default_rng(seed)
    mean = model + noise_mean
    samples = rng.multivariate_normal(
        mean, cov_model + cov_noise, size=n_simulations, method="eigh"
    )

    span = float(obs_idx[-1] * to_years)
    rates = {"offset": 0.0, "rate": 0.0}
    scales = {"offset": 0.0, "rate": 0.0}
    for part, column in (("offset", "offset"), ("rate", "rate_change")):
        sizes = cps[column].dropna().to_numpy()
        if part in simulate_changes and len(sizes) and span > 0:
            rates[part] = len(sizes) / span
            scales[part] = float(np.mean(np.abs(sizes)))
    t_last = float(obs_idx[-1] * to_years)
    horizon = float(t_future.max() - t_last)
    for part in ("offset", "rate"):
        if rates[part] == 0.0:
            continue
        counts = rng.poisson(rates[part] * horizon, size=n_simulations)
        for s in np.flatnonzero(counts):
            when = t_last + rng.uniform(0.0, horizon, counts[s])
            size = rng.laplace(0.0, scales[part], counts[s])
            if part == "offset":
                samples[s] += ((t_future[:, None] >= when) * size).sum(axis=1)
            else:
                samples[s] += (np.maximum(0.0, t_future[:, None] - when) * size).sum(
                    axis=1
                )

    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail], axis=0)
    table = pd.DataFrame(
        {
            "lead_years": t_future - t_last,
            "model": model,
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "sigma_model": np.sqrt(np.clip(np.diag(cov_model), 0.0, None)),
            "sigma_noise": np.sqrt(np.clip(np.diag(cov_noise), 0.0, None)),
        },
        index=pd.DatetimeIndex(future, name="date"),
    )
    return Forecast(
        table=table,
        samples=samples,
        level=level,
        change_rates=rates,
        change_scales=scales,
        changepoints=changepoints,
    )


def backtest(
    dates: ArrayLike,
    values: ArrayLike,
    horizon_years: float,
    *,
    cutoffs: ArrayLike | None = None,
    n_cutoffs: int = 5,
    min_history_years: float = 4.0,
    n_bins: int = 5,
    forecaster: Callable[..., Any] | None = None,
    **forecast_kwargs: Any,
) -> Backtest:
    """Score `forecast_trend` on the data that followed past cutoffs.

    At each cutoff the whole fit (change points, noise, forecast) is
    redone on the observations up to it, and the forecast is compared
    with the observations in the following `horizon_years`.

    Parameters
    ----------
    dates, values : array-like
        The full series.
    horizon_years : float
        Forecast length scored after each cutoff.
    cutoffs : array-like of datetime64, optional
        Cutoff epochs. Default `n_cutoffs` evenly spaced between
        `min_history_years` after the first epoch and `horizon_years`
        before the last.
    n_cutoffs : int
        Number of default cutoffs. Default 5.
    min_history_years : float
        Shortest record fitted. Default 4.
    n_bins : int
        Number of lead-time bins in `metrics`. Default 5.
    forecaster : callable, optional
        What to score, called as ``forecaster(dates, values,
        future_dates=..., **forecast_kwargs)`` and returning an object
        whose ``table`` has the columns of `Forecast.table`, such as
        `geepers.smooth.forecast_smooth`. Default `forecast_trend`.
    **forecast_kwargs
        Passed to the forecaster (e.g. ``sampling_days``, ``level``).

    Returns
    -------
    Backtest

    Raises
    ------
    ValueError
        If the series is too short for any cutoff.

    """
    epochs = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    y_all = np.asarray(values, dtype=float)
    good = np.isfinite(y_all)
    epochs, y = epochs[good], y_all[good]
    horizon = pd.Timedelta(days=horizon_years * DAYS_PER_YEAR)
    if cutoffs is None:
        first = epochs[0] + pd.Timedelta(days=min_history_years * DAYS_PER_YEAR)
        last = epochs[-1] - horizon
        if last < first:
            msg = (
                f"A {min_history_years}-year history and a {horizon_years}-year"
                " horizon do not fit in the record"
            )
            raise ValueError(msg)
        cut = pd.DatetimeIndex(pd.date_range(first, last, periods=n_cutoffs))
    else:
        cut = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(np.asarray(cutoffs))))

    rows = []
    for cutoff in cut:
        past = epochs <= cutoff
        later = (epochs > cutoff) & (epochs <= cutoff + horizon)
        if not later.any():
            continue
        fc = (forecaster or forecast_trend)(
            epochs[past], y[past], future_dates=epochs[later], **forecast_kwargs
        )
        table = fc.table
        rows.append(
            pd.DataFrame(
                {
                    "cutoff": cutoff,
                    "date": table.index,
                    "lead_years": table["lead_years"].to_numpy(),
                    "observed": y[later],
                    "mean": table["mean"].to_numpy(),
                    "lower": table["lower"].to_numpy(),
                    "upper": table["upper"].to_numpy(),
                }
            )
        )
    predictions = pd.concat(rows, ignore_index=True)
    predictions["error"] = predictions["observed"] - predictions["mean"]
    predictions["covered"] = predictions["observed"].between(
        predictions["lower"], predictions["upper"]
    )

    edges = np.linspace(0.0, predictions["lead_years"].max(), n_bins + 1)
    bins = pd.cut(predictions["lead_years"], edges, include_lowest=True)
    grouped = predictions.groupby(bins, observed=True)
    metrics = pd.DataFrame(
        {
            "lead_years": [interval.right for interval in grouped.size().index],
            "n": grouped.size().to_numpy(),
            "rmse": grouped["error"].apply(lambda e: np.sqrt(np.mean(e**2))).to_numpy(),
            "mae": grouped["error"].apply(lambda e: np.mean(np.abs(e))).to_numpy(),
            "coverage": grouped["covered"].mean().to_numpy(),
            "width": (grouped["upper"].mean() - grouped["lower"].mean()).to_numpy(),
        }
    )
    return Backtest(
        predictions=predictions,
        metrics=metrics,
        coverage=float(predictions["covered"].mean()),
    )

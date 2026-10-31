# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Estimate a smooth, time-varying rate without assuming its shape.

Decays, droughts, accelerating subsidence and multi-year cycles are not
piecewise linear, and their shape is not known in advance. This module
fits

    y = X b + Z u + e,    u ~ N(0, s2 lam I),    e ~ N(0, s2 C(phi)),

where ``X`` holds the intercept, trend, periodic terms and steps, ``Z``
the hinges ``max(0, t - knot)`` at regularly spaced knots, and ``C`` the
flicker + white noise covariance with white fraction ``phi``. Independent
slope changes make the rate a random walk and the displacement an
integrated random walk, the state-space form of a cubic-like smoothing
spline (Wahba, G., 1978, Improper priors, spline smoothing and the problem
of guarding against model errors in regression, J. R. Stat. Soc. B, 40(3),
364-372, https://doi.org/10.1111/j.2517-6161.1978.tb01050.x; Brumback,
B. A., Ruppert, D., & Wand, M. P., 1999, Variable selection and function
estimation in additive nonparametric regression using a data-based prior,
J. Am. Stat. Assoc., 94(447), 794-797). The smoothness ``lam`` and ``phi``
are estimated together by restricted maximum likelihood (Patterson, H. D.,
& Thompson, R., 1971, Recovery of inter-block information when block sizes
are unequal, Biometrika, 58(3), 545-554,
https://doi.org/10.1093/biomet/58.3.545).

Over one record a smooth signal and temporally correlated noise look
alike, so a flexible fit can report noise as signal. The fit therefore
uses the smooth trend only if its likelihood gain exceeds `min_evidence`,
calibrated on simulated linear series. Otherwise it is the piecewise-linear
model of `geepers.changepoints.detect_changepoints`, a straight line where
that finds nothing.

The gain is measured against a line whose noise may also hold a random
walk, up to `max_random_walk_mm`: a free random walk can explain any single
record, so its amplitude has to be limited by what monuments can do. A
signal needing more than that is taken as deformation. Noise beyond the
limit, and deformation below it, are still mistaken for each other; only
other records (neighboring stations) can tell them apart.

A forecast does not extend the curve: the last rate continues, plus the
periodic terms and the persisting part of the noise, and it never
predicts a turning point. Future slope changes enter through the
uncertainty, which grows with lead time at the size this record has
shown.

Example:
-------
>>> fit = fit_smooth_trend(dates, up, units="m", sampling_days=30)   # doctest: +SKIP
>>> fit.rate, fit.rate_sigma, fit.shape                             # doctest: +SKIP
>>> fc = forecast_smooth(dates, up, 3, sampling_days=30, fit=fit)   # doctest: +SKIP
>>> fc.table[["mean", "lower", "upper"]]                       # doctest: +SKIP

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg, optimize, stats

from geepers.changepoints import detect_changepoints
from geepers.forecast import _future_regressors
from geepers.trend import (
    DAYS_PER_YEAR,
    _design_matrix,
    _nll_core,
    _powerlaw_covariance,
    _regressor_columns,
)

__all__ = ["SmoothForecast", "SmoothResult", "fit_smooth_trend", "forecast_smooth"]

# Coarse search of the likelihood surface, polished afterwards. The white
# fraction runs close to both ends: real records range from nearly pure
# flicker to nearly pure white.
_PHIS = (0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.999)
_LOG_LAMBDAS = np.linspace(np.log(1e-5), np.log(1e5), 16)
# Share of the correlated noise that is random walk, for the line's noise
_RANDOM_WALK_SHARES = (0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.7, 0.95)
_UNITS_PER_MM = {"mm": 1.0, "m": 1e-3}


@dataclass
class _Solution:
    """What a forecast needs from the fit."""

    epochs: pd.DatetimeIndex
    y: np.ndarray
    obs_idx: np.ndarray
    sampling_days: float
    periods_years: tuple[float, ...]
    n_steps: int
    rate_change_years: np.ndarray
    regressor_names: list[str]
    knots: np.ndarray
    knot_step: float
    coefficients: np.ndarray  # base terms, then slope changes
    covariance: np.ndarray
    s2: float
    phi: float
    lam: float


@dataclass
class SmoothResult:
    """Output of `fit_smooth_trend`.

    Attributes
    ----------
    model : np.ndarray
        Fitted model at the observed (finite) epochs: trend, steps and
        periodic terms.
    rate, rate_sigma : float
        Rate at the last epoch and its 1-sigma, input units per year.
    rates, rates_sigma : pd.Series
        The same at every observed epoch.
    shape : {"linear", "piecewise", "smooth"}
        The trend used: "smooth" if `evidence` reached `min_evidence`,
        otherwise "piecewise" where `detect_changes` found rate changes
        and "linear" where it did not.
    evidence : float
        Twice the restricted log-likelihood gain of the smooth trend over
        a straight line (with the steps), each with its own noise estimate;
        the line's may hold a random walk up to `max_random_walk_mm`.
    sigma_slope_change : float
        Standard deviation of the rate change between knots, input units
        per year (0 unless `shape` is "smooth").
    sigma_white, sigma_flicker : float
        Noise amplitudes, input units.
    step_dates : list of pd.Timestamp
        Steps in the model: given and detected.
    rate_change_dates : list of pd.Timestamp
        Rate changes in the model; empty unless `shape` is "piecewise".
    parameters : dict[str, tuple[float, float]]
        Base terms as ``name -> (value, 1-sigma)``. With a smooth trend,
        ``trend`` is the rate at the first epoch.

    """

    model: np.ndarray
    rate: float
    rate_sigma: float
    rates: pd.Series
    rates_sigma: pd.Series
    shape: Literal["linear", "piecewise", "smooth"]
    evidence: float
    sigma_slope_change: float
    sigma_white: float
    sigma_flicker: float
    step_dates: list[pd.Timestamp] = field(default_factory=list)
    rate_change_dates: list[pd.Timestamp] = field(default_factory=list)
    parameters: dict[str, tuple[float, float]] = field(default_factory=dict)
    _solution: _Solution | None = field(default=None, repr=False)


@dataclass
class SmoothForecast:
    """Output of `forecast_smooth`.

    Attributes
    ----------
    table : pd.DataFrame
        Indexed by forecast epoch, with the columns of
        `geepers.forecast.Forecast.table`: ``lead_years``, ``model``,
        ``mean``, ``lower``, ``upper``, ``sigma_model`` (parameters, past
        and future slope changes) and ``sigma_noise``.
    level : float
        Coverage of the interval.
    fit : SmoothResult
        The fit that was extrapolated.

    """

    table: pd.DataFrame
    level: float
    fit: SmoothResult


def _unique_steps(
    steps: ArrayLike | None, epochs: pd.DatetimeIndex
) -> list[pd.Timestamp]:
    """Keep one step per first epoch after it; a fit can only estimate those.

    Steps outside the record have no data on one side, and steps inside
    one data gap would be the same column.
    """
    if steps is None:
        return []
    dates = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(np.asarray(steps))))
    kept: dict[int, pd.Timestamp] = {}
    for date in sorted(dates):
        i = int(epochs.searchsorted(date))
        if 0 < i < len(epochs):
            kept.setdefault(i, date)
    return list(kept.values())


def _noise_mixture(flicker: np.ndarray, phi: float) -> np.ndarray:
    return phi * np.eye(len(flicker)) + (1.0 - phi) * flicker


def _search_noise(
    flicker: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """Find the white fraction of the straight line: its likelihood and value."""
    best = min((_nll_core(_noise_mixture(flicker, p), x, y, True), p) for p in _PHIS)
    low, high = np.log(_PHIS[0]), np.log(_PHIS[-1])
    res = optimize.minimize_scalar(
        lambda v: _nll_core(_noise_mixture(flicker, float(np.exp(v))), x, y, True),
        bounds=(max(low, np.log(best[1]) - 1.5), min(high, np.log(best[1]) + 1.5)),
        method="bounded",
        options={"xatol": 0.02},
    )
    if res.fun < best[0]:
        return float(res.fun), float(np.exp(res.x))
    return best


def _search_random_walk_noise(
    flicker: np.ndarray,
    random_walk: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    years_per_step: float,
    max_random_walk: float,
) -> float:
    """Find the best likelihood of a line with limited random-walk noise.

    The amplitude depends on the variance scale, which the likelihood
    profiles out, so it is checked after the fact for each mixture.
    """
    best = np.inf
    for share in _RANDOM_WALK_SHARES:
        correlated = (1.0 - share) * flicker + share * random_walk
        for phi in _PHIS:
            cov = _noise_mixture(correlated, phi)
            value = _nll_core(cov, x, y, True)
            if value >= best:
                continue
            cho = linalg.cho_factor(cov, lower=True, check_finite=False)
            ci_x = linalg.cho_solve(cho, x, check_finite=False)
            beta = linalg.solve(x.T @ ci_x, ci_x.T @ y, assume_a="pos")
            r = y - x @ beta
            chi2 = float(r @ linalg.cho_solve(cho, r, check_finite=False))
            s2 = chi2 / (len(y) - x.shape[1])
            amplitude = np.sqrt(s2 * (1.0 - phi) * share / years_per_step)
            if amplitude <= max_random_walk:
                best = value
    return best


def _search_smooth(
    flicker: np.ndarray, zzt: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float]:
    """Find the likelihood, white fraction and smoothness of the smooth model."""

    def nll(log_phi: float, log_lam: float) -> float:
        cov = _noise_mixture(flicker, float(np.exp(log_phi)))
        return _nll_core(cov + float(np.exp(log_lam)) * zzt, x, y, True)

    value, log_phi, log_lam = min(
        (nll(np.log(p), v), np.log(p), v) for p in _PHIS for v in _LOG_LAMBDAS
    )
    bounds = [
        (np.log(_PHIS[0]), np.log(_PHIS[-1])),
        (_LOG_LAMBDAS[0], _LOG_LAMBDAS[-1]),
    ]
    res = optimize.minimize(
        lambda v: nll(v[0], v[1]),
        x0=[log_phi, log_lam],
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 0.05, "fatol": 1e-3, "maxfev": 60},
    )
    if res.fun < value:
        value, log_phi, log_lam = float(res.fun), res.x[0], res.x[1]
    return value, float(np.exp(log_phi)), float(np.exp(log_lam))


def fit_smooth_trend(
    dates: ArrayLike,
    values: ArrayLike,
    *,
    units: Literal["m", "mm"],
    sampling_days: float = 1.0,
    periods_years: tuple[float, ...] = (1.0, 0.5),
    step_dates: ArrayLike | None = None,
    detect_changes: bool = True,
    knots_per_year: float = 1.0,
    min_evidence: float = 6.0,
    max_random_walk_mm: float | None = 3.0,
    regressors: pd.DataFrame | Mapping[str, ArrayLike] | None = None,
) -> SmoothResult:
    """Fit a smooth trend, its smoothness estimated from the data.

    Parameters
    ----------
    dates : array-like of datetime64
        Observation epochs; gaps are allowed.
    values : array-like of float
        Observations in `units`; NaNs are dropped.
    units : {"m", "mm"}
        Unit of `values`. Required, because `max_random_walk_mm` is a
        physical limit and no default would suit both: geepers' GNSS series
        are in meters, and a limit taken 1000 times too large or too small
        changes the results without any error.
    sampling_days : float
        Nominal sampling interval in days. Each likelihood evaluation
        factorizes an n x n matrix about 150 times per fit, so average
        daily series to weekly or monthly first. Default 1.
    periods_years : tuple of float
        Periodic terms. Default annual and semi-annual.
    step_dates : array-like of datetime64, optional
        Known offsets. A step inside a data gap shared with another, or
        outside the record, is dropped.
    detect_changes : bool
        Also look for unknown changes with
        `geepers.changepoints.detect_changepoints`. Offsets are kept in
        every model, because a smooth trend cannot follow a jump; rate
        changes only if the smooth trend is rejected, which it otherwise
        replaces. Default True.
    knots_per_year : float
        Knot spacing. Default 1, which cannot represent an annual cycle:
        with denser knots the rate followed the seasonal cycle wherever
        its amplitude varies (groundwater pumping), and the present-day
        rate then depended on the month the record ended (P056: a scatter
        of 19 mm/yr at 4 knots, 7 at 1). The price is a coarser response
        to fast changes, such as a multi-year cycle.
    min_evidence : float
        Smallest `SmoothResult.evidence` at which the smooth trend is
        used. Default 6: about 1 % of simulated linear series with flicker
        + white noise reach it, and at most 3 % for any record length from
        5 to 20 years and any flicker amplitude tried.

    max_random_walk_mm : float or None
        Largest random-walk noise the straight line may be given, in
        mm/sqrt(yr) whatever the `units`. Default 3. None compares against
        flicker + white noise only.

        Differences of same-class station pairs within 25 km have medians
        from 0.7 (deep-braced) to 2.0 mm/sqrt(yr) (masts), though 13-31 % of
        stations exceed 3, so 3 is a judgement. On simulated linear series the smooth
        trend is chosen on 2 % with a true random walk of 3 and on 14 % at
        5, against 28 % and 43 % without the limit. Signals near the limit
        are missed more often (1 in 5 of 30 mm decays over 12 years) and
        get the piecewise model instead.

        For InSAR pixels use about 8. See ``docs/analysis-modules.md``.
    regressors : pd.DataFrame or mapping of name to array-like, optional
        Explanatory series, as in `geepers.trend.estimate_trend`, in every
        model: a signal the driver explains is not taken for a smooth
        trend. Amplitudes are in `SmoothResult.parameters` as ``reg_<name>``.

    Returns
    -------
    SmoothResult

    Raises
    ------
    ValueError
        If there are fewer than 10 finite values, `units` is unknown, or a
        regressor does not have one value per epoch.

    """
    if units not in _UNITS_PER_MM:
        msg = f"units must be 'm' or 'mm', got {units!r}"
        raise ValueError(msg)
    epochs = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    y_all = np.asarray(values, dtype=float)
    reg_names, reg_all = _regressor_columns(regressors, len(y_all))
    good = np.isfinite(y_all) & np.all(np.isfinite(reg_all), axis=1)
    epochs, y, reg = epochs[good], y_all[good], reg_all[good]
    drivers = dict(zip(reg_names, reg.T, strict=True)) or None
    if len(y) < 10:
        msg = f"Need at least 10 finite observations, got {len(y)}"
        raise ValueError(msg)

    # Same epoch grid and time axis as `estimate_trend`
    days = np.asarray((epochs - epochs[0]).total_seconds()) / 86400.0
    obs_idx = np.round(days / sampling_days).astype(int)
    t = obs_idx * sampling_days / DAYS_PER_YEAR

    known = _unique_steps(step_dates, epochs)
    steps = known
    screened_rates: list[pd.Timestamp] = []
    if detect_changes:
        found = detect_changepoints(
            epochs,
            y,
            sampling_days=sampling_days,
            periods_years=periods_years,
            step_dates=known or None,
            regressors=drivers,
        ).changepoints
        offsets = list(found.loc[found["offset"].notna(), "date"])
        steps = _unique_steps([*known, *offsets], epochs)
        screened_rates = list(found.loc[found["rate_change"].notna(), "date"])
    first_after = epochs.searchsorted(pd.DatetimeIndex(steps)) if steps else []
    step_years = np.array([t[i] for i in first_after])
    reg_labels = [f"reg_{name}" for name in reg_names]
    x, names = _design_matrix(t, 1, periods_years, step_years, [], [])
    x, names = np.column_stack([x, reg]), names + reg_labels

    knot_step = 1.0 / knots_per_year
    knots = np.arange(t[0] + knot_step, t[-1], knot_step)
    z = np.maximum(0.0, t[:, None] - knots)
    flicker = _powerlaw_covariance(-1.0, int(obs_idx[-1]) + 1)[np.ix_(obs_idx, obs_idx)]

    y0 = y - y.mean()
    nll_line, _ = _search_noise(flicker, x, y0)
    nll_smooth, phi, lam = _search_smooth(flicker, z @ z.T, x, y0)
    evidence = max(0.0, 2.0 * (nll_line - nll_smooth))
    # Only a series that would go smooth needs the slower comparison
    if max_random_walk_mm is not None and evidence >= min_evidence:
        random_walk = _powerlaw_covariance(-2.0, int(obs_idx[-1]) + 1)[
            np.ix_(obs_idx, obs_idx)
        ]
        nll_limited = _search_random_walk_noise(
            flicker,
            random_walk,
            x,
            y0,
            sampling_days / DAYS_PER_YEAR,
            max_random_walk_mm * _UNITS_PER_MM[units],
        )
        evidence = max(0.0, 2.0 * (min(nll_line, nll_limited) - nll_smooth))
    rate_dates: list[pd.Timestamp] = []
    if evidence < min_evidence:
        rate_dates = screened_rates
        rate_at = (
            epochs.searchsorted(pd.DatetimeIndex(rate_dates)) if rate_dates else []
        )
        rate_years = np.array([t[i] for i in rate_at])
        x, names = _design_matrix(t, 1, periods_years, step_years, [], [], rate_years)
        x, names = np.column_stack([x, reg]), names + reg_labels
        _, phi = _search_noise(flicker, x, y0)
        lam = 0.0
    else:
        rate_years = np.array([])

    cov_noise = _noise_mixture(flicker, phi)
    cho = linalg.cho_factor(cov_noise, lower=True, check_finite=False)
    ci_x = linalg.cho_solve(cho, x, check_finite=False)
    if lam > 0:
        ci_z = linalg.cho_solve(cho, z, check_finite=False)
        normal = np.block(
            [
                [x.T @ ci_x, x.T @ ci_z],
                [z.T @ ci_x, z.T @ ci_z + np.eye(z.shape[1]) / lam],
            ]
        )
        design = np.column_stack([x, z])
    else:
        normal = x.T @ ci_x
        design = x
    unscaled = linalg.inv(normal)
    coefficients = unscaled @ (design.T @ linalg.cho_solve(cho, y, check_finite=False))
    model = design @ coefficients

    # Restricted estimate of the variance scale, from the marginal model
    cho_v = linalg.cho_factor(cov_noise + lam * z @ z.T, lower=True, check_finite=False)
    n_base = x.shape[1]
    r = y - x @ coefficients[:n_base]
    s2 = float(r @ linalg.cho_solve(cho_v, r, check_finite=False)) / (len(y) - n_base)
    covariance = s2 * unscaled

    # The rate at each epoch: the trend plus the slope changes before it
    itrend = names.index("trend")
    weights = np.zeros((len(t), len(coefficients)))
    weights[:, itrend] = 1.0
    for i, tc in enumerate(rate_years):
        weights[:, names.index(f"rate_change_{i}")] = (t >= tc).astype(float)
    if lam > 0:
        weights[:, n_base:] = (t[:, None] >= knots).astype(float)
    rates = weights @ coefficients
    rates_sigma = np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance, weights))

    return SmoothResult(
        model=model,
        rate=float(rates[-1]),
        rate_sigma=float(rates_sigma[-1]),
        rates=pd.Series(rates, index=epochs),
        rates_sigma=pd.Series(rates_sigma, index=epochs),
        shape="smooth" if lam > 0 else ("piecewise" if rate_dates else "linear"),
        evidence=evidence,
        sigma_slope_change=float(np.sqrt(s2 * lam)),
        sigma_white=float(np.sqrt(s2 * phi)),
        sigma_flicker=float(np.sqrt(s2 * (1.0 - phi))),
        step_dates=list(steps),
        rate_change_dates=list(rate_dates),
        parameters={
            name: (float(coefficients[i]), float(np.sqrt(covariance[i, i])))
            for i, name in enumerate(names)
        },
        _solution=_Solution(
            epochs=epochs,
            y=y,
            obs_idx=obs_idx,
            sampling_days=sampling_days,
            periods_years=periods_years,
            n_steps=len(steps),
            rate_change_years=rate_years,
            regressor_names=reg_names,
            knots=knots if lam > 0 else knots[:0],
            knot_step=knot_step,
            coefficients=coefficients,
            covariance=covariance,
            s2=s2,
            phi=phi,
            lam=lam,
        ),
    )


def forecast_smooth(
    dates: ArrayLike,
    values: ArrayLike,
    horizon_years: float | None = None,
    *,
    future_dates: ArrayLike | None = None,
    fit: SmoothResult | None = None,
    include_noise: bool = True,
    level: float = 0.95,
    sampling_days: float = 1.0,
    future_regressors: pd.DataFrame | Mapping[str, ArrayLike] | None = None,
    **fit_kwargs: Any,
) -> SmoothForecast:
    """Extrapolate a smooth trend with a forecast interval.

    The last rate continues, with the periodic terms. The interval adds
    to the parameter uncertainty the slope changes still to come, at the
    size the record has shown, so it grows faster than linearly with lead
    time. A piecewise fit instead expects further rate changes at the rate
    and size of its own, and a straight line expects none. With
    `include_noise`, flicker
    noise is predicted from the past residuals as in
    `geepers.forecast.forecast_trend`. Pass this function as `forecaster`
    to `geepers.forecast.backtest` to score it.

    Parameters
    ----------
    dates, values : array-like
        The series, as in `fit_smooth_trend`.
    horizon_years : float, optional
        Forecast every `sampling_days` up to this many years after the
        last epoch. Give this or `future_dates`.
    future_dates : array-like of datetime64, optional
        Epochs to forecast, all after the last observation.
    fit : SmoothResult, optional
        A `fit_smooth_trend` result for this series; computed if not given.
    include_noise : bool
        Forecast an observation (True, default) or the noise-free signal.
    level : float
        Coverage of the interval. Default 0.95.
    sampling_days : float
        Nominal sampling interval in days. Default 1.
    future_regressors : pd.DataFrame or mapping of name to array-like, optional
        One value per forecast epoch of each regressor of the fit. Required
        if it has any; taken as exact, so the interval leaves out their
        uncertainty.
    **fit_kwargs
        Passed to `fit_smooth_trend`, which needs `units` unless `fit` is
        given.

    Returns
    -------
    SmoothForecast

    Raises
    ------
    ValueError
        If neither or both of `horizon_years` and `future_dates` are
        given, a forecast epoch is not after the last observation, or
        `future_regressors` does not match the fit or the forecast epochs.

    """
    if (horizon_years is None) == (future_dates is None):
        msg = "Give exactly one of horizon_years and future_dates"
        raise ValueError(msg)
    if fit is None:
        fit = fit_smooth_trend(dates, values, sampling_days=sampling_days, **fit_kwargs)
    sol = fit._solution
    assert sol is not None
    start, step = sol.epochs[0], sol.sampling_days
    if future_dates is None:
        assert horizon_years is not None
        n_ahead = int(np.floor(horizon_years * DAYS_PER_YEAR / step))
        fut_idx = sol.obs_idx[-1] + np.arange(1, n_ahead + 1)
        future = start + pd.to_timedelta(fut_idx * step, unit="D")
    else:
        future = pd.DatetimeIndex(pd.to_datetime(np.asarray(future_dates)))
        days = np.asarray((future - start).total_seconds()) / 86400.0
        fut_idx = np.round(days / step).astype(int)
    if len(fut_idx) == 0 or fut_idx.min() <= sol.obs_idx[-1]:
        msg = "Forecast epochs must all come after the last observation"
        raise ValueError(msg)

    to_years = step / DAYS_PER_YEAR
    t_future = fut_idx * to_years
    t_last = float(sol.obs_idx[-1] * to_years)
    # Every step has happened by now
    x_f, _ = _design_matrix(
        t_future,
        1,
        sol.periods_years,
        np.zeros(sol.n_steps),
        [],
        [],
        sol.rate_change_years,
    )
    _, future_reg = _future_regressors(
        sol.regressor_names, future_regressors, len(t_future)
    )
    design = np.column_stack(
        [x_f, future_reg, np.maximum(0.0, t_future[:, None] - sol.knots)]
    )
    model = design @ sol.coefficients
    var_model = np.einsum("ij,jk,ik->i", design, sol.covariance, design)
    if sol.lam > 0:
        # Slope changes still to come have zero mean and the fitted variance
        last = sol.knots[-1] if len(sol.knots) else t_last
        coming = np.arange(last + sol.knot_step, t_future.max(), sol.knot_step)
        ramps = np.maximum(0.0, t_future[:, None] - coming)
        var_model = var_model + sol.s2 * sol.lam * np.sum(ramps**2, axis=1)

    n_rates = len(sol.rate_change_years)
    if n_rates and t_last > 0:
        # As `forecast_trend` simulates: Poisson-timed changes at the record's
        # rate with Laplace sizes of its mean magnitude, in closed form
        # A piecewise fit has no slope changes: its last terms are the rates
        # and regressors
        n_reg = len(sol.regressor_names)
        sizes = sol.coefficients[-(n_rates + n_reg) : len(sol.coefficients) - n_reg]
        scale = float(np.mean(np.abs(sizes)))
        var_model = var_model + (n_rates / t_last) * 2 * scale**2 * (
            (t_future - t_last) ** 3 / 3.0
        )

    mean = model.copy()
    var_noise = np.zeros(len(t_future))
    if include_noise:
        n = len(sol.y)
        idx = np.r_[sol.obs_idx, fut_idx]
        flicker = _powerlaw_covariance(-1.0, int(idx.max()) + 1)[np.ix_(idx, idx)]
        cov = _noise_mixture(flicker, sol.phi)
        cho = linalg.cho_factor(cov[:n, :n], lower=True, check_finite=False)
        gain = linalg.cho_solve(cho, cov[:n, n:], check_finite=False).T
        mean = model + gain @ (sol.y - fit.model)
        var_noise = sol.s2 * np.diag(cov[n:, n:] - gain @ cov[:n, n:])

    sigma_model = np.sqrt(np.clip(var_model, 0.0, None))
    sigma_noise = np.sqrt(np.clip(var_noise, 0.0, None))
    half = stats.norm.ppf(0.5 + level / 2.0) * np.hypot(sigma_model, sigma_noise)
    table = pd.DataFrame(
        {
            "lead_years": t_future - t_last,
            "model": model,
            "mean": mean,
            "lower": mean - half,
            "upper": mean + half,
            "sigma_model": sigma_model,
            "sigma_noise": sigma_noise,
        },
        index=pd.DatetimeIndex(future, name="date"),
    )
    return SmoothForecast(table=table, level=level, fit=fit)

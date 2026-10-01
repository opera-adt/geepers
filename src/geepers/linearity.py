"""Test whether a single rate describes a time series, and for how long.

A velocity is only worth extrapolating if the series is linear. This
module compares a linear model against a quadratic and a continuous
piecewise-linear one (a hinge with a free breakpoint) under the *same*
noise covariance, so that no model wins by absorbing a different share
of the temporally correlated noise. That covariance is the one
`geepers.trend.estimate_trend` estimates around the quadratic model:
around a straight line, real curvature is indistinguishable from - and
gets absorbed into - a steeper noise spectrum. `validity_horizon` then bounds how
far the rate can be carried before the extrapolation is off by a stated
tolerance.

The comparison uses the Bayesian information criterion (Schwarz, G.,
1978, Estimating the dimension of a model, Ann. Statist., 6(2), 461-464,
https://doi.org/10.1214/aos/1176344136). Because the breakpoint is
searched for, the piecewise test statistic is the supremum over the
candidate breakpoints, which does not follow a chi-square distribution
(Davies, R. B., 1987, Hypothesis testing when a nuisance parameter is
present only under the alternative, Biometrika, 74(1), 33-43,
https://doi.org/10.1093/biomet/74.1.33); its p-value is therefore
obtained by simulating the statistic under the fitted noise model.

Example:
-------
>>> result = linearity_test(dates, up_mm, sampling_days=12)   # doctest: +SKIP
>>> result.model, result.departure                            # doctest: +SKIP
>>> validity_horizon(result, tolerance=10.0).years            # doctest: +SKIP

"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg, stats

from geepers.trend import (
    DAYS_PER_YEAR,
    TrendResult,
    _design_matrix,
    _powerlaw_covariance,
    estimate_trend,
)

__all__ = ["LinearityResult", "ValidityHorizon", "linearity_test", "validity_horizon"]

Model = Literal["linear", "quadratic", "piecewise"]


@dataclass
class LinearityResult:
    """Outcome of the linearity test.

    Attributes
    ----------
    model : {"linear", "quadratic", "piecewise"}
        Model with the lowest BIC.
    trend : TrendResult
        The linear fit and its own noise model (`trend.velocity`,
        `trend.velocity_uncertainty`): the rate one would extrapolate.
        Where the series is not linear its uncertainty is inflated,
        because the non-linearity counts as noise.
    bic : dict[str, float]
        BIC of each model, up to a common constant.
    p_values : dict[str, float]
        p-value of each alternative against the linear model. The
        piecewise one accounts for the search over breakpoints.
    acceleration, acceleration_sigma : float
        Acceleration of the quadratic model and its 1-sigma, in input
        units per year squared (reported whichever model is preferred).
    breakpoint : pd.Timestamp or None
        Best-fitting breakpoint of the piecewise model (None if the
        series is too short for one).
    rate_before, rate_after : float
        Rates on either side of `breakpoint`, input units per year.
    rate_change_sigma : float
        1-sigma of ``rate_after - rate_before`` at the fitted breakpoint.
    departure : float
        Largest separation between the preferred model and the straight
        line over the observed span, in input units (0 for "linear").

    """

    model: Model
    trend: TrendResult
    bic: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    acceleration: float = np.nan
    acceleration_sigma: float = np.nan
    breakpoint: pd.Timestamp | None = None
    rate_before: float = np.nan
    rate_after: float = np.nan
    rate_change_sigma: float = np.nan
    departure: float = 0.0


@dataclass
class ValidityHorizon:
    """How far a rate can be extrapolated within a tolerance.

    Attributes
    ----------
    years : float
        Time until the expected extrapolation error reaches the tolerance.
    from_rate : float
        The same horizon if only the rate uncertainty counted.
    driver : {"rate uncertainty", "non-linearity", "capped"}
        What sets `years`.
    tolerance : float
        The tolerance used, in input units.

    """

    years: float
    from_rate: float
    driver: str
    tolerance: float


def _noise_covariance(
    trend: TrendResult, obs_idx: np.ndarray, n_epochs: int
) -> np.ndarray:
    """Noise covariance of a fitted `TrendResult` at the observed epochs."""
    select = np.ix_(obs_idx, obs_idx)
    components = (
        (trend.sigma_powerlaw, trend.kappa),
        (trend.sigma_flicker, -1.0),
        (trend.sigma_randomwalk, -2.0),
    )
    C = trend.sigma_white**2 * np.eye(len(obs_idx))
    for sigma, kappa in components:
        # Components a noise model does not have are NaN
        if np.isfinite(sigma) and sigma > 0:
            C += sigma**2 * _powerlaw_covariance(kappa, n_epochs)[select]
    return C


def _fit(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares on whitened data: coefficients, covariance, chi-square."""
    beta, *_ = linalg.lstsq(design, y, check_finite=False)
    resid = y - design @ beta
    cov = linalg.inv(design.T @ design)
    return beta, cov, float(resid @ resid)


def linearity_test(
    dates: ArrayLike,
    values: ArrayLike,
    *,
    sampling_days: float = 1.0,
    periods_years: tuple[float, ...] = (1.0, 0.5),
    step_dates: ArrayLike | None = None,
    noise_model: Literal["PLWN", "PL", "WN", "FNWN", "RWFNWN"] = "FNWN",
    method: Literal["exact", "whittle"] = "exact",
    min_segment_years: float = 1.5,
    n_breakpoints: int = 50,
    n_simulations: int = 2000,
    seed: int | np.random.Generator | None = 0,
) -> LinearityResult:
    """Decide whether a single rate describes a time series.

    Estimates the noise around a quadratic model with
    `geepers.trend.estimate_trend`, holds that covariance fixed, and
    compares a linear, a quadratic and a continuous piecewise-linear
    model, preferring the lowest BIC. A free breakpoint is charged as
    two parameters.

    Curvature and strongly correlated noise look alike over one record,
    so the verdict depends on the noise model. The default is flicker +
    white ("FNWN"), the usual model of GNSS noise. Letting the spectral
    index go free ("PLWN") is not a safer choice: on real station series
    with obvious rate changes, offsets or accelerating subsidence the
    index runs to its steep limit and explains all of it as noise, so
    every series is called linear (with a correspondingly large rate
    uncertainty). The price of the fixed index is that a linear series
    whose noise really is a random walk is flagged about one time in
    five. An unmodeled offset is usually reported as non-linearity
    (one at mid-record instead biases the rate unnoticed); pass known
    offsets as `step_dates`. A rate change near mid-record is usually reported as
    "quadratic": the two shapes are close, and the hinge costs one more
    parameter.

    Parameters
    ----------
    dates : array-like of datetime64
        Observation epochs; gaps are allowed.
    values : array-like of float
        Observations (any unit); NaNs are dropped.
    sampling_days : float
        Nominal sampling interval in days. Default 1.
    periods_years : tuple of float
        Periodic terms included in every model. Default annual and
        semi-annual.
    step_dates : array-like of datetime64, optional
        Known offsets, estimated in every model.
    noise_model : {"PLWN", "PL", "WN", "FNWN", "RWFNWN"}
        Noise model of `estimate_trend`. Default "FNWN".
    method : {"exact", "whittle"}
        Noise estimation method of `estimate_trend`.
    min_segment_years : float
        Shortest segment allowed on either side of a breakpoint; a hinge
        near the ends just fits the last few epochs. Default 1.5.
    n_breakpoints : int
        Number of candidate breakpoints, evenly spaced. Default 50.
    n_simulations : int
        Noise realizations for the piecewise p-value. Default 2000.
    seed : int or np.random.Generator, optional
        Seed for those realizations. Default 0 (reproducible).

    Returns
    -------
    LinearityResult

    """
    epochs = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    y_all = np.asarray(values, dtype=float)
    good = np.isfinite(y_all)
    epochs, y = epochs[good], y_all[good]

    fit = partial(
        estimate_trend,
        epochs,
        y,
        sampling_days=sampling_days,
        periods_years=periods_years,
        step_dates=step_dates,
        noise_model=noise_model,
        method=method,
    )
    trend = fit(poly_deg=1)
    # Noise for the comparison comes from the quadratic fit: estimated around
    # a straight line, real curvature would be absorbed into the noise as a
    # steeper spectrum and the test would have little power
    noise = fit(poly_deg=2)

    # Same epoch grid and time axis as `estimate_trend`
    t_days = np.asarray((epochs - epochs[0]).total_seconds()) / 86400.0
    obs_idx = np.round(t_days / sampling_days).astype(int)
    t = obs_idx * sampling_days / DAYS_PER_YEAR
    steps = pd.DatetimeIndex(
        pd.to_datetime(
            np.atleast_1d(np.asarray([] if step_dates is None else step_dates))
        )
    )
    step_years = (
        np.asarray((steps - epochs[0]).total_seconds()) / 86400.0 / DAYS_PER_YEAR
    )
    A_lin, names = _design_matrix(t, 1, periods_years, step_years, [], [])
    itrend = names.index("trend")

    # Whiten with that one noise covariance: every model below is then an
    # ordinary least-squares problem with unit-variance residuals
    C = _noise_covariance(noise, obs_idx, int(obs_idx[-1]) + 1)
    chol = linalg.cholesky(C, lower=True, check_finite=False)

    def whiten(x: np.ndarray) -> np.ndarray:
        return linalg.solve_triangular(chol, x, lower=True, check_finite=False)

    n = len(y)
    k_lin = A_lin.shape[1]
    yw, Aw = whiten(y), whiten(A_lin)
    beta_lin, _, chi2_lin = _fit(Aw, yw)
    linear_fit = A_lin @ beta_lin

    # --- quadratic, centered so the trend term stays the mid-span rate
    quad = (t - t.mean()) ** 2
    A_quad = np.column_stack([A_lin, quad])
    beta_q, cov_q, chi2_q = _fit(whiten(A_quad), yw)

    bic = {"linear": k_lin * np.log(n) + chi2_lin}
    bic["quadratic"] = (k_lin + 1) * np.log(n) + chi2_q
    p_values = {"quadratic": float(stats.chi2.sf(max(chi2_lin - chi2_q, 0.0), 1))}
    result = LinearityResult(
        model="linear",
        trend=trend,
        acceleration=float(2 * beta_q[-1]),
        acceleration_sigma=float(2 * np.sqrt(cov_q[-1, -1])),
    )

    # --- continuous piecewise linear: + c * max(0, t - t_break)
    lo, hi = t[0] + min_segment_years, t[-1] - min_segment_years
    hinge_fit = None
    if hi > lo:
        breaks = np.linspace(lo, hi, n_breakpoints)
        hinges = np.maximum(0.0, t[:, None] - breaks)
        # Part of each whitened hinge the linear model cannot explain
        q, _ = linalg.qr(Aw, mode="economic", check_finite=False)
        hw = whiten(hinges)
        hw -= q @ (q.T @ hw)
        norms = np.linalg.norm(hw, axis=0)
        usable = norms > 1e-10 * np.linalg.norm(whiten(hinges), axis=0)
        unit = hw[:, usable] / norms[usable]
        gain = (unit.T @ (yw - q @ (q.T @ yw))) ** 2  # chi-square drop per break
        best = int(np.argmax(gain))
        t_break = float(breaks[usable][best])

        A_hinge = np.column_stack([A_lin, np.maximum(0.0, t - t_break)])
        beta_h, cov_h, chi2_h = _fit(whiten(A_hinge), yw)
        bic["piecewise"] = (k_lin + 2) * np.log(n) + chi2_h
        hinge_fit = A_hinge @ beta_h

        # Null distribution of the best gain over all breakpoints: the same
        # statistic on pure whitened noise
        rng = np.random.default_rng(seed)
        null = ((rng.standard_normal((n_simulations, n)) @ unit) ** 2).max(axis=1)
        p_values["piecewise"] = float(
            (1 + np.sum(null >= gain[best])) / (1 + n_simulations)
        )

        result.breakpoint = epochs[0] + pd.Timedelta(days=t_break * DAYS_PER_YEAR)
        result.rate_before = float(beta_h[itrend])
        result.rate_after = float(beta_h[itrend] + beta_h[-1])
        result.rate_change_sigma = float(np.sqrt(cov_h[-1, -1]))

    result.bic = {name: float(v) for name, v in bic.items()}
    result.p_values = p_values
    models: tuple[Model, ...] = ("linear", "quadratic", "piecewise")
    result.model = min((m for m in models if m in bic), key=lambda m: bic[m])
    if result.model == "quadratic":
        result.departure = float(np.max(np.abs(A_quad @ beta_q - linear_fit)))
    elif result.model == "piecewise":
        assert hinge_fit is not None
        result.departure = float(np.max(np.abs(hinge_fit - linear_fit)))
    return result


def validity_horizon(
    result: LinearityResult,
    tolerance: float,
    *,
    cap_years: float = np.inf,
) -> ValidityHorizon:
    """Bound how far the rate can be extrapolated within `tolerance`.

    The expected extrapolation error after ``T`` years is the rate
    uncertainty, ``sigma_v * T``, plus the non-linearity the test found:

    - "quadratic": ``0.5 * |a| * T**2`` when the acceleration is resolved
      at 2 sigma;
    - "piecewise": ``|rate_after - rate_before| * T`` - a rate that has
      changed once can change again, so the observed change is taken as
      the scale of what may happen next.

    The horizon is the ``T`` at which the sum reaches `tolerance`.

    Parameters
    ----------
    result : LinearityResult
        Output of `linearity_test`.
    tolerance : float
        Acceptable extrapolation error, in the units of the series.
    cap_years : float
        Upper limit on the reported horizon. Default no limit.

    Returns
    -------
    ValidityHorizon

    """
    sigma = result.trend.velocity_uncertainty
    from_rate = tolerance / sigma if sigma > 0 else np.inf

    years = from_rate
    accel = abs(result.acceleration)
    if result.model == "quadratic" and accel > 2 * result.acceleration_sigma:
        # Positive root of 0.5 |a| T^2 + sigma T - tolerance = 0
        years = (-sigma + np.sqrt(sigma**2 + 2 * accel * tolerance)) / accel
    elif result.model == "piecewise":
        years = tolerance / (sigma + abs(result.rate_after - result.rate_before))

    driver = "non-linearity" if years < from_rate else "rate uncertainty"
    if years >= cap_years:
        years, driver = cap_years, "capped"
    return ValidityHorizon(
        years=float(years),
        from_rate=float(from_rate),
        driver=driver,
        tolerance=tolerance,
    )

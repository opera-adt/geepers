# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Find an unknown number of offsets and rate changes in a time series.

`geepers.trend.estimate_trend` needs the epochs of offsets and rate
changes, `geepers.steps.detect_steps` finds offsets only, and
`geepers.linearity.linearity_test` allows a single hinge. This module
estimates up to `max_changepoints` change points at once. Each one is an
offset, a rate change (the series stays continuous and its slope
changes), or both, and is fitted together with the trend, the periodic
terms, the known steps and temporally correlated noise. The rate after
the last change point is the present-day rate.

Two estimators share one model:

- `detect_changepoints` searches by forward-backward stepwise selection
  on the Bayesian information criterion (Schwarz, G., 1978, Estimating
  the dimension of a model, Ann. Statist., 6(2), 461-464,
  https://doi.org/10.1214/aos/1176344136). Every model is compared under
  the same noise covariance from `estimate_trend`, which is re-estimated
  with the change points included until the selection stops changing:
  estimated without them, offsets and rate changes are absorbed into a
  larger flicker amplitude.
- `sample_changepoints` samples the posterior distribution of the number,
  epochs and kinds of change points by reversible-jump Markov chain Monte
  Carlo (Green, P. J., 1995, Reversible jump Markov chain Monte Carlo
  computation and Bayesian model determination, Biometrika, 82(4),
  711-732, https://doi.org/10.1093/biomet/82.4.711). The offset and
  rate-change sizes have Gaussian priors and the noise covariance is held
  at the `detect_changepoints` estimate, so every amplitude is integrated
  out analytically. The chain then only moves the change points, and
  each step costs a solve the size of the model, not of the series.

The model and its Bernoulli-per-slot prior on the number of change
points follow DiscoTimeS (Oelsmann, J., Passaro, M., Sanchez, L.,
Dettmering, D., Schwatke, C., & Seitz, F., 2022, Bayesian modelling of
piecewise trends and discontinuities to improve the estimation of coastal
vertical land motion, J. Geod., 96, 62,
https://doi.org/10.1007/s00190-022-01645-6). That package samples every
parameter with PyMC3; no DiscoTimeS (GPL) code is included here, and
this implementation is written from the paper. It differs in its noise
model (the power-law noise of `estimate_trend` with fixed parameters,
instead of AR(1) estimated jointly) and in placing change points on a
grid of candidate epochs.

Example:
-------
>>> found = detect_changepoints(dates, up_mm, sampling_days=30)   # doctest: +SKIP
>>> found.changepoints, found.rate, found.rate_sigma              # doctest: +SKIP
>>> post = sample_changepoints(dates, up_mm, sampling_days=30)    # doctest: +SKIP
>>> post.n_probability, post.rate, post.rate_sigma                # doctest: +SKIP

"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg, special

from geepers.linearity import _noise_covariance
from geepers.trend import DAYS_PER_YEAR, TrendResult, _design_matrix, estimate_trend

__all__ = [
    "ChangepointPosterior",
    "ChangepointResult",
    "detect_changepoints",
    "sample_changepoints",
]

Kind = Literal["offset", "rate", "both"]
NoiseModel = Literal["PLWN", "PL", "WN", "FNWN", "RWFNWN"]
_PARTS: dict[str, tuple[str, ...]] = {
    "offset": ("offset",),
    "rate": ("rate",),
    "both": ("offset", "rate"),
}
# A change point is the index of its candidate epoch plus its kind
Config = tuple[tuple[int, str], ...]


@dataclass
class ChangepointResult:
    """Change points found by `detect_changepoints`.

    Attributes
    ----------
    changepoints : pd.DataFrame
        One row per change point, in time order, with columns ``date``
        (first epoch after the change), ``kind`` ("offset", "rate" or
        "both"), ``offset`` and ``offset_sigma`` (input units),
        ``rate_change`` and ``rate_change_sigma`` (input units per year).
        A part a change point does not have is NaN.
    segments : pd.DataFrame
        The trend between consecutive change points: ``start``, ``end``,
        ``rate`` and ``rate_sigma`` (input units per year).
    rate, rate_sigma : float
        Rate of the last segment (the present-day rate) and its 1-sigma.
    trend : TrendResult
        `estimate_trend` fit with the change points included; its noise
        parameters are the ones the selection used.
    bic, bic_none : float
        Criterion (BIC unless `penalty` is set) of the selected model and
        of the model without change points, under the same noise
        covariance, up to a common constant.
    model : np.ndarray
        Fitted model at the observed (finite) epochs.

    """

    changepoints: pd.DataFrame
    segments: pd.DataFrame
    rate: float
    rate_sigma: float
    trend: TrendResult
    bic: float
    bic_none: float
    model: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ChangepointPosterior:
    """Posterior distribution of change points from `sample_changepoints`.

    Attributes
    ----------
    changepoints : pd.DataFrame
        Summary configuration: the posterior-mode number of change
        points, placed where the posterior concentrates. Columns
        ``date``, ``kind`` (most frequent kind near that date),
        ``probability`` (fraction of samples with a change point within
        half `min_segment_years` of it), ``date_low`` and ``date_high``
        (95 % interval of that change point's epoch), and ``offset``,
        ``offset_sigma``, ``rate_change``, ``rate_change_sigma``
        (posterior given this configuration).
    rate, rate_sigma : float
        Posterior mean and standard deviation of the present-day rate,
        averaged over all configurations.
    rate_samples : np.ndarray
        Draws of the present-day rate, one per sample.
    n_probability : pd.Series
        Posterior probability of each number of change points.
    probability : pd.DataFrame
        Indexed by candidate epoch: posterior probability that a change
        point sits there (``any``), or one with an offset (``offset``)
        or a rate change (``rate``).
    acceptance_rate : float
        Fraction of proposals accepted.
    rhat : dict[str, float]
        Gelman-Rubin statistic of the number of change points (``n``) and
        of the present-day rate (``rate``) across chains; values above
        about 1.05 mean the chains have not mixed.
    detection : ChangepointResult
        The `detect_changepoints` result that supplied the noise
        covariance and the starting configuration.

    """

    changepoints: pd.DataFrame
    rate: float
    rate_sigma: float
    rate_samples: np.ndarray
    n_probability: pd.Series
    probability: pd.DataFrame
    acceptance_rate: float
    rhat: dict[str, float]
    detection: ChangepointResult


@dataclass
class _Series:
    """A series on the `estimate_trend` time axis, with candidate epochs."""

    epochs: pd.DatetimeIndex
    y: np.ndarray
    obs_idx: np.ndarray
    t: np.ndarray
    base: np.ndarray
    names: list[str]
    known_steps: list[pd.Timestamp]
    candidates: np.ndarray  # observation index of the first epoch after a change

    def column(self, j: int, part: str) -> np.ndarray:
        i = self.candidates[j]
        if part == "offset":
            return (np.arange(len(self.y)) >= i).astype(float)
        return np.maximum(0.0, self.t - self.t[i])

    def date(self, j: int) -> pd.Timestamp:
        return self.epochs[self.candidates[j]]


def _prepare(
    dates: ArrayLike,
    values: ArrayLike,
    sampling_days: float,
    periods_years: tuple[float, ...],
    step_dates: ArrayLike | None,
    min_segment_years: float,
    n_candidates: int,
) -> _Series:
    epochs = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    y_all = np.asarray(values, dtype=float)
    good = np.isfinite(y_all)
    epochs, y = epochs[good], y_all[good]

    # Same epoch grid and time axis as `estimate_trend`
    t_days = np.asarray((epochs - epochs[0]).total_seconds()) / 86400.0
    obs_idx = np.round(t_days / sampling_days).astype(int)
    t = obs_idx * sampling_days / DAYS_PER_YEAR
    known = list(
        pd.DatetimeIndex(
            pd.to_datetime(
                np.atleast_1d(np.asarray([] if step_dates is None else step_dates))
            )
        )
    )
    step_years = np.array(
        [(s - epochs[0]).total_seconds() / 86400.0 / DAYS_PER_YEAR for s in known]
    )
    base, names = _design_matrix(t, 1, periods_years, step_years, [], [])

    inside = np.flatnonzero(
        (t >= t[0] + min_segment_years) & (t <= t[-1] - min_segment_years)
    )
    if len(inside) > n_candidates:
        pick = np.round(np.linspace(0, len(inside) - 1, n_candidates)).astype(int)
        inside = inside[np.unique(pick)]
    return _Series(epochs, y, obs_idx, t, base, names, known, inside)


class _Normal:
    """Normal equations of every candidate model under one noise covariance.

    The base design and all candidate columns are whitened once; a model
    is then a subset of rows and columns of `gram`.
    """

    def __init__(
        self, series: _Series, covariance: np.ndarray, penalty: float | None = None
    ) -> None:
        chol = linalg.cholesky(covariance, lower=True, check_finite=False)
        n_cand = len(series.candidates)
        extra = [series.column(j, "offset") for j in range(n_cand)] + [
            series.column(j, "rate") for j in range(n_cand)
        ]
        X = np.column_stack([series.base, *extra]) if extra else series.base
        Xw = linalg.solve_triangular(chol, X, lower=True, check_finite=False)
        yw = linalg.solve_triangular(chol, series.y, lower=True, check_finite=False)
        self.gram = Xw.T @ Xw
        self.proj = Xw.T @ yw
        self.yty = float(yw @ yw)
        self.n_obs = len(series.y)
        self.penalty = np.log(self.n_obs) if penalty is None else penalty
        self.n_base = series.base.shape[1]
        self.n_cand = n_cand

    def columns(self, config: Config) -> tuple[np.ndarray, list[tuple[int, str]]]:
        """Column indices of a configuration and the (change point, part) of each."""
        labels = [(j, part) for j, kind in config for part in _PARTS[kind]]
        extra = [
            self.n_base + j + (self.n_cand if part == "rate" else 0)
            for j, part in labels
        ]
        return np.r_[np.arange(self.n_base), extra].astype(int), labels

    def solve(
        self, cols: np.ndarray, prior_precision: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Coefficients, their covariance, ``y' y - b' beta`` and ``log|Lambda|``."""
        lam = self.gram[np.ix_(cols, cols)]
        if prior_precision is not None:
            lam = lam + np.diag(prior_precision)
        cho = linalg.cho_factor(lam, lower=True, check_finite=False)
        b = self.proj[cols]
        beta = linalg.cho_solve(cho, b, check_finite=False)
        cov = linalg.cho_solve(cho, np.eye(len(cols)), check_finite=False)
        logdet = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
        return beta, cov, self.yty - float(b @ beta), logdet

    def bic(self, config: Config) -> float:
        cols, _ = self.columns(config)
        try:
            *_, chi2, _ = self.solve(cols)
        except linalg.LinAlgError:
            return np.inf
        # Each change point costs its amplitudes plus its epoch
        k = len(cols) + len(config)
        return chi2 + k * self.penalty


def _separated(series: _Series, config: Config, j: int, min_years: float) -> bool:
    t_j = series.t[series.candidates[j]]
    return all(
        abs(series.t[series.candidates[i]] - t_j) >= min_years
        for i, _ in config
        if i != j
    )


def _stepwise(
    normal: _Normal,
    series: _Series,
    kinds: tuple[str, ...],
    max_changepoints: int,
    min_segment_years: float,
) -> Config:
    """Stepwise selection: add the best change point, then revise the others.

    After each addition, every change point may be dropped, given another
    kind or moved, whichever lowers the BIC most, until none does: an
    early change point otherwise keeps a rate change or a position that
    only compensated for one found later. BIC strictly decreases with
    every accepted move, so the search terminates.
    """

    def additions(config: Config) -> list[Config]:
        return [
            tuple(sorted((*config, (j, kind))))
            for j in range(normal.n_cand)
            if j not in {i for i, _ in config}
            and _separated(series, config, j, min_segment_years)
            for kind in kinds
        ]

    def revisions(config: Config) -> list[Config]:
        out = []
        for i in range(len(config)):
            rest = config[:i] + config[i + 1 :]
            out.append(rest)
            out.extend(additions(rest))  # every other kind and position
        return out

    def best(trials: list[Config], current: float) -> tuple[float, Config] | None:
        scored = [(normal.bic(trial), trial) for trial in trials]
        top = min(scored, default=None, key=lambda s: s[0])
        return top if top is not None and top[0] < current else None

    config: Config = ()
    bic = normal.bic(config)
    while len(config) < max_changepoints:
        step = best(additions(config), bic)
        if step is None:
            break
        bic, config = step
        while (step := best(revisions(config), bic)) is not None:
            bic, config = step
    return config


def _fit_trend(
    series: _Series,
    config: Config,
    sampling_days: float,
    periods_years: tuple[float, ...],
    noise_model: NoiseModel,
    method: Literal["exact", "whittle"],
) -> TrendResult:
    def at(years: float) -> pd.Timestamp:
        return series.epochs[0] + pd.Timedelta(days=years * DAYS_PER_YEAR)

    offsets, rates = [], []
    for j, kind in config:
        i = series.candidates[j]
        if "offset" in _PARTS[kind]:
            # Between two epochs, so the step column is the same whichever
            # way `estimate_trend` rounds the epoch
            offsets.append(at(0.5 * (series.t[i - 1] + series.t[i])))
        if "rate" in _PARTS[kind]:
            rates.append(at(series.t[i]))
    return estimate_trend(
        series.epochs,
        series.y,
        sampling_days=sampling_days,
        periods_years=periods_years,
        step_dates=series.known_steps + offsets,
        rate_change_dates=rates,
        noise_model=noise_model,
        method=method,
    )


def _amplitudes(
    series: _Series,
    labels: list[tuple[int, str]],
    beta: np.ndarray,
    cov: np.ndarray,
    config: Config,
) -> pd.DataFrame:
    rows = []
    for j, kind in config:
        row: dict = {"date": series.date(j), "kind": kind}
        for part, name in (("offset", "offset"), ("rate", "rate_change")):
            if (j, part) in labels:
                k = series.base.shape[1] + labels.index((j, part))
                row[name], row[f"{name}_sigma"] = beta[k], np.sqrt(cov[k, k])
            else:
                row[name] = row[f"{name}_sigma"] = np.nan
        rows.append(row)
    columns = [
        "date",
        "kind",
        "offset",
        "offset_sigma",
        "rate_change",
        "rate_change_sigma",
    ]
    return pd.DataFrame(rows, columns=columns)


def _rate_vector(
    series: _Series, labels: list[tuple[int, str]], until: int | None = None
) -> np.ndarray:
    """Weights summing the coefficients to the rate after `until` change points."""
    n_base = series.base.shape[1]
    c = np.zeros(n_base + len(labels))
    c[series.names.index("trend")] = 1.0
    order = sorted({j for j, _ in labels})
    keep = set(order if until is None else order[:until])
    for k, (j, part) in enumerate(labels):
        if part == "rate" and j in keep:
            c[n_base + k] = 1.0
    return c


def _check_kinds(kinds: tuple[str, ...]) -> tuple[Kind, ...]:
    unknown = set(kinds) - set(_PARTS)
    if unknown or not kinds:
        msg = f"kinds must be a non-empty subset of {tuple(_PARTS)}, got {kinds}"
        raise ValueError(msg)
    return tuple(dict.fromkeys(kinds))  # type: ignore[arg-type]


def detect_changepoints(
    dates: ArrayLike,
    values: ArrayLike,
    *,
    sampling_days: float = 1.0,
    periods_years: tuple[float, ...] = (1.0, 0.5),
    step_dates: ArrayLike | None = None,
    kinds: tuple[Kind, ...] = ("offset", "rate", "both"),
    max_changepoints: int = 5,
    min_segment_years: float = 1.0,
    n_candidates: int = 300,
    noise_model: NoiseModel = "FNWN",
    method: Literal["exact", "whittle"] = "exact",
    max_iterations: int = 3,
    penalty: float | None = None,
) -> ChangepointResult:
    """Find offsets and rate changes by stepwise BIC selection.

    Starting from no change points, repeatedly adds the (epoch, kind)
    that lowers the BIC most and drops any change point whose removal
    lowers it, until no addition helps. An offset or a rate change costs
    two parameters (amplitude and epoch), one with both three. The noise
    covariance is estimated with `geepers.trend.estimate_trend` around
    the current selection and the search is repeated, up to
    `max_iterations` times, until the selection stops changing.

    The BIC is computed under the estimated noise covariance, so the
    default flicker + white noise ("FNWN") matters as it does for
    `geepers.linearity.linearity_test`: a free spectral index ("PLWN")
    tends to explain offsets and rate changes as steep noise. Change
    points are searched on `n_candidates` epochs, so their dates are
    resolved to the spacing of that grid.

    Parameters
    ----------
    dates : array-like of datetime64
        Observation epochs; gaps are allowed.
    values : array-like of float
        Observations (any unit); NaNs are dropped.
    sampling_days : float
        Nominal sampling interval in days. Default 1.
    periods_years : tuple of float
        Periodic terms in every model. Default annual and semi-annual.
    step_dates : array-like of datetime64, optional
        Known offsets (equipment changes, earthquakes), estimated in
        every model and not counted as change points.
    kinds : tuple of {"offset", "rate", "both"}
        Kinds of change point allowed. Default all three.
    max_changepoints : int
        Upper limit on the number of change points. Default 5.
    min_segment_years : float
        Shortest separation between change points and between a change
        point and either end of the record. Default 1.
    n_candidates : int
        Number of candidate epochs, evenly spaced among the observations.
        Default 300.
    noise_model : {"PLWN", "PL", "WN", "FNWN", "RWFNWN"}
        Noise model of `estimate_trend`. Default "FNWN".
    method : {"exact", "whittle"}
        Noise estimation method of `estimate_trend`.
    max_iterations : int
        Times the noise is re-estimated. Default 3.
    penalty : float, optional
        Cost of each parameter in the criterion. Default ``log(n)``, the
        BIC. On long, temporally correlated series (daily GNSS) ``log(n)``
        counts samples that carry little independent information and the
        selection is conservative; 2 gives the AIC.

    Returns
    -------
    ChangepointResult

    Raises
    ------
    ValueError
        If `kinds` names an unknown kind, or (from `estimate_trend`) the
        series has fewer than 10 finite values.

    """
    kinds = _check_kinds(kinds)
    series = _prepare(
        dates,
        values,
        sampling_days,
        periods_years,
        step_dates,
        min_segment_years,
        n_candidates,
    )
    n_epochs = int(series.obs_idx[-1]) + 1

    def refit(config: Config) -> tuple[TrendResult, _Normal]:
        trend = _fit_trend(
            series, config, sampling_days, periods_years, noise_model, method
        )
        C = _noise_covariance(trend, series.obs_idx, n_epochs)
        return trend, _Normal(series, C, penalty)

    config: Config = ()
    trend, normal = refit(config)
    for _ in range(max_iterations):
        selected = _stepwise(normal, series, kinds, max_changepoints, min_segment_years)
        if selected == config:
            break
        config = selected
        trend, normal = refit(config)

    cols, labels = normal.columns(config)
    beta, cov, *_ = normal.solve(cols)
    design = np.column_stack(
        [series.base, *(series.column(j, part) for j, part in labels)]
    )

    bounds = [series.epochs[0], *(series.date(j) for j, _ in config), series.epochs[-1]]
    segments = []
    for k in range(len(config) + 1):
        c = _rate_vector(series, labels, until=k)
        segments.append(
            {
                "start": bounds[k],
                "end": bounds[k + 1],
                "rate": float(c @ beta),
                "rate_sigma": float(np.sqrt(c @ cov @ c)),
            }
        )
    return ChangepointResult(
        changepoints=_amplitudes(series, labels, beta, cov, config),
        segments=pd.DataFrame(segments),
        rate=segments[-1]["rate"],
        rate_sigma=segments[-1]["rate_sigma"],
        trend=trend,
        bic=float(normal.bic(config)),
        bic_none=float(normal.bic(())),
        model=design @ beta,
    )


def _gelman_rubin(chains: np.ndarray) -> float:
    """Potential scale reduction of a (chains, draws) array (Gelman & Rubin, 1992)."""
    n = chains.shape[1]
    within = float(np.mean(np.var(chains, axis=1, ddof=1)))
    between = n * float(np.var(np.mean(chains, axis=1), ddof=1))
    if within == 0.0:
        return 1.0 if between == 0.0 else np.inf
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def sample_changepoints(
    dates: ArrayLike,
    values: ArrayLike,
    *,
    sampling_days: float = 1.0,
    periods_years: tuple[float, ...] = (1.0, 0.5),
    step_dates: ArrayLike | None = None,
    kinds: tuple[Kind, ...] = ("offset", "rate", "both"),
    max_changepoints: int = 5,
    min_segment_years: float = 1.0,
    n_candidates: int = 300,
    p_change: float = 0.1,
    offset_scale: float | None = None,
    rate_scale: float | None = None,
    n_chains: int = 4,
    n_samples: int = 5000,
    n_burn: int = 1000,
    noise_model: NoiseModel = "FNWN",
    method: Literal["exact", "whittle"] = "exact",
    detection: ChangepointResult | None = None,
    seed: int | np.random.Generator | None = 0,
) -> ChangepointPosterior:
    """Sample the posterior of the number, epochs and kinds of change points.

    The prior gives each of `max_changepoints` slots a change point with
    probability `p_change`, places change points uniformly on the
    candidate epochs (at least `min_segment_years` apart), and draws each
    kind uniformly from `kinds`. Offsets and rate changes are Gaussian
    with standard deviations `offset_scale` and `rate_scale`; trend,
    periodic terms and known steps have flat priors. The noise covariance
    is that of `detect_changepoints`, held fixed.

    With the amplitudes integrated out, the chain proposes adding a
    change point, removing one, or moving one or changing its kind,
    accepted by the Metropolis-Hastings-Green ratio. Half of the chains
    start from the `detect_changepoints` selection and half from none.
    The present-day rate is drawn, for every sample, from its Gaussian
    posterior given that sample's change points.

    The prior scales set how large a change must be to be believed, as
    for any Bayes factor: wider priors penalize change points more.

    Parameters
    ----------
    dates, values, sampling_days, periods_years, step_dates, kinds, \
max_changepoints, min_segment_years, n_candidates, noise_model, method
        As in `detect_changepoints`.
    p_change : float
        Prior probability of each change-point slot being used.
        Default 0.1 (DiscoTimeS' default).
    offset_scale : float, optional
        Prior standard deviation of an offset, input units. Default
        three times the standard deviation of the `detect_changepoints`
        residuals.
    rate_scale : float, optional
        Prior standard deviation of a rate change, input units per year.
        Default the standard deviation of those residuals, per year.
    n_chains : int
        Independent chains. Default 4.
    n_samples, n_burn : int
        Iterations kept and discarded per chain. Defaults 5000 and 1000.
    detection : ChangepointResult, optional
        A `detect_changepoints` result for the same series and grid
        settings, to skip running it again.
    seed : int or np.random.Generator, optional
        Seed of the chains. Default 0 (reproducible).

    Returns
    -------
    ChangepointPosterior

    Raises
    ------
    ValueError
        If `kinds` names an unknown kind or `p_change` is not in (0, 1).

    """
    kinds = _check_kinds(kinds)
    if not 0.0 < p_change < 1.0:
        msg = f"p_change must be in (0, 1), got {p_change}"
        raise ValueError(msg)
    series = _prepare(
        dates,
        values,
        sampling_days,
        periods_years,
        step_dates,
        min_segment_years,
        n_candidates,
    )
    if detection is None:
        detection = detect_changepoints(
            dates,
            values,
            sampling_days=sampling_days,
            periods_years=periods_years,
            step_dates=step_dates,
            kinds=kinds,
            max_changepoints=max_changepoints,
            min_segment_years=min_segment_years,
            n_candidates=n_candidates,
            noise_model=noise_model,
            method=method,
        )
    n_epochs = int(series.obs_idx[-1]) + 1
    normal = _Normal(
        series, _noise_covariance(detection.trend, series.obs_idx, n_epochs)
    )
    spread = float(np.std(detection.trend.residuals))
    offset_scale = 3.0 * spread if offset_scale is None else offset_scale
    rate_scale = spread if rate_scale is None else rate_scale
    variance = {"offset": offset_scale**2, "rate": rate_scale**2}

    n_cand, n_kinds = normal.n_cand, len(kinds)
    log_prior_n = [
        (
            float(
                np.log(special.comb(max_changepoints, k))
                + k * np.log(p_change)
                + (max_changepoints - k) * np.log1p(-p_change)
                - np.log(special.comb(n_cand, k))
                - k * np.log(n_kinds)
            )
            if k <= n_cand
            else -np.inf
        )
        for k in range(max_changepoints + 1)
    ]
    # Each visited configuration: log posterior and the present-day rate's
    # conditional mean and variance
    cache: dict[Config, tuple[float, float, float]] = {}

    def evaluate(config: Config) -> tuple[float, float, float]:
        if config not in cache:
            cols, labels = normal.columns(config)
            prior_var = np.array([variance[part] for _, part in labels])
            precision = np.r_[np.zeros(normal.n_base), 1.0 / prior_var]
            beta, cov, quad, logdet = normal.solve(cols, precision)
            log_ml = -0.5 * (logdet + np.sum(np.log(prior_var)) + quad)
            c = _rate_vector(series, labels)
            cache[config] = (
                log_ml + log_prior_n[len(config)],
                float(c @ beta),
                float(c @ cov @ c),
            )
        return cache[config]

    def allowed(config: Config, j: int) -> bool:
        return _separated(series, config, j, min_segment_years)

    def propose(
        config: Config, rng: np.random.Generator
    ) -> tuple[Config | None, float]:
        """Propose a configuration; return it and the log reverse/forward ratio."""
        k = len(config)
        occupied = {j for j, _ in config}
        move = rng.integers(3)
        if move == 0:  # birth
            free = [j for j in range(n_cand) if j not in occupied]
            if k == max_changepoints or not free:
                return None, 0.0
            j = int(free[rng.integers(len(free))])
            if not allowed(config, j):
                return None, 0.0
            new = tuple(sorted((*config, (j, kinds[rng.integers(n_kinds)]))))
            return new, float(np.log(len(free) * n_kinds) - np.log(k + 1))
        if k == 0:
            return None, 0.0
        i = int(rng.integers(k))
        rest = config[:i] + config[i + 1 :]
        if move == 1:  # death
            return rest, float(np.log(k) - np.log((n_cand - k + 1) * n_kinds))
        j, kind = config[i]
        # Symmetric updates: shift locally, jump anywhere, or change the kind
        update = rng.integers(3 if n_kinds > 1 else 2)
        if update == 2:
            kind = [kd for kd in kinds if kd != kind][rng.integers(n_kinds - 1)]
        elif update == 1:
            j = int(rng.integers(n_cand))
        else:
            width = max(1, n_cand // 20)
            j += int(rng.integers(1, width + 1)) * (1 if rng.random() < 0.5 else -1)
        if not 0 <= j < n_cand or j in occupied - {config[i][0]}:
            return None, 0.0
        if not allowed(rest, j):
            return None, 0.0
        return tuple(sorted((*rest, (j, kind)))), 0.0

    rng = np.random.default_rng(seed)
    start = tuple(
        sorted(
            (
                int(np.argmin(np.abs(series.epochs[series.candidates] - date))),
                kind,
            )
            for date, kind in zip(
                detection.changepoints["date"],
                detection.changepoints["kind"],
                strict=True,
            )
        )
    )
    draws: list[Config] = []
    n_per_chain = np.empty((n_chains, n_samples))
    rate_draws = np.empty((n_chains, n_samples))
    rate_mean = np.empty((n_chains, n_samples))
    rate_var = np.empty((n_chains, n_samples))
    accepted = 0
    for chain in range(n_chains):
        config = start if chain % 2 == 0 else ()
        current = evaluate(config)
        for it in range(n_burn + n_samples):
            new, log_q = propose(config, rng)
            if new is not None:
                candidate = evaluate(new)
                if np.log(rng.random()) < candidate[0] - current[0] + log_q:
                    config, current = new, candidate
                    accepted += 1
            if it >= n_burn:
                s = it - n_burn
                draws.append(config)
                n_per_chain[chain, s] = len(config)
                rate_mean[chain, s], rate_var[chain, s] = current[1], current[2]
                rate_draws[chain, s] = rng.normal(current[1], np.sqrt(current[2]))

    n_draws = len(draws)
    dates_grid = series.epochs[series.candidates]
    occupancy = {part: np.zeros((n_draws, n_cand), bool) for part in ("offset", "rate")}
    for s, config in enumerate(draws):
        for j, kind in config:
            for part in _PARTS[kind]:
                occupancy[part][s, j] = True
    any_change = occupancy["offset"] | occupancy["rate"]
    probability = pd.DataFrame(
        {
            "any": any_change.mean(axis=0),
            "offset": occupancy["offset"].mean(axis=0),
            "rate": occupancy["rate"].mean(axis=0),
        },
        index=pd.DatetimeIndex(dates_grid, name="date"),
    )
    counts = Counter(len(config) for config in draws)
    n_probability = pd.Series(
        [counts.get(k, 0) / n_draws for k in range(max_changepoints + 1)],
        index=pd.RangeIndex(max_changepoints + 1, name="n_changepoints"),
    )

    # Summary configuration: the most probable number of change points,
    # placed at the peaks of the windowed posterior probability
    t_grid = series.t[series.candidates]
    half = 0.5 * min_segment_years
    near = np.abs(t_grid[:, None] - t_grid[None, :]) < half
    windowed = (any_change.astype(float) @ near > 0).mean(axis=0)
    summary: list[tuple[int, str, float, pd.Timestamp, pd.Timestamp]] = []
    n_best = int(n_probability.idxmax())
    for peak in np.argsort(-windowed, kind="stable"):
        if len(summary) == n_best or windowed[peak] == 0:
            break
        epochs_near, kinds_near = [], []
        for config in draws:
            close = [(abs(t_grid[i] - t_grid[peak]), i, kd) for i, kd in config]
            close = [c for c in close if c[0] < half]
            if close:
                _, i, kd = min(close)
                epochs_near.append(i)
                kinds_near.append(kd)
        # The windowed probability is flat wherever the window covers the
        # mass, so the date is the median of the sampled epochs, not the peak
        low, j, high = (
            int(q)
            for q in np.quantile(epochs_near, [0.025, 0.5, 0.975], method="nearest")
        )
        if any(abs(t_grid[j] - t_grid[i]) < min_segment_years for i, *_ in summary):
            continue
        summary.append(
            (
                j,
                Counter(kinds_near).most_common(1)[0][0],
                float(windowed[j]),
                series.date(low),
                series.date(high),
            )
        )
    summary.sort()
    config = tuple((j, kind) for j, kind, *_ in summary)
    cols, labels = normal.columns(config)
    prior_var = np.array([variance[part] for _, part in labels])
    beta, cov, *_ = normal.solve(cols, np.r_[np.zeros(normal.n_base), 1.0 / prior_var])
    table = _amplitudes(series, labels, beta, cov, config)
    table.insert(2, "probability", [p for *_, p, _, _ in summary])
    table.insert(3, "date_low", [lo for *_, lo, _ in summary])
    table.insert(4, "date_high", [hi for *_, hi in summary])

    return ChangepointPosterior(
        changepoints=table,
        rate=float(rate_mean.mean()),
        # Law of total variance over the visited configurations
        rate_sigma=float(np.sqrt(rate_var.mean() + rate_mean.var())),
        rate_samples=rate_draws.ravel(),
        n_probability=n_probability,
        probability=probability,
        acceptance_rate=accepted / (n_chains * (n_burn + n_samples)),
        rhat={
            "n": _gelman_rubin(n_per_chain),
            "rate": _gelman_rubin(rate_draws),
        },
        detection=detection,
    )

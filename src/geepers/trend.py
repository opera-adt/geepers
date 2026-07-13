"""Maximum-likelihood trend estimation with colored (power-law) noise.

This is a clean-room implementation of GNSS trend estimation under a
combined power-law + white noise model, following the published equations
in the references below. It estimates a linear (or higher-degree) trend,
periodic signals, offsets, and postseismic terms simultaneously with the
noise parameters, and returns trend uncertainties that account for the
temporal correlation of the noise (unlike ordinary least squares, which
can underestimate GNSS velocity uncertainties by a factor of 5-10).

Credit
------
The methodology follows Hector / HectorP by Machiel Bos and colleagues
(https://gitlab.com/machielsimonbos/hectorp); HectorP served as the
reference implementation for validating this module's outputs. No
Hector/HectorP (GPL) code is included here - the implementation is
written from the published papers:

- Bos, M. S., Fernandes, R. M. S., Williams, S. D. P., & Bastos, L.
  (2013). Fast error analysis of continuous GNSS observations with
  missing data. Journal of Geodesy, 87(4), 351-360.
  doi:10.1007/s00190-012-0605-0
- Williams, S. D. P. (2003). The effect of coloured noise on the
  uncertainties of rates estimated from geodetic time series.
  Journal of Geodesy, 76(9-10), 483-494. doi:10.1007/s00190-002-0283-4
- Hosking, J. R. M. (1981). Fractional differencing. Biometrika, 68(1),
  165-176. doi:10.1093/biomet/68.1.165
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg, optimize

__all__ = ["TrendResult", "estimate_trend", "estimate_trend_many"]

logger = logging.getLogger("geepers")

DAYS_PER_YEAR = 365.25


@dataclass
class TrendResult:
    """Results from maximum-likelihood trend estimation.

    Attributes
    ----------
    velocity : float
        Linear trend (units of the observations per year).
    velocity_uncertainty : float
        1-sigma trend uncertainty accounting for the estimated noise
        covariance (same units as `velocity`).
    parameters : dict[str, tuple[float, float]]
        All estimated design-matrix parameters as
        ``name -> (value, 1-sigma uncertainty)``.
    kappa : float
        Estimated spectral index of the power-law noise
        (0 = white, -1 = flicker, -2 = random walk).
    sigma_powerlaw : float
        Power-law noise amplitude (observation units, per sampling
        interval to the power ``-kappa/4``). NaN for the fixed-index
        mixture models ("FNWN", "RWFNWN").
    sigma_white : float
        White noise amplitude (observation units).
    sigma_flicker : float
        Flicker noise (kappa = -1) amplitude. Only set by the "FNWN"
        and "RWFNWN" noise models; NaN otherwise.
    sigma_randomwalk : float
        Random-walk noise (kappa = -2) amplitude. Only set by the
        "RWFNWN" noise model; NaN otherwise.
    log_likelihood : float
        Log-likelihood at the optimum.
    aic : float
        Akaike information criterion.
    bic : float
        Bayesian information criterion.
    residuals : np.ndarray
        Observation-minus-model residuals at the observed epochs.
    model : np.ndarray
        Fitted model evaluated at the observed epochs.

    """

    velocity: float
    velocity_uncertainty: float
    parameters: dict[str, tuple[float, float]] = field(default_factory=dict)
    kappa: float = np.nan
    sigma_powerlaw: float = np.nan
    sigma_white: float = np.nan
    sigma_flicker: float = np.nan
    sigma_randomwalk: float = np.nan
    log_likelihood: float = np.nan
    aic: float = np.nan
    bic: float = np.nan
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    model: np.ndarray = field(default_factory=lambda: np.array([]))


def _fractional_diff_coeffs(d: float, n: int) -> np.ndarray:
    """Coefficients psi_i of (1-B)^{-d} (Hosking, 1981, eq. for the MA form).

    psi_0 = 1 and psi_i = psi_{i-1} * (d + i - 1) / i.
    """
    psi = np.empty(n)
    psi[0] = 1.0
    if n > 1:
        i = np.arange(1, n)
        psi[1:] = np.cumprod((d + i - 1.0) / i)
    return psi


def _powerlaw_covariance(kappa: float, n: int) -> np.ndarray:
    """Unit-amplitude covariance of power-law noise on a regular grid.

    Uses the non-stationary formulation C = U U^T where U is the lower
    triangular Toeplitz matrix of fractional-differencing coefficients
    (Bos et al., 2013, section 2). Entries are built in O(n^2) with
    per-diagonal cumulative sums:
    C[j+l, j] = sum_{m=0..j} psi_{m+l} psi_m.
    """
    d = -kappa / 2.0
    psi = _fractional_diff_coeffs(d, n)
    C = np.empty((n, n))
    for lag in range(n):
        diag = np.cumsum(psi[lag:] * psi[: n - lag])
        idx = np.arange(n - lag)
        C[idx + lag, idx] = diag
        C[idx, idx + lag] = diag
    return C


def _design_matrix(
    t_years: np.ndarray,
    poly_deg: int,
    periods_years: tuple[float, ...],
    step_years: np.ndarray,
    log_terms: list[tuple[float, float]],
    exp_terms: list[tuple[float, float]],
) -> tuple[np.ndarray, list[str]]:
    """Build the design matrix and matching parameter names.

    Parameters are in years relative to the first epoch.
    """
    cols: list[np.ndarray] = []
    names: list[str] = []

    for p in range(poly_deg + 1):
        cols.append(t_years**p)
        names.append({0: "intercept", 1: "trend"}.get(p, f"poly_{p}"))

    for period in periods_years:
        w = 2.0 * np.pi / period
        cols.append(np.cos(w * t_years))
        cols.append(np.sin(w * t_years))
        names.append(f"cos_{period:g}yr")
        names.append(f"sin_{period:g}yr")

    for i, ts in enumerate(step_years):
        cols.append((t_years >= ts).astype(float))
        names.append(f"step_{i}")

    for i, (t0, tau) in enumerate(log_terms):
        dt = np.clip(t_years - t0, 0.0, None)
        cols.append(np.log1p(dt / tau) * (t_years >= t0))
        names.append(f"log_{i}")

    for i, (t0, tau) in enumerate(exp_terms):
        dt = np.clip(t_years - t0, 0.0, None)
        cols.append((1.0 - np.exp(-dt / tau)) * (t_years >= t0))
        names.append(f"exp_{i}")

    return np.column_stack(cols), names


# Fixed spectral indices of the named mixture components. "powerlaw"
# (used by PLWN/PL) is the only component with a *free* kappa.
_FIXED_KAPPA = {"white": 0.0, "flicker": -1.0, "randomwalk": -2.0}
_MIXTURE_COMPONENTS = {
    "FNWN": ("flicker", "white"),
    "RWFNWN": ("randomwalk", "flicker", "white"),
}


def _softmax_weights(z: np.ndarray) -> np.ndarray:
    """Map K-1 free parameters to K positive weights summing to 1.

    The last component's logit is pinned to 0.
    """
    logits = np.append(np.asarray(z, dtype=float), 0.0)
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def _nll_core(C: np.ndarray, A: np.ndarray, y: np.ndarray, use_rmle: bool) -> float:
    """Profiled negative log-likelihood of GLS under noise covariance `C`.

    The overall variance is profiled out analytically
    (Bos et al., 2013, eq. 2-5). Returns 1e12 on numerical failure.
    """
    n = len(y)
    try:
        cho = linalg.cho_factor(C, lower=True, check_finite=False)
    except linalg.LinAlgError:
        return 1e12

    logdet_C = 2.0 * np.sum(np.log(np.diag(cho[0])))
    Ci_A = linalg.cho_solve(cho, A, check_finite=False)
    Ci_y = linalg.cho_solve(cho, y, check_finite=False)
    AtCiA = A.T @ Ci_A
    try:
        beta = linalg.solve(AtCiA, A.T @ Ci_y, assume_a="pos")
    except linalg.LinAlgError:
        return 1e12
    r = y - A @ beta
    rCir = float(r @ linalg.cho_solve(cho, r, check_finite=False))
    if rCir <= 0:
        return 1e12

    n_p = A.shape[1] if use_rmle else 0
    sigma2 = rCir / (n - n_p)
    nll = 0.5 * (logdet_C + (n - n_p) * np.log(sigma2) + rCir / sigma2)
    if use_rmle:
        sign, logdet_N = np.linalg.slogdet(AtCiA)
        if sign <= 0:
            return 1e12
        nll += 0.5 * logdet_N
    return nll


def _negative_log_likelihood(
    params: np.ndarray,
    C_cache: dict,
    obs_idx: np.ndarray,
    n_epochs: int,
    A: np.ndarray,
    y: np.ndarray,
    use_rmle: bool,
) -> float:
    """Profiled negative log-likelihood for (kappa, logit white fraction)."""
    kappa, z = params
    if not (-3.0 <= kappa <= 0.01):
        return 1e12
    phi = 1.0 / (1.0 + np.exp(-z))  # white-noise fraction in (0, 1)

    key = round(float(kappa), 10)
    if key not in C_cache:
        C_cache.clear()  # only the latest kappa is ever reused
        C_cache[key] = _powerlaw_covariance(kappa, n_epochs)[np.ix_(obs_idx, obs_idx)]
    C_pl = C_cache[key]

    n = len(y)
    C = (1.0 - phi) * C_pl + phi * np.eye(n)
    return _nll_core(C, A, y, use_rmle)


def _negative_log_likelihood_mixture(
    z: np.ndarray,
    covs: list[np.ndarray],
    A: np.ndarray,
    y: np.ndarray,
    use_rmle: bool,
) -> float:
    """Profiled negative log-likelihood over mixture weights.

    ``z`` holds K-1 free logits for K fixed-spectral-index components
    whose unit covariances `covs` are precomputed (flicker, random walk,
    white), so each evaluation costs only one Cholesky factorization.
    """
    weights = _softmax_weights(z)
    C = sum(w * Ck for w, Ck in zip(weights, covs, strict=True))
    return _nll_core(C, A, y, use_rmle)


def _whittle_noise_estimate(
    resid_grid: np.ndarray,
    noise_model: Literal["PLWN", "PL"],
) -> tuple[float, float]:
    """Estimate (kappa, phi) from the periodogram of gridded residuals.

    Whittle's frequency-domain approximation to the Gaussian likelihood:
    each evaluation is O(n) on the FFT periodogram instead of an O(n^3)
    covariance factorization, and is asymptotically equivalent to exact
    maximum likelihood (Bos et al., 2013, section 5 use the same
    spectral form for power-law noise).

    Parameters
    ----------
    resid_grid : np.ndarray
        Detrended residuals on the *complete* regular sampling grid
        (gaps filled by interpolation).
    noise_model : {"PLWN", "PL"}
        Whether to include a white-noise floor.

    Returns
    -------
    tuple[float, float]
        Estimated ``(kappa, phi)``: spectral index and white-noise
        variance fraction.

    """
    n = len(resid_grid)
    pgram = np.abs(np.fft.rfft(resid_grid - resid_grid.mean())) ** 2 / n
    freqs = np.fft.rfftfreq(n)
    # Skip f=0; also drop the lowest bin, which detrending biases low
    start = 2 if len(freqs) > 20 else 1
    pgram = pgram[start:]
    # Spectrum of (1-B)^{-d} driven by unit white noise: (2 sin(pi f))^kappa
    log_two_sin = np.log(2.0 * np.sin(np.pi * freqs[start:]))
    n_freqs = len(pgram)

    def nll(params: np.ndarray) -> float:
        kappa, z = params
        if not (-3.0 <= kappa <= 0.01):
            return 1e12
        phi = 0.0 if noise_model == "PL" else 1.0 / (1.0 + np.exp(-z))
        g = (1.0 - phi) * np.exp(kappa * log_two_sin) + phi
        sigma2 = float(np.mean(pgram / g))
        if not np.isfinite(sigma2) or sigma2 <= 0:
            return 1e12
        return float(np.sum(np.log(g))) + n_freqs * np.log(sigma2)

    z0_grid = [0.0] if noise_model == "PL" else [-2.0, 0.0, 2.0]
    best = None
    for k0 in (-0.4, -1.0, -1.6):
        for z0 in z0_grid:
            res = optimize.minimize(
                nll,
                x0=np.array([k0, z0]),
                method="Nelder-Mead",
                options={"xatol": 1e-3, "fatol": 1e-8, "maxiter": 500},
            )
            if best is None or res.fun < best.fun:
                best = res
    kappa = float(np.clip(best.x[0], -3.0, 0.0))
    phi = 0.0 if noise_model == "PL" else float(1.0 / (1.0 + np.exp(-best.x[1])))
    return kappa, phi


def _whittle_mixture_estimate(
    resid_grid: np.ndarray,
    kappas: tuple[float, ...],
) -> np.ndarray:
    """Estimate mixture weights of fixed-spectral-index noise components.

    Whittle likelihood on the periodogram, like `_whittle_noise_estimate`,
    but for a sum of components with *fixed* kappas (e.g. random walk -2,
    flicker -1, white 0). Only the K-1 relative weights are optimized.

    Parameters
    ----------
    resid_grid : np.ndarray
        Detrended residuals on the complete regular sampling grid.
    kappas : tuple of float
        Spectral index of each component.

    Returns
    -------
    np.ndarray
        Estimated variance-fraction weights, summing to 1.

    """
    n = len(resid_grid)
    pgram = np.abs(np.fft.rfft(resid_grid - resid_grid.mean())) ** 2 / n
    freqs = np.fft.rfftfreq(n)
    start = 2 if len(freqs) > 20 else 1
    pgram = pgram[start:]
    two_sin = 2.0 * np.sin(np.pi * freqs[start:])
    # (n_comps, n_freqs) unit spectra
    spectra = np.stack([two_sin**k for k in kappas])
    n_freqs = len(pgram)

    def nll(z: np.ndarray) -> float:
        weights = _softmax_weights(z)
        g = weights @ spectra
        sigma2 = float(np.mean(pgram / g))
        if not np.isfinite(sigma2) or sigma2 <= 0:
            return 1e12
        return float(np.sum(np.log(g))) + n_freqs * np.log(sigma2)

    n_free = len(kappas) - 1
    grid = [-2.0, 0.0, 2.0]
    best = None
    for z0 in np.stack(np.meshgrid(*([grid] * n_free))).reshape(n_free, -1).T:
        res = optimize.minimize(
            nll,
            x0=np.asarray(z0, dtype=float),
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-8, "maxiter": 500},
        )
        if best is None or res.fun < best.fun:
            best = res
    return _softmax_weights(best.x)


def estimate_trend(
    dates: ArrayLike,
    values: ArrayLike,
    *,
    sampling_days: float = 1.0,
    poly_deg: int = 1,
    periods_years: tuple[float, ...] = (1.0, 0.5),
    step_dates: ArrayLike | None = None,
    postseismic_log: list[tuple] | None = None,
    postseismic_exp: list[tuple] | None = None,
    noise_model: Literal["PLWN", "PL", "WN", "FNWN", "RWFNWN"] = "PLWN",
    use_rmle: bool = True,
    method: Literal["exact", "whittle"] = "exact",
) -> TrendResult:
    """Estimate a trend and its realistic uncertainty from a time series.

    Fits a deterministic model (polynomial + periodic + steps +
    postseismic terms) by generalized least squares while simultaneously
    estimating power-law + white noise parameters by (restricted)
    maximum likelihood.

    Parameters
    ----------
    dates : array-like of datetime64 / pd.Timestamp
        Observation epochs. Need not be continuous; gaps are handled by
        evaluating the noise covariance at the observed epochs only.
    values : array-like of float
        Observations (any unit; outputs are in the same unit). NaNs are
        dropped.
    sampling_days : float
        Nominal sampling interval in days. Default is 1 (daily).
    poly_deg : int
        Degree of the polynomial part. Default 1 (intercept + trend).
    periods_years : tuple of float
        Periodic signal periods in years. Default annual + semi-annual.
    step_dates : array-like of datetime64, optional
        Epochs of instantaneous offsets (equipment changes, earthquakes).
    postseismic_log, postseismic_exp : list of (date, tau_years), optional
        Postseismic relaxation terms: logarithmic ``log(1 + dt/tau)`` or
        exponential ``1 - exp(-dt/tau)`` with fixed relaxation time
        `tau_years`, starting at `date`. Amplitudes are estimated.
    noise_model : {"PLWN", "PL", "WN", "FNWN", "RWFNWN"}
        Noise model:

        - ``"PLWN"`` (default): power-law (estimated spectral index
          kappa) + white.
        - ``"PL"``: power-law only.
        - ``"WN"``: white only (equivalent to weighted OLS).
        - ``"FNWN"``: flicker (kappa fixed at -1) + white - the
          standard GNSS noise model. Cheaper than "PLWN" because no
          spectral-index search is needed.
        - ``"RWFNWN"``: random walk (kappa = -2) + flicker + white.
    use_rmle : bool
        Use restricted maximum likelihood, which corrects the noise
        estimates for the degrees of freedom absorbed by the
        deterministic model. Default True.
    method : {"exact", "whittle"}
        How the noise parameters (kappa, phi) are estimated:

        - ``"exact"`` (default): time-domain (R)MLE. Every optimizer
          iteration factorizes an n x n covariance (O(n^3)), so long
          daily series take minutes.
        - ``"whittle"``: frequency-domain (Whittle) likelihood on the
          periodogram of the OLS residuals - O(n log n) - followed by a
          single exact GLS solve at the optimum for the parameters and
          their colored-noise uncertainties. Orders of magnitude faster
          with near-identical results; preferred when fitting many
          series (see `estimate_trend_many`). Gaps are filled by linear
          interpolation for the noise estimation stage only.

    Returns
    -------
    TrendResult
        Estimated trend, uncertainties, noise parameters, and fit
        diagnostics. Velocity is per **year**.

    """
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    y_all = np.asarray(values, dtype=float)
    good = np.isfinite(y_all)
    dates, y = dates[good], y_all[good]
    if len(y) < 10:
        msg = f"Need at least 10 finite observations, got {len(y)}"
        raise ValueError(msg)

    # Epoch indices on the regular sampling grid
    t_days = (dates - dates[0]).total_seconds() / 86400.0
    obs_idx = np.round(t_days / sampling_days).astype(int)
    if len(np.unique(obs_idx)) != len(obs_idx):
        msg = "Duplicate epochs after rounding to the sampling interval"
        raise ValueError(msg)
    n_epochs = obs_idx[-1] + 1
    t_years = obs_idx * sampling_days / DAYS_PER_YEAR

    def _to_years(d) -> float:
        return (pd.Timestamp(d) - dates[0]).total_seconds() / 86400.0 / DAYS_PER_YEAR

    step_years = np.array(
        [_to_years(d) for d in (step_dates if step_dates is not None else [])]
    )
    log_terms = [(_to_years(d), float(tau)) for d, tau in (postseismic_log or [])]
    exp_terms = [(_to_years(d), float(tau)) for d, tau in (postseismic_exp or [])]

    A, names = _design_matrix(
        t_years, poly_deg, periods_years, step_years, log_terms, exp_terms
    )

    # Remove the mean to keep the likelihood well-conditioned
    y_mean = float(np.mean(y))
    y0 = y - y_mean

    n = len(y0)
    C_cache: dict = {}
    mixture_weights: np.ndarray | None = None
    comp_names: tuple[str, ...] = ()

    if noise_model in _MIXTURE_COMPONENTS:
        # Fixed-spectral-index mixture (flicker / random walk / white):
        # unit covariances are built once, only the weights are estimated.
        comp_names = _MIXTURE_COMPONENTS[noise_model]
        covs = [
            np.eye(n)
            if name == "white"
            else _powerlaw_covariance(_FIXED_KAPPA[name], n_epochs)[
                np.ix_(obs_idx, obs_idx)
            ]
            for name in comp_names
        ]
        if method == "whittle":
            beta_ols, *_ = np.linalg.lstsq(A, y0, rcond=None)
            resid_grid = np.interp(np.arange(n_epochs), obs_idx, y0 - A @ beta_ols)
            kappas = tuple(_FIXED_KAPPA[name] for name in comp_names)
            mixture_weights = _whittle_mixture_estimate(resid_grid, kappas)
            w = np.clip(mixture_weights, 1e-12, None)
            z_hat = np.log(w[:-1] / w[-1])
            nll = _negative_log_likelihood_mixture(z_hat, covs, A, y0, use_rmle)
        else:
            n_free = len(comp_names) - 1
            grid = [-2.0, 0.0, 2.0]
            args = (covs, A, y0, use_rmle)
            z_starts = np.stack(np.meshgrid(*([grid] * n_free))).reshape(n_free, -1).T
            starts = sorted(
                (_negative_log_likelihood_mixture(z0, *args), tuple(z0))
                for z0 in z_starts
            )
            best = None
            for _, z0 in starts[:2]:
                res = optimize.minimize(
                    _negative_log_likelihood_mixture,
                    x0=np.asarray(z0),
                    args=args,
                    method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 300},
                )
                if best is None or res.fun < best.fun:
                    best = res
            mixture_weights = _softmax_weights(best.x)
            nll = float(best.fun)
        kappa_hat = -1.0 if noise_model == "FNWN" else np.nan
        phi_hat = float(mixture_weights[-1])  # white fraction
    elif noise_model == "WN":
        kappa_hat, phi_hat = 0.0, 1.0 - 1e-12
        nll = _negative_log_likelihood(
            np.array([0.0, 30.0]), C_cache, obs_idx, n_epochs, A, y0, use_rmle
        )
    elif method == "whittle":
        # Fast path: noise parameters from the periodogram of the OLS
        # residuals (O(n log n)), then one exact GLS solve below.
        beta_ols, *_ = np.linalg.lstsq(A, y0, rcond=None)
        resid_grid = np.interp(np.arange(n_epochs), obs_idx, y0 - A @ beta_ols)
        kappa_hat, phi_hat = _whittle_noise_estimate(resid_grid, noise_model)
        phi_c = float(np.clip(phi_hat, 1e-12, 1.0 - 1e-12))
        z_hat = float(np.log(phi_c / (1.0 - phi_c)))
        # One exact likelihood evaluation for comparable AIC/BIC values
        nll = _negative_log_likelihood(
            np.array([kappa_hat, z_hat]), C_cache, obs_idx, n_epochs, A, y0, use_rmle
        )
    else:
        # Coarse grid scan for a good starting point, then one polish run.
        # The PLWN likelihood surface in (kappa, phi) is smooth; this is
        # much cheaper than multi-start Nelder-Mead.
        z0_grid = [30.0] if noise_model == "PL" else [-2.0, 0.0, 2.0]
        args = (C_cache, obs_idx, n_epochs, A, y0, use_rmle)
        starts = sorted(
            (
                (_negative_log_likelihood(np.array([k0, z0]), *args), k0, z0)
                for k0 in (-0.4, -1.0, -1.6)
                for z0 in z0_grid
            ),
        )
        # Polish from the two best grid points: the profiled likelihood can
        # have a flat kappa/phi trade-off ridge, so a single Nelder-Mead run
        # occasionally converges to a secondary basin.
        best = None
        for _, k0, z0 in starts[:2]:
            res = optimize.minimize(
                _negative_log_likelihood,
                x0=np.array([k0, z0]),
                args=args,
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 300},
            )
            if best is None or res.fun < best.fun:
                best = res
        kappa_hat = float(best.x[0])
        phi_hat = float(1.0 / (1.0 + np.exp(-best.x[1])))
        if noise_model == "PL":
            phi_hat = 0.0
        nll = float(best.fun)

    # Final GLS solve at the optimum (reuse the cached covariance if the
    # last likelihood evaluation was already at kappa_hat)
    if mixture_weights is not None:
        C = sum(w * Ck for w, Ck in zip(mixture_weights, covs, strict=True))
    else:
        C_pl = C_cache.get(round(float(kappa_hat), 10))
        if C_pl is None:
            C_pl = _powerlaw_covariance(kappa_hat, n_epochs)[np.ix_(obs_idx, obs_idx)]
        C = (1.0 - phi_hat) * C_pl + phi_hat * np.eye(n)
    cho = linalg.cho_factor(C, lower=True, check_finite=False)
    Ci_A = linalg.cho_solve(cho, A, check_finite=False)
    AtCiA = A.T @ Ci_A
    beta = linalg.solve(AtCiA, A.T @ linalg.cho_solve(cho, y0), assume_a="pos")
    r = y0 - A @ beta
    n_p = A.shape[1] if use_rmle else 0
    sigma2 = float(r @ linalg.cho_solve(cho, r, check_finite=False)) / (n - n_p)
    beta_cov = sigma2 * linalg.inv(AtCiA)
    beta_sig = np.sqrt(np.diag(beta_cov))

    # Un-shift the intercept
    beta_out = beta.copy()
    beta_out[names.index("intercept")] += y_mean

    sigma_fn = sigma_rw = np.nan
    if mixture_weights is not None:
        comp_sigma = {
            name: float(np.sqrt(sigma2 * w))
            for name, w in zip(comp_names, mixture_weights, strict=True)
        }
        sigma_pl = np.nan
        sigma_w = comp_sigma["white"]
        sigma_fn = comp_sigma["flicker"]
        sigma_rw = comp_sigma.get("randomwalk", np.nan)
    else:
        sigma_pl = float(np.sqrt(sigma2 * (1.0 - phi_hat)))
        sigma_w = float(np.sqrt(sigma2 * phi_hat))
    n_noise_params = {"PLWN": 3, "PL": 2, "WN": 1, "FNWN": 2, "RWFNWN": 3}[noise_model]
    k_total = A.shape[1] + n_noise_params
    log_l = -nll - 0.5 * n * np.log(2.0 * np.pi)

    itrend = names.index("trend") if "trend" in names else -1
    return TrendResult(
        velocity=float(beta_out[itrend]) if itrend >= 0 else np.nan,
        velocity_uncertainty=float(beta_sig[itrend]) if itrend >= 0 else np.nan,
        parameters={
            nm: (float(b), float(s))
            for nm, b, s in zip(names, beta_out, beta_sig, strict=True)
        },
        kappa=kappa_hat,
        sigma_powerlaw=sigma_pl,
        sigma_white=sigma_w,
        sigma_flicker=sigma_fn,
        sigma_randomwalk=sigma_rw,
        log_likelihood=float(log_l),
        aic=float(2 * k_total - 2 * log_l),
        bic=float(k_total * np.log(n) - 2 * log_l),
        residuals=r,
        model=A @ beta + y_mean,
    )


def _fit_one(
    args: tuple[np.ndarray, np.ndarray, str, dict],
) -> dict[str, float]:
    """Fit a single series for `estimate_trend_many` (must be picklable)."""
    dates, values, name, kwargs = args
    nan_row = dict.fromkeys(
        [
            "velocity",
            "velocity_uncertainty",
            "kappa",
            "sigma_powerlaw",
            "sigma_white",
            "sigma_flicker",
            "sigma_randomwalk",
            "log_likelihood",
            "aic",
            "bic",
        ],
        np.nan,
    )
    try:
        res = estimate_trend(dates, values, **kwargs)
    except (ValueError, linalg.LinAlgError) as e:
        logger.warning("Trend fit failed for %s: %s", name, e)
        return {"name": name, **nan_row}
    return {
        "name": name,
        "velocity": res.velocity,
        "velocity_uncertainty": res.velocity_uncertainty,
        "kappa": res.kappa,
        "sigma_powerlaw": res.sigma_powerlaw,
        "sigma_white": res.sigma_white,
        "sigma_flicker": res.sigma_flicker,
        "sigma_randomwalk": res.sigma_randomwalk,
        "log_likelihood": res.log_likelihood,
        "aic": res.aic,
        "bic": res.bic,
    }


def estimate_trend_many(
    dates: ArrayLike,
    values: ArrayLike | pd.DataFrame,
    *,
    n_jobs: int | None = None,
    method: Literal["exact", "whittle"] = "whittle",
    **kwargs,
) -> pd.DataFrame:
    """Estimate trends for many series sharing the same epochs, in parallel.

    Designed for scaling `estimate_trend` over networks of stations or
    stacks of InSAR pixels. Uses the fast Whittle noise estimator by
    default and distributes series over worker processes.

    Parameters
    ----------
    dates : array-like of datetime64
        Observation epochs, shared by all series. Ignored if `values`
        is a DataFrame with a DatetimeIndex.
    values : array-like or pd.DataFrame
        Either a 2D array of shape ``(n_series, n_epochs)``, or a
        DataFrame with one column per series (its index supplies the
        dates when it is a DatetimeIndex). NaNs are allowed and dropped
        per series.
    n_jobs : int, optional
        Number of worker processes. Default is ``os.cpu_count()``.
        Use 1 to run serially (no subprocesses).
    method : {"exact", "whittle"}
        Noise-estimation method passed to `estimate_trend`.
        Default here is ``"whittle"`` (fast).
    **kwargs
        Passed through to `estimate_trend` (e.g. ``step_dates``,
        ``periods_years``, ``noise_model``).

    Returns
    -------
    pd.DataFrame
        One row per series (indexed by column name or integer position)
        with columns ``velocity``, ``velocity_uncertainty``, ``kappa``,
        ``sigma_powerlaw``, ``sigma_white``, ``log_likelihood``,
        ``aic``, ``bic``.

    Notes
    -----
    Worker processes each use BLAS internally; when running many
    workers, set ``OMP_NUM_THREADS=1`` in the environment to avoid
    thread oversubscription.

    """
    if isinstance(values, pd.DataFrame):
        names = list(values.columns)
        if isinstance(values.index, pd.DatetimeIndex):
            dates = values.index
        matrix = values.to_numpy().T
    else:
        matrix = np.atleast_2d(np.asarray(values, dtype=float))
        names = list(range(matrix.shape[0]))

    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    if matrix.shape[1] != len(dates):
        msg = (
            f"values has {matrix.shape[1]} epochs per series but"
            f" {len(dates)} dates were given"
        )
        raise ValueError(msg)

    kwargs = {**kwargs, "method": method}
    tasks = [
        (dates, row, name, kwargs) for row, name in zip(matrix, names, strict=True)
    ]

    if n_jobs == 1:
        rows = [_fit_one(t) for t in tasks]
    else:
        max_workers = n_jobs if n_jobs and n_jobs > 0 else os.cpu_count()
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            rows = list(pool.map(_fit_one, tasks, chunksize=4))

    return pd.DataFrame(rows).set_index("name").rename_axis(None)

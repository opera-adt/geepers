# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Python conversion of MIDAS velocity calculation.

Original code: https://geodesy.unr.edu/MIDAS_release.tar
Original author: Geoff Blewitt.  Copyright (C) 2015.

When using these velocities in publications please cite:

Blewitt, G., C. Kreemer, W.C. Hammond, and J. Gazeaux (2016). MIDAS robust trend
estimator for accurate GPS station velocities without step detection, Journal
of Geophysical Research, 121, doi:10.1002/2015JB012552

"""

from dataclasses import dataclass, field
from functools import cache, wraps

import numpy as np


@dataclass
class MidasResult:
    """Results from MIDAS velocity estimation.

    Attributes
    ----------
    velocity : float
        Median velocity estimate after outlier removal (m/year)
    velocity_uncertainty : float
        Uncertainty in velocity estimate (m/year)
    reference_position : float
        Position at reference epoch (m)
    outlier_fraction : float
        Fraction of pairs removed as outliers
    velocity_scatter : float
        Scatter in velocity estimates from pairs (m/year)
    residuals : np.ndarray
        Residuals to linear fit (m)

    """

    velocity: float = field(  # v50 in original code
        metadata={"help": "Median velocity estimate after outlier removal (m/year)"}
    )
    velocity_uncertainty: float = field(  # sv
        metadata={"help": "Uncertainty in velocity estimate (m/year)"}
    )
    reference_position: float = field(  # x50
        metadata={"help": "Position at reference epoch (m)"}
    )
    outlier_fraction: float = field(  # f
        metadata={"help": "Fraction of pairs removed as outliers"}
    )
    velocity_scatter: float = field(  # sd
        metadata={"help": "Scatter in velocity estimates from pairs (m/year)"}
    )
    residuals: np.ndarray = field(metadata={"help": "Residuals to linear fit (m)"})  # r

    def __mul__(self, x):
        if not isinstance(x, int | float):
            return NotImplemented

        return MidasResult(
            velocity=self.velocity * x,
            velocity_uncertainty=self.velocity_uncertainty * x,
            velocity_scatter=self.velocity_scatter * x,
            residuals=self.residuals * x,
            # Dont multiply these positions/fractions
            reference_position=self.reference_position,
            outlier_fraction=self.outlier_fraction,
        )

    def __rmul__(self, x):
        return self.__mul__(x)


def np_cache(function):
    """Cache results where inputs are numpy arrays."""

    @cache
    def cached_wrapper(*args, **kwargs):
        args = [np.array(a) if isinstance(a, tuple) else a for a in args]
        kwargs = {
            k: np.array(v) if isinstance(v, tuple) else v for k, v in kwargs.items()
        }

        return function(*args, **kwargs)

    @wraps(function)
    def wrapper(*args, **kwargs):
        args = [tuple(a) if isinstance(a, np.ndarray) else a for a in args]
        kwargs = {
            k: tuple(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
        }
        return cached_wrapper(*args, **kwargs)

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


@np_cache
def select_pairs(
    times: np.ndarray, max_n: int, tol: float, step_times: np.ndarray | None = None
) -> tuple[int, np.ndarray]:
    """Select pairs of time points separated by approximately one year.

    Parameters
    ----------
    times : np.ndarray
        Array of time points
    max_n : int
        Maximum number of pairs to select
    tol : float
        Tolerance for matching pairs (in years)
    step_times : np.ndarray, optional
        Array of step epoch times to avoid spanning

    Returns
    -------
    n : int
        Number of pairs selected
    pairs : np.ndarray
        Array of shape (n,2) containing indices of selected pairs

    """
    m = len(times)
    pairs = np.zeros((max_n, 2), dtype=int)
    k = 0
    n = 0
    istep = 0

    if step_times is None:
        step_times = np.array([])
    nstep = len(step_times)

    for i in range(m):
        if n >= max_n:
            break
        if times[i] > (times[-1] + tol - 1.0):
            break

        while istep < nstep and times[i] >= step_times[istep] + tol:
            istep += 1

        if istep < nstep and times[i] > (step_times[istep] + tol - 1.0):
            continue

        for j in range(i + 1, m):
            if k < j:
                k = j
            if istep < nstep and times[j] > (step_times[istep] - tol):
                break

            dt = times[j] - times[i]
            fdt = dt - 1.0

            if fdt < -tol:
                continue

            if fdt < tol:
                i2 = j
            else:
                i2 = k
                dt = times[i2] - times[i]
                if istep < nstep and times[i2] > (step_times[istep] - tol):
                    k = 0
                    continue
                if k == m - 1:
                    k = 0
                k += 1

            n += 1
            pairs[n - 1] = [i, i2]
            break

    return n, pairs[:n]


def midas(
    times: np.ndarray, values: np.ndarray, step_times: np.ndarray | None = None
) -> MidasResult:
    """MIDAS (Median Interannual Difference Adjusted for Skewness) algorithm.

    Parameters
    ----------
    times : np.ndarray
        Time points in decimal years
    values : np.ndarray
        Measurements of one component
    step_times : np.ndarray, optional
        Times of step epochs to avoid spanning

    Returns
    -------
    MidasResult
        Object containing velocity (v50), uncertainty (sv), intercept (x50),
        outlier fraction (f), scatter estimate (sd), and residuals (r)

    """
    m = len(times)
    max_n = 19999
    tol = 0.001

    # Forward time pairs
    n1, pairs1 = select_pairs(times, max_n, tol, step_times)

    # Backward time pairs
    rev_times = -times[::-1]
    if step_times is not None:
        rev_steps = -step_times[::-1]
    else:
        rev_steps = None
    n2, pairs2 = select_pairs(rev_times, max_n, tol, rev_steps)

    # Correct backward pairs indices
    pairs2 = m - 1 - pairs2[:, ::-1]

    # Combine pairs
    pairs = np.vstack([pairs1, pairs2])
    n = len(pairs)
    if n < 1:
        return MidasResult(
            velocity=np.nan,
            velocity_uncertainty=np.nan,
            reference_position=np.nan,
            outlier_fraction=np.nan,
            velocity_scatter=np.nan,
            residuals=np.array([]),
        )

    # Compute velocities
    dt = times[pairs[:, 1]] - times[pairs[:, 0]]
    v = (values[pairs[:, 1]] - values[pairs[:, 0]]) / dt

    # Initial median velocity
    v50 = np.median(v)

    # Absolute deviations
    d = np.abs(v - v50)

    # MAD estimate
    d50 = np.median(d)

    # Standard deviation
    sd = 1.4826 * d50

    # Trim outliers
    c = 2.0 * sd
    v_clean = v[d < c]

    # Check if any pairs remain after outlier removal
    if len(v_clean) == 0:
        return MidasResult(
            velocity=np.nan,
            velocity_uncertainty=np.nan,
            reference_position=np.nan,
            outlier_fraction=1.0,
            velocity_scatter=np.nan,
            residuals=np.full(len(times), np.nan),
        )

    # Recompute median
    v50 = np.median(v_clean)

    # Final MAD
    d = np.abs(v_clean - v50)
    d50 = np.median(d)

    # Standard error
    sv = 3.0 * 1.2533 * 1.4826 * d50 / np.sqrt(len(v_clean) / 4.0)

    # Compute intercept
    dt = times - times[0]
    r = values - v50 * dt
    x50 = np.median(r)

    # Compute residuals
    r -= x50

    # Fraction of pairs removed
    f = (n - len(v_clean)) / n

    return MidasResult(
        velocity=v50,
        velocity_uncertainty=sv,
        reference_position=x50,
        outlier_fraction=f,
        velocity_scatter=sd,
        residuals=r,
    )

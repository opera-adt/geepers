"""Residual-based step (jump) detection for GNSS and InSAR timeseries.

Complements the catalogued steps from `UnrSource.steps` (equipment changes,
earthquakes) with a detector that finds *uncatalogued* jumps directly in the
data - e.g. undocumented antenna changes in GPS, or phase-unwrapping errors
in InSAR series sampled at station locations.

For every candidate epoch, two models are fit to a sliding window of the
series: a line, and a line plus a Heaviside step at that epoch. The
difference in the Akaike Information Criterion (AIC) between the two fits
measures how strongly the data favor a step. Local maxima of the AIC
improvement above a threshold are reported, provided the step is also
larger than a multiple of the series' white-noise level: the AIC test
assumes white noise, so on its own it reports the wander of temporally
correlated (flicker, random-walk) noise as steps, the more so the more
samples a window holds. This model-comparison approach follows the
step-detection concept of

    Köhne, T., Riel, B., & Simons, M. (2023). Decomposition and Inference
    of Sources through Spatiotemporal Analysis of Network Signals: The
    DISSTANS Python package. Computers & Geosciences, 170, 105247.
    https://doi.org/10.1016/j.cageo.2022.105247

(clean-room implementation from the published description).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["detect_steps", "detect_steps_enu"]


def _aic(rss: float, n: int, k: int) -> float:
    """AIC for a least-squares fit with `k` parameters and `n` samples."""
    return n * np.log(max(rss, 1e-20) / n) + 2 * k


def white_noise_sigma(values: np.ndarray) -> float:
    """Robust white-noise level of a series from its first differences.

    Differencing removes the trend and most of the temporally correlated
    noise; the difference of two independent samples has variance
    ``2 sigma**2``, and the scaled MAD keeps steps and outliers out of
    the estimate.

    Parameters
    ----------
    values : np.ndarray
        Series values in time order, without NaNs.

    Returns
    -------
    float
        1-sigma white-noise level, in the units of `values`.

    """
    diffs = np.diff(values)
    return float(1.4826 * np.median(np.abs(diffs - np.median(diffs))) / np.sqrt(2))


def detect_steps(
    series: pd.Series,
    window_days: int = 60,
    min_step_ratio: float = 20.0,
    min_separation_days: int = 30,
    min_step_sigma: float = 3.0,
) -> pd.DataFrame:
    """Detect step discontinuities in a single timeseries.

    Parameters
    ----------
    series : pd.Series
        Values indexed by a DatetimeIndex. NaNs are dropped.
    window_days : int
        Length of the sliding window (days) centered on each candidate
        epoch. Each side must contain at least 5 observations.
    min_step_ratio : float
        Minimum AIC improvement (``aic_line - aic_step``) for an epoch to
        qualify as a step. Larger means fewer, more confident detections.
    min_separation_days : int
        Merge detections closer than this, keeping the strongest.
    min_step_sigma : float
        Minimum step size, in units of the series' white-noise level
        (`white_noise_sigma`). Rejects the small apparent steps that
        temporally correlated noise produces; steps below about 3 sigma
        cannot be told apart from such noise. Default 3; 0 disables the
        check (AIC test only).

    Returns
    -------
    pd.DataFrame
        One row per detected step with columns ``date``, ``step_size``
        (signed, same units as the input), and ``delta_aic``.

    """
    clean = series.dropna()
    if len(clean) < 10:
        return pd.DataFrame(columns=["date", "step_size", "delta_aic"])

    values = clean.to_numpy(dtype=float)
    days = (clean.index - clean.index[0]).days.to_numpy(dtype=float)
    n = len(values)
    half = window_days / 2
    min_size = min_step_sigma * white_noise_sigma(values)

    dates, sizes, delta_aics = [], [], []
    for i in range(1, n):
        t0 = days[i]
        in_window = (days >= t0 - half) & (days < t0 + half)
        n_before = int(np.sum(in_window & (days < t0)))
        n_after = int(np.sum(in_window & (days >= t0)))
        if n_before < 5 or n_after < 5:
            continue

        t = days[in_window]
        y = values[in_window]
        m = len(y)

        # Line: y = a + b t
        design_line = np.column_stack([np.ones(m), t])
        _, rss_line, *_ = np.linalg.lstsq(design_line, y, rcond=None)
        rss_line = float(rss_line[0]) if len(rss_line) else 0.0

        # Line + Heaviside step at t0
        heaviside = (t >= t0).astype(float)
        design_step = np.column_stack([np.ones(m), t, heaviside])
        coeffs, rss_step, *_ = np.linalg.lstsq(design_step, y, rcond=None)
        rss_step = float(rss_step[0]) if len(rss_step) else 0.0

        delta = _aic(rss_line, m, 2) - _aic(rss_step, m, 3)
        if delta > min_step_ratio and abs(coeffs[2]) > min_size:
            dates.append(clean.index[i])
            sizes.append(float(coeffs[2]))
            delta_aics.append(float(delta))

    df = pd.DataFrame({"date": dates, "step_size": sizes, "delta_aic": delta_aics})
    if df.empty:
        return df

    # Non-maximum suppression: keep the strongest detection within
    # min_separation_days of each local cluster
    df = df.sort_values("delta_aic", ascending=False).reset_index(drop=True)
    kept: list[int] = []
    for idx, row in df.iterrows():
        if all(
            abs((row["date"] - df.loc[j, "date"]).days) >= min_separation_days
            for j in kept
        ):
            kept.append(idx)
    return df.loc[kept].sort_values("date").reset_index(drop=True)


def detect_steps_enu(
    df: pd.DataFrame,
    components: tuple[str, ...] = ("east", "north", "up"),
    date_column: str = "date",
    **kwargs,
) -> pd.DataFrame:
    """Detect steps in each component of a station observation DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Observation table with a date column and component columns
        (e.g. from `geepers.gps_sources` or `geepers.synthetic`).
    components : tuple of str
        Column names to scan.
    date_column : str
        Name of the date column.
    **kwargs
        Passed through to `detect_steps`.

    Returns
    -------
    pd.DataFrame
        Concatenated detections with an extra ``component`` column.

    """
    results = []
    indexed = df.set_index(date_column)
    for comp in components:
        found = detect_steps(indexed[comp], **kwargs)
        found["component"] = comp
        results.append(found)
    # Drop components with no detections before concat: concatenating empty /
    # all-NA frames is deprecated in pandas and changes result dtypes.
    non_empty = [r for r in results if not r.empty]
    if not non_empty:
        return pd.DataFrame(columns=["date", "step_size", "delta_aic", "component"])
    return pd.concat(non_empty, ignore_index=True)

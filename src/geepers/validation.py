"""GNSS-vs-InSAR validation utilities.

One-stop module for validating an InSAR displacement product against
GNSS: apples-to-apples velocity fitting (the *same* estimator applied
to both series), comparison statistics (bias, MAD, RMSE, R²), the
pairwise structure function / semivariogram of the misfit, and the
per-epoch network RMSE.

The structure-function pieces are re-exported from `geepers.analysis`;
the plot helpers accept log-log or linear axes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

from geepers.analysis import (
    binned_rmse_profile,
    epoch_rmse,
    pairwise_differential_rmse,
)
from geepers.midas import midas

__all__ = [
    "binned_rmse_profile",
    "comparison_stats",
    "epoch_rmse",
    "fit_velocities",
    "misfit_semivariogram",
    "pairwise_differential_rmse",
    "plot_semivariogram",
    "plot_velocity_scatter",
]

logger = logging.getLogger("geepers")

DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------------------
# Velocity comparison
# ---------------------------------------------------------------------------
def _fit_one(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    method: str,
    step_years: np.ndarray | None,
) -> tuple[float, float]:
    """Velocity and 1-sigma for a single series (input units per year)."""
    good = np.isfinite(values)
    if good.sum() < 2:
        return np.nan, np.nan
    t = (
        (dates[good] - dates[good][0]).total_seconds().to_numpy()
        / 86400
        / DAYS_PER_YEAR
    )
    v = values[good]

    if method == "midas":
        res = midas(t, v, step_years)
        return res.velocity, res.velocity_uncertainty
    if method == "lsq":
        A = np.c_[t, np.ones_like(t)]
        coef, residuals, *_ = np.linalg.lstsq(A, v, rcond=None)
        if len(v) > 2 and residuals.size:
            dof = len(v) - 2
            cov = residuals[0] / dof * np.linalg.inv(A.T @ A)
            return float(coef[0]), float(np.sqrt(cov[0, 0]))
        return float(coef[0]), np.nan
    if method == "trend":
        from geepers.trend import estimate_trend

        res = estimate_trend(dates[good], v, periods_years=(1.0, 0.5))
        return res.velocity, res.velocity_uncertainty
    msg = f"Unknown method: {method}"
    raise ValueError(msg)


def fit_velocities(
    station_to_merged_df: Mapping[str, pd.DataFrame],
    method: Literal["midas", "lsq", "trend"] = "midas",
    step_dates: Mapping[str, list] | None = None,
) -> pd.DataFrame:
    """Fit the *same* velocity estimator to the GPS and InSAR series.

    Comparing a MIDAS GPS rate against a least-squares InSAR rate mixes
    estimator differences into the product assessment; fitting both
    series identically isolates the product error.

    Parameters
    ----------
    station_to_merged_df
        Mapping from station name to a dataframe indexed by date with
        ``los_gps`` and ``los_insar`` columns.
    method : {"midas", "lsq", "trend"}
        Velocity estimator applied to *both* series. "midas" needs
        roughly a year of data to form pairs; "lsq" is ordinary least
        squares (fast, outlier-sensitive); "trend" is the power-law +
        white noise MLE (`geepers.trend`, slow but honest sigmas).
    step_dates
        Optional mapping from station name to a list of step epochs
        (passed to MIDAS so pairs never span them).

    Returns
    -------
    pd.DataFrame
        Indexed by station with ``gps_velocity``, ``gps_sigma``,
        ``insar_velocity``, ``insar_sigma`` (input units per year) and
        ``n_dates``.

    """
    rows = {}
    for station, df in station_to_merged_df.items():
        dates = pd.DatetimeIndex(df.index)
        steps_yr = None
        if step_dates and station in step_dates and len(dates):
            t0 = dates.min()
            steps_yr = np.sort(
                (pd.DatetimeIndex(pd.to_datetime(step_dates[station])) - t0)
                .total_seconds()
                .to_numpy()
                / 86400
                / DAYS_PER_YEAR
            )
        vg, sg = _fit_one(dates, df["los_gps"].to_numpy(float), method, steps_yr)
        vi, si = _fit_one(dates, df["los_insar"].to_numpy(float), method, steps_yr)
        rows[station] = {
            "gps_velocity": vg,
            "gps_sigma": sg,
            "insar_velocity": vi,
            "insar_sigma": si,
            "n_dates": int(
                np.isfinite(df[["los_gps", "los_insar"]].to_numpy(float))
                .all(axis=1)
                .sum()
            ),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def comparison_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Agreement statistics between two paired samples.

    Parameters
    ----------
    x, y : np.ndarray
        Paired values (e.g. InSAR and GPS velocities). NaN pairs are
        dropped.

    Returns
    -------
    dict
        ``bias`` (mean of y - x), ``mad`` (1.4826-scaled median absolute
        deviation of y - x about its median), ``rmse``, ``r2``
        (squared Pearson correlation) and ``n``.

    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2:
        return {
            "bias": np.nan,
            "mad": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "n": len(x),
        }
    d = y - x
    r = np.corrcoef(x, y)[0, 1]
    return {
        "bias": float(np.mean(d)),
        "mad": float(1.4826 * np.median(np.abs(d - np.median(d)))),
        "rmse": float(np.sqrt(np.mean(d**2))),
        "r2": float(r**2),
        "n": len(x),
    }


def plot_velocity_scatter(
    velocity_df: pd.DataFrame,
    units: str = "m/yr",
    scale: float = 1.0,
    annotate: bool = True,
    ax=None,
):
    """Scatter of GPS vs InSAR velocities with 1:1 line and statistics.

    Parameters
    ----------
    velocity_df : pd.DataFrame
        Output of `fit_velocities` (or any frame with ``insar_velocity``
        / ``gps_velocity`` and optionally the sigma columns).
    units : str
        Axis unit label after applying `scale`. Default "m/yr".
    scale : float
        Multiply values before plotting (e.g. 1000 for m -> mm).
    annotate : bool
        Label each point with the station name.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes

    """
    import matplotlib.pyplot as plt

    ok = velocity_df[["insar_velocity", "gps_velocity"]].dropna() * scale
    stats = comparison_stats(ok["insar_velocity"], ok["gps_velocity"])

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    if ok.empty:
        ax.text(0.5, 0.5, "no finite velocity pairs", ha="center")
        return ax

    lim = np.nanmax(np.abs(ok.to_numpy())) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="1:1")

    xerr = yerr = None
    if {"insar_sigma", "gps_sigma"} <= set(velocity_df.columns):
        sig = velocity_df.loc[ok.index, ["insar_sigma", "gps_sigma"]] * scale
        xerr, yerr = sig["insar_sigma"], sig["gps_sigma"]
    ax.errorbar(
        ok["insar_velocity"],
        ok["gps_velocity"],
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        ms=6,
        capsize=2,
        zorder=3,
    )
    if annotate:
        for sid, row in ok.iterrows():
            ax.annotate(
                str(sid),
                (row["insar_velocity"], row["gps_velocity"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.set_xlabel(f"InSAR velocity [{units}]")
    ax.set_ylabel(f"GPS velocity [{units}]")
    ax.set_title(
        f"bias={stats['bias']:.2f}  MAD={stats['mad']:.2f}  "
        f"RMSE={stats['rmse']:.2f} {units}   R²={stats['r2']:.3f}  n={stats['n']}"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")
    return ax


# ---------------------------------------------------------------------------
# Semivariogram of the misfit
# ---------------------------------------------------------------------------
def misfit_semivariogram(
    rmse_df: pd.DataFrame,
    n_bins: int = 12,
    log_bins: bool = False,
) -> pd.DataFrame:
    """Semivariogram of the InSAR-GPS differential misfit.

    For a pair (A, B), the pairwise RMSE of the differential residual
    relates to the semivariance as ``gamma = rmse**2 / 2`` — so the
    semivariogram is derived directly from
    `pairwise_differential_rmse` output and binned by distance.

    Parameters
    ----------
    rmse_df : pd.DataFrame
        Output of `pairwise_differential_rmse`.
    n_bins : int
        Number of distance bins. Default 12.
    log_bins : bool
        Log-spaced distance bins (recommended with log-log plotting).

    Returns
    -------
    pd.DataFrame
        Per non-empty bin: ``distance_km`` (bin center),
        ``gamma_median``, ``gamma_mean`` (squared series units) and
        ``n_pairs``.

    """
    if rmse_df.empty:
        return pd.DataFrame()
    d = rmse_df["distance_km"].to_numpy()
    gamma = 0.5 * rmse_df["rmse"].to_numpy() ** 2

    if log_bins:
        lo = max(d.min() * 0.999, 1e-3)
        edges = np.geomspace(lo, d.max() * 1.001, n_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(0.0, d.max() * 1.001, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(d, edges) - 1

    rows = []
    for b in range(n_bins):
        sel = which == b
        if not sel.any():
            continue
        rows.append(
            {
                "distance_km": float(centers[b]),
                "gamma_median": float(np.median(gamma[sel])),
                "gamma_mean": float(np.mean(gamma[sel])),
                "n_pairs": int(sel.sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_semivariogram(
    rmse_df: pd.DataFrame,
    binned: pd.DataFrame | None = None,
    scale: Literal["loglog", "logx", "linear"] = "loglog",
    units: str = "m",
    ax=None,
):
    """Semivariogram of the misfit vs distance.

    Parameters
    ----------
    rmse_df : pd.DataFrame
        Output of `pairwise_differential_rmse` (pairs are drawn as
        points with ``gamma = rmse**2 / 2``).
    binned : pd.DataFrame, optional
        Output of `misfit_semivariogram`; adds the binned median curve.
    scale : {"loglog", "logx", "linear"}
        Axis scaling. Default "loglog".
    units : str
        Base unit of the series; the y label shows ``units²``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes

    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    gamma = 0.5 * rmse_df["rmse"] ** 2
    ax.scatter(
        rmse_df["distance_km"],
        gamma,
        s=25,
        c="0.2",
        alpha=0.6,
        zorder=3,
        label="station pairs",
    )
    if binned is not None and len(binned):
        ax.plot(
            binned["distance_km"],
            binned["gamma_median"],
            "r-o",
            ms=4,
            lw=1.5,
            zorder=4,
            label="median per bin",
        )

    if scale in ("loglog", "logx"):
        ax.set_xscale("log")
    if scale == "loglog":
        ax.set_yscale("log")
    ax.set_xlabel("Station separation (km)")
    ax.set_ylabel(f"Semivariance γ ({units}²)")  # noqa: RUF001  (greek gamma intended)
    ax.set_title("Semivariogram of InSAR - GPS differential misfit")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    return ax

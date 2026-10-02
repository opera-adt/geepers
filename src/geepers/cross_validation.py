# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Spatially blocked cross-validation for velocity-field interpolators.

Scores an interpolator on stations it has not seen, to choose its
parameters (correlation length, damping, ...) and to compare methods
(`geepers.collocation`, `geepers.gps_imaging`, ...) on equal terms.

Neighboring stations are correlated, so holding out stations at random
leaves a near-duplicate of every test station in the training set and
the score comes out too optimistic. Holding out whole spatial blocks
instead measures how well the method predicts *away* from the data
(Roberts et al., 2017, Cross-validation strategies for data with
temporal, spatial, hierarchical, or phylogenetic structure, Ecography,
40(8), 913-929, https://doi.org/10.1111/ecog.02881). The block-balanced
fold assignment follows ``verde.BlockKFold`` (Uieda, 2018,
https://doi.org/10.21105/joss.00957).

Example:
-------
>>> def predict(train, test):                                # doctest: +SKIP
...     out = ordinary_kriging(lon[train], lat[train], v[train], s[train],
...                            lon[test], lat[test], parameters)
...     return out.signal[:, 0]
>>> cv = cross_validate(predict, lon, lat, v, block_km=100)  # doctest: +SKIP
>>> cv.rmse                                                  # doctest: +SKIP

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from pyproj import Transformer

__all__ = ["CrossValidationResult", "block_kfold", "cross_validate", "spatial_blocks"]


def spatial_blocks(lon: ArrayLike, lat: ArrayLike, block_km: float) -> np.ndarray:
    """Label each point with the square block that contains it.

    Parameters
    ----------
    lon, lat : array-like
        Point coordinates in degrees.
    block_km : float
        Block edge length in kilometers.

    Returns
    -------
    np.ndarray of int
        Block label per point, numbered 0..n_blocks-1 (only blocks that
        contain points are numbered).

    """
    lon = np.asarray(lon, float).ravel()
    lat = np.asarray(lat, float).ravel()
    # Equal-area, so every block covers the same ground area
    transformer = Transformer.from_crs(
        "EPSG:4326",
        {
            "proj": "laea",
            "lat_0": float(lat.mean()),
            "lon_0": float(lon.mean()),
            "datum": "WGS84",
            "units": "km",
        },
        always_xy=True,
    )
    x, y = transformer.transform(lon, lat)
    cells = np.c_[np.floor(x / block_km), np.floor(y / block_km)].astype(int)
    _, labels = np.unique(cells, axis=0, return_inverse=True)
    return labels.ravel()


def block_kfold(
    lon: ArrayLike,
    lat: ArrayLike,
    *,
    block_km: float | None,
    n_splits: int = 5,
    seed: int | np.random.Generator | None = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split points into K folds made of whole spatial blocks.

    Blocks are shuffled and dealt, largest first, to the fold that
    currently holds the fewest points, so folds end up with similar
    numbers of points even when the network is clustered.

    Parameters
    ----------
    lon, lat : array-like
        Point coordinates in degrees.
    block_km : float or None
        Block edge length in kilometers; should exceed the correlation
        length of the field. None puts every point in its own block,
        i.e. ordinary (random) K-fold.
    n_splits : int
        Number of folds. Default 5.
    seed : int or np.random.Generator, optional
        Seed for the block shuffle. Default 0 (reproducible).

    Returns
    -------
    list of (train, test)
        Index arrays for each fold. Every point is in exactly one test
        set.

    Raises
    ------
    ValueError
        If there are fewer occupied blocks than folds.

    """
    n_points = np.asarray(lon).size
    if block_km is None:
        labels = np.arange(n_points)
    else:
        labels = spatial_blocks(lon, lat, block_km)
    n_blocks = int(labels.max()) + 1
    if n_blocks < n_splits:
        msg = (
            f"Only {n_blocks} occupied blocks for {n_splits} folds - "
            "decrease block_km or n_splits"
        )
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    sizes = np.bincount(labels, minlength=n_blocks)
    order = rng.permutation(n_blocks)
    # Stable sort keeps the shuffled order among equally sized blocks
    order = order[np.argsort(-sizes[order], kind="stable")]

    fold_of_block = np.empty(n_blocks, dtype=int)
    fold_sizes = np.zeros(n_splits, dtype=int)
    for block in order:
        fold = int(np.argmin(fold_sizes))
        fold_of_block[block] = fold
        fold_sizes[fold] += sizes[block]

    fold_of_point = fold_of_block[labels]
    return [
        (np.flatnonzero(fold_of_point != k), np.flatnonzero(fold_of_point == k))
        for k in range(n_splits)
    ]


@dataclass
class CrossValidationResult:
    """Out-of-fold predictions and their misfit.

    Attributes
    ----------
    predictions : np.ndarray
        Prediction for every point, made by a model that did not see it;
        same shape as the values.
    residuals : np.ndarray
        ``values - predictions``.
    fold : np.ndarray of int
        Fold in which each point was held out.
    rmse : np.ndarray
        Root-mean-square residual per component, shape (n_components,).
    mad : np.ndarray
        Robust spread per component: 1.4826 x median absolute deviation
        of the residuals about their median.
    fold_rmse : np.ndarray
        RMSE per fold and component, shape (n_splits, n_components).

    """

    predictions: np.ndarray
    residuals: np.ndarray
    fold: np.ndarray
    rmse: np.ndarray
    mad: np.ndarray
    fold_rmse: np.ndarray


def cross_validate(
    predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    lon: ArrayLike,
    lat: ArrayLike,
    values: ArrayLike,
    *,
    block_km: float | None,
    n_splits: int = 5,
    seed: int | np.random.Generator | None = 0,
) -> CrossValidationResult:
    """Score an interpolator on spatially held-out stations.

    Parameters
    ----------
    predict : callable
        ``predict(train, test) -> predictions``: fit on the stations with
        indices `train` and return the prediction at the stations with
        indices `test`, shaped like ``values[test]``. The callable owns
        the data (coordinates, values, uncertainties), which keeps this
        function independent of any interpolator's signature.
    lon, lat : array-like
        Station coordinates in degrees (used to build the folds).
    values : array-like
        Observed values, shape (n,) for a scalar field or (n, k) for k
        components (e.g. east and north).
    block_km : float or None
        Block edge length passed to `block_kfold`; None gives ordinary
        K-fold.
    n_splits : int
        Number of folds. Default 5.
    seed : int or np.random.Generator, optional
        Seed for the fold assignment. Default 0.

    Returns
    -------
    CrossValidationResult

    """
    values = np.asarray(values, float)
    table = values.reshape(len(values), -1)
    predictions = np.full(table.shape, np.nan)
    fold = np.empty(len(table), dtype=int)

    folds = block_kfold(lon, lat, block_km=block_km, n_splits=n_splits, seed=seed)
    for k, (train, test) in enumerate(folds):
        predicted = np.asarray(predict(train, test), float)
        predictions[test] = predicted.reshape(len(test), -1)
        fold[test] = k

    residuals = table - predictions
    with np.errstate(invalid="ignore"):
        fold_rmse = np.array(
            [
                np.sqrt(np.nanmean(residuals[fold == k] ** 2, axis=0))
                for k in range(n_splits)
            ]
        )
        rmse = np.sqrt(np.nanmean(residuals**2, axis=0))
    mad = 1.4826 * np.nanmedian(
        np.abs(residuals - np.nanmedian(residuals, axis=0)), axis=0
    )
    return CrossValidationResult(
        predictions=predictions.reshape(values.shape),
        residuals=residuals.reshape(values.shape),
        fold=fold,
        rmse=rmse,
        mad=mad,
        fold_rmse=fold_rmse,
    )

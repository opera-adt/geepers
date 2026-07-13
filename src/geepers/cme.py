"""Common-mode error (CME) estimation for station networks.

Regional GNSS networks share a coherent noise component (reference-frame
wobble, large-scale atmosphere/loading) that inflates the scatter of every
station equally. Removing this "common mode" - estimated here from the
network's residuals via principal component analysis (or optionally ICA) -
substantially tightens station repeatability and, in geepers' context, the
GPS <-> InSAR comparison statistics.

The approach is the network-decomposition strategy popularized by

    Wdowinski, S., et al. (1997). Southern California permanent GPS geodetic
    array: Spatial filtering of daily positions for estimating coseismic and
    postseismic displacements induced by the 1992 Landers earthquake.
    JGR, 102(B8). https://doi.org/10.1029/97JB01378

and implemented as PCA/ICA decomposition in DISSTANS (Köhne et al., 2023;
clean-room implementation from the published description).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["CMEResult", "estimate_cme", "remove_cme"]


@dataclass
class CMEResult:
    """Result of a common-mode estimation.

    Attributes
    ----------
    common_mode : pd.DataFrame
        Temporal common-mode signal(s), indexed like the input, one column
        per component (``cme_0``, ``cme_1``, ...). Units of the input.
    spatial_response : pd.DataFrame
        Per-station response (unit-norm spatial eigenvector entries),
        stations as index, one column per component.
    cleaned : pd.DataFrame
        Input with the common-mode reconstruction subtracted (NaNs of the
        input are preserved).
    explained_variance : np.ndarray
        Fraction of total variance explained by each returned component.

    """

    common_mode: pd.DataFrame
    spatial_response: pd.DataFrame
    cleaned: pd.DataFrame
    explained_variance: np.ndarray


def estimate_cme(
    residuals: pd.DataFrame,
    n_components: int = 1,
    method: str = "pca",
    min_coverage: float = 0.5,
) -> CMEResult:
    """Estimate the common-mode signal of a network from residual series.

    Parameters
    ----------
    residuals : pd.DataFrame
        Detrended residuals: rows indexed by date, one column per station.
        Should have trends/seasonals already removed (e.g. the residuals
        of `geepers.trend` fits, or GPS-InSAR difference series).
    n_components : int
        Number of common-mode components to estimate.
    method : str
        ``"pca"`` (numpy SVD, default) or ``"ica"`` (requires scikit-learn).
    min_coverage : float
        Drop stations observed in fewer than this fraction of epochs.

    Returns
    -------
    CMEResult
        Temporal common mode, per-station response, cleaned residuals, and
        explained variance fractions.

    Raises
    ------
    ValueError
        If fewer than 3 stations pass the coverage filter, or `method` is
        unknown.

    """
    coverage = residuals.notna().mean()
    used = residuals.loc[:, coverage >= min_coverage]
    if used.shape[1] < 3:
        msg = (
            f"Need at least 3 stations with >= {min_coverage:.0%} coverage;"
            f" got {used.shape[1]}"
        )
        raise ValueError(msg)

    # Center each station and zero-fill gaps so the SVD ignores them
    means = used.mean()
    centered = (used - means).fillna(0.0)
    matrix = centered.to_numpy()  # (n_epochs, n_stations)

    if method == "pca":
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        temporal = u[:, :n_components] * s[:n_components]
        spatial = vt[:n_components].T  # (n_stations, n_components)
        total_var = float(np.sum(s**2))
        if total_var > 0:
            explained = (s[:n_components] ** 2) / total_var
        else:
            explained = np.zeros(n_components)
    elif method == "ica":
        try:
            from sklearn.decomposition import FastICA
        except ImportError as e:
            msg = "method='ica' requires scikit-learn"
            raise ImportError(msg) from e
        ica = FastICA(n_components=n_components, random_state=0)
        temporal = ica.fit_transform(matrix)
        spatial = ica.mixing_
        recon_var = np.var(temporal @ spatial.T, axis=0).sum()
        explained = np.full(n_components, recon_var / matrix.var(axis=0).sum())
    else:
        msg = f"Unknown method: {method!r}. Use 'pca' or 'ica'"
        raise ValueError(msg)

    # Sign convention: majority-positive spatial response per component
    for j in range(n_components):
        if np.median(spatial[:, j]) < 0:
            spatial[:, j] *= -1
            temporal[:, j] *= -1

    columns = [f"cme_{j}" for j in range(n_components)]
    common_mode = pd.DataFrame(temporal, index=used.index, columns=columns)
    spatial_response = pd.DataFrame(spatial, index=used.columns, columns=columns)

    reconstruction = pd.DataFrame(
        temporal @ spatial.T, index=used.index, columns=used.columns
    )
    cleaned = residuals.copy()
    cleaned.loc[:, used.columns] = used - reconstruction

    return CMEResult(
        common_mode=common_mode,
        spatial_response=spatial_response,
        cleaned=cleaned,
        explained_variance=explained,
    )


def remove_cme(
    residuals: pd.DataFrame,
    n_components: int = 1,
    method: str = "pca",
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Return `residuals` with the estimated common mode removed.

    Convenience wrapper around `estimate_cme`; see it for parameters.
    """
    return estimate_cme(
        residuals,
        n_components=n_components,
        method=method,
        min_coverage=min_coverage,
    ).cleaned

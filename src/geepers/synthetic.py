"""Generate synthetic GNSS networks and timeseries for testing and validation.

Produces station tables matching `PointSchema` and observation tables
matching `StationObservationSchema`, built from a trajectory model
(velocity + seasonal + steps) plus white and power-law (colored) noise,
optionally sharing a common-mode signal across a network.

The power-law noise generator follows the fractional-differencing
(Hosking) recursion as described in

    Williams, S. D. P. (2003), The effect of coloured noise on the
    uncertainties of rates estimated from geodetic time series,
    J. Geod., 76, 483-494. https://doi.org/10.1007/s00190-002-0283-4
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd

from geepers.schemas import PointSchema, StationObservationSchema

__all__ = [
    "SyntheticStep",
    "power_law_noise",
    "synthetic_network_timeseries",
    "synthetic_stations",
    "synthetic_timeseries",
]

DAYS_PER_YEAR = 365.25


@dataclass
class SyntheticStep:
    """A step (offset) added to a synthetic timeseries.

    Attributes
    ----------
    date : str or pd.Timestamp
        Epoch at which the offset is applied (inclusive).
    east, north, up : float
        Offset amplitudes in meters.

    """

    date: str | pd.Timestamp
    east: float = 0.0
    north: float = 0.0
    up: float = 0.0


def power_law_noise(
    n: int,
    spectral_index: float = -1.0,
    scale: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Generate power-law noise via fractional differencing.

    Parameters
    ----------
    n : int
        Number of samples.
    spectral_index : float
        Spectral index kappa of the power law ``P(f) ~ f**kappa``.
        0 is white noise, -1 flicker noise, -2 random walk.
    scale : float
        Standard deviation of the driving white noise.
    seed : int or np.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    np.ndarray
        Noise samples of length `n`.

    Examples
    --------
    >>> noise = power_law_noise(100, spectral_index=-1, seed=1234)
    >>> noise.shape
    (100,)

    """
    rng = np.random.default_rng(seed)
    white = rng.normal(scale=scale, size=n)
    if spectral_index == 0:
        return white
    # Hosking recursion for the filter coefficients h_i (Williams 2003, eq. 8)
    d = -spectral_index / 2
    h = np.empty(n)
    h[0] = 1.0
    idx = np.arange(1, n)
    h[1:] = np.cumprod((d + idx - 1) / idx)
    return np.convolve(white, h)[:n]


def synthetic_timeseries(
    dates: pd.DatetimeIndex,
    velocity_enu: tuple[float, float, float] = (0.0, 0.0, 0.0),
    annual_enu: tuple[float, float, float] = (0.0, 0.0, 0.0),
    semiannual_enu: tuple[float, float, float] = (0.0, 0.0, 0.0),
    steps: list[SyntheticStep] | None = None,
    white_sigma: float = 0.001,
    colored_sigma: float = 0.0,
    spectral_index: float = -1.0,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate one synthetic E/N/U timeseries.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Observation epochs.
    velocity_enu : tuple of float
        Linear velocities in meters/year for east, north, up.
    annual_enu, semiannual_enu : tuple of float
        Amplitudes (meters) of annual and semiannual sinusoids per component.
    steps : list of SyntheticStep, optional
        Offsets to add.
    white_sigma : float
        White-noise standard deviation in meters (also used as the
        reported per-epoch sigma).
    colored_sigma : float
        Driving standard deviation of the power-law noise, in meters.
        0 disables colored noise.
    spectral_index : float
        Spectral index for the colored noise (-1 flicker, -2 random walk).
    seed : int or np.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        Validated against `StationObservationSchema`: columns ``date``,
        ``east``, ``north``, ``up``, per-component sigmas and correlations.

    """
    rng = np.random.default_rng(seed)
    n = len(dates)
    t_years = (dates - dates[0]).days.to_numpy() / DAYS_PER_YEAR
    omega1 = 2 * np.pi * t_years
    omega2 = 4 * np.pi * t_years

    data = {}
    for i, comp in enumerate(["east", "north", "up"]):
        series = (
            velocity_enu[i] * t_years
            + annual_enu[i] * np.sin(omega1 + rng.uniform(0, 2 * np.pi))
            + semiannual_enu[i] * np.sin(omega2 + rng.uniform(0, 2 * np.pi))
            + rng.normal(scale=white_sigma, size=n)
        )
        if colored_sigma > 0:
            series += power_law_noise(
                n, spectral_index=spectral_index, scale=colored_sigma, seed=rng
            )
        for step in steps or []:
            series += np.where(
                dates >= pd.Timestamp(step.date), getattr(step, comp), 0.0
            )
        data[comp] = series

    df = pd.DataFrame({"date": dates, **data})
    sigma = max(white_sigma, 1e-6)
    for comp in ["east", "north", "up"]:
        df[f"sigma_{comp}"] = sigma
    for corr in ["corr_en", "corr_eu", "corr_nu"]:
        df[corr] = 0.0
    return StationObservationSchema.validate(df, lazy=True)


def synthetic_stations(
    n: int,
    bbox: tuple[float, float, float, float] = (-120.0, 34.0, -118.0, 36.0),
    seed: int | np.random.Generator | None = None,
) -> gpd.GeoDataFrame:
    """Generate random station locations inside a bounding box.

    Parameters
    ----------
    n : int
        Number of stations.
    bbox : tuple of float
        (west, south, east, north) in degrees.
    seed : int or np.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    gpd.GeoDataFrame
        Columns ``id``, ``lon``, ``lat``, ``alt`` and point geometry,
        validated against `PointSchema`.

    """
    rng = np.random.default_rng(seed)
    west, south, east, north = bbox
    lons = rng.uniform(west, east, size=n)
    lats = rng.uniform(south, north, size=n)
    alts = rng.uniform(0, 2000, size=n)
    ids = [f"S{i:03d}" for i in range(n)]
    gdf = gpd.GeoDataFrame(
        {"id": ids, "lon": lons, "lat": lats, "alt": alts},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    )
    return PointSchema.validate(gdf, lazy=True)


@dataclass
class SyntheticNetwork:
    """A synthetic GNSS network: stations, per-station series, common mode.

    Attributes
    ----------
    stations : gpd.GeoDataFrame
        Station metadata (id, lon, lat, alt, geometry).
    observations : dict[str, pd.DataFrame]
        Station id -> observation DataFrame
        (`StationObservationSchema` columns).
    common_mode : pd.DataFrame
        The injected common-mode signal per component, indexed by date
        (all zeros when no common mode was requested).

    """

    stations: gpd.GeoDataFrame
    observations: dict[str, pd.DataFrame]
    common_mode: pd.DataFrame = field(default_factory=pd.DataFrame)


def synthetic_network_timeseries(
    n_stations: int = 10,
    dates: pd.DatetimeIndex | None = None,
    bbox: tuple[float, float, float, float] = (-120.0, 34.0, -118.0, 36.0),
    velocity_enu: tuple[float, float, float] = (0.0, 0.0, 0.0),
    white_sigma: float = 0.001,
    common_mode_sigma: float = 0.0,
    seed: int | np.random.Generator | None = None,
) -> SyntheticNetwork:
    """Generate a network of synthetic stations sharing a common-mode signal.

    Parameters
    ----------
    n_stations : int
        Number of stations.
    dates : pd.DatetimeIndex, optional
        Observation epochs. Default is 3 years of daily solutions
        starting 2020-01-01.
    bbox : tuple of float
        (west, south, east, north) in degrees.
    velocity_enu : tuple of float
        Common linear velocity (m/yr) applied to every station.
    white_sigma : float
        Per-station white noise sigma in meters.
    common_mode_sigma : float
        Standard deviation (meters) of a flicker-noise common-mode signal
        added identically to every station. 0 disables it.
    seed : int or np.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    SyntheticNetwork
        Stations, per-station observations, and the injected common mode.

    """
    rng = np.random.default_rng(seed)
    if dates is None:
        dates = pd.date_range("2020-01-01", periods=3 * 365, freq="D")

    stations = synthetic_stations(n_stations, bbox=bbox, seed=rng)

    components = ["east", "north", "up"]
    if common_mode_sigma > 0:
        cme = {
            c: power_law_noise(
                len(dates), spectral_index=-1, scale=common_mode_sigma, seed=rng
            )
            for c in components
        }
    else:
        cme = {c: np.zeros(len(dates)) for c in components}
    common_mode = pd.DataFrame(cme, index=dates)

    observations = {}
    for sid in stations["id"]:
        df = synthetic_timeseries(
            dates,
            velocity_enu=velocity_enu,
            white_sigma=white_sigma,
            seed=rng,
        )
        for c in components:
            df[c] += common_mode[c].to_numpy()
        observations[sid] = df

    return SyntheticNetwork(
        stations=stations, observations=observations, common_mode=common_mode
    )

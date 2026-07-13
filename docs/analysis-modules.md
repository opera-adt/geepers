# Analysis modules

Beyond the GNSS-vs-InSAR comparison workflow, geepers ships a set of
standalone analysis modules for GNSS velocity fields. All of them
operate on plain numpy/pandas inputs, so they compose freely with the
`geepers.gps_sources` downloaders.

For a runnable end-to-end demonstration with plots, see the
[tour notebook](notebooks/geepers_tour.ipynb).

## Velocities: MIDAS (`geepers.midas`)

Robust trend estimation insensitive to seasonal signals, outliers, and
(known) steps — Blewitt et al. (2016).

```python
import numpy as np
from geepers.midas import midas

# times in decimal years from the first epoch, values in meters
result = midas(t_years, up_meters, step_times)   # step_times optional
print(result.velocity, result.velocity_uncertainty)  # m/yr
```

## Velocities with realistic uncertainties (`geepers.trend`)

Maximum-likelihood trend estimation under power-law + white noise —
the colored noise of GNSS series makes ordinary least-squares
uncertainties 5–10× too optimistic. Clean-room implementation of the
published method (Bos et al., 2013; Williams, 2003), validated against
HectorP.

```python
from geepers.trend import estimate_trend

res = estimate_trend(
    dates, values,                 # pandas dates, values in any unit
    periods_years=(1.0, 0.5),      # annual + semi-annual estimated
    step_dates=["2019-07-06"],     # offsets estimated, not spanned
    noise_model="PLWN",            # power-law + white (default)
)
res.velocity, res.velocity_uncertainty   # per year, honest sigma
res.kappa, res.sigma_powerlaw, res.sigma_white  # noise model
res.parameters                            # every estimated term
```

Two estimation methods:

- `method="exact"` (default): full time-domain (R)MLE — the validated
  reference. Runtime grows as O(n³) with series length (~30 s for
  8 years daily).
- `method="whittle"`: fast spectral (Whittle) approximation of the
  noise parameters — near-instant even for decades-long series, at the
  cost of slightly coarser kappa estimates. Use it for networks.

For many series at once (station networks, InSAR pixel stacks):

```python
from geepers.trend import estimate_trend_many

# values: (n_series, n_epochs) array or DataFrame with one column per series
df = estimate_trend_many(dates, values, method="whittle", n_jobs=8)
df[["velocity", "velocity_uncertainty", "kappa"]]
```

Notes:

- `noise_model="WN"` gives a fast OLS-equivalent first pass.
- `use_rmle=True` (default) applies the restricted-likelihood
  correction for the parameters absorbed by the deterministic model.

## Station stability metrics (`geepers.variability`)

Quality metrics extracted from the AUS GNSS screening workflow:

```python
from geepers.variability import (
    temporal_velocity_variability, ssf_per_station, spatial_variability,
)

# How stable is the velocity in time? (sliding-window MIDAS spread)
tv = temporal_velocity_variability(dates, enu_dataframe)
tv.variability          # {"east": ..., "north": ..., "up": ...} per yr

# Does each station agree with its Delaunay neighbors?
sv = spatial_variability(lon, lat, up_velocities)   # rms + mad columns
ssf_scores = ssf_per_station(lon, lat, {"up": up_velocities})
```

Gap screening lives in `geepers.quality`:

```python
from geepers.quality import gap_percentage
pct = gap_percentage(dates, sampling_days=1, start="20100101", end="20250101")
```

Known step epochs (equipment changes, earthquakes) come from UNR:

```python
from geepers.gps_sources import UnrSource
steps = UnrSource().steps(station_ids=["P123", "ALBY"])
```

## GPS Imaging (`geepers.gps_imaging`)

Robust interpolation of (typically vertical) velocity fields with
SSF-weighted medians over Delaunay neighborhoods — Python port of the
GPS Imaging MATLAB codes (Hammond et al., 2016; please cite when used).
Outlier-resistant and edge-preserving; no explicit smoothing.

```python
from geepers.gps_imaging import make_ssf, median_spatial_filter, msf_interpolate

ssf = make_ssf(lon, lat, v_up, sigma_up)          # data-driven weights
v_filtered = median_spatial_filter(lon, lat, v_up, sigma_up, ssf)
grid = msf_interpolate(lon, lat, v_up, sigma_up, lon_grid, lat_grid, ssf)
grid[["value", "sigma_robust", "n_stations"]]
```

## Least-squares collocation (`geepers.collocation`)

Geostatistical interpolation of horizontal (or scalar) velocity fields
with full covariance propagation — like kriging, with on-sphere angular
cross-covariance terms for velocity fields.

```python
from geepers.collocation import (
    empirical_covariance, interpolate_velocities, create_regular_grid,
)

# 1. Fit a covariance model to the data (C0, d0 of a Gauss-Markov model)
emp = empirical_covariance(lon, lat, ve, vn, sigma_e, sigma_n)

# 2. Interpolate onto a regular grid with uncertainties
lon_g, lat_g = create_regular_grid(lon0=16.0, lat0=45.0,
                                   width_km=400, height_km=300, dx_km=25)
at_stations, at_grid = interpolate_velocities(
    lon, lat, ve, vn, sigma_e, sigma_n, lon_g, lat_g,
    covariance_parameters=emp.parameters,
)
at_grid.signal, at_grid.signal_sigma    # (n, 2) east/north + 1-sigma
```

`collocate`/`predict` give lower-level control (custom covariance
matrices, scalar fields, signal/noise separation at the stations).

**Ordinary kriging** is also available with the same covariance models —
mean-invariant (no demeaning needed) and with a per-station nugget from
the measurement uncertainties:

```python
from geepers.collocation import ordinary_kriging
out = ordinary_kriging(lon, lat, v, sigma, lon_g, lat_g, emp.parameters)
out.signal, out.signal_sigma
```

**Plate-boundary constraint**: interpolating across a plate boundary
smears discontinuous motion. `separate_plates` builds a working
coordinate space where each plate's points are pushed ≥1500 km apart,
so distance-decaying covariances treat stations on different plates as
uncorrelated. Remove per-plate rigid motion first, interpolate in moved
coordinates, then restore:

```python
from geepers.collocation import separate_plates
lon_m, lat_m, plate_idx = separate_plates(
    np.r_[lon, lon_grid], np.r_[lat, lat_grid], plates_gdf,
    min_separation_km=1500,
)
# ... interpolate using lon_m/lat_m, attach results to original coords
```

See the full worked example (with a verified before/after comparison on
a synthetic two-plate field) in
[How-To Guides: interpolating across plate boundaries](how-to-guides.md#how-to-interpolate-across-plate-boundaries-plate-separation-constraint).

## Euler poles and plate motion (`geepers.euler`)

Estimate a rigid-plate rotation from horizontal GNSS velocities and
predict it anywhere - the remove/restore step of plate-boundary-aware
interpolation. Velocities in mm/yr, rates in deg/Myr.

```python
from geepers.euler import EulerPole, estimate_euler_pole, predict_plate_motion

# Estimate from stations on one plate (velocities in mm/yr, ITRF frame)
pole = estimate_euler_pole(lon, lat, ve, vn, sigma_e, sigma_n)
pole.lon, pole.lat, pole.rate        # deg, deg, deg/Myr
pole.wrms, pole.reduced_chi2         # fit quality
pole.uncertainty()                   # pole error ellipse + rate sigma

# Or use published ITRF2014-PMM values directly:
aus = EulerPole(lon=38.0, lat=32.5, rate=0.63)

# Remove the rigid motion before interpolating, restore after
ve_plate, vn_plate = predict_plate_motion(pole, lon, lat)
ve_resid = ve - ve_plate
```

## Common-mode error (`geepers.cme`)

Regional networks share a coherent noise component (reference-frame
wobble, large-scale atmosphere/loading). Estimate it from detrended
residuals by PCA (or ICA) and remove it — typically shrinking scatter
by 30-50% and sharpening every downstream comparison:

```python
from geepers.cme import estimate_cme, remove_cme

# residuals: DataFrame indexed by date, one column per station,
# trends/seasonals already removed
cme = estimate_cme(residuals, n_components=1)
cme.common_mode                 # temporal common-mode signal(s)
cme.spatial_response            # per-station response
cme.explained_variance          # fraction captured per component
cleaned = cme.cleaned           # residuals minus the common mode
# (remove_cme(residuals) is a one-call shortcut that returns `cleaned`)
```

## Step detection (`geepers.steps`)

Finds *uncatalogued* jumps directly in the data (undocumented antenna
changes, InSAR unwrapping errors), complementing the catalogued steps
from `UnrSource.steps()`. AIC-based sliding-window test:

```python
from geepers.steps import detect_steps, detect_steps_enu

found = detect_steps(series, window_days=60)      # pd.Series in, one
found[["date", "step_size", "delta_aic"]]         # row per detection

# All three ENU components of a station at once:
found = detect_steps_enu(df)                      # east/north/up columns
```

Feed the detected dates into `estimate_trend(step_dates=...)` or
`midas(step_times=...)` so they are estimated, not smeared.

## Strain rates (`geepers.strain`)

Differentiate a gridded horizontal velocity field (e.g. collocation or
GPS Imaging output) into the 2D strain-rate tensor with spherical
metric factors:

```python
from geepers.strain import strain_rate_field

ds = strain_rate_field(lon_1d, lat_1d, ve_grid, vn_grid)
ds.max_shear, ds.dilatation, ds.second_invariant, ds.rotation
```

## Synthetic data (`geepers.synthetic`)

Trajectory-model + colored-noise generators producing schema-valid
tables — for testing pipelines and validating estimators against known
truth:

```python
import pandas as pd
from geepers.synthetic import synthetic_timeseries, synthetic_network_timeseries

dates = pd.date_range("2020-01-01", periods=1000, freq="D")
df = synthetic_timeseries(dates, velocity_enu=(0.005, -0.002, 0.001),
                          colored_sigma=0.002, spectral_index=-1.0, seed=1)
net = synthetic_network_timeseries(n_stations=20, common_mode_sigma=0.002)
net.stations           # GeoDataFrame (PointSchema)
net.observations       # {station_id: StationObservationSchema DataFrame}
```

## Product validation (`geepers.validation`)

One-stop module for verifying an InSAR displacement product (OPERA
DISP) against GPS:

```python
import numpy as np
from geepers.validation import (
    fit_velocities, plot_velocity_scatter,             # velocities
    pairwise_differential_rmse, binned_rmse_profile,   # structure fn
    misfit_semivariogram, plot_semivariogram,          # semivariogram
    epoch_rmse,                                        # per-epoch QC
)
from geepers.plotting import plot_rmse_vs_distance

# station_to_merged: {station: DataFrame indexed by date with
#                     los_gps, los_insar}  (built by the main workflow)
# coords: {station: (lon, lat)}

# 1. Velocity comparison with the SAME estimator applied to both
#    series (midas / lsq / trend) - scatter with bias, MAD, RMSE, R2:
vel = fit_velocities(station_to_merged, method="midas")
plot_velocity_scatter(vel, units="mm/yr", scale=1000)

# 2. Structure function: relative InSAR-GPS RMSE for every station
#    pair vs separation. Reference-datum free by construction.
rmse_df = pairwise_differential_rmse(station_to_merged, coords)

# 3. Binned profile + requirement compliance,
#    e.g. "3 mm + 0.5 mm * sqrt(d_km)" with series in meters:
req = lambda d: (3 + 0.5 * np.sqrt(d)) / 1000
profile = binned_rmse_profile(rmse_df, requirement=req)
plot_rmse_vs_distance(rmse_df, profile, requirement=req)  # log-log
#                                        loglog=False for linear axes

# 4. Semivariogram of the misfit (gamma = rmse^2 / 2):
svg = misfit_semivariogram(rmse_df, log_bins=True)
plot_semivariogram(rmse_df, svg, scale="loglog")  # or "logx", "linear"

# 5. Per-epoch network misfit - spikes flag bad acquisitions
#    (ionosphere, unwrapping, snow):
per_epoch = epoch_rmse(station_to_merged)
```

The structure function, binned profile, and epoch RMSE are computed
automatically by the `geepers` CLI workflow
(see [How-To Guides](how-to-guides.md#how-to-validate-an-opera-disp-product-against-gps));
the [validation notebook](notebooks/gnss_insar_validation.ipynb)
demonstrates everything on real data.

## Choosing between GPS Imaging and collocation

- **GPS Imaging**: robust to outliers, preserves sharp boundaries, no
  covariance model to fit — best for vertical land motion maps and
  messy networks.
- **Collocation**: full covariance propagation, handles east/north
  jointly with cross-correlation, gives statistically rigorous
  uncertainties — best when the field is smooth and the covariance
  model fits.

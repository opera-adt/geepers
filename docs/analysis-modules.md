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

### External regressors

Hydrological loading, well levels or temperature enter the trajectory
model as extra columns, estimated with the trend under the same noise
model. A regressor that itself trends takes that part of the rate.
`lagged_regressor` samples a driver at fixed lags for delayed responses
(an aquifer after rainfall):

```python
from geepers.trend import estimate_trend, lagged_regressor

res = estimate_trend(dates, up_mm, regressors={"grace": grace_mm})
res.parameters["reg_grace"]          # mm per mm of loading, 1-sigma

lags = lagged_regressor(dates, rain_dates, rain_mm, [0, 30, 60, 90], name="rain")
res = estimate_trend(dates, up_mm, regressors=lags)
```

## Is a single rate enough? (`geepers.linearity`)

A velocity is only worth extrapolating if the series is linear.
`linearity_test` compares a linear, a quadratic and a continuous
piecewise-linear model (free breakpoint) under one colored-noise
covariance and picks the lowest BIC; `validity_horizon` turns the
result into the time over which the rate stays within a tolerance:

```python
from geepers.linearity import linearity_test, validity_horizon

result = linearity_test(dates, up_mm, sampling_days=12)
result.model                      # "linear" | "quadratic" | "piecewise"
result.trend.velocity, result.trend.velocity_uncertainty
result.p_values                   # each alternative against linear
result.breakpoint, result.rate_before, result.rate_after
result.acceleration, result.departure    # mm/yr^2, mm off the line

horizon = validity_horizon(result, tolerance=10.0)   # 10 mm
horizon.years, horizon.driver
```

Curvature and strongly correlated noise look alike over one record, so
the verdict depends on the noise model. The default is flicker + white.
A free spectral index (`noise_model="PLWN"`) explains real offsets and
rate changes as very steep noise and calls everything linear; the fixed
index instead flags about one in five linear series whose noise really
is a random walk. Declare known offsets with `step_dates`, or they
count as non-linearity. The web viewer's linearity test estimates the
offsets its step detector finds.

## Offsets and rate changes at unknown epochs (`geepers.changepoints`)

`estimate_trend` needs the offset epochs; `detect_steps` finds offsets
only and `linearity_test` one hinge. `geepers.changepoints` estimates up
to `max_changepoints` change points at once, each an offset, a rate
change, or both, together with the trend, periodic terms, known steps
and colored noise. The rate after the last change point is the
present-day rate. The model follows DiscoTimeS (Oelsmann et al., 2022,
J. Geod., https://doi.org/10.1007/s00190-022-01645-6), without PyMC:

```python
from geepers.changepoints import (
    detect_changepoints, fit_sparse_changepoints, sample_changepoints,
)

found = detect_changepoints(dates, up_mm, sampling_days=30)
sparse = fit_sparse_changepoints(dates, up_mm, sampling_days=30)  # same result type
found.changepoints            # date, kind, offset, rate_change (+ sigmas)
found.segments                # rate per segment
found.rate, found.rate_sigma  # present-day rate

post = sample_changepoints(dates, up_mm, sampling_days=30, detection=found)
post.n_probability            # P(number of change points)
post.probability              # P(change point at each candidate epoch)
post.changepoints             # summary, with 95 % epoch intervals
post.rate, post.rate_sigma    # averaged over all configurations
post.rhat                     # chain convergence
```

- `detect_changepoints` is stepwise BIC selection under the
  `estimate_trend` noise covariance, re-estimated with the change points
  in. On long daily series `log(n)` overcounts correlated samples and
  the selection is conservative; `penalty=2` gives the AIC.
- `fit_sparse_changepoints` is the L1 (lasso) alternative: an offset and
  a rate change at every candidate epoch, selected along a penalty path
  by the same BIC after an unpenalized refit, then refined with the
  moves of the stepwise search. Without that refinement the lasso
  selection had 2.5-3.8 times the forecast error on smooth signals and
  missed change points on busy stations; with it the two mostly agree.
- `fit_sparse_changepoints_many` runs the sparse fit on many series that
  share their epochs (InSAR pixels, a station network on one grid): one
  shared noise shape (median of `estimate_trend` on a sample of series,
  or `noise=`) with a per-series amplitude, whitened once, and all lasso
  paths solved together. About 50 ms per series against 0.7 s for
  `fit_sparse_changepoints`; the noise is not re-estimated with the
  change points in and the stepwise refinement is skipped, so treat it
  as a screen for piecewise-linear signals. Series with any NaN are
  skipped.
- `sample_changepoints` is reversible-jump MCMC over the number, epochs
  and kinds of change points, with Gaussian priors on their sizes
  (`offset_scale`, `rate_scale`) integrated out analytically and the
  noise held at the `detect_changepoints` estimate. A few seconds for a
  ten-year series. The prior scales act like any Bayes factor: wider
  priors demand larger changes.
- Change points sit on `n_candidates` (default 300) epochs at least
  `min_segment_years` apart and from the ends. Pass known offsets as
  `step_dates`. The flicker + white default matters for the same reason
  as in `linearity_test`.

## Smooth, time-varying rates (`geepers.smooth`)

Decays, droughts, accelerating subsidence and multi-year cycles are not
piecewise linear. `fit_smooth_trend` estimates the rate as a random walk
(a penalized linear spline) with its smoothness and the noise estimated
together by REML, and needs no shape in advance. A smooth trend is used
only with evidence for one; otherwise the result is the
`detect_changepoints` model, a straight line if that finds nothing:

```python
from geepers.forecast import backtest
from geepers.smooth import fit_smooth_trend, forecast_smooth

fit = fit_smooth_trend(dates, up_mm, sampling_days=30, step_dates=known_steps)
fit.shape                     # "linear" | "piecewise" | "smooth"
fit.evidence                  # gain over the line (allowed a limited random
                              # walk, `max_random_walk`); smooth from 6
fit.rate, fit.rate_sigma      # present-day rate
fit.rates, fit.rates_sigma    # rate at every epoch

fc = forecast_smooth(dates, up_mm, 3, sampling_days=30, fit=fit)
fc.table[["mean", "lower", "upper"]]

backtest(dates, up_mm, 3, forecaster=forecast_smooth, sampling_days=30)
```

A forecast continues the last rate with the periodic terms and the
persisting noise. It never predicts a turning point: future slope changes
only widen the interval, which grows faster than linearly with lead time.

Relative to `detect_changepoints`, on 30 synthetic series per shape, the
3-year forecast error is 0.48-0.79 on sigmoids, decays, acceleration and
multi-year cycles, and 1.00-1.09 on lines and hinges, with coverage of
the 95 % interval rising from 0.60-0.75 to 0.87 where a signal keeps
changing. On 13 real stations it is better by over 10 % on 11 % of 55
cutoffs and worse on 9 %: identical on most stable, glacial rebound and
tectonic records, mostly better on subsidence and volcanoes (0.55-1.00),
worst at COSO (1.33).

Limits:

- A single record cannot separate a smooth signal from a random walk, so
  the evidence is measured against a line allowed random-walk noise up to
  `max_random_walk`. **The default of 3 assumes millimeters: pass 0.003
  for meters**, or nearly every series stays non-smooth. On simulated
  linear series the smooth trend is chosen on 1-2 % with a true random walk
  up to 3 mm/sqrt(yr) and on 14 % at 5, against 8-28 % and 43 % without the
  limit. Differences of station pairs within 25 km give a median of 2.7
  mm/sqrt(yr), including local signals, so 3 is a generous limit on
  monument noise, but not a measurement of it. About 1 in 5 moderate signals
  (a 30 mm decay over 12 years) is missed and gets the piecewise model.
- Knots default to one per year, so the rate is a yearly staircase and
  cannot follow an annual cycle of varying amplitude; denser knots let the
  present-day rate depend on the month the record ends.
- The rate is less precise than a line's where a line or hinge is the
  truth (0.53 against 0.20 mm/yr on a hinge), and on strongly unsteady
  records the methods disagree on the present-day rate: at LEMA -112 ± 15
  against -62 ± 5 mm/yr. Read such a rate with its uncertainty.
- Each fit factorizes an n x n matrix about 150 times, so average daily
  series to weekly or monthly first.

## Forecasts and backtests (`geepers.forecast`)

`forecast_trend` extrapolates the `detect_changepoints` model (present-day
rate and periodic terms) with an interval built from three parts: the
parameter uncertainty, the colored noise predicted from the past
residuals (flicker persists, so its conditional mean is not zero), and
future rate changes simulated at the rate and mean size this series has
shown, as in Prophet (Taylor & Letham, 2018). `backtest` redoes the whole
fit at past cutoffs and scores the forecasts against what followed:

```python
from geepers.forecast import backtest, forecast_trend

fc = forecast_trend(dates, up_mm, horizon_years=5, sampling_days=30)
fc.table[["mean", "lower", "upper", "sigma_model", "sigma_noise"]]
fc.change_rates, fc.change_scales     # simulated future rate changes

bt = backtest(dates, up_mm, horizon_years=3, sampling_days=30)
bt.metrics                            # rmse, mae, coverage, width by lead
```

- Only rate changes are simulated by default. A past offset is usually an
  event whose recurrence the record cannot predict, and simulating offsets
  made the 95 % intervals cover 100 % at three times the needed width;
  `simulate_changes=("offset", "rate")` turns them on.
- On synthetic flicker + white series without change points the 95 %
  intervals cover 92-96 % of the next three years. A record that has never
  shown a rate change simulates none, so a first change is not covered
  until it has been in the record long enough to be detected (coverage
  0.24 half a year after a -4 mm/yr change, 0.97 after 2.5 years).

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

## Trend surfaces: remove-restore (`geepers.surface`)

Collocation and the elastic spline model a zero-mean signal, so a
regional offset or tilt left in the data is pulled toward zero away
from the stations. Fit a low-degree polynomial surface, interpolate the
residuals, and add the surface back:

```python
from geepers.surface import fit_polynomial_surface

surface = fit_polynomial_surface(lon, lat, vu, sigma_u, degree=1)
emp = empirical_covariance(lon, lat, surface.residuals, surface.residuals,
                           sigma_u, sigma_u)
# ... interpolate surface.residuals onto lon_g/lat_g -> signal_g
vu_g = signal_g + surface.predict(lon_g, lat_g)
```

Keep `degree` at 0-2. For horizontal velocities remove the rigid plate
rotation instead (see `geepers.euler` below). Ordinary kriging already
handles an unknown constant mean, but not a tilt.

## Elastic vector spline (`geepers.spline`)

Interpolates east and north velocities *jointly* as the response of a
thin elastic sheet to point forces at the stations (Sandwell & Wessel,
2016 — the method behind GMT's `gpsgridder`), so the components are
coupled through Poisson's ratio. Suited to deforming zones; remove the
rigid plate motion first (see `geepers.euler`).

```python
from geepers.spline import fit_vector_spline

spline = fit_vector_spline(lon, lat, ve, vn, sigma_e, sigma_n,
                           poisson=0.5, damping=0.1)
ve_g, vn_g = spline.predict(lon_g, lat_g)
spline.residuals            # (n, 2) data - model at the stations
```

Without `damping` the spline passes through every station; with it the
field is smoothed and the sigmas act as weights. There is no formal
prediction uncertainty: judge the fit, and choose `damping`, on
held-out stations.

## Cross-validating an interpolation (`geepers.cross_validation`)

Scores any interpolator on stations it has not seen — to choose its
parameters or to compare methods. Stations are held out in whole
spatial blocks: with random hold-out a close neighbor of every test
station stays in training and the score is too optimistic.

```python
from geepers.collocation import ordinary_kriging
from geepers.cross_validation import cross_validate

def predict(train, test):          # fit on `train`, predict at `test`
    out = ordinary_kriging(lon[train], lat[train], v[train], sigma[train],
                           lon[test], lat[test], parameters)
    return out.signal[:, 0]

cv = cross_validate(predict, lon, lat, v, block_km=100, n_splits=5)
cv.rmse, cv.mad             # per component; mad is the robust spread
cv.residuals                # out-of-fold misfit at every station
```

Use a `block_km` larger than the correlation length of the field, and
loop over candidate parameters keeping the lowest `cv.rmse`.
`block_kfold` returns the train/test index sets directly.

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

A detection must also exceed `min_step_sigma` (default 3) times the
series' white-noise level: the AIC test assumes white noise and, on its
own, reports the wander of flicker or random-walk noise as steps.
Smaller steps cannot be told apart from such noise; pass
`min_step_sigma=0` for the AIC test alone.

`clean_step_dates` turns a mixed list of detected and catalogued steps
into one a fit can use (inside the record, no two within one sample).
Pass `epochs=` when the series has gaps: two steps inside one gap are the
same column of the design matrix, and the fit would raise.

Feed the detected dates into `estimate_trend(step_dates=...)` or
`midas(step_times=...)` so they are estimated, not smeared.

## Strain rates (`geepers.strain`)

Differentiate a gridded horizontal velocity field (e.g. collocation or
GPS Imaging output) into the 2D strain-rate tensor with spherical
metric factors and curvature terms (a rigid plate rotation is
strain-free, so velocities need not be in a plate-fixed frame):

```python
from geepers.strain import strain_rate_field

ds = strain_rate_field(lon_1d, lat_1d, ve_grid, vn_grid)
ds.max_shear, ds.dilatation, ds.second_invariant, ds.rotation
```

## Masking unsupported grid nodes (`geepers.masks`)

Interpolators return a value everywhere, however far from the data.
Blank the nodes that no station supports before mapping velocities or
differentiating them into strain rates:

```python
from geepers.masks import distance_mask, convex_hull_mask

near = distance_mask(lon, lat, lon_g, lat_g, max_distance_km=75)
inside = convex_hull_mask(lon, lat, lon_g, lat_g, buffer_km=20)
ve_g = np.where(near & inside, ve_g, np.nan)
```

`distance_mask` keeps nodes with a station (or `min_stations` of them)
within a great-circle radius; `convex_hull_mask` keeps nodes the network
surrounds, i.e. where the field is interpolated rather than extrapolated.

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
- **Elastic vector spline**: couples east and north physically, with no
  covariance model to fit — best for horizontal velocities in deforming
  zones; no formal uncertainties.
- **Collocation**: full covariance propagation, handles east/north
  jointly with cross-correlation, gives statistically rigorous
  uncertainties — best when the field is smooth and the covariance
  model fits.

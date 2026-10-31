# Changelog

All notable changes to geepers. Format: [Keep a Changelog](https://keepachangelog.com/);
versions follow PEP 440 via setuptools_scm.

## [Unreleased]

### Changed

- **Dependencies are split into a lean core and extras.** `pip install geepers`
  now installs only what GNSS retrieval needs (numpy, pandas, scipy, pyproj,
  requests, tqdm, tyro). `geepers[grid]` adds shapely (GPS Imaging
  re-interpolation with exclusion areas, Euler poles); `geepers[analysis]`
  adds geopandas, pyogrio, xarray, dask, rasterio, rioxarray, zarr, pandera
  and lxml (InSAR comparison, MIDAS, strain, validation, schemas);
  `geepers[plot]` is unchanged; `geepers[all]` is everything. **If you relied
  on `pip install geepers` bringing xarray/geopandas, install
  `geepers[analysis]`.** pixi environments mirror the tiers: `default`,
  `test`, `docs`, `dev` carry grid+analysis as before; `ops` is the core
  only; `core-test` is core + test tooling.
- Without geopandas, `BaseGpsSource.stations()` / `timeseries_many()` return
  plain DataFrames (lon/lat columns kept) instead of GeoDataFrames; without
  pandera, schema validation is skipped. With the extras installed nothing
  changes. The seams live in `geepers._optional`.
- `import geepers` no longer imports `geepers.gps`, `geepers.io`,
  `geepers.schemas` and `geepers.uncertainty` eagerly; their public names
  resolve on first access (PEP 562), so `from geepers import XarrayReader`
  still works.
- `EPS` moved to `geepers.constants` (still re-exported by `geepers.schemas`).
- mypy targets Python 3.12 (numpy ≥ 2.3 stubs); runtime support stays ≥ 3.11.
  pre-commit runs mypy 2.3.1 without site packages; 21 type findings fixed
  in `trend`, `validation`, `io` and `scripts/create-geojson.py` (the latter
  defaulted to UNR grid version 0.2, which does not exist; now 0.3).

### Added

- `geepers.smooth`: `fit_smooth_trend` estimates a time-varying rate
  without assuming its shape (penalized linear spline, smoothness and noise
  by REML), used only with evidence over the piecewise-linear model;
  `forecast_smooth` extrapolates it.
- `geepers.forecast.backtest(forecaster=...)` scores any forecaster.
- `geepers.changepoints.fit_sparse_changepoints_many`: the sparse fit for
  many series on one time grid (shared noise shape, per-series amplitude,
  all lasso paths solved together), returning a summary and a long table
  of change points.
- `geepers.changepoints.fit_sparse_changepoints`: change points selected
  with an L1 (lasso) penalty along a path (FISTA with adaptive restart),
  scored by BIC; returns a `ChangepointResult` like `detect_changepoints`.
- `geepers.forecast`: `forecast_trend` extrapolates the change-point model
  with intervals from the parameters, the predicted colored noise and
  simulated future rate changes (Prophet's trend uncertainty); `backtest`
  scores it at past cutoffs (RMSE, MAE, interval coverage by lead time).
- `TrendResult.covariance`: the full parameter covariance.
- `geepers.changepoints`: an unknown number of offsets and rate changes
  with the trend, periodic terms and colored noise (the DiscoTimeS model,
  Oelsmann et al. 2022, without PyMC). `detect_changepoints` selects them
  stepwise by BIC under the `estimate_trend` noise covariance;
  `sample_changepoints` samples their posterior by reversible-jump MCMC
  with the amplitudes integrated out. Both report the present-day rate.
- `estimate_trend(rate_change_dates=...)`: trend changes without an offset
  (hinge terms `rate_change_<i>`).
- `estimate_trend(regressors=...)`: external explanatory series (loading,
  well levels, temperature) estimated as `reg_<name>` with the trend, and
  `geepers.trend.lagged_regressor` for delayed responses.
- `geepers.euler`: ITRF2020-PMM and ITRF2014-PMM plate-motion tables
  (`geepers/data/`), `load_plate_motion_model`, `plate_pole`,
  `plate_velocity_enu` and `PLATE_CODES` (UNR two-letter codes → PMM names).
- `geepers.gps_imaging.reinterpolate_nodes`: replace grid-node values inside
  exclusion polygons by median-spatial-filter estimates from the nodes
  outside (GNSS-side remove-restore for the DISP-CAL calibration grid).
- `tests/test_core_imports.py` and a `core-only` CI job guard the lean core;
  the pytest CI job runs per dependency tier.
- Shared engineering kit: pre-commit (check-toml, nbstripout keeping outputs,
  SPDX header hook), GitHub issue/PR templates, `CONTRIBUTING.md`, working
  sections in `CLAUDE.md`, `docs/dependency_audit.md`.

### Fixed

- `estimate_trend` and `linearity_test` started a step dated on an
  observation one epoch late when rounding to the sampling grid moved that
  epoch earlier than its timestamp (e.g. monthly data). Steps now start at
  the first epoch on or after their date.
- A freshly solved environment failed 13 tests on the `affine` 3
  PendingDeprecationWarning raised through rasterio ≤ 1.5.1; it is now
  filtered (as in Venti).

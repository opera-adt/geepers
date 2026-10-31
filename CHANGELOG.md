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

- A freshly solved environment failed 13 tests on the `affine` 3
  PendingDeprecationWarning raised through rasterio ≤ 1.5.1; it is now
  filtered (as in Venti).

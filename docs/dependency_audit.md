# Dependency audit (2026-10-05)

Why: the operational DISP-CAL image must stay small (Venti `docs/specs.md`
R-O6, §4.4), and cal-disp only needs UNR grid/station retrieval from geepers,
plus GPS Imaging re-interpolation and Euler-pole plate motion. Everything else
(MIDAS, strain, cross-validation, plotting, zarr/dask workflows) belongs to the
validation package. This audit (plan task T12) is the basis for the
core / `[grid]` / `[analysis]` / `[all]` split in T13–T15.

Generated with a small `ast` walk over `src/geepers` (module-level imports vs
imports inside functions or `TYPE_CHECKING`). Heavy packages are **bold**.

## Import graph

| module | module-level third-party imports | lazy / TYPE_CHECKING only |
|---|---|---|
| `geepers` | — | — |
| `_types` | numpy, pandas | — |
| `_version` | — | — |
| `analysis` | numpy, pandas, pyproj | — |
| `cli` | tyro | — |
| `cme` | numpy, pandas | sklearn |
| `collocation` | numpy, pyproj, scipy | shapely |
| `constants` | — | — |
| `cross_validation` | numpy, pyproj | — |
| `euler` | numpy, pyproj | — |
| `gps` | **geopandas**, pandas | **rasterio** |
| `gps_imaging` | numpy, pandas, scipy | — |
| `gps_sources` | — | — |
| `gps_sources.base` | **geopandas**, pandas, requests, tqdm | — |
| `gps_sources.sideshow` | **geopandas**, numpy, pandas | — |
| `gps_sources.unr` | **geopandas**, numpy, pandas, requests | — |
| `gps_sources.unr_grid` | **geopandas**, pandas, requests, tqdm | — |
| `io` | numpy, pandas, pyproj, **rasterio**, **xarray** | — |
| `linearity` | numpy, pandas, scipy | — |
| `masks` | numpy, pyproj, scipy | shapely |
| `midas` | numpy | — |
| `plotting` | **matplotlib**, numpy, pandas | — |
| `processing` | numpy, pandas, tqdm, **xarray** | — |
| `quality` | numpy, pandas | — |
| `rates` | **geopandas**, numpy, pandas | — |
| `schemas` | pandas, **pandera** | — |
| `spline` | numpy, pyproj, scipy | — |
| `steps` | numpy, pandas | — |
| `strain` | numpy, **xarray** | — |
| `surface` | numpy, pyproj | — |
| `synthetic` | **geopandas**, numpy, pandas | — |
| `trend` | numpy, pandas, scipy | — |
| `uncertainty` | numpy, pandas, **pandera** | — |
| `utils` | **geopandas**, numpy, pandas | — |
| `validation` | numpy, pandas | **matplotlib** |
| `variability` | numpy, pandas, scipy | — |
| `workflows` | numpy, pandas, **rasterio**, requests, tqdm, tyro | **matplotlib** |

Declared today (`pyproject.toml` `dependencies`): pandas, lxml, geopandas,
pyogrio, xarray, dask, pyproj, rasterio, rioxarray, scipy, shapely, requests,
tqdm, zarr, tyro, pandera. Not imported anywhere at module level: **dask**,
**zarr**, **rioxarray**, **lxml**, **pyogrio** (they are used by pandas/xarray
backends through `engine=` arguments and by `geopandas.read_file`).

## What the operational consumer needs

cal-disp (via Venti) imports:

- `gps_sources.unr_grid.UnrGridSource`, `gps_sources.unr.UnrSource`
  → `gps_sources.base`, `schemas`, `utils`;
- later (T14): `gps_imaging` (re-interpolation of excluded nodes) and `euler`
  (plate motion).

Nothing else. In particular not `io`, `processing`, `workflows`, `plotting`,
`strain`, `cross_validation`, `midas`, `rates`, `synthetic`.

## Blockers for a lean core

1. **`schemas.py` imports pandera at module level** and is imported by
   `gps_sources.*`. pandera is heavy (pulls in a validation stack). → make
   validation optional: a thin `validate(df, schema_name)` that is a no-op
   when pandera is absent (T13.2).
2. **`gps_sources.base.stations()` returns a `GeoDataFrame`**, and `utils`
   imports geopandas at module level. geopandas drags in pyogrio/GDAL. →
   core returns a `pandas.DataFrame` with `lon`/`lat` columns; a `GeoDataFrame`
   is produced only when geopandas is importable (T13.3). The bbox filter is a
   pandas comparison, not a spatial join.
3. `utils` mixes core helpers (date conversion) with geopandas helpers → split
   into `utils` (core) and `geo_utils` (`[grid]`/`[analysis]`).

## Proposed partition

| extra | modules | runtime deps | consumer |
|---|---|---|---|
| **core** (no extra) | `gps_sources/*`, `schemas` (pandera optional), `utils` (split), `_types`, `constants`, `steps`, `trend`, `quality` | numpy, pandas, scipy, pyproj, requests, tqdm | cal-disp image, Venti core |
| **`[grid]`** | `gps_imaging`, `euler`, `surface`, `spline`, `masks`, `collocation`, `geo_utils` | shapely (+ core) | cal-disp image (R-G5, R-G6), Venti `[calibration]` |
| **`[analysis]`** | `io`, `processing`, `workflows`, `cli`, `rates`, `midas`, `cross_validation`, `linearity`, `variability`, `strain`, `cme`, `uncertainty`, `synthetic`, `validation`, `analysis`, `gps` | xarray, dask, zarr, rasterio, rioxarray, geopandas, pyogrio, pandera, lxml, tyro, scikit-learn | validation package, research |
| **`[plot]`** (exists) | `plotting` | matplotlib, contextily | notebooks |
| **`[all]`** | everything | | |

Four extras plus `[all]`, which is the limit set in the PRD (§4.4). `midas`
only imports numpy and could sit in core, but its consumer is validation, so
it stays in `[analysis]` to keep the core's *purpose* narrow, not just its
imports.

## Checks to add (T13.4, T15.2)

- CI job `core-only`: `pip install .` (no extras) in a clean env, then
  `python -c "from geepers.gps_sources import UnrGridSource, UnrSource"` and a
  cassette-based download test.
- CI matrix over `core`, `grid`, `analysis`, `all` with pytest markers.
- A unit test that imports every core module and asserts none of
  {dask, zarr, pandera, rasterio, geopandas, matplotlib} appears in
  `sys.modules` afterwards (regression guard for the partition).

## Decision needed (T12.3)

Owner sign-off on the partition above before T13 starts; the only judgement
calls are `midas`→`[analysis]` and splitting `utils`.

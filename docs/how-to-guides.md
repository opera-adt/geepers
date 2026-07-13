# How-To Guides

Task-oriented recipes. Each assumes `geepers` is installed
([Getting started](getting-started.md)) and, where GPS data is
downloaded, network access to `geodesy.unr.edu` or
`sideshow.jpl.nasa.gov`. Downloads are cached under
`~/.cache/geepers/` (override with `cache_dir=` on any source).

## How to validate an OPERA DISP product against GPS

One command runs the full comparison — time series, velocities, and the
structure-function requirement verification:

```bash
geepers \
    --los los_enu.tif \
    --timeseries-stack disp_product.zarr \
    --stack-data-var displacement \
    --wavelength 0.2384 \
    --insar-buffer-meters 100 \
    --requirement-mm 3 0.5 \
    -o ./GPS_validation
```

Key choices:

- `--insar-buffer-meters R` samples a **circular** footprint of radius
  R meters around *every* GPS station (reference and secondary alike)
  and takes the NaN-ignoring median — e.g. 100 m to suppress
  single-pixel noise without mixing in different ground motion.
  (`--insar-buffer N` is the older pixel-count rectangular variant.)
- `--requirement-mm A B` draws and checks the requirement curve
  ``A + B·√(distance_km)`` mm on the structure function (e.g. `3 0.5`).
- `--gps-time-window D` applies a D-day rolling median to the GPS LOS
  series (default 10; 0 disables).
- If the product stores displacement in meters, `--wavelength` is
  ignored; it only matters for stacks in radians (pass `0.2384` for
  NISAR L-band).
- `--ref STATION` fixes the reference station; otherwise the station
  with the best temporal coherence / data coverage is auto-selected.

Outputs in `GPS_validation/`:

| file | contents |
|---|---|
| `combined_data.csv` | tidy per-station GPS + InSAR LOS series |
| `relative_comparison.csv` | both series relative to the reference station |
| `station_summary.csv` | per-station GPS (MIDAS) and InSAR rates + quality metrics |
| `pairwise_rmse.csv` | structure function: relative RMSE + bias for every station pair vs separation |
| `rmse_profile.csv` | binned median/mean/p90 RMSE per distance bin, `fraction_passing` if a requirement was given |
| `epoch_rmse.csv` | network misfit per acquisition date — spikes flag bad epochs (ionosphere, unwrapping, snow) |
| `structure_function.png` | pairs + binned curve + requirement line |

The pairwise structure function differences the two stations' series
*before* comparing, so any common reference or datum wobble cancels —
it measures the product's relative accuracy the way the OPERA
requirements are written. For programmatic access to the same metrics
see [Analysis modules](analysis-modules.md#product-validation-geepersvalidation).

## How to screen a GPS network before analysis

```python
import numpy as np
from geepers.gps_sources import UnrSource
from geepers.quality import gap_percentage
from geepers.variability import temporal_velocity_variability

src = UnrSource()
stations = src.stations(bbox=(112, -39, 158, -10))  # W, S, E, N

good = []
for sid in stations["id"]:
    df = src.timeseries(sid, start_date="2010-01-01")
    # 1. Enough data, few gaps
    span_yr = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    if span_yr < 3 or gap_percentage(df["date"]) > 35:
        continue
    # 2. Velocity stable in time
    tv = temporal_velocity_variability(df["date"], df[["east", "north", "up"]])
    if tv.variability["up"] * 1000 > 2.0:  # mm/yr
        continue
    good.append(sid)
```

Add per-station spatial checks with
`geepers.variability.spatial_variability` (neighbor RMS/MAD) and
`ssf_per_station` once you have velocities for the whole network.

Two more cleaning steps that pay off before velocity fitting:

```python
from geepers.steps import detect_steps_enu
from geepers.cme import estimate_cme, remove_cme

# 1. Find uncatalogued jumps; feed them to the trend/MIDAS estimators
found = detect_steps_enu(df)                # complements UnrSource.steps()

# 2. Estimate and remove the network common mode from detrended residuals
cme = estimate_cme(residuals_by_station)    # epochs x stations DataFrame
cleaned = cme.cleaned                       # or remove_cme(residuals_by_station)
```

## How to estimate a velocity with a realistic uncertainty

```python
from geepers.gps_sources import UnrSource
from geepers.trend import estimate_trend

src = UnrSource()
df = src.timeseries("P123", start_date="2015-01-01")
steps = src.steps(station_ids=["P123"])

res = estimate_trend(
    df["date"],
    df["up"] * 1000,                      # meters -> mm
    step_dates=steps["date"].tolist(),    # estimate offsets, don't span them
    noise_model="PLWN",                   # power-law + white
)
print(f"v = {res.velocity:.2f} ± {res.velocity_uncertainty:.2f} mm/yr "
      f"(kappa = {res.kappa:.2f})")
```

Use `noise_model="WN"` for a fast first pass (OLS-equivalent sigma,
typically 5-10× too small), or `geepers.midas.midas` when robustness to
unknown steps matters more than the noise model.

## How to interpolate a velocity field onto a grid

Robust, edge-preserving (GPS Imaging — cite Hammond et al., 2016):

```python
import numpy as np
from geepers.gps_imaging import make_ssf, msf_interpolate

ssf = make_ssf(lon, lat, v_up, sigma_up)
gx, gy = np.meshgrid(np.arange(-120, -114, 0.1), np.arange(36, 41, 0.1))
grid = msf_interpolate(lon, lat, v_up, sigma_up, gx.ravel(), gy.ravel(), ssf)
v_grid = grid["value"].to_numpy().reshape(gx.shape)
```

Geostatistical, with covariance-propagated uncertainties (collocation):

```python
from geepers.collocation import empirical_covariance, interpolate_velocities

emp = empirical_covariance(lon, lat, ve, vn, se, sn)
at_stations, at_grid = interpolate_velocities(
    lon, lat, ve, vn, se, sn, gx.ravel(), gy.ravel(),
    covariance_parameters=emp.parameters,
)
```

Rules of thumb: GPS Imaging for vertical land motion, outlier-prone
networks, and sharp boundaries; collocation for smooth horizontal
fields where you need rigorous sigmas. See
[Analysis modules](analysis-modules.md#choosing-between-gps-imaging-and-collocation).

## How to interpolate across plate boundaries (plate-separation constraint)

A covariance model only knows *distance*: stations 50 km apart on
opposite sides of a plate boundary look "close", and interpolation
smears the velocity discontinuity across the boundary.
`separate_plates` fixes this by building a working coordinate space in
which each plate's points are pushed far apart (≥ 1500 km by default),
so any distance-decaying covariance treats cross-plate pairs as
uncorrelated — while within-plate geometry is preserved.

The recipe: **assign & separate → interpolate in moved coordinates →
attach results back to the true coordinates.**

Self-contained synthetic example (two plates with discontinuous motion):

```python
import geopandas as gpd
import numpy as np
from shapely.geometry import box

from geepers.collocation import (
    ordinary_kriging, separate_plates,
)

rng = np.random.default_rng(1)

# Two plates meeting at lon = 5°, moving differently
plates = gpd.GeoDataFrame(
    {"code": ["A", "B"]},
    geometry=[box(0, 0, 5, 10), box(5, 0, 10, 10)],
    crs="EPSG:4326",
)
n = 80
lon = np.r_[rng.uniform(0.5, 4.5, n // 2), rng.uniform(5.5, 9.5, n // 2)]
lat = rng.uniform(1, 9, n)
v = np.where(lon < 5, -2.0, 3.0) + rng.normal(0, 0.3, n)   # 5 mm/yr jump
sv = np.full(n, 0.3)

# Evaluation grid spanning the boundary
gx, gy = np.meshgrid(np.linspace(0.5, 9.5, 60), np.linspace(1, 9, 40))

# --- 1. Separate stations AND grid points together (one call, so both
#        get consistent working coordinates)
all_lon = np.r_[lon, gx.ravel()]
all_lat = np.r_[lat, gy.ravel()]
lon_m, lat_m, plate_idx = separate_plates(
    all_lon, all_lat, plates, min_separation_km=1500
)
slon_m, slat_m = lon_m[:n], lat_m[:n]          # stations, moved
glon_m, glat_m = lon_m[n:], lat_m[n:]          # grid, moved

# --- 2. Interpolate in the moved space
params = np.array([1.0, 200.0])                 # (C0, d0) — or fit with
                                                # empirical_covariance on
                                                # the *moved* coordinates
unconstrained = ordinary_kriging(lon, lat, v, sv, gx.ravel(), gy.ravel(), params)
constrained   = ordinary_kriging(slon_m, slat_m, v, sv, glon_m, glat_m, params)

# --- 3. Results map back trivially: constrained.signal[i] belongs to
#        the true coordinate (gx.ravel()[i], gy.ravel()[i])
v_unc = unconstrained.signal.reshape(gx.shape)  # boundary smeared
v_con = constrained.signal.reshape(gx.shape)    # sharp step at lon=5°
```

Plotting `v_unc` vs `v_con` shows the difference: without the
constraint the −2 → +3 mm/yr step is blurred over ~2 x d0; with it,
each grid node is informed only by stations on its own plate.

The same moved coordinates work for `interpolate_velocities` (LSC) and
`geepers.gps_imaging.msf_interpolate` — pass `slon_m/slat_m` and
`glon_m/glat_m` wherever station/evaluation coordinates go, and always
fit `empirical_covariance` / `make_ssf` on the **moved** coordinates so
cross-plate pairs don't contaminate the covariogram.

Real-data notes:

- Plate polygons: use e.g. the Bird (2003) plate-boundary model, or
  your project's plate shapefile —
  `plates = gpd.read_file("plates.shp")`. Points falling outside every
  polygon are assigned to the nearest plate (the original workflow
  silently dropped them).
- **Horizontal velocities**: remove each plate's rigid rotation
  *before* interpolating and restore it afterward — otherwise the plate
  motion itself dominates the field. Use `geepers.euler`:

  ```python
  from geepers.euler import estimate_euler_pole, predict_plate_motion

  for k in np.unique(plate_idx[:n]):        # per plate
      sel = plate_idx[:n] == k
      pole = estimate_euler_pole(lon[sel], lat[sel], ve[sel], vn[sel],
                                 se[sel], sn[sel])
      vp_e, vp_n = predict_plate_motion(pole, lon[sel], lat[sel])
      ve[sel] -= vp_e; vn[sel] -= vp_n      # remove (restore after)
  ```

  (or build the `EulerPole` from published ITRF2014-PMM values).
  Vertical velocities need no such correction, which makes them the
  simplest case.
- Choose `min_separation_km` several times larger than the fitted
  correlation length `d0` (the 1500 km default suits d0 ≲ 300 km).

## How to browse UNR grid time series interactively

```bash
cd scripts/
python create-geoparquet.py --bbox -110 28 -101 36 --start-date 2016-01-01
python -m http.server 8123
# open http://localhost:8123/browse_unr_grid.html
```

The viewer loads the Parquet directly in the browser (no server-side
processing) with a date slider, per-point time-series charts, and WebM
animation export. See `scripts/README.md` for details.

## How to point the download cache somewhere else

Every data source accepts `cache_dir`:

```python
src = UnrSource(cache_dir="/big/disk/gps_cache")
```

or set `XDG_CACHE_HOME` to relocate all caches.

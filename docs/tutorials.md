# Tutorials

## GNSS vs InSAR validation (notebook)

The [GNSS-InSAR validation notebook](notebooks/gnss_insar_validation.ipynb)
runs the complete comparison workflow on the real Sentinel-1 Kīlauea test
dataset that ships with the repository: metric-buffer sampling,
per-station time series, velocity scatter, the structure function with a
requirement curve, per-epoch network misfit, and the effect of the
sampling buffer radius. It is the executable companion to
[How to validate an OPERA DISP product against GPS](how-to-guides.md#how-to-validate-an-opera-disp-product-against-gps).

## Tour of the analysis modules (notebook)

The [geepers tour notebook](notebooks/geepers_tour.ipynb) is a runnable,
self-contained walkthrough — synthetic data with known truth, so every
result can be checked against what went in:

1. robust velocities with **MIDAS**;
2. velocities with **realistic uncertainties** under power-law + white
   noise, and why OLS sigmas are 5-10× too optimistic;
3. **temporal velocity variability** (sliding-window MIDAS);
4. spotting bad stations with **spatial variability**;
5. interpolating a velocity field two ways — robust **GPS Imaging**
   vs. geostatistical **collocation** — on a field with a sharp
   boundary and a corrupted station.

Run it locally with:

```bash
jupyter lab docs/notebooks/geepers_tour.ipynb
```

## Your first GPS-InSAR comparison

1. **Gather inputs**: a stack of displacement rasters (one per date,
   dates in the filenames) or a Zarr/NetCDF cube, plus a 3-band
   line-of-sight ENU raster on the same grid.

2. **Run the workflow**:

   ```bash
   geepers --los los_enu.tif --timeseries-files displacement_*.tif -o GPS
   ```

   GPS stations inside the raster bounds are discovered and downloaded
   automatically (cached under `~/.cache/geepers`).

3. **Look at the outputs** in `GPS/`:

   - `combined_data.csv` — every GPS and InSAR LOS series, tidy format;
   - `relative_comparison.csv` — both series referenced to a common
     station;
   - `station_summary.csv` — per-station GPS (MIDAS) and InSAR rates,
     their difference, and quality metrics.

4. **Plot a station** (pandas one-liner):

   ```python
   import pandas as pd
   df = pd.read_csv("GPS/combined_data.csv", parse_dates=["date"])
   (df[df.id == "P123"]
      .pivot_table(index="date", columns="measurement", values="value")
      [["los_gps", "los_insar"]]
      .plot(style=".", ms=3))
   ```

For recipes on screening stations, estimating velocities, and gridding
velocity fields, see the [How-To Guides](how-to-guides.md); for the
underlying methods, see [Background theory](background-theory.md).

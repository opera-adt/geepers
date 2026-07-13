# OPERA UNR Grid Web Browser

Interactive MapLibre GL viewer for UNR gridded GPS time series.
The gridded data are produced by the Nevada Geodetic Laboratory (UNR),
funded by the JPL-led [OPERA](https://www.jpl.nasa.gov/go/opera) project;
the viewer is developed at JPL.

> **Disclaimer**: the viewer and the underlying gridded GPS products are
> research tools provided "as is", without warranty of any kind.
> Displacements, uncertainties, and derived velocities are experimental
> and may contain errors or artifacts. Use of this tool does not imply
> endorsement by JPL/Caltech, NASA, or the University of Nevada, Reno.
Loads a single Parquet file directly in the browser (via
[hyparquet](https://github.com/hyparam/hyparquet)) and scrubs through dates
with GPU-driven color updates — no per-date files, no re-fetching.

## Live site

Hosted on GitHub Pages: **https://opera-adt.github.io/geepers/**

The default dataset is the **global UNR grid, all 28,358 points at monthly
sampling** (2014→2026, ~95 MB), served same-origin from the `gh-pages`
branch. See `deploy-pages.sh` for how the site is (re)built and pushed.

### Viewing the full daily-resolution grid

The full daily grid is too large for browser hosting — GitHub has no
surface that serves it cross-origin (Pages caps files at 100 MB; Release
assets and LFS send no CORS headers). It ships instead as a **local
artifact** for offline viewing:

```bash
cd scripts/
# OPERA_UNR_GNSS_grid_full.parquet: all 28,358 points, daily, 2014→2026
# (~850 MB; viewer-minimal columns date_idx/point_idx/E/N/U, no sigmas)
python -m http.server 8123
# Open http://localhost:8123/browse_unr_grid.html and use "Open .parquet…"
# in the Data panel to pick OPERA_UNR_GNSS_grid_full.parquet, or:
#   browse_unr_grid.html?data=OPERA_UNR_GNSS_grid_full.parquet
```

At full daily resolution the viewer needs ~1.5 GB of browser memory (it
will ask to confirm); use the **Date stride** selector (or `?stride=N`) to
subsample and lighten it. The `_full` file omits the sigma columns, so the
±σ chart band is unavailable there — use a smaller/regional export (with
sigmas) if you need uncertainties.

## Setup

```bash
cd scripts/
# 1. Download data and build the viewer-ready Parquet file (example bbox):
python create-geoparquet.py --bbox -110 28 -101 36 --start-date 2016-01-01
# Creates unr_grid.parquet

# 2. Serve and open:
python -m http.server 8123
# Visit http://localhost:8123/browse_unr_grid.html
```

Useful options:

- `--source grid|stations` — UNR gridded (interpolated) product, or real
  UNR GPS station positions (.tenv3). Default `grid`.
- `--gridded-type constant|variable` — time-constant vs time-variable UNR
  product (version 0.3 only; grid source only; default `variable`).
- `--output-file my_area.parquet` then open
  `browse_unr_grid.html?data=my_area.parquet`.
- `--clear-cache` — wipe the geepers download cache first (forces fresh
  downloads).
- `--zero-by mean|start|none` — zero each point's series by its mean, its
  first epochs, or `none` to keep values exactly as published (default
  `mean`).
- If no file is found, the page offers a local file picker (drag any
  compatible `.parquet` in — nothing is uploaded, parsing is in-browser).

The viewer is a single self-contained HTML file — MapLibre GL v5, uPlot and
hyparquet are inlined, so only the basemap/terrain tiles need the network.

## Viewer features

- Date slider + playback (2–30 fps), keyboard: `←`/`→` step, `space` play.
- Click a grid point → East/North/Up time series chart (uPlot) with an
  optional ±1σ shaded band; **Shift+click** a second point for a comparison
  chart. Charts are resizable (drag the corner), zoomable (drag box, mouse
  wheel, double-click resets), and clicking a sample jumps the map to that
  date.
- Component selector, colormaps (RdBu, BrBG, Viridis, Turbo, Magma),
  invert, symmetric/robust (p2–p98) or manual range in mm; `live` re-runs
  the auto range on every date change while scrubbing/playing.
- Velocity mode: color points by per-point linear trend (least-squares,
  mm/yr) instead of per-date displacement.
- Vector overlay: horizontal (E+N) and/or vertical (Up, red up / blue
  down) quiver arrows over the points, with an arrow-scale slider and a
  scale legend above the colorbar. Arrow scaling is automatic (p90 of the
  data, follows the date like the color `live` mode) or fixed via typed
  reference magnitudes (e.g. H 3, V 1 mm/yr). Arrows show the same field
  as the colors (per-date displacement, or velocity in velocity mode).
  The globe view gets a dark space backdrop.
- Each chart has a `csv` button (dates + E/N/U ± σ of that point, in mm,
  with the current referencing applied).
- Find ID box zooms to a grid point / station by identifier.
- Large files: a memory estimate is checked before loading, and a "Date
  stride" selector (`?stride=N`) loads only every Nth date to bound
  memory (e.g. the ~1 GB time-variable CA file fits comfortably with
  stride 5).
- Reference modes: none (values exactly as stored in the file), per-point
  temporal mean, first date, or any chosen date (displacement relative to
  that date).
- Basemaps: Carto light/dark, OSM, Esri satellite; globe (default) or
  Mercator projection; optional 3D terrain (AWS terrain tiles) with
  adjustable exaggeration — right-drag / Ctrl+drag to tilt and rotate.
- Tectonic plate boundaries overlay (Bird 2003, via
  [fraxen/tectonicplates](https://github.com/fraxen/tectonicplates));
  loads `PB2002_boundaries.json` next to the HTML if present, else from
  GitHub raw.
- Data panel: load another `.parquet` (local file or URL) without reloading
  the page, and a "Clear cache & reload" button. Fetches are keyed to the
  file's `Last-Modified`/`ETag`, so regenerating a parquet under the same
  name can never serve stale cached byte ranges.

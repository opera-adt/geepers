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
branch. The "GPS sites" switch loads a global `--source midas` file (all
~21,800 UNR stations, 1994→now, ~80 MB), with a note on the page saying it
is a velocity-based simplification. See `deploy-pages.sh` for how the site
is (re)built and pushed:

```bash
cd scripts/
python create-geoparquet.py --bbox -180 -90 180 90 --source midas \
    --start-date 1994-01-01 --output-file OPERA_UNR_GNSS_stations_midas.parquet
./deploy-pages.sh                        # grid + stations, default demo note
./deploy-pages.sh --note "Custom text"   # your own note
./deploy-pages.sh --no-note              # no note
./deploy-pages.sh --no-stations          # grid only
```

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

- `--source grid|stations|midas` — UNR gridded (interpolated) product,
  real UNR GPS station positions (.tenv3), or `midas`: monthly straight
  lines from UNR's IGS20 MIDAS station velocities over each station's
  observing span. `midas` downloads one ~5 MB table instead of every
  station's series, so it suits a global demo (the GitHub Pages "GPS
  sites"), but it has no seasonal signal, offsets, noise or sigmas.
  Default `grid`.
- `--gridded-type constant|variable` — time-constant vs time-variable UNR
  product (version 0.3 only; grid source only; default `variable`).
- `--output-file my_area.parquet` then open
  `browse_unr_grid.html?data=my_area.parquet`.
- `--clear-cache` — wipe the geepers download cache first (forces fresh
  downloads).
- `--zero-by mean|start|none` — zero each point's series by its mean, its
  first epochs, or `none` to keep values exactly as published (default
  `mean`).
- Viewer URL options: `?data=<file>` (grid), `?stations=<file>` (GPS
  sites, default `unr_stations.parquet`), `?stride=N`, and
  `?note=<text>` to show a note panel (empty `?note=` hides one set at
  deploy time).
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
- Component selector. Colors: click the colorbar (in Display, or the one
  on the map) for the colormap (19: diverging RdBu, BrBG, RdYlBu, PuOr,
  PiYG, Spectral, Coolwarm, Vik, Roma, Balance; sequential Viridis, Magma,
  Plasma, Inferno, Cividis, Batlow, Thermal, Turbo, Greys),
  invert, manual range, or Auto with a percentile stretch (p2–p98 …
  min–max), symmetric or not; `live` re-runs the auto range on every date
  change while scrubbing/playing. Both colorbars stay in step.
- Velocity mode: color points by per-point linear trend (least-squares,
  mm/yr) instead of per-date displacement.
- Vector overlay: horizontal (E+N) and/or vertical (Up, red up / blue
  down) quiver arrows over the points, with an arrow-scale slider and a
  scale legend above the colorbar. Arrow scaling is automatic (p90 of the
  data, follows the date like the color `live` mode) or fixed via typed
  reference magnitudes (e.g. H 3, V 1 mm/yr). Arrows show the same field
  as the colors (per-date displacement, or velocity in velocity mode).
  The globe view gets a dark space backdrop.
- Chart `fit` option: least-squares trajectory model per component —
  polynomial of order 0–3 (1 = velocity), annual and semi-annual
  sinusoids, and Heaviside steps at typed dates. Fitted curves are drawn
  dashed and the estimates listed under the chart. Uncertainties assume
  white noise, so they are optimistic for GPS (typically several times
  smaller than MIDAS's).
- Analysis section (sidebar), "Run on view" for the points in the current
  view (hover any parameter's label for what it does):
  - Field compared between neighbors: velocity, displacement at the date,
    or the whole time series (for two points, the RMS of their relative
    series, mean removed, over common dates).
  - Delaunay network: links colored by length (optional km labels), links
    over "Max link" dropped. Click a link to disable / re-enable it, or
    load breaklines (zipped shapefile, .shp or GeoJSON, lon/lat) to cut
    every link that crosses one; the network metrics update.
  - Map metrics (colorbar click: colormap, invert, range): spatial RMS /
    MAD of neighbor differences, neighbor similarity (median correlation
    of detrended series), SSF score, median link length, temporal velocity
    variability (MIDAS in sliding windows), gap % (against the dates in
    the file, over each point's span or the Record span), detected steps.
  - A field-level spatial structure function (SSF) chart.
  - Detected steps of the clicked point are marked on its chart and listed
    with their sizes; the chart fit's `detect` button fills its steps.
  The metrics are ports of `geepers.variability`, `geepers.gps_imaging`,
  `geepers.quality`, `geepers.steps` and `geepers.midas` and give the
  same numbers. The SSF is the GPS Imaging one (Hammond et al., 2016:
  great-circle separation, scatter forced to grow with distance) and the
  SSF score is the field SSF at each point's link lengths (Hammond et
  al., 2021), near 1 where the network resolves the field. On large
  views the SSF is built from a random subsample of the points. Gap %
  counts against the file's own
  dates rather than a fixed step, which matters for monthly data.
- Each chart has a `csv` button: a `#` header (grid or GPS site, id,
  lat, lon, reference, velocity ± σ per component from the fit model if
  one is on, else a straight line, plus the fit terms), then dates + E/N/U
  ± σ of that point in mm with the current referencing applied. Read it
  with `pandas.read_csv(path, comment="#")`.
- Grid / GPS sites / Both switch (top left): swaps between the gridded
  product and a stations file (`unr_stations.parquet` next to the HTML,
  or `?stations=<file>`; build one with `--source stations`), keeping
  the map view and the nearest date. "Both" draws grid nodes as squares
  and stations as circles over the union of their dates; on each date
  a set shows its own nearest date if that lies within half its date
  spacing. Outside its record a site is grey (no data); a grid node is
  hidden, or faded if the grid is drawn in a single color. Grid and
  sites share the colormap and range, have separate size sliders, and
  each has a swatch beside its slider: click it to draw that set in a
  single color, double-click to choose the color.
- Charts open on the clicked point's own record (stations start and
  stop at different times); zoom out or scroll to see the full axis.
- Search box (top left): `lat lon` or `lon lat` coordinates, a grid
  point / station id (matched locally as you type), or a place name
  (looked up on Enter via OpenStreetMap Nominatim).
- Large files: a memory estimate is checked before loading, and a "Date
  stride" selector (`?stride=N`) loads only every Nth date to bound
  memory (e.g. the ~1 GB time-variable CA file fits comfortably with
  stride 5).
- Sidebar: the `–` button in its title collapses the whole panel; the
  Record and Data sections start collapsed (Data opens itself when a
  load fails).
- Record start / end (sidebar): filled from the data and editable; the
  date slider and playback run between them (`Full` restores the whole
  record).
- Reference modes: none (values exactly as stored in the file), per-point
  temporal mean, first date, or any chosen date (displacement relative to
  that date).
- Basemaps: OpenFreeMap light/dark (Positron / Dark Matter vector styles,
  no API key or account), OSM, Esri satellite, Google satellite hybrid;
  globe (default) or Mercator projection; optional 3D terrain (AWS
  terrain tiles) with adjustable exaggeration — right-drag / Ctrl+drag
  to tilt and rotate. These are icon buttons in the lower-right corner
  of the map, above the zoom control.
- Tectonic plates overlay: its button cycles boundaries, boundaries
  with plate names, off (Bird 2003, via
  [fraxen/tectonicplates](https://github.com/fraxen/tectonicplates));
  loads `PB2002_boundaries.json` next to the HTML if present, else from
  GitHub raw.
- Data panel: load another `.parquet` (local file or URL) without reloading
  the page, and a "Clear cache & reload" button. Fetches are keyed to the
  file's `Last-Modified`/`ETag`, so regenerating a parquet under the same
  name can never serve stale cached byte ranges.

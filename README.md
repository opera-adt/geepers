# OPERA UNR Grid Viewer (GitHub Pages)

Static deployment of `scripts/browse_unr_grid.html`, served at
https://opera-adt.github.io/geepers/ . Rebuilt by `scripts/deploy-pages.sh`.

- `index.html` — self-contained viewer (MapLibre GL, uPlot, hyparquet inlined)
- `OPERA_UNR_GNSS_grid.parquet.zip` — RAW PARQUET (not a zip; the extension disables the Pages
  CDN gzip that would corrupt range reads — rename to .parquet after download)
- `PB2002_boundaries.json` — tectonic plate boundaries (Bird 2003)
- `docs/` — mkdocs project documentation (served at `.../geepers/docs/`)

Other datasets can be viewed with `?data=<url>` (host must allow CORS + ranges).

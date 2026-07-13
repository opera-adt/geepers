#!/usr/bin/env bash
# Rebuild and deploy the GitHub Pages site for the OPERA UNR grid viewer.
#
# Publishes a single self-contained page plus its default dataset to the
# `gh-pages` branch of opera-adt/geepers, served at
#   https://opera-adt.github.io/geepers/
#
# The branch is rebuilt as ONE fresh orphan commit each run so old (large)
# data blobs never accumulate in history.
#
# Usage:
#   ./deploy-pages.sh [DATA_PARQUET]
# DATA_PARQUET defaults to OPERA_UNR_GNSS_grid_monthly.parquet (the global
# monthly grid). It must be < 100 MB (GitHub Pages per-file limit).
set -euo pipefail

REPO="opera-adt/geepers"
REMOTE="https://github.com/${REPO}"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_SRC="${1:-${SCRIPTS_DIR}/OPERA_UNR_GNSS_grid_monthly.parquet}"
# The .zip extension is deliberate: it stops the GitHub Pages CDN from
# gzipping the file, which would corrupt hyparquet's HTTP range reads.
DATA_DEST="OPERA_UNR_GNSS_grid.parquet.zip"
BOUNDARIES="${SCRIPTS_DIR}/PB2002_boundaries.json"

[ -f "$DATA_SRC" ] || { echo "error: data file not found: $DATA_SRC" >&2; exit 1; }
size_mb=$(( $(stat -c%s "$DATA_SRC") / 1024 / 1024 ))
if [ "$size_mb" -ge 100 ]; then
    echo "error: $DATA_SRC is ${size_mb} MB; GitHub Pages caps files at 100 MB." >&2
    exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# 1. Viewer: point the default DATA_URL at the hosted (renamed) dataset, and
#    inject a "Docs" link (deploy-only, so running the HTML locally has no
#    dead docs/ link).
python3 - "$SCRIPTS_DIR/browse_unr_grid.html" "$WORK/index.html" "$DATA_DEST" <<'PY'
import sys
src, dst, data = sys.argv[1:4]
html = open(src).read()
old = "const DATA_URL = params.get('data') || 'unr_grid.parquet';"
new = f"const DATA_URL = params.get('data') || '{data}';"
assert html.count(old) == 1, "could not find DATA_URL default in viewer"
html = html.replace(old, new)
# Deploy-only Docs link (present only when docs/ is published alongside).
credit = "funded by the JPL-led OPERA project. Viewer: JPL."
if credit in html:
    html = html.replace(
        credit,
        credit + '\n            &middot; '
        '<a href="docs/" style="color:var(--accent)">Docs</a>',
    )
open(dst, "w").write(html)
PY

# 2. Static assets.
cp "$DATA_SRC" "$WORK/$DATA_DEST"
[ -f "$BOUNDARIES" ] && cp "$BOUNDARIES" "$WORK/PB2002_boundaries.json"
# .nojekyll (at root) disables Jekyll for the whole site, so the mkdocs
# assets under docs/ (e.g. _mkdocstrings.css) are served too.
: > "$WORK/.nojekyll"

# 2b. mkdocs docs at /docs/ (viewer stays at the site root). Built only when
# mkdocs is available (run from an env with the docs deps + geepers importable);
# skipped with a warning otherwise so a viewer-only deploy still works.
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
if command -v mkdocs >/dev/null 2>&1 && [ -f "$REPO_ROOT/mkdocs.yml" ]; then
    echo "Building docs → docs/ …"
    ( cd "$REPO_ROOT" && PYTHONPATH=src mkdocs build --quiet --site-dir "$WORK/docs" )
else
    echo "warning: mkdocs not found; deploying viewer only (no docs/)." >&2
fi
cat > "$WORK/README.md" <<EOF
# OPERA UNR Grid Viewer (GitHub Pages)

Static deployment of \`scripts/browse_unr_grid.html\`, served at
https://opera-adt.github.io/geepers/ . Rebuilt by \`scripts/deploy-pages.sh\`.

- \`index.html\` — self-contained viewer (MapLibre GL, uPlot, hyparquet inlined)
- \`$DATA_DEST\` — RAW PARQUET (not a zip; the extension disables the Pages
  CDN gzip that would corrupt range reads — rename to .parquet after download)
- \`PB2002_boundaries.json\` — tectonic plate boundaries (Bird 2003)
- \`docs/\` — mkdocs project documentation (served at \`.../geepers/docs/\`)

Other datasets can be viewed with \`?data=<url>\` (host must allow CORS + ranges).
EOF

# 3. Fresh single-commit orphan branch, force-pushed.
cd "$WORK"
git init -q -b gh-pages
git add -A
git commit -q -m "Deploy OPERA UNR grid viewer ($(basename "$DATA_SRC"), ${size_mb} MB) + docs"
git remote add origin "$REMOTE"
git push -f origin gh-pages

echo "Deployed. Pages will rebuild shortly: https://opera-adt.github.io/geepers/"

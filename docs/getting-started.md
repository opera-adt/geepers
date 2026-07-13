## Install


`geepers` is available on conda-forge:

```bash
mamba install -c conda-forge geepers
```


## Usage

### Compare an InSAR time series to GPS (main workflow)

The `geepers` command samples an InSAR displacement stack at every GPS
station inside its bounds, projects the GPS East/North/Up series onto the
radar line of sight, and writes per-station comparison tables:

```bash
geepers \
    --los path/to/los_enu.tif \
    --timeseries-files displacement_*.tif \
    --gps-source unr \
    --wavelength 0.2384 \
    -o ./GPS
```

- `--los` is a 3-band raster of the line-of-sight unit vector (ENU).
- `--wavelength` (meters) is used only when the stack units are radians;
  the default is Sentinel-1 C-band (~0.0555 m) — pass `0.2384` for
  NISAR L-band.
- Outputs: `combined_data.csv` (tidy GPS+InSAR series),
  `relative_comparison.csv` (relative to a reference station,
  auto-selected if not given via `--ref`), and `station_summary.csv`
  (per-station rates and quality metrics).

Run `geepers --help` for all options.

### Download GPS data programmatically

```python
from geepers.gps_sources import UnrSource, UnrGridSource

src = UnrSource()
stations = src.stations(bbox=(-120, 35, -115, 40))     # (W, S, E, N)
df = src.timeseries("P123", start_date="2015-01-01")   # one station
gdf = src.timeseries_many(bbox=(-118, 36, -117, 37))   # all stations in box
steps = src.steps(station_ids=["P123"])                # known offsets
```

`UnrGridSource` provides the UNR interpolated grid products the same way,
and `SideshowSource` the JPL series.

### Estimate velocities

```python
from geepers.trend import estimate_trend

res = estimate_trend(df["date"], df["up"] * 1000, step_dates=steps["date"])
print(f"{res.velocity:.2f} ± {res.velocity_uncertainty:.2f} mm/yr")
```

See [Analysis modules](analysis-modules.md) for the full velocity /
interpolation / quality-metric toolbox, and the
[tour notebook](notebooks/geepers_tour.ipynb) for a runnable demo.

## Setup for Developers

To contribute to the development of `geepers`, you can fork the repository and install the package in development mode.
We encourage new features to be developed on a new branch of your fork, and then submitted as a pull request to the main repository.

To install locally,

1. Download source code:
```bash
git clone https://github.com/isce-framework/geepers
```
2. Install dependencies:
```bash
mamba env create --file environment.yml
```

or if you have an existing environment:
```bash
mamba env update --name my-existing-env --file environment.yml
```

3. Install `geepers` via pip:
```bash
mamba activate geepers-env
python -m pip install -e .
```


The extra packages required for testing and building the documentation can be installed:
```bash
# Run "pip install -e" to install with extra development requirements
python -m pip install -e ".[docs,test]"
```

We use [`pre-commit`](https://pre-commit.com/) to automatically run linting and formatting:
```bash
# Get pre-commit hooks so that linting/formatting is done automatically
pre-commit install
```
This will set up the linters and formatters to run on any staged files before you commit them.

After making functional changes, you can rerun the existing tests and any new ones you have added using:
```bash
python -m pytest
```

### Creating Documentation

We use [MKDocs](https://www.mkdocs.org/) to generate the documentation.
The reference documentation is generated from the code docstrings using [mkdocstrings](mkdocstrings.github.io/).

When adding new documentation, you can build and serve the documentation locally using:

```
mkdocs serve
```
then open http://localhost:8000 in your browser.
Creating new files or updating existing files will automatically trigger a rebuild of the documentation while `mkdocs serve` is running.

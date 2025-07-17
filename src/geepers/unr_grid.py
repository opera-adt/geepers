from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

from geepers.io import XarrayReader

LOOKUP_FILE_URL = "https://geodesy.unr.edu/grid_timeseries/grid_latlon_lookup.txt"
GRID_DATA_BASE_URL = (
    "https://geodesy.unr.edu/grid_timeseries/time_variable_gridded/IGS14/"
)


def get_grid_geometry() -> gpd.GeoSeries:
    """Get the grid geometry."""
    df = _read_grid_file()
    return gpd.GeoSeries.from_xy(df.longitude, df.latitude, crs="EPSG:4326")


@cache
def _read_grid_file() -> pd.DataFrame:
    """Download the UNR grid latitude/longitude lookup table."""
    df = pd.read_csv(
        LOOKUP_FILE_URL,
        delim_whitespace=True,
        names=["grid_point", "longitude", "latitude"],
    )
    return df.set_index("grid_point")


@cache
def list_remote_data_files() -> list[str]:
    """Retrieve available .tenv8 filenames from the UNR grid data directory.

    Returns
    -------
    List[str]
        Filenames matching the pattern '######_IGS14.tenv8'.

    """
    response = requests.get(GRID_DATA_BASE_URL)
    response.raise_for_status()

    import re

    pattern = re.compile(r"(\d{6}_IGS14\.tenv8)")
    files = set(pattern.findall(response.text))
    return sorted(files)


def get_grid_within_image(reader: XarrayReader) -> gpd.GeoDataFrame:
    """Find grid points within a given geocoded image.

    Parameters
    ----------
    reader : XarrayReader
        Reader object containing the geocoded DataArray.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing information about the GPS stations within the image.

    Notes
    -----
    This function assumes the image is in a geographic coordinate system (lat/lon).

    """
    import rasterio.warp
    from shapely import box

    if reader.crs != "EPSG:4326":
        bounds = rasterio.warp.transform_bounds(
            reader.crs, "EPSG:4326", *reader.da.rio.bounds()
        )
    else:
        bounds = reader.da.rio.bounds()
    bounds_poly = box(*bounds)

    # Get all GPS stations
    gdf_all = _read_grid_file()

    gdf_within = gdf_all.clip(bounds_poly)

    # Reset index for cleaner output
    gdf_within.reset_index(drop=True, inplace=True)
    return gdf_within


def download_data_files(
    output_dir: Path,
    file_list: list[str] | None = None,
    max_workers: int = 8,
) -> None:
    """Download .tenv8 data files in parallel, showing progress.

    Parameters
    ----------
    output_dir : Path
        Directory to store downloaded data files.
    file_list : Optional[List[str]]
        Specific filenames to download. If None, files are listed remotely.
    max_workers : int
        Number of threads to use for downloading in parallel.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if file_list is None:
        file_list = list_remote_data_files()

    def _download(fname: str) -> None:
        url = f"{GRID_DATA_BASE_URL}{fname}"
        dest = output_dir / fname
        if not dest.exists():
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download, fn): fn for fn in file_list}
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading data files"
        ):
            pass


def parse_lookup_file(file_path: Path) -> pd.DataFrame:
    """Parse lookup table to DataFrame of grid point coordinates.

    Parameters
    ----------
    file_path : Path
        Path to the lookup text file.

    Returns
    -------
    pd.DataFrame
        Columns: ['grid_point', 'longitude', 'latitude'].

    """
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        names=["grid_point", "longitude", "latitude"],
        dtype={"grid_point": str, "longitude": float, "latitude": float},
    )
    return df


def parse_data_file(file_path: Path) -> pd.DataFrame:
    """Parse a .tenv8 time-series data file into a DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the .tenv8 file.

    Returns
    -------
    pd.DataFrame
        Pandas datafram with columns
        [decimal_year, east, north, up, sigma_east, sigma_north, sigma_up, rapid_flag].

    """
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        header=None,
        names=[
            "decimal_year",
            "east",
            "north",
            "up",
            "sigma_east",
            "sigma_north",
            "sigma_up",
            "rapid_flag",
        ],
    )
    return df

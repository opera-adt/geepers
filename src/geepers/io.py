from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
import xarray as xr
from affine import cached_property
from numpy.typing import ArrayLike
from pyproj import Transformer
from rasterio.crs import CRS

from geepers._types import DatetimeLike

from .utils import get_dates

__all__ = ["XarrayReader"]

logger = logging.getLogger("geepers")


if TYPE_CHECKING:
    from ._types import Index


@dataclass
class XarrayReader:
    """A wrapper for an `xarray.DataArray` with georeferencing/windowed reading."""

    da: xr.DataArray

    def __post_init__(self):
        # Check that the DataArray has the required coordinates
        da = self.da
        if "band" in da.coords and da.coords["band"].size == 1:
            self.da = da.sel(band=da.coords["band"][0]).drop_vars("band")

        # Normalize so we have 'x' and 'y' coordinates
        if "lon" in self.da.coords:
            self.da = self.da.rename({"lon": "x"})
            if self.crs is not None and self.crs != "EPSG:4326":
                msg = "CRS is not EPSG:4326, but 'lon' coordinate is present."
                raise ValueError(msg)
            self.da.rio.write_crs("EPSG:4326", inplace=True)
        if "lat" in self.da.coords:
            self.da = self.da.rename({"lat": "y"})

        if "x" not in self.da.coords or "y" not in self.da.coords:
            msg = "DataArray must have 'x' and 'y' coordinates."
            raise ValueError(msg)

        if self.crs is None:
            msg = "CRS is not set."
            raise ValueError(msg)

        if not hasattr(self.da, "units"):
            msg = "Units are not set."
            raise ValueError(msg)

    @classmethod
    def from_file(
        cls,
        filename: Path | str,
        data_var: str | None = None,
        engine: str | None = None,
        nodata: float | None = None,
        crs: CRS | None = None,
        units: str | None = None,
    ) -> Self:
        """Create a XarrayReader from one file.

        Can be a 2D XarrayReader from a single-band GDAL-readable file,
        or a 3D XarrayReader from data cube (e.g. NetCDF, Zarr).

        Parameters
        ----------
        filename : Path | str
            Path to the file to load.
        data_var : str
            Name of the variable to load.
        engine : str | None
            Xarray engine to use for opening the file.
        nodata : float | None
            Nodata value to use.
        crs : rasterio.crs.CRS | None
            CRS to use.
        units : str | None
            Units to use.

        Returns
        -------
        XarrayReader
            A 2D XarrayReader with the data from the file.

        """
        if Path(filename).suffix == ".zarr":
            ds = xr.open_zarr(filename, consolidated=False)
        else:
            engine = cls._guess_engine(filename)
            ds = xr.open_dataset(filename, engine=engine)

        if data_var is not None:
            da = ds[data_var]
        else:
            if len([var for var in ds.data_vars if ds[var].ndim >= 2]) != 1:
                msg = (
                    "Multiple data variables found in file. Please specify which one to"
                    " use."
                )
                raise ValueError(msg)
            da = ds[next(var for var in ds.data_vars if ds[var].ndim >= 2)]

        # Apply overrides, if given
        if crs is not None:
            da.rio.write_crs(crs, inplace=True)
        elif da.rio.crs is None and "spatial_ref" in ds:
            da.rio.write_crs(ds["spatial_ref"].crs_wkt, inplace=True)
        if nodata is not None:
            da.rio.write_nodata(nodata, inplace=True)
        if units is not None:
            da.attrs["units"] = units
        return cls(da)

    @classmethod
    def from_file_list(
        cls,
        file_list: Sequence[Path | str],
        file_date_fmt: str = "%Y%m%d",
        file_date_idx: int = 1,
        units: str | None = None,
    ) -> Self:
        """Create a 3D XarrayReader from a list of single-band GDAL-readable files.

        Parameters
        ----------
        file_list : Sequence[Path | str]
            List of files to load.
        file_date_fmt : str
            Format string for parsing dates from file names.
        file_date_idx : int
            Index of the date in the file name.
        units : str | None
            Units for the output data (default ``"unitless"``).

        Returns
        -------
        XarrayReader
            A 3D XarrayReader with the data from the files.

        """
        files = sorted(file_list)
        logger.info(f"Loading {len(files)} files from {files[0]}")

        def preprocess(ds: xr.Dataset) -> xr.Dataset:
            """Preprocess individual dataset when loading with open_mfdataset."""
            fname = ds.encoding["source"]
            date = get_dates(fname, fmt=file_date_fmt)[file_date_idx]
            if len(ds.band) == 1:
                ds = ds.sel(band=ds.band[0]).drop_vars("band")
            return ds.expand_dims(time=[pd.to_datetime(date)])

        ds = xr.open_mfdataset(files, engine="rasterio", preprocess=preprocess)
        if units:
            ds.band_data.attrs["units"] = units
        return cls(ds.band_data)

    @staticmethod
    def _guess_engine(filename: str | Path) -> str | None:
        # TODO: Figure out why Xarray is bad at guessing, and uses `h5netcdf`
        # when i pass zarr or geotiffs...
        match Path(filename).suffix:
            case "h5netcdf":
                return "h5netcdf"
            case ".zarr":
                return "zarr"
            case ".tif":
                return "rasterio"
            case _:
                return None

    @classmethod
    def from_range_file_list(
        cls,
        file_list: Sequence[str | Path],
        target_times: Sequence[DatetimeLike],
        file_date_fmt: str = "%Y%m%d",
        units: str | None = None,
    ) -> Self:
        """Create a 3D reader from a list of range-based rasters.

        Build a reader whose 3-D array has a `time` axis identical to
        `target_times`, but each slice comes from the single quality raster
        whose filename-encoded date-range covers that epoch.

        Parameters
        ----------
        file_list : Sequence[str | Path]
            List of files to load.
        target_times : Sequence[DatetimeLike]
            The time epochs you want on the output `time` axis.
            Can come from a `XarrayReader`'s `time` coordinate.
        file_date_fmt : str
            Format used by ``get_dates`` (default ``"%Y%m%d"``).
        units : str | None
            Units for the output data (default ``"unitless"``).

        Notes
        -----
        Broadcasting is lazy: every epoch that maps to the same file
          references the same dask array.

        """
        times = pd.to_datetime(target_times)  # type: ignore[arg-type]

        # Create a mapping from each target time to its corresponding file
        time_to_file: dict[int, Path | str] = {}
        for fp in sorted(file_list):
            t0, t1 = get_dates(fp, fmt=file_date_fmt)[:2]  # start, end
            t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)

            # Find the epochs that fall inside [t0, t1]
            mask = (times >= t0) & (times <= t1)
            if not mask.any():
                continue

            # Map each matching time to this file
            for time_idx in np.where(mask)[0]:
                time_to_file[time_idx] = fp

        if not time_to_file:
            msg = "None of the files cover any requested epoch."
            raise ValueError(msg)

        # Group times by file to minimize file opening
        file_to_times: dict[str | Path, list[int]] = {}
        for time_idx, fp in time_to_file.items():
            if fp not in file_to_times:
                file_to_times[fp] = []
            file_to_times[fp].append(time_idx)

        # Build layers without duplicates
        layers: list[xr.DataArray] = []
        for fp, time_indices in file_to_times.items():
            # lazily open once, drop the 'band' dim if present
            da = xr.open_dataset(
                fp, engine="rasterio", chunks="auto"
            ).band_data.squeeze("band", drop=True)

            # broadcast onto the matching epochs without data copy
            matching_times = times[time_indices]
            layers.append(da.expand_dims(time=matching_times))

        # Stack all the mini-arrays
        da_out = xr.concat(layers, dim="time").reindex(time=times)

        if units:
            # Make sure a units attribute exists so __post_init__ is happy
            da_out.attrs["units"] = units

        return cls(da_out)

    @property
    def ndim(self):
        return self.da.ndim

    @property
    def shape(self):
        return self.da.shape

    @property
    def dtype(self):
        return self.da.dtype

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        return self.da[key].values

    @property
    def crs(self):
        return self.da.rio.crs

    def read_lon_lat(
        self, lons: float | ArrayLike, lats: float | ArrayLike
    ) -> list[xr.DataArray]:
        """Read values at given longitudes and latitudes.

        Parameters
        ----------
        lons : float | ArrayLike
            Longitudes to read.
        lats : float | ArrayLike
            Latitudes to read.

        Returns
        -------
        xr.DataArray | list[xr.DataArray]
            Values at the given longitudes and latitudes.
            If a single longitude and latitude is provided, returns a single
            `xr.DataArray`. Otherwise, returns a list of `xr.DataArray`
            objects.

        """
        if self.crs != "EPSG:4326":
            x, y = self._transformer_from_lonlat.transform(lons, lats)
        else:
            x, y = np.asarray(lons), np.asarray(lats)

        xa, ya = np.asarray(x), np.asarray(y)
        if xa.size != ya.size:
            msg = "x and y must have the same length"
            raise ValueError(msg)

        if xa.size == 1:
            return self.da.sel(x=xa, y=ya, method="nearest")

        return [
            self.da.sel(x=xx, y=yy, method="nearest")
            for xx, yy in zip(xa, ya, strict=False)
        ]

    @cached_property
    def _transformer_from_lonlat(self):
        return Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    def read_window(
        self, lon: float, lat: float, buffer_pixels: int = 0, op=round
    ) -> np.ndarray:
        """Read values in a window around the given longitude and latitude.

        Parameters
        ----------
        lon : float
            Longitude to read.
        lat : float
            Latitude to read.
        buffer_pixels : int, optional
            Number of pixels to read around the given longitude and latitude.
            Default is 0.
        op : callable, optional
            Function to apply to the longitude and latitude to get the row and column.
            Default is `round`.

        Returns
        -------
        np.ndarray
            Values in the window around the given longitude and latitude.
            Dimension of output is equal to `self.ndim`.

        """
        if self.crs != "EPSG:4326":
            x, y = self._transformer_from_lonlat.transform(lon, lat)
        else:
            x, y = lon, lat
        # Use the inverse transform to get row, col
        col_float, row_float = ~(self.da.rio.transform()) * (x, y)
        col, row = op(col_float), op(row_float)
        x_slice = slice(col - buffer_pixels, col + buffer_pixels + 1)
        y_slice = slice(row - buffer_pixels, row + buffer_pixels + 1)
        print(x_slice, y_slice)
        return self.da.isel(x=x_slice, y=y_slice).values

    @property
    def units(self) -> str:
        return self.da.units

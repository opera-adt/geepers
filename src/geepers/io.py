from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import rasterio as rio
import rasterio.windows
from numpy.typing import ArrayLike
from rasterio.vrt import WarpedVRT
from tqdm.contrib.concurrent import thread_map

from ._types import PathOrStr

__all__ = ["DatasetReader", "StackReader", "RasterReader", "RasterStackReader"]


if TYPE_CHECKING:
    from dolphin._types import Index


@runtime_checkable
class DatasetReader(Protocol):
    """An array-like interface for reading input datasets.

    `DatasetReader` defines the abstract interface that types must conform to in order
    to be read by functions which iterate in blocks over the input data.
    Such objects must export NumPy-like `dtype`, `shape`, and `ndim` attributes,
    and must support NumPy-style slice-based indexing.

    Note that this protocol allows objects to be passed to `dask.array.from_array`
    which needs `.shape`, `.ndim`, `.dtype` and support numpy-style slicing.
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __getitem__(self, key: tuple[Index, ...], /) -> ArrayLike:
        """Read a block of data."""
        ...


@runtime_checkable
class StackReader(DatasetReader, Protocol):
    """An array-like interface for reading a 3D stack of input datasets.

    `StackReader` defines the abstract interface that types must conform to in order
    to be valid inputs to be read in functions like [dolphin.ps.create_ps][].
    It is a specialization of [DatasetReader][] that requires a 3D shape.
    """

    ndim: int = 3
    """int : Number of array dimensions."""

    shape: tuple[int, int, int]
    """tuple of int : Tuple of array dimensions."""

    def __len__(self) -> int:
        """Int : Number of images in the stack."""
        return self.shape[0]


def _mask_array(arr: np.ndarray, nodata_value: float | None) -> np.ma.MaskedArray:
    """Mask an array based on a nodata value."""
    if np.isnan(nodata_value):
        return np.ma.masked_invalid(arr)
    return np.ma.masked_equal(arr, nodata_value)


@dataclass
class RasterReader(DatasetReader):
    """A single raster band of a GDAL-compatible dataset.

    See Also
    --------
    BinaryReader
    HDF5

    Notes
    -----
    If `keep_open=True`, this class does not store an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.
    If passing the `RasterReader` to multiple spawned processes, it is recommended
    to set `keep_open=False` .

    """

    filename: PathOrStr
    """PathOrStr : The file path."""

    band: int
    """int : Band index (1-based)."""

    driver: str
    """str : Raster format driver name."""

    crs: rio.crs.CRS
    """rio.crs.CRS : The dataset's coordinate reference system."""

    transform: rio.transform.Affine
    """
    rasterio.transform.Affine : The dataset's georeferencing transformation matrix.

    This transform maps pixel row/column coordinates to coordinates in the dataset's
    coordinate reference system.
    """

    shape: tuple[int, int]
    dtype: np.dtype

    nodata: Optional[float] = None
    """Optional[float] : Value to use for nodata pixels."""

    keep_open: bool = False
    """bool : If True, keep the rasterio file handle open for faster reading."""

    chunks: Optional[tuple[int, int]] = None
    """Optional[tuple[int, int]] : Chunk shape of the dataset, or None if unchunked."""

    @classmethod
    def from_file(
        cls,
        filename: PathOrStr,
        band: int = 1,
        nodata: Optional[float] = None,
        keep_open: bool = False,
        **options,
    ) -> RasterReader:
        with rio.open(filename, "r", **options) as src:
            shape = (src.height, src.width)
            dtype = np.dtype(src.dtypes[band - 1])
            driver = src.driver
            crs = src.crs
            nodata = nodata or src.nodatavals[band - 1]
            transform = src.transform
            chunks = src.block_shapes[band - 1]

            return cls(
                filename=filename,
                band=band,
                driver=driver,
                crs=crs,
                transform=transform,
                shape=shape,
                dtype=dtype,
                nodata=nodata,
                keep_open=keep_open,
                chunks=chunks,
            )

    def __post_init__(self):
        if self.keep_open:
            self._src = rio.open(self.filename, "r")

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """Int : Number of array dimensions."""
        return 2

    def __array__(self) -> np.ndarray:
        return self[:, :]

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        if key is ... or key == ():
            key = (slice(None), slice(None))

        if not isinstance(key, tuple):
            msg = "Index must be a tuple of slices or integers."
            raise TypeError(msg)

        r_slice, c_slice = _ensure_slices(*key[-2:])
        window = rasterio.windows.Window.from_slices(
            r_slice,
            c_slice,
            height=self.shape[0],
            width=self.shape[1],
        )
        if self.keep_open:
            out = self._src.read(self.band, window=window)

        with rio.open(self.filename) as src:
            out = src.read(self.band, window=window)
        out_masked = _mask_array(out, self.nodata) if self.nodata is not None else out
        # Note that Rasterio doesn't use the `step` of a slice, so we need to
        # manually slice the output array.
        r_step, c_step = r_slice.step or 1, c_slice.step or 1
        return out_masked[::r_step, ::c_step].squeeze()

    def read_lon_lat(self, lons, lats, masked: bool = False) -> np.ndarray:
        """Get pixel values from a raster file for given longitudes and latitudes.

        Parameters
        ----------
        raster_path : str
            Path to the raster file.
        lons : float
            Longitude of the points.
        lats : float
            Latitude of the points.
        masked : bool, optional
            If True, reads in data as a MaskedArray, masking nodata values.
            Default is False.

        Returns
        -------
        np.ndarray
            pixel_values at the nearest point

        """
        with ExitStack() as stack:
            if not self.keep_open:
                src = stack.enter_context(rio.open(self.filename))
            else:
                src = self._src

            lon_list = [lons] if np.isscalar(lons) else lons
            lat_list = [lats] if np.isscalar(lats) else lats

            with WarpedVRT(src, crs="EPSG:4326") as vrt:
                return np.array(
                    list(vrt.sample(xy=zip(lon_list, lat_list), masked=masked))
                )

    def read_window(
        self,
        lon: float,
        lat: float,
        buffer_pixels: int = 0,
        op=round,
        masked: bool = False,
    ):
        """Get a window of pixel values for a given longitude and latitude.

        Parameters
        ----------
        lon : float
            Longitude of the central point.
        lat : float
            Latitude of the central point.
        buffer_pixels : int
            Number of pixels to buffer around the central point.
        op : callable, optional
            Operation to use when calculating the central pixel. Default is round.
            Options are "round", "math.floor", "math.ceil"
        masked : bool, optional
            If True, reads in data as a MaskedArray, masking nodata values.
            Default is False.

        Returns
        -------
        np.ndarray
            Window of pixel values around the specified point.

        """
        with ExitStack() as stack:
            if not self.keep_open:
                src = stack.enter_context(rio.open(self.filename))
            else:
                src = self._src

            # Transform the lon/lat to the raster's CRS
            x, y = rio.warp.transform("EPSG:4326", src.crs, [lon], [lat])

            # Get the row and column of the central pixel
            row, col = src.index(x[0], y[0], op=op)

            # Calculate the window boundaries
            window = rasterio.windows.Window(
                col - buffer_pixels,
                row - buffer_pixels,
                2 * buffer_pixels + 1,
                2 * buffer_pixels + 1,
            )

            return src.read(self.band, window=window, masked=masked)


def _read_3d(
    key: tuple[Index, ...], readers: Sequence[DatasetReader], num_threads: int = 1
):
    bands, r_slice, c_slice = _unpack_3d_slices(key)

    if isinstance(bands, slice):
        # convert the bands to -1-indexed list
        total_num_bands = len(readers)
        band_idxs = list(range(*bands.indices(total_num_bands)))
    elif isinstance(bands, int):
        band_idxs = [bands]
    else:
        msg = "Band index must be an integer or slice."
        raise TypeError(msg)

    # Get only the bands we need
    if num_threads == 1:
        out = np.stack([readers[i][r_slice, c_slice] for i in band_idxs], axis=0)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(lambda i: readers[i][r_slice, c_slice], band_idxs)
        out = np.stack(list(results), axis=0)

    # TODO: Do i want a "keep_dims" option to not collapse singleton dimensions?
    return np.squeeze(out)


@dataclass
class BaseStackReader(StackReader):
    """Base class for stack readers."""

    file_list: Sequence[PathOrStr]
    readers: Sequence[DatasetReader]
    num_threads: int = 1
    nodata: Optional[float] = None

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        return _read_3d(key, self.readers, num_threads=self.num_threads)

    @property
    def shape_2d(self):
        return self.readers[0].shape

    @property
    def shape(self):
        return (len(self.file_list), *self.shape_2d)

    @property
    def dtype(self):
        return self.readers[0].dtype


@dataclass
class RasterStackReader(BaseStackReader):
    """A stack of datasets for any GDAL-readable rasters.

    See Also
    --------
    BinaryStackReader
    HDF5StackReader

    Notes
    -----
    If `keep_open=True`, this class stores an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.

    """

    readers: Sequence[RasterReader]

    @classmethod
    def from_file_list(
        cls,
        file_list: Sequence[PathOrStr],
        bands: int | Sequence[int] = 1,
        keep_open: bool = False,
        num_threads: int = 1,
        nodata: Optional[float] = None,
    ) -> "RasterStackReader":
        """Create a RasterStackReader from a list of files.

        Parameters
        ----------
        file_list : Sequence[PathOrStr]
            List of paths to the files to read.
        bands : int | Sequence[int]
            Band to read from each file.
            If a single int, will be used for all files.
            Default = 1.
        keep_open : bool, optional (default False)
            If True, keep the rasterio file handles open for faster reading.
        num_threads : int, optional (default 1)
            Number of threads to use for reading.
        nodata : float, optional
            Manually set value to use for nodata pixels, by default None

        Returns
        -------
        RasterStackReader
            The RasterStackReader object.

        """
        if isinstance(bands, int):
            bands = [bands] * len(file_list)

        readers = [
            RasterReader.from_file(f, band=b, keep_open=keep_open)
            for (f, b) in zip(file_list, bands)
        ]
        # Check if nodata values were found in the files
        nds = {r.nodata for r in readers}
        if len(nds) == 1:
            nodata = nds.pop()
        return cls(file_list, readers, num_threads=num_threads, nodata=nodata)

    def read_lon_lat(
        self, lons, lats, masked: bool = False, max_workers: int = 4
    ) -> np.ndarray:
        """Read in raster values located at `lons`, `lats`."""

        def _read_single(reader):
            return reader.read_lon_lat(lons, lats, masked=masked)

        results = thread_map(
            _read_single,
            self.readers,
            max_workers=max_workers,
            desc="Reading points from time series",
        )
        return np.array(list(results))

    def read_window(
        self,
        lon: float,
        lat: float,
        buffer_pixels: int = 0,
        op=round,
        masked: bool = False,
    ):
        """Get a window of pixel values for a given longitude and latitude."""
        return np.array(
            [
                reader.read_window(
                    lon, lat, buffer_pixels=buffer_pixels, op=op, masked=masked
                )
                for reader in self.readers
            ]
        )


def _ensure_slices(rows: Index, cols: Index) -> tuple[slice, slice]:
    def _parse(key: Index):
        if isinstance(key, int):
            return slice(key, key + 1)
        elif key is ...:
            return slice(None)
        else:
            return key

    return _parse(rows), _parse(cols)


def _unpack_3d_slices(key: tuple[Index, ...]) -> tuple[Index, slice, slice]:
    # Check that it's a tuple of slices
    if not isinstance(key, tuple):
        msg = "Index must be a tuple of slices."
        raise TypeError(msg)
    if len(key) not in (1, 3):
        msg = "Index must be a tuple of 1 or 3 slices."
        raise TypeError(msg)
    # If only the band is passed (e.g. stack[0]), convert to (0, :, :)
    if len(key) == 1:
        key = (key[0], slice(None), slice(None))
    # unpack the slices
    bands, rows, cols = key
    # convert the rows/cols to slices
    r_slice, c_slice = _ensure_slices(rows, cols)
    return bands, r_slice, c_slice


def get_raster_units(filename: PathOrStr, band: int = 1) -> str | None:
    with rio.open(filename) as src:
        return src.units[band - 1]

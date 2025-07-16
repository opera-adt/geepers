"""InSAR data processing functions."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from dask import compute
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback

from geepers._types import DatetimeLike
from geepers.constants import SENTINEL_1_WAVELENGTH
from geepers.io import XarrayReader

logger = logging.getLogger("geepers")

PHASE_TO_METERS = float(SENTINEL_1_WAVELENGTH) / (4.0 * np.pi)


def process_insar_data(
    *,
    reader: XarrayReader,
    df_gps_stations: pd.DataFrame,
    reader_temporal_coherence: XarrayReader | None = None,
    reader_similarity: XarrayReader | None = None,
) -> dict[str, pd.DataFrame]:
    """Sample InSAR rasters at all station locations in one pass.

    Parameters
    ----------
    reader : RasterStackReader
        *RasterStackReader* opened on the displacement stack.
    df_gps_stations
        DataFrame indexed by station name with at least ``lon`` and ``lat``
        columns (decimal degrees).
    reader_temporal_coherence, reader_similarity
        Optional readers to sample alongside displacement to
        compute temporal coherence and phase similarity.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A mapping from station name to a dataframe that contains *los_insar*
        (in **metres**), *temporal_coherence* and *similarity* columns indexed
        by acquisition date.

    """
    lons = df_gps_stations.lon.to_numpy()
    lats = df_gps_stations.lat.to_numpy()

    # los_insar gets stacks as (n_stations, len(time) ), each row one station
    with TqdmCallback(desc="Sampling InSAR data locations"):
        r = compute(reader.read_lon_lat(lons, lats))
    los_insar = np.stack(r).squeeze()

    if reader_temporal_coherence is not None:
        with TqdmCallback(desc="Sampling temporal coherence data locations"):
            r = compute(reader_temporal_coherence.read_lon_lat(lons, lats))
        temp_coh = np.stack(r).squeeze()
    else:
        temp_coh = None

    if reader_similarity is not None:
        with TqdmCallback(desc="Sampling similarity data locations"):
            r = compute(reader_similarity.read_lon_lat(lons, lats))
        similarity = np.stack(r).squeeze()
    else:
        similarity = None

    if reader.units not in ("meters", "m"):
        logger.warning("Converting InSAR displacement to meters.")
        los_insar *= PHASE_TO_METERS

    station_to_insar: dict[str, pd.DataFrame] = {}
    for i, station in tqdm(
        enumerate(df_gps_stations.index), total=len(df_gps_stations)
    ):
        data = {
            "los_insar": los_insar[i],
        }
        if similarity is not None:
            data["similarity"] = similarity[i]
        if temp_coh is not None:
            data["temporal_coherence"] = temp_coh[i]

        station_to_insar[station] = pd.DataFrame(index=reader.da.time, data=data)

    return station_to_insar


def get_quality_reader(
    quality_files: Sequence[str | Path] | None,
    time_array: Sequence[DatetimeLike],
    file_date_fmt: str = "%Y%m%d",
) -> XarrayReader | None:
    """Create a quality reader from file list and time array.

    Parameters
    ----------
    quality_files
        List of quality files (e.g., temporal coherence, similarity).
    time_array
        Array of time values for the stack.
    file_date_fmt
        Format string for parsing dates from filenames.

    Returns
    -------
    XarrayReader | None
        Quality reader, or None if no files provided.

    """
    if quality_files is None:
        return None

    # If there is only one file per ministack, then we need to use the range reader
    if len(quality_files) < len(time_array):
        return XarrayReader.from_range_file_list(
            quality_files,
            time_array,
            file_date_fmt=file_date_fmt,
            units="unitless",
        )
    else:
        # Otherwise, the reader will be like other readers
        return XarrayReader.from_file_list(
            quality_files, file_date_fmt, units="unitless"
        )

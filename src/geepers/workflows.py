"""High-level workflows and CLI orchestration."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import rasterio.warp
import requests
import tyro
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

import geepers.rates
from geepers.analysis import compare_relative_gps_insar, create_tidy_df
from geepers.gps_sources import SideshowSource, UnrSource
from geepers.io import XarrayReader
from geepers.processing import get_quality_reader, process_insar_data
from geepers.quality import select_gps_reference
from geepers.uncertainty import get_sigma_los_df

logger = logging.getLogger("geepers")


def main(
    *,
    los_enu_file: Annotated[str | Path, tyro.conf.arg(aliases=["--los"])],
    timeseries_files: Sequence[str | Path] | None = None,
    timeseries_stack: str | Path | None = None,
    output_dir: Annotated[Path, tyro.conf.arg(aliases=["-o"])] = Path("./GPS"),
    file_date_fmt: str = "%Y%m%d",
    gps_source: Literal["unr", "jpl"] = "unr",
    stack_data_var: str | None = "displacement",
    reference_station: Annotated[str | None, tyro.conf.arg(aliases=["--ref"])] = None,
    temporal_coherence_files: Sequence[str | Path] | None = None,
    similarity_files: Sequence[str | Path] | None = None,
    insar_buffer: int = 0,
    gps_time_window: int = 10,
) -> None:
    """Compare InSAR time series to GPS observations along the line-of-sight.

    Parameters
    ----------
    los_enu_file
        Three-band GeoTIFF with the line-of-sight unit vector expressed in the
        local East-North-Up coordinate frame.
        LOS convention is that the unit vectors point from the ground toward
        the satellite (i.e. the "up" component is positive).
    timeseries_files
        List of wrapped-phase (or displacement) rasters, one per acquisition.
        File names must encode the reference and secondary dates using
        `file_date_fmt`.
    timeseries_stack
        Path to an xarray stack of wrapped-phase (or displacement) rasters.
        Alternative to `timeseries_files`
    output_dir : Path | str
        Directory where CSV outputs will be written.  Will be created if it does
        not exist.
    file_date_fmt : str
        ``strftime`` pattern describing how dates are embedded in
        `timeseries_files`.
        Default is "%Y%m%d"
    gps_source : str
        GNSS data source.
        Choices are "unr" or "jpl". See `geepers.gps_sources`.
    stack_data_var : str
        Name of the variable in the timeseries stack to use for InSAR data.
        If `None`, the `timeseries_stack` must have only one data variable.
    reference_station : str
        Optional GPS station name - if provided, relative displacements are
        computed with respect to this station.
    temporal_coherence_files, similarity_files
        Optional rasters providing per-pixel temporal coherence and phase
        similarity which will be sampled at station locations.
    insar_buffer
        Number of pixels to buffer around each GPS station when sampling InSAR
        data. Uses median averaging, ignoring NaN values, to reduce noise through
        spatial averaging. Default is 0 (single pixel).
    gps_time_window
        Number of days for GPS rolling average window to reduce temporal noise.
        Default is 30 days.

    Notes
    -----
    The script writes three CSV files into `output_dir`:

    combined_data.csv
        Tidy table stacking raw GPS and InSAR series for each station.
    relative_comparison.csv
        Relative GPS/InSAR displacements if `reference_station` was specified,
        if `reference_station` is not `None`.
    station_summary.csv
        Per-station linear rates (mm/yr) computed from the combined table.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if timeseries_files is None:
        if timeseries_stack is None:
            msg = "Must provide either timeseries_files or timeseries_stack"
            raise ValueError(msg)
        insar_reader = XarrayReader.from_file(timeseries_stack, data_var=stack_data_var)
        reader_temporal_coherence: XarrayReader | None = XarrayReader.from_file(
            timeseries_stack, data_var="temporal_coherence"
        )
        reader_similarity: XarrayReader | None = XarrayReader.from_file(
            timeseries_stack, data_var="phase_similarity"
        )

    else:
        insar_reader = XarrayReader.from_file_list(timeseries_files, file_date_fmt)
        reader_temporal_coherence = get_quality_reader(
            temporal_coherence_files, insar_reader.da.time, file_date_fmt
        )
        reader_similarity = get_quality_reader(
            similarity_files, insar_reader.da.time, file_date_fmt
        )
    logger.info("Created %s", insar_reader)
    if reader_temporal_coherence is not None:
        logger.info("Created %s", reader_temporal_coherence)
    else:
        logger.info("No temporal coherence reader provided.")
    if reader_similarity is not None:
        logger.info("Created %s", reader_similarity)
    else:
        logger.info("No similarity reader provided.")

    # Get GPS stations within image bounds using new API
    source = UnrSource() if gps_source == "unr" else SideshowSource()
    if insar_reader.crs != "EPSG:4326":
        bounds = rasterio.warp.transform_bounds(
            insar_reader.crs, "EPSG:4326", *insar_reader.da.rio.bounds()
        )
    else:
        bounds = insar_reader.da.rio.bounds()
    df_gps_stations = source.stations(bbox=bounds)
    df_gps_stations.set_index("id", inplace=True)

    start_date = insar_reader.da.time[0].to_pandas()
    end_date = insar_reader.da.time[-1].to_pandas()

    def _load_or_none(name: str) -> pd.DataFrame | None:
        try:
            return source.timeseries(
                name, frame="ENU", start_date=start_date, end_date=end_date
            )
        except requests.HTTPError:
            return None

    df_gps_list = thread_map(
        _load_or_none,
        df_gps_stations.index,
        max_workers=10,
        desc="Downloading GPS station data",
    )

    los_reader = XarrayReader.from_file(los_enu_file, units="unitless")
    station_to_los_gps: dict[str, pd.DataFrame] = {}

    for station_row, df in tqdm(
        zip(df_gps_stations.itertuples(), df_gps_list, strict=True),
        total=len(df_gps_stations),
        desc="Projecting GPS -> LOS",
    ):
        if df is None:
            warnings.warn(f"Failed to download {station_row}; skipping.", stacklevel=2)
            continue

        enu_vec = np.nan_to_num(
            los_reader.read_window(station_row.lon, station_row.lat, 0)
        ).squeeze()
        if enu_vec.size == 0 or np.allclose(enu_vec, 0):
            logger.info(f"{station_row} lies outside LOS raster bounds; skipping.")
            continue
        assert enu_vec.shape == (3,)

        e, n, u = enu_vec
        df["los_gps"] = df.east * e + df.north * n + df.up * u
        if df["los_gps"].size > 0:
            df["los_gps"] -= np.nanmean(df["los_gps"])  # remove arbitrary offset

        # Apply rolling average if requested
        if gps_time_window > 0:
            df["los_gps"] = (
                df["los_gps"]
                .rolling(window=gps_time_window, center=True, min_periods=1)
                .median()
            )
        # Compute the LOS uncertainty
        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            df["sigma_los"] = get_sigma_los_df(df, enu_vec)

        df_subset = df[["date", "los_gps", "sigma_los"]].set_index("date")
        station_to_los_gps[station_row.Index] = df_subset

    # Sample InSAR rasters at station locations
    logger.info("Sampling InSAR rasters at station locations")
    station_to_insar = process_insar_data(
        reader=insar_reader,
        df_gps_stations=df_gps_stations,
        reader_temporal_coherence=reader_temporal_coherence,
        reader_similarity=reader_similarity,
        insar_buffer=insar_buffer,
    )

    # Merge GPS and InSAR tables per station
    logger.info("Merging GPS and InSAR tables per station")
    station_to_merged: dict[str, pd.DataFrame] = {}
    for station_id in tqdm(station_to_los_gps, desc="Merging GPS and InSAR"):
        # Use asof merge in case GPS is datetime and insar is date
        station_to_merged[station_id] = pd.merge_asof(
            left=station_to_los_gps[station_id],
            right=station_to_insar[station_id],
            tolerance=pd.Timedelta("1D"),
            direction="nearest",
            left_index=True,
            right_index=True,
        )

    # Save results
    combined_df = create_tidy_df(station_to_merged)
    combined_df.to_csv(output_dir / "combined_data.csv", index=False)

    if not reference_station:
        # Automatic reference selection (if the user didn't supply --ref)
        if reference_station is None:
            reference_station = select_gps_reference(station_to_merged)
            logger.info("Auto-selected %s as reference station", reference_station)

        # Compute relative comparison
        logger.info("Comparing GPS and InSAR relative to %s", reference_station)
        rel_df = compare_relative_gps_insar(
            station_to_merged, reference_station=reference_station
        )

    logger.info("Comparing GPS and InSAR relative to %s", reference_station)
    rel_df = compare_relative_gps_insar(
        station_to_merged, reference_station=reference_station
    )
    rel_df.to_csv(output_dir / "relative_comparison.csv", index=False)

    df_rates = geepers.rates.calculate_rates(df=combined_df, to_mm=True)
    df_rates.to_csv(output_dir / "station_summary.csv")

    logger.info("Finished - results written to %s", output_dir)

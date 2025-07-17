"""High-level workflows and CLI orchestration."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import requests
import tyro
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

import geepers.gps
import geepers.rates
from geepers.analysis import compare_relative_gps_insar, create_tidy_df
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
    stack_data_var: str | None = "displacement",
    reference_station: Annotated[str | None, tyro.conf.arg(aliases=["--ref"])] = None,
    temporal_coherence_files: Sequence[str | Path] | None = None,
    similarity_files: Sequence[str | Path] | None = None,
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
    output_dir
        Directory where CSV outputs will be written.  Will be created if it does
        not exist.
    file_date_fmt
        ``strftime`` pattern describing how dates are embedded in
        `timeseries_files`.
    stack_data_var
        Name of the variable in the timeseries stack to use for InSAR data.
        If `None`, the `timeseries_stack` must have only one data variable.
    reference_station
        Optional GPS station name - if provided, relative displacements are
        computed with respect to this station.
    temporal_coherence_files, similarity_files
        Optional rasters providing per-pixel temporal coherence and phase
        similarity which will be sampled at station locations.

    Notes
    -----
    The script writes three CSV files into *output_dir*:

    ``combined_data.csv``
        Tidy table stacking raw GPS and InSAR series for each station.
    ``relative_comparison.csv``
        Relative GPS/InSAR displacements if *reference_station* was specified,
        if `reference_station` is not `None`.
    ``station_summary.csv``
        Per-station linear rates (mm/yr) computed from the combined table.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if timeseries_files is None:
        if timeseries_stack is None:
            msg = "Must provide either timeseries_files or timeseries_stack"
            raise ValueError(msg)
        insar_reader = XarrayReader.from_file(timeseries_stack, data_var=stack_data_var)
    else:
        insar_reader = XarrayReader.from_file_list(timeseries_files, file_date_fmt)

    logger.info("Created %s", insar_reader)
    reader_temporal_coherence = get_quality_reader(
        temporal_coherence_files, insar_reader.da.time, file_date_fmt
    )
    reader_similarity = get_quality_reader(
        similarity_files, insar_reader.da.time, file_date_fmt
    )

    df_gps_stations = geepers.gps.get_stations_within_image(
        insar_reader, mask_invalid=False
    )
    df_gps_stations.set_index("name", inplace=True)

    start_date = insar_reader.da.time[0].to_pandas()
    end_date = insar_reader.da.time[-1].to_pandas()

    def _load_or_none(name: str) -> pd.DataFrame | None:
        try:
            return geepers.gps.load_station_enu(
                station_name=name, start_date=start_date, end_date=end_date
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
            warnings.warn(
                f"Failed to download {station_row.Index}; skipping.", stacklevel=2
            )
            continue

        enu_vec = np.nan_to_num(
            los_reader.read_lon_lat(station_row.lon, station_row.lat)
        )
        assert enu_vec.shape == (3,)
        if np.allclose(enu_vec, 0):
            logger.info(f"{station_row.Index} lies has nodata in LOS raster; skipping.")
            continue

        e, n, u = enu_vec
        df["los_gps"] = df.east * e + df.north * n + df.up * u
        if df["los_gps"].size > 0:
            df["los_gps"] -= np.nanmean(df["los_gps"])  # remove arbitrary offset
        # Compute the LOS uncertainty
        # RuntimeWarning
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
    )

    # Merge GPS and InSAR tables per station
    logger.info("Merging GPS and InSAR tables per station")
    station_to_merged: dict[str, pd.DataFrame] = {}
    for name in tqdm(station_to_los_gps, desc="Merging GPS and InSAR"):
        station_to_merged[name] = pd.merge(
            station_to_los_gps[name],
            station_to_insar[name],
            how="left",
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

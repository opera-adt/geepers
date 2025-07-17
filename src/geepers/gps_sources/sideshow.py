"""JPL Sideshow GPS data source implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import pandas as pd

from .base import BaseGpsSource

if TYPE_CHECKING:
    pass

__all__ = ["SideshowSource"]

# JPL Sideshow constants
SITE_LIST_URL = "https://sideshow.jpl.nasa.gov/post/tables/table2.html"
STATION_URL_BASE = (
    "https://sideshow.jpl.nasa.gov/pub/JPL_GPS_Timeseries/repro2018a/post/point/"
)
GPS_BASE_URL = f"{STATION_URL_BASE}{{station}}.series"
STEPS_URL = "https://sideshow.jpl.nasa.gov/post/tables/table3.html"

logger = logging.getLogger("geepers")


class SideshowSource(BaseGpsSource):
    """JPL Sideshow GPS data source."""

    def timeseries(
        self,
        station_id: str,
        /,
        frame: Literal["ENU", "XYZ"] = "ENU",
        start_date: str | None = None,
        end_date: str | None = None,
        download_if_missing: bool = True,
    ) -> pd.DataFrame:
        """Load GPS station time series data.

        Parameters
        ----------
        station_id : str
            The station identifier.
        frame : {"ENU", "XYZ"}, optional
            Coordinate frame for the data. Default is "ENU".
        start_date : str, optional
            Start date for data filtering (ISO format).
        end_date : str, optional
            End date for data filtering (ISO format).
        download_if_missing : bool, optional
            Whether to download data if not found locally.

        Returns
        -------
        pd.DataFrame
            DataFrame with ENU time series data validated against schema.

        Raises
        ------
        NotImplementedError
            Sideshow station data loading not yet implemented.

        Notes
        -----
        This is a placeholder implementation. Full implementation would require:
        1. Downloading .series files from JPL Sideshow
        2. Parsing the specific format (see constants for column descriptions)
        3. Converting to standardized DataFrame format
        4. Validating against StationObservationSchema

        """
        if frame == "XYZ":
            msg = "ECEF frame not supported for Sideshow data"
            raise ValueError(msg)

        msg = f"Sideshow station data loading not yet implemented for {station_id}"
        raise NotImplementedError(msg)

    def _read_station_data(self) -> gpd.GeoDataFrame:
        """Read raw station data from the source.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with station metadata including lon, lat, alt columns.

        Raises
        ------
        NotImplementedError
            Sideshow station list reading not yet implemented.

        Notes
        -----
        This is a placeholder implementation. Full implementation would require:
        1. Parsing the JPL Sideshow station list from table2.html
        2. Extracting station names and coordinates
        3. Converting to standardized DataFrame format

        """
        msg = "Sideshow station list reading not yet implemented"
        raise NotImplementedError(msg)

    def download_station_data(
        self,
        station_id: str,
        output_dir: str | None = None,
    ) -> None:
        """Download GPS station data from JPL Sideshow.

        Parameters
        ----------
        station_id : str
            The station identifier.
        output_dir : str, optional
            Directory to save downloaded files.

        Raises
        ------
        NotImplementedError
            Sideshow station data download not yet implemented.

        Notes
        -----
        This is a placeholder implementation. Full implementation would:
        1. Download .series file from JPL Sideshow
        2. Save to local cache directory
        3. Handle error cases (station not found, etc.)

        """
        msg = f"Sideshow station data download not yet implemented for {station_id}"
        raise NotImplementedError(msg)

    def parse_series_file(self, file_path: str) -> pd.DataFrame:
        """Parse a JPL Sideshow .series file into a DataFrame.

        Parameters
        ----------
        file_path : str
            Path to the .series file.

        Returns
        -------
        pd.DataFrame
            DataFrame with parsed time series data.

        Raises
        ------
        NotImplementedError
            Sideshow series file parsing not yet implemented.

        Notes
        -----
        This is a placeholder implementation. Full implementation would parse:
        - Column 1: Decimal_YR
        - Columns 2-4: East(m) North(m) Vert(m)
        - Columns 5-7: E_sig(m) N_sig(m) V_sig(m)
        - Columns 8-10: E_N_cor E_V_cor N_V_cor
        - Column 11: Time in Seconds past J2000
        - Columns 12-17: Time in YEAR MM DD HR MN SS

        """
        msg = f"Sideshow series file parsing not yet implemented for {file_path}"
        raise NotImplementedError(msg)


# Create instance for backward compatibility
_sideshow_source = SideshowSource()

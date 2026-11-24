# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""A gridded GNSS product held locally: a regenerated or frozen UNR-style grid.

`UnrGridSource` is UNR's product as UNR serves it: the lookup table and the node
files come from geodesy.unr.edu. `CustomGridSource` reads the same file formats
from disk only, for a grid that is not UNR's (a regeneration whose node ids
extend UNR's, a frozen snapshot). It never downloads: a node without a file is an
error, not a silent fetch of UNR's value for that id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from geepers._optional import to_point_frame

from .unr_grid import UnrGridSource

if TYPE_CHECKING:
    import geopandas as gpd
    import requests

__all__ = ["CustomGridSource"]

DEFAULT_TEMPLATE = "{grid_id:06d}_{plate}.tenv8"
PlateName = Literal["NA", "PA", "IGS14", "IGS20"]


class CustomGridSource(UnrGridSource):
    """A local gridded product in UNR's formats (lookup table + ``.tenv8`` files).

    Examples
    --------
    >>> grid = CustomGridSource.from_snapshot("snapshot_dir")  # doctest: +SKIP
    >>> grid.stations(bbox=(-122.6, 36.3, -120.3, 38.3))  # doctest: +SKIP
    >>> grid.timeseries("043500")  # doctest: +SKIP

    """

    def __init__(
        self,
        lookup_file: str | Path,
        data_dir: str | Path,
        *,
        filename_template: str = DEFAULT_TEMPLATE,
        plate: PlateName = "IGS20",
        gridded_type: Literal["constant", "variable"] = "constant",
    ):
        """Open a local grid from its lookup table and node directory.

        Parameters
        ----------
        lookup_file : str | Path
            Node table, one ``id lon lat`` per line (UNR's ``grid_latlon_lookup.txt``
            format). Longitudes may be 0-360 or -180-180.
        data_dir : str | Path
            Directory holding one ``.tenv8`` file per node.
        filename_template : str, optional
            Node file name in ``data_dir``, formatted with ``grid_id`` (int) and
            ``plate``. Default ``"{grid_id:06d}_{plate}.tenv8"``.
        plate : {"NA", "PA", "IGS14", "IGS20"}, optional
            Reference frame of the files, used in the file names. Default "IGS20".
        gridded_type : {"constant", "variable"}, optional
            Recorded for reference. Default "constant".

        Raises
        ------
        FileNotFoundError
            If the lookup file or the data directory does not exist.

        """
        self.lookup_file = Path(lookup_file)
        self.data_dir = Path(data_dir)
        self.filename_template = filename_template
        self.plate = plate
        if not self.lookup_file.is_file():
            msg = f"lookup file not found: {self.lookup_file}"
            raise FileNotFoundError(msg)
        if not self.data_dir.is_dir():
            msg = f"data directory not found: {self.data_dir}"
            raise FileNotFoundError(msg)
        super().__init__(
            version="0.3", gridded_type=gridded_type, cache_dir=self.data_dir
        )

    @classmethod
    def from_snapshot(cls, snapshot_dir: str | Path) -> CustomGridSource:
        """Open a frozen grid snapshot.

        The directory holds ``grid_latlon_lookup.txt``, ``snapshot.json`` and
        ``nodes/<id:06d>_<frame>_<grid_type>.tenv8`` (one file per lookup line).
        """
        snapshot_dir = Path(snapshot_dir)
        info = json.loads((snapshot_dir / "snapshot.json").read_text())
        frame = info.get("reference_frame", "IGS20")
        grid_type = (info.get("grid_types") or ["constant"])[0]
        return cls(
            snapshot_dir / "grid_latlon_lookup.txt",
            snapshot_dir / "nodes",
            filename_template="{grid_id:06d}_{plate}_" + grid_type + ".tenv8",
            plate=frame,
            gridded_type=grid_type,
        )

    def _read_station_data(self) -> gpd.GeoDataFrame:
        df = pd.read_csv(self.lookup_file, sep=r"\s+", names=["id", "lon", "lat"])
        df["lon"] = ((df["lon"] + 180) % 360) - 180
        df["alt"] = 0.0
        return to_point_frame(df)

    def node_path(self, grid_id: str | int, plate: PlateName | None = None) -> Path:
        """Path of one node's ``.tenv8`` file (FileNotFoundError if missing)."""
        name = self.filename_template.format(
            grid_id=int(grid_id), plate=plate or self.plate
        )
        path = self.data_dir / name
        if not path.is_file():
            msg = f"grid node {int(grid_id):06d}: no file {path}"
            raise FileNotFoundError(msg)
        return path

    def _download_file(
        self,
        grid_id: str,
        plate: PlateName = "IGS14",
        output_dir: Path | None = None,  # noqa: ARG002
        session: requests.Session | None = None,  # noqa: ARG002
        version: Literal["0.1", "0.3"] = "0.3",  # noqa: ARG002
        gridded_type: Literal["constant", "variable"] = "variable",  # noqa: ARG002
    ) -> Path:
        # The inherited readers fetch through here: return the node's own file,
        # never UNR's file for the same id (UnrGridSource's IGS14 default -> ours).
        return self.node_path(grid_id, self.plate if plate == "IGS14" else plate)

    def timeseries(
        self,
        station_id: str,
        /,
        frame: Literal["ENU", "XYZ"] = "ENU",
        start_date: str | None = None,
        end_date: str | None = None,
        zero_by: Literal["mean", "start", "none"] = "mean",
        download_if_missing: bool = True,  # noqa: ARG002 (local only)
        *,
        plate: PlateName | None = None,
    ) -> pd.DataFrame:
        """Load one node's time series (ENU, meters) from its local file.

        Same output as `UnrGridSource.timeseries`; ``download_if_missing`` is
        ignored, nothing is ever downloaded.
        """
        return super().timeseries(
            station_id,
            frame=frame,
            start_date=start_date,
            end_date=end_date,
            zero_by=zero_by,
            download_if_missing=True,  # routes through _download_file: the local file
            plate=plate or self.plate,
        )

    def node_series(
        self, grid_id: int, *, plate: PlateName | None = None
    ) -> pd.DataFrame:
        """Load one node in mm, indexed by calendar day.

        See `UnrGridSource.node_series`.
        """
        return super().node_series(grid_id, plate=plate or self.plate)

    def download_data_files(self, *args, **kwargs) -> list[Path]:
        """Not available: a custom grid is read from disk only."""
        msg = "CustomGridSource reads local files only; nothing to download"
        raise NotImplementedError(msg)

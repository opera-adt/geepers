# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Copyright (c) 2023 Scott Staniewicz. All rights reserved.

geepers: Download GPS data and compare to InSAR
"""

from __future__ import annotations

import importlib
from typing import Any

from geepers._version import version as __version__

# Public names that used to be re-exported eagerly from the analysis modules.
# They are resolved on first access (PEP 562) so that `import geepers` and the
# GNSS retrieval core do not pull in xarray/rasterio (io), geopandas (gps) or
# pandera (schemas, uncertainty). `from geepers import XarrayReader` still
# works; it just imports geepers.io at that moment.
_LAZY_EXPORTS: dict[str, str] = {
    # geepers.gps (deprecated station helpers)
    "download_station_data": "geepers.gps",
    "get_stations_within_image": "geepers.gps",
    "load_station_enu": "geepers.gps",
    "read_station_llas": "geepers.gps",
    "station_lonlat": "geepers.gps",
    # geepers.io
    "XarrayReader": "geepers.io",
    # geepers.schemas
    "GPSUncertaintySchema": "geepers.schemas",
    "GridCellSchema": "geepers.schemas",
    "RatesSchema": "geepers.schemas",
    "StationObservationSchema": "geepers.schemas",
    "StationSchema": "geepers.schemas",
    # geepers.uncertainty
    "build_covariance_matrix": "geepers.uncertainty",
    "get_sigma_los": "geepers.uncertainty",
    "get_sigma_los_df": "geepers.uncertainty",
}

__all__ = [
    "GPSUncertaintySchema",
    "GridCellSchema",
    "RatesSchema",
    "StationObservationSchema",
    "StationSchema",
    "XarrayReader",
    "__version__",
    "build_covariance_matrix",
    "download_station_data",
    "get_sigma_los",
    "get_sigma_los_df",
    "get_stations_within_image",
    "load_station_enu",
    "read_station_llas",
    "station_lonlat",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        msg = f"module 'geepers' has no attribute {name!r}"
        raise AttributeError(msg)
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value  # cache so the lookup happens once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

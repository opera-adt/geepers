"""Strain-rate and rotation fields from gridded horizontal velocity fields.

Differentiates a gridded east/north velocity field (e.g. the output of
`geepers.collocation` or `geepers.gps_imaging`) into the 2D infinitesimal
strain-rate tensor and rotation rate, using spherical-Earth metric factors
to convert per-degree gradients into per-meter gradients.

Outputs (units: 1/yr when velocities are m/yr; multiply by 1e9 for
nanostrain/yr):

- ``exx``, ``eyy``, ``exy`` : strain-rate tensor components
- ``rotation``              : vertical-axis rotation rate
- ``dilatation``            : areal strain rate (exx + eyy)
- ``max_shear``             : maximum shear strain rate
- ``second_invariant``      : sqrt(exx^2 + eyy^2 + 2 exy^2)
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["strain_rate_field"]

EARTH_RADIUS = 6_371_000.0  # meters


def strain_rate_field(
    lon: np.ndarray,
    lat: np.ndarray,
    v_east: np.ndarray,
    v_north: np.ndarray,
) -> xr.Dataset:
    """Compute strain-rate and rotation fields from a gridded velocity field.

    Parameters
    ----------
    lon : np.ndarray
        1D longitudes (degrees) of the grid columns, ascending.
    lat : np.ndarray
        1D latitudes (degrees) of the grid rows.
    v_east, v_north : np.ndarray
        2D velocity component grids with shape ``(len(lat), len(lon))``,
        in meters/year (any consistent unit works; outputs are per year
        of that unit per meter).

    Returns
    -------
    xr.Dataset
        Gridded ``exx``, ``eyy``, ``exy``, ``rotation``, ``dilatation``,
        ``max_shear`` and ``second_invariant`` on (lat, lon) coordinates.

    Examples
    --------
    A uniform velocity field has no strain:

    >>> import numpy as np
    >>> lon, lat = np.linspace(0, 1, 5), np.linspace(0, 1, 4)
    >>> v = np.ones((4, 5))
    >>> ds = strain_rate_field(lon, lat, v, v)
    >>> bool(np.allclose(ds.second_invariant, 0))
    True

    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    v_east = np.asarray(v_east, dtype=float)
    v_north = np.asarray(v_north, dtype=float)
    if v_east.shape != (lat.size, lon.size) or v_north.shape != v_east.shape:
        msg = (
            f"Velocity grids must have shape (len(lat), len(lon)) ="
            f" ({lat.size}, {lon.size}); got {v_east.shape} and {v_north.shape}"
        )
        raise ValueError(msg)

    # Metric conversion: degrees -> meters (dx shrinks with cos(lat))
    deg2m = np.deg2rad(1.0) * EARTH_RADIUS
    cos_lat = np.cos(np.deg2rad(lat))[:, np.newaxis]

    # np.gradient(field, lat_coord, lon_coord) -> (d/dlat, d/dlon)
    dve_dlat, dve_dlon = np.gradient(v_east, lat, lon)
    dvn_dlat, dvn_dlon = np.gradient(v_north, lat, lon)

    dve_dx = dve_dlon / (deg2m * cos_lat)
    dvn_dx = dvn_dlon / (deg2m * cos_lat)
    dve_dy = dve_dlat / deg2m
    dvn_dy = dvn_dlat / deg2m

    exx = dve_dx
    eyy = dvn_dy
    exy = 0.5 * (dve_dy + dvn_dx)
    rotation = 0.5 * (dvn_dx - dve_dy)
    dilatation = exx + eyy
    max_shear = np.sqrt(((exx - eyy) / 2) ** 2 + exy**2)
    second_invariant = np.sqrt(exx**2 + eyy**2 + 2 * exy**2)

    coords = {"lat": lat, "lon": lon}
    dims = ("lat", "lon")
    return xr.Dataset(
        {
            "exx": (dims, exx),
            "eyy": (dims, eyy),
            "exy": (dims, exy),
            "rotation": (dims, rotation),
            "dilatation": (dims, dilatation),
            "max_shear": (dims, max_shear),
            "second_invariant": (dims, second_invariant),
        },
        coords=coords,
        attrs={
            "units": "1/yr (for input velocities in m/yr)",
            "description": "2D infinitesimal strain-rate and rotation fields",
        },
    )

# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Published plate-motion models: poles and rigid-plate velocities.

Ported from Venti's `models/load_itrf.py` tables (plan T14.3a). The DISP-CAL
`plate_motion` layer is built from `plate_velocity_enu`, so the sanity checks
here are geophysical: pole positions and plate speeds at the benchmark sites
must match the well-known ITRF values.
"""

from __future__ import annotations

import numpy as np
import pytest

from geepers.euler import (
    PLATE_CODES,
    EulerPole,
    load_plate_motion_model,
    plate_pole,
    plate_velocity_enu,
)


def test_models_load_with_expected_plates():
    p20 = load_plate_motion_model("ITRF2020-PMM")
    p14 = load_plate_motion_model("ITRF2014-PMM")
    assert {"NOAM", "PCFC", "CARB", "EURA"} <= set(p20)
    assert len(p20) == 13
    assert len(p14) == 11
    assert all(k in p20["NOAM"] for k in ("name", "omega_x", "omega_y", "omega_z"))
    with pytest.raises(ValueError, match="Unknown plate-motion model"):
        load_plate_motion_model("ITRF2008-PMM")  # type: ignore[arg-type]


def test_every_code_maps_to_a_2020_plate():
    plates = load_plate_motion_model()
    assert set(PLATE_CODES.values()) <= set(plates)


def test_north_america_pole():
    """ITRF2020-PMM North America: pole near 86 W, 8 S, 0.187 deg/Myr (0.67 mas/yr).

    (The ITRF2014 pole is at ~88 W, 5 S; the two models differ by ~3 degrees
    in pole latitude, which is why the model is an explicit argument.)
    """
    pole = plate_pole("NA")
    assert isinstance(pole, EulerPole)
    assert pole.lon == pytest.approx(-86.1, abs=1.0)
    assert pole.lat == pytest.approx(-8.35, abs=0.5)
    assert pole.rate == pytest.approx(0.187, abs=0.005)
    pole14 = plate_pole("NA", model="ITRF2014-PMM")
    assert pole14.lat == pytest.approx(-5.2, abs=0.5)
    assert pole14.rate == pytest.approx(0.194, abs=0.005)
    # code and name are the same plate
    assert plate_pole("NOAM").rate == pole.rate


def test_unknown_plate_raises():
    with pytest.raises(ValueError, match="not in ITRF2020-PMM"):
        plate_pole("XX")


@pytest.mark.parametrize(
    ("plate", "lon", "lat", "speed_range", "sign_e", "sign_n"),
    [
        # Houston, North America: slow WSW motion in ITRF
        ("NA", -95.4, 29.7, (8.0, 20.0), -1, -1),
        # Hilo, Pacific plate: fast NW motion (~70 mm/yr)
        ("PA", -155.1, 19.7, (60.0, 80.0), -1, +1),
        # San Juan, Caribbean plate: ENE, ~15-20 mm/yr
        ("CA", -66.1, 18.4, (8.0, 25.0), +1, 0),
    ],
)
def test_benchmark_site_velocities(plate, lon, lat, speed_range, sign_e, sign_n):
    ve, vn, vu = plate_velocity_enu(lon, lat, plate)
    speed = float(np.hypot(ve, vn)[0])
    assert speed_range[0] <= speed <= speed_range[1], speed
    assert np.sign(ve[0]) == sign_e
    if sign_n:
        assert np.sign(vn[0]) == sign_n
    assert vu[0] == 0.0  # rigid rotation has no vertical component


def test_vectorised_over_arrays_and_relative_motion():
    lons = np.array([-120.0, -118.0, -116.0])
    lats = np.array([34.0, 34.0, 34.0])
    ve_na, vn_na, vu = plate_velocity_enu(lons, lats, "NA")
    ve_pa, vn_pa, _ = plate_velocity_enu(lons, lats, "PA")
    assert ve_na.shape == vn_na.shape == vu.shape == (3,)
    # PA relative to NA across southern California: ~45-50 mm/yr to the NW
    rel = np.hypot(ve_pa - ve_na, vn_pa - vn_na)
    assert np.all((rel > 40.0) & (rel < 55.0)), rel

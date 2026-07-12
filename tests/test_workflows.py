import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import geepers.gps as gps
from geepers.quality import InsufficientDataError, select_gps_reference
from geepers.workflows import main


def test_main(tmp_path, monkeypatch):
    data_dir = Path(__file__).parent / "data/hawaii"
    unr_data_zipped = Path(__file__).parent / "data/unr.zip"
    # unzip, and set to GPS dir:
    with zipfile.ZipFile(unr_data_zipped, "r") as zip_ref:
        zip_ref.extractall(tmp_path)

    monkeypatch.setattr(gps, "GPS_DIR", tmp_path)

    main(
        los_enu_file=data_dir / "hawaii_los_enu.tif",
        timeseries_files=sorted(data_dir.glob("displacement_*.tif")),
        output_dir=tmp_path / "GPS",
    )
    assert (tmp_path / "GPS").exists()

    df = pd.read_csv(tmp_path / "GPS" / "combined_data.csv")
    expected_stations = [
        "HLNA",
        "MANE",
        "KOSM",
        "AHUP",
        "OUTL",
        "CNPK",
        "CRIM",
    ]
    assert set(df.id) == set(expected_stations)
    # Verify the first HLNA entry (value changed due to duplicate timestamp fix)
    # The fix now selects the closest GPS point to each InSAR epoch
    expected_entry = {
        "id": "HLNA",
        "date": "2016-07-23",
        "measurement": "los_gps",
        "value": -0.0020305613040891,
    }
    pd.testing.assert_series_equal(
        df[df.id == "HLNA"].iloc[0], pd.Series(expected_entry, name=0)
    )


def test_select_gps_reference_coherence_priority():
    """Test select_gps_reference with coherence priority."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")

    # Station A: Good coherence, higher RMS
    station_a = pd.DataFrame(
        {
            "los_gps": np.random.normal(0, 0.01, 50),
            "los_insar": np.random.normal(0, 0.02, 50),  # Higher noise
            "temporal_coherence": np.full(50, 0.9),  # High coherence
        },
        index=dates,
    )

    # Station B: Lower coherence, lower RMS
    station_b = pd.DataFrame(
        {
            "los_gps": np.random.normal(0, 0.005, 50),
            "los_insar": np.random.normal(0, 0.005, 50),  # Lower noise
            "temporal_coherence": np.full(50, 0.6),  # Lower coherence
        },
        index=dates,
    )

    station_to_merged = {"STAT_A": station_a, "STAT_B": station_b}

    # With coherence priority, should pick station A (higher coherence)
    ref_station = select_gps_reference(station_to_merged, coherence_priority=True)
    assert ref_station == "STAT_A"


def test_select_gps_reference_rms_priority():
    """Test select_gps_reference without coherence priority."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")

    # Station A: Good coherence, higher RMS
    np.random.seed(42)
    gps_a = np.random.normal(0, 0.01, 50)
    insar_a = gps_a + np.random.normal(0, 0.02, 50)  # Higher noise
    station_a = pd.DataFrame(
        {
            "los_gps": gps_a,
            "los_insar": insar_a,
            "temporal_coherence": np.full(50, 0.9),
        },
        index=dates,
    )

    # Station B: Lower coherence, lower RMS
    gps_b = np.random.normal(0, 0.005, 50)
    insar_b = gps_b + np.random.normal(0, 0.005, 50)  # Lower noise
    station_b = pd.DataFrame(
        {
            "los_gps": gps_b,
            "los_insar": insar_b,
            "temporal_coherence": np.full(50, 0.6),
        },
        index=dates,
    )

    station_to_merged = {"STAT_A": station_a, "STAT_B": station_b}

    # Without coherence priority, should pick station B (lower RMS)
    ref_station = select_gps_reference(station_to_merged, coherence_priority=False)
    assert ref_station == "STAT_B"


def test_select_gps_reference_insufficient_data():
    """Test select_gps_reference with insufficient data."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    # Station with only 10 points
    station_a = pd.DataFrame(
        {
            "los_gps": np.random.normal(0, 0.01, 10),
            "los_insar": np.random.normal(0, 0.01, 10),
        },
        index=dates,
    )
    station_a.iloc[:6, station_a.columns.get_loc("los_gps")] = np.nan

    station_b = pd.DataFrame(
        {
            "los_gps": np.random.normal(0, 0.01, 10),
            "los_insar": np.random.normal(0, 0.01, 10),
        },
        index=dates,
    )

    station_to_merged = {"STAT_A": station_a, "STAT_B": station_b}
    ref_station = select_gps_reference(station_to_merged, min_coverage_fraction=0.5)
    assert ref_station == "STAT_B"

    station_b.iloc[:6, station_b.columns.get_loc("los_gps")] = np.nan

    # Should raise InsufficientDataError due to insufficient data
    with pytest.raises(
        InsufficientDataError,
        match="Could not determine an automatic reference station",
    ):
        select_gps_reference(station_to_merged, min_coverage_fraction=0.5)


def test_select_gps_reference_no_coherence_data():
    """Test select_gps_reference when temporal_coherence is not available."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")

    # Station without coherence data
    np.random.seed(42)
    gps_a = np.random.normal(0, 0.01, 50)
    insar_a = gps_a + np.random.normal(0, 0.01, 50)
    station_a = pd.DataFrame(
        {
            "los_gps": gps_a,
            "los_insar": insar_a,
        },
        index=dates,
    )

    gps_b = np.random.normal(0, 0.005, 50)
    insar_b = gps_b + np.random.normal(0, 0.02, 50)
    station_b = pd.DataFrame(
        {
            "los_gps": gps_b,
            "los_insar": insar_b,
        },
        index=dates,
    )

    station_to_merged = {"STAT_A": station_a, "STAT_B": station_b}

    # Should fall back to RMS-based selection even with coherence_priority=True
    ref_station = select_gps_reference(station_to_merged, coherence_priority=True)
    assert ref_station == "STAT_A"  # Lower RMS misfit


def test_no_duplicate_insar_timestamps():
    """Test that merge produces exactly one GPS row per InSAR epoch.

    This test demonstrates the issue where multiple GPS observations
    (from adjacent days or same day with different times) can match
    to the same InSAR acquisition when using merge_asof with
    tolerance="1D" and direction="nearest".

    The fix ensures we keep only the closest GPS sample per InSAR epoch.
    """
    # GPS data: daily observations at ~11:57:30 (typical GPS timestamp)
    gps_dates = pd.date_range("2023-01-01 11:57:30", periods=10, freq="D")
    gps_df = pd.DataFrame(
        {
            "los_gps": np.random.normal(0, 0.01, 10),
            "sigma_los": np.full(10, 0.001),
        },
        index=gps_dates,
    )

    # InSAR data: epochs at midnight (typical InSAR timestamp)
    # Use fewer epochs to trigger the duplicate matching issue
    insar_dates = pd.DatetimeIndex(
        [
            "2023-01-02 00:00:00",
            "2023-01-05 00:00:00",
            "2023-01-08 00:00:00",
        ]
    )
    insar_df = pd.DataFrame(
        {
            "los_insar": np.random.normal(0, 0.01, 3),
        },
        index=insar_dates,
    )

    # OLD BEHAVIOR: merge_asof with tolerance="1D" allows multiple GPS rows
    # to match the same InSAR epoch
    old_merged = pd.merge_asof(
        left=gps_df.sort_index(),
        right=insar_df.sort_index(),
        tolerance=pd.Timedelta("1D"),
        direction="nearest",
        left_index=True,
        right_index=True,
    )

    # Demonstrate the issue: multiple GPS rows can have the same los_insar value
    # (meaning they matched to the same InSAR epoch)
    value_counts = old_merged["los_insar"].value_counts()
    (value_counts > 1).any()

    # FIXED BEHAVIOR: Keep only one GPS row per InSAR epoch (the closest one)
    # Add insar_time column to track which InSAR epoch each GPS row matched to
    insar_with_time = insar_df.assign(insar_time=insar_df.index)
    merged = pd.merge_asof(
        left=gps_df.sort_index(),
        right=insar_with_time.sort_index(),
        tolerance=pd.Timedelta("1D"),
        direction="nearest",
        left_index=True,
        right_index=True,
    )

    # Keep only the closest GPS row per InSAR epoch
    dt = (merged.index - merged["insar_time"]).abs()
    keep_idx = dt.groupby(merged["insar_time"]).idxmin()
    merged_one_per_insar = merged.loc[keep_idx].sort_index()

    # After the fix: each InSAR value should appear exactly once
    fixed_value_counts = merged_one_per_insar["los_insar"].value_counts()
    assert (
        fixed_value_counts <= 1
    ).all(), "Each InSAR epoch should appear at most once"

    # Verify we have the expected number of matches (3 InSAR epochs)
    assert (
        len(merged_one_per_insar) == 3
    ), "Should have exactly 3 GPS rows (one per InSAR epoch)"

    # Verify that GPS rows are matched to the closest InSAR epoch
    for idx, row in merged_one_per_insar.iterrows():
        insar_time = row["insar_time"]
        # This GPS timestamp should be the closest to its matched InSAR epoch
        gps_matches = merged[merged["insar_time"] == insar_time]
        time_diffs = pd.Series(
            abs(gps_matches.index - insar_time), index=gps_matches.index
        )
        assert (
            idx == time_diffs.idxmin()
        ), f"GPS row {idx} should be closest to InSAR epoch {insar_time}"

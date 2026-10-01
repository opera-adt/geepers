import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import geepers.gps as gps
from geepers.quality import InsufficientDataError, select_gps_reference
from geepers.workflows import main, merge_gps_insar


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
    expected_entry = {
        "id": "HLNA",
        "date": "2016-07-23",
        "measurement": "los_gps",
        "value": 0.0010796103397325,
    }
    # Compare row content independent of its absolute position in the CSV
    # (station block order depends on the spatial-filter method).
    pd.testing.assert_series_equal(
        df[df.id == "HLNA"].iloc[0],
        pd.Series(expected_entry, name=0),
        check_names=False,
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


@pytest.mark.parametrize("gps_hour", [0, 12])
def test_merge_gps_insar_one_row_per_epoch(gps_hour):
    """Regression: each InSAR epoch was copied onto 2-3 adjacent GPS days."""
    gps_dates = pd.date_range("2020-01-01", periods=40, freq="D") + pd.Timedelta(
        hours=gps_hour
    )
    df_gps = pd.DataFrame({"los_gps": np.arange(40.0)}, index=gps_dates)
    insar_dates = pd.date_range("2020-01-05", periods=3, freq="12D")
    df_insar = pd.DataFrame(
        {"los_insar": [1.0, 2.0, 3.0], "temporal_coherence": 0.9}, index=insar_dates
    )

    merged = merge_gps_insar(df_gps, df_insar)

    assert merged.index.equals(df_gps.index)
    assert merged["los_gps"].tolist() == df_gps["los_gps"].tolist()
    matched = merged.dropna(subset="los_insar")
    assert matched["los_insar"].tolist() == [1.0, 2.0, 3.0]
    assert (abs(matched.index - insar_dates) <= pd.Timedelta("12h")).all()


def test_merge_gps_insar_respects_tolerance():
    df_gps = pd.DataFrame(
        {"los_gps": [0.0, 1.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"])
    )
    df_insar = pd.DataFrame({"los_insar": [5.0]}, index=pd.to_datetime(["2020-01-10"]))
    assert merge_gps_insar(df_gps, df_insar)["los_insar"].isna().all()

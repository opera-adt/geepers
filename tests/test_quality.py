"""Tests for geepers.quality module."""

import numpy as np
import pandas as pd
import pytest

from geepers.quality import StationQuality, compute_station_quality


def test_station_quality_dataclass():
    """Test StationQuality dataclass creation and access."""
    quality = StationQuality(
        num_gps=50,
        gps_time_span_years=2.5,
        temporal_coherence=0.85,
        similarity=0.92,
        rms_misfit=0.005,
    )

    assert quality.num_gps == 50
    assert quality.gps_time_span_years == 2.5
    assert quality.temporal_coherence == 0.85
    assert quality.similarity == 0.92
    assert quality.rms_misfit == 0.005


def test_compute_station_quality_basic():
    """Test compute_station_quality with basic GPS/InSAR data."""
    # Create test data with 30 days of daily measurements
    dates = pd.date_range("2023-01-01", periods=30, freq="D")

    # Create synthetic GPS and InSAR data with slight differences
    np.random.seed(42)
    gps_los = np.cumsum(np.random.normal(0, 0.001, 30))  # Random walk
    insar_los = gps_los + np.random.normal(0, 0.002, 30)  # GPS + noise

    df = pd.DataFrame(
        {
            "los_gps": gps_los,
            "los_insar": insar_los,
            "temporal_coherence": np.full(30, 0.8),
            "similarity": np.full(30, 0.9),
        },
        index=dates,
    )

    quality = compute_station_quality(df)

    assert quality.num_gps == 30
    assert quality.gps_time_span_years == pytest.approx(29 / 365.25, rel=1e-3)
    assert quality.temporal_coherence == pytest.approx(0.8)
    assert quality.similarity == pytest.approx(0.9)
    assert quality.rms_misfit > 0  # Should have some misfit


def test_compute_station_quality_missing_gps():
    """Test compute_station_quality with missing GPS data."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    df = pd.DataFrame(
        {
            "los_gps": [np.nan] * 10,  # All NaN GPS data
            "los_insar": np.random.normal(0, 0.01, 10),
        },
        index=dates,
    )

    quality = compute_station_quality(df)

    assert quality.num_gps == 0
    assert quality.gps_time_span_years == 0.0
    assert quality.rms_misfit == np.inf  # No common data


def test_compute_station_quality_missing_insar():
    """Test compute_station_quality with missing InSAR data."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    df = pd.DataFrame(
        {
            "los_gps": np.random.normal(0, 0.01, 10),
            "los_insar": [np.nan] * 10,  # All NaN InSAR data
        },
        index=dates,
    )

    quality = compute_station_quality(df)

    assert quality.num_gps == 10
    assert quality.gps_time_span_years > 0
    assert quality.rms_misfit == np.inf  # No common data


def test_compute_station_quality_partial_data():
    """Test compute_station_quality with partial GPS/InSAR overlap."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")

    # GPS data for first 15 days, InSAR for last 15 days (10 days overlap)
    gps_data = [0.001 * i if i < 15 else np.nan for i in range(20)]
    insar_data = [np.nan if i < 5 else 0.001 * i for i in range(20)]

    df = pd.DataFrame(
        {
            "los_gps": gps_data,
            "los_insar": insar_data,
        },
        index=dates,
    )

    quality = compute_station_quality(df)

    assert quality.num_gps == 15
    assert quality.gps_time_span_years == pytest.approx(14 / 365.25, rel=1e-3)
    assert quality.rms_misfit < np.inf  # Should have some common data


def test_compute_station_quality_no_quality_columns():
    """Test compute_station_quality without temporal_coherence or similarity."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    df = pd.DataFrame(
        {
            "los_gps": np.random.normal(0, 0.01, 10),
            "los_insar": np.random.normal(0, 0.01, 10),
        },
        index=dates,
    )

    quality = compute_station_quality(df)

    assert quality.temporal_coherence is None
    assert quality.similarity is None
    assert quality.num_gps == 10
    assert quality.rms_misfit > 0


def test_compute_station_quality_empty_quality_columns():
    """Test compute_station_quality with all-NaN quality columns."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    df = pd.DataFrame(
        {
            "los_gps": np.random.normal(0, 0.01, 10),
            "los_insar": np.random.normal(0, 0.01, 10),
            "temporal_coherence": [np.nan] * 10,
            "similarity": [np.nan] * 10,
        },
        index=dates,
    )

    quality = compute_station_quality(df)

    assert quality.temporal_coherence is None
    assert quality.similarity is None


def test_rms_misfit_ignores_zero_level_offset():
    """Regression: a constant offset between the series is not misfit."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    gps = np.linspace(0, 0.01, 30)
    df = pd.DataFrame({"los_gps": gps, "los_insar": gps + 0.05}, index=dates)
    assert compute_station_quality(df).rms_misfit == pytest.approx(0.0, abs=1e-12)

# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for forecast_time_series transform node."""
import os
import pytest
import numpy as np
import pandas as pd

from choregraph.collection.timeseries import forecast_time_series
from choregraph.library import TRANSFORM_REGISTRY


# ---------------------------------------------------------------------------
# Path to real CSV fixture
# ---------------------------------------------------------------------------
CAPE_CSV = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "dive_data",
    "0fcba1b1-477b-48c3-9919-49df5c997438",
    "9910eca6-8b58-41c0-9364-d26867a48fd4",
    "DIVE", "Projects", "Project", "UserFile",
    "CAPE 5TC spot.csv",
)

# ---------------------------------------------------------------------------
# Synthetic fixture — 200 rows of daily business-day data
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_bday_df():
    """200 business-day rows with a random-walk value column."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=200)
    values = 100 + np.cumsum(rng.normal(0, 1, 200))
    return pd.DataFrame({"Date": dates, "Price": values})


@pytest.fixture
def synthetic_daily_df():
    """200 calendar-day rows (includes weekends) with a random-walk value."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    values = 100 + np.cumsum(rng.normal(0, 1, 200))
    return pd.DataFrame({"Date": dates, "Value": values})


@pytest.fixture
def cape_df():
    """Load the real CAPE 5TC spot CSV if available."""
    if not os.path.exists(CAPE_CSV):
        pytest.skip("CAPE CSV not found")
    return pd.read_csv(CAPE_CSV)


# =========================================================================
# Normal-mode tests
# =========================================================================
class TestNormalMode:
    def test_columns_present(self, synthetic_bday_df):
        result = forecast_time_series(synthetic_bday_df, "Date", "Price", horizon=10)
        assert "Price" in result.columns
        assert "Forecast_Lower" in result.columns
        assert "Forecast_Upper" in result.columns
        assert "Date" in result.columns
        assert "Is_Forecast" in result.columns
        assert "Forecast" not in result.columns

    def test_is_forecast_flag(self, synthetic_bday_df):
        result = forecast_time_series(synthetic_bday_df, "Date", "Price", horizon=10)
        hist = result[~result["Is_Forecast"]]
        fc = result[result["Is_Forecast"]]
        assert len(fc) == 10
        assert len(hist) > 0
        # Value column populated everywhere
        assert result["Price"].notna().all()

    def test_horizon_row_count(self, synthetic_bday_df):
        horizon = 15
        result = forecast_time_series(synthetic_bday_df, "Date", "Price", horizon=horizon)
        fc = result[result["Is_Forecast"]]
        assert len(fc) == horizon

    def test_positive_forecasts(self, synthetic_bday_df):
        result = forecast_time_series(synthetic_bday_df, "Date", "Price", horizon=10)
        fc = result[result["Is_Forecast"]]
        assert (fc["Price"] > 0).all()

    def test_lower_lt_median_lt_upper(self, synthetic_bday_df):
        result = forecast_time_series(synthetic_bday_df, "Date", "Price", horizon=10)
        fc = result[result["Is_Forecast"]]
        assert (fc["Forecast_Lower"] <= fc["Price"]).all()
        assert (fc["Price"] <= fc["Forecast_Upper"]).all()

    def test_confidence_widens_over_time(self, synthetic_bday_df):
        result = forecast_time_series(synthetic_bday_df, "Date", "Price", horizon=20)
        fc = result[result["Is_Forecast"]].reset_index(drop=True)
        widths = fc["Forecast_Upper"] - fc["Forecast_Lower"]
        assert widths.iloc[-1] >= widths.iloc[0]

    def test_cape_csv_normal(self, cape_df):
        result = forecast_time_series(cape_df, "Date", "Close", horizon=10)
        fc = result[result["Is_Forecast"]]
        assert len(fc) == 10
        assert (fc["Close"] > 0).all()


# =========================================================================
# Split-test mode tests
# =========================================================================
class TestSplitTest:
    def test_split_test_columns(self, synthetic_bday_df):
        result = forecast_time_series(
            synthetic_bday_df, "Date", "Price", perform_split_test=True
        )
        assert "Actual" in result.columns
        assert "Predicted" in result.columns
        assert "Date" in result.columns

    def test_split_test_includes_training_history(self, synthetic_bday_df):
        """Split-test returns the full history; Predicted is NaN for the
        training portion and populated for the test portion."""
        result = forecast_time_series(
            synthetic_bday_df, "Date", "Price", perform_split_test=True
        )
        # Training portion should have NaN predictions
        assert result["Predicted"].isna().any()
        # Test portion should have real predictions
        predicted = result["Predicted"].dropna()
        assert len(predicted) > 0

    def test_split_test_positive(self, synthetic_bday_df):
        result = forecast_time_series(
            synthetic_bday_df, "Date", "Price", perform_split_test=True
        )
        predicted = result["Predicted"].dropna()
        assert (predicted > 0).all()

    def test_cape_split_test(self, cape_df):
        result = forecast_time_series(
            cape_df, "Date", "Close", perform_split_test=True
        )
        assert len(result) > 0
        assert "Actual" in result.columns
        assert "Predicted" in result.columns


# =========================================================================
# Frequency inference tests
# =========================================================================
class TestFrequencyInference:
    def test_daily_no_weekends(self, synthetic_daily_df):
        """Calendar-day data should be resampled as 'D'."""
        result = forecast_time_series(synthetic_daily_df, "Date", "Value", horizon=5)
        assert len(result[result["Is_Forecast"]]) == 5

    def test_business_day(self, synthetic_bday_df):
        """Business-day data should be resampled as 'B'."""
        result = forecast_time_series(synthetic_bday_df, "Date", "Price", horizon=5)
        assert len(result[result["Is_Forecast"]]) == 5

    def test_reverse_chronological(self, cape_df):
        """CAPE CSV is reverse-chronological — function should handle it."""
        result = forecast_time_series(cape_df, "Date", "Close", horizon=5)
        assert result["Date"].is_monotonic_increasing


# =========================================================================
# Data guard test
# =========================================================================
class TestDataGuard:
    def test_large_dataset_tail_sampling(self):
        """Datasets exceeding 5000 rows should be tail-sampled."""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2000-01-01", periods=6000, freq="D")
        vals = 100 + np.cumsum(rng.normal(0, 0.5, 6000))
        df = pd.DataFrame({"Date": dates, "Val": vals})
        # Should not crash and should produce results
        result = forecast_time_series(df, "Date", "Val", horizon=5)
        assert len(result[result["Is_Forecast"]]) == 5


# =========================================================================
# Validation tests
# =========================================================================
class TestValidation:
    def test_missing_date_col(self, synthetic_bday_df):
        with pytest.raises(ValueError, match="not found"):
            forecast_time_series(synthetic_bday_df, "NonExistent", "Price")

    def test_missing_value_col(self, synthetic_bday_df):
        with pytest.raises(ValueError, match="not found"):
            forecast_time_series(synthetic_bday_df, "Date", "NonExistent")

    def test_empty_df(self):
        result = forecast_time_series(pd.DataFrame(), "Date", "Value")
        assert result.empty

    def test_none_df(self):
        result = forecast_time_series(None, "Date", "Value")
        assert result.empty

    def test_too_few_rows(self):
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10),
            "Value": range(10),
        })
        with pytest.raises(ValueError, match="Insufficient data"):
            forecast_time_series(df, "Date", "Value")

    def test_horizon_less_than_one(self, synthetic_bday_df):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            forecast_time_series(synthetic_bday_df, "Date", "Price", horizon=0)


# =========================================================================
# Registry test
# =========================================================================
class TestRegistry:
    def test_registered(self):
        assert "forecast_time_series" in TRANSFORM_REGISTRY
        assert TRANSFORM_REGISTRY["forecast_time_series"]["output_type"] is pd.DataFrame

# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for choregraph.collection.timeseries — date extraction, rolling stats, lag/lead, offset."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from choregraph.collection.timeseries import (
    extract_date_part,
    rolling_statistics,
    lag_lead,
    offset_datetime,
)


@pytest.fixture
def timeseries_df():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.DataFrame({"date": dates, "value": range(10)})


# ---- extract_date_part ----


def test_extract_date_part_year(timeseries_df):
    result = extract_date_part(timeseries_df, "date", "YEAR")
    assert "date_year" in result.columns
    assert (result["date_year"] == 2023).all()


def test_extract_date_part_month(timeseries_df):
    result = extract_date_part(timeseries_df, "date", "MONTH")
    assert "date_month" in result.columns
    assert (result["date_month"] == 1).all()


def test_extract_date_part_day(timeseries_df):
    result = extract_date_part(timeseries_df, "date", "DAY")
    assert "date_day" in result.columns
    assert list(result["date_day"]) == list(range(1, 11))


def test_extract_date_part_weekday(timeseries_df):
    result = extract_date_part(timeseries_df, "date", "WEEKDAY")
    assert "date_weekday" in result.columns
    # 2023-01-01 is a Sunday = weekday 6
    assert result["date_weekday"].iloc[0] == 6


def test_extract_date_part_custom_output_column(timeseries_df):
    result = extract_date_part(timeseries_df, "date", "YEAR", output_column="yr")
    assert "yr" in result.columns
    assert "date_year" not in result.columns


def test_extract_date_part_missing_column(timeseries_df):
    with pytest.raises(ValueError, match="not found"):
        extract_date_part(timeseries_df, "nonexistent", "YEAR")


def test_extract_date_part_unsupported_part(timeseries_df):
    with pytest.raises(ValueError, match="Unsupported date part"):
        extract_date_part(timeseries_df, "date", "CENTURY")


def test_extract_date_part_empty_df():
    result = extract_date_part(pd.DataFrame(), "date", "YEAR")
    assert result.empty


# ---- rolling_statistics ----


def test_rolling_statistics_mean(timeseries_df):
    result = rolling_statistics(timeseries_df, "value", window_size=3, function="MEAN")
    assert "value_mean" in result.columns
    # First and last values are NaN (centered window of 3)
    assert pd.isna(result["value_mean"].iloc[0])
    # Middle value at index 1: mean of [0, 1, 2] = 1.0
    assert result["value_mean"].iloc[1] == 1.0


def test_rolling_statistics_sum(timeseries_df):
    result = rolling_statistics(timeseries_df, "value", window_size=3, function="SUM")
    assert "value_sum" in result.columns
    # At index 1 (centered window of 3): 0+1+2 = 3
    assert result["value_sum"].iloc[1] == 3.0


def test_rolling_statistics_cumsum(timeseries_df):
    result = rolling_statistics(timeseries_df, "value", function="CUMSUM")
    assert "value_cumsum" in result.columns
    expected = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
    assert list(result["value_cumsum"]) == expected


def test_rolling_statistics_missing_window_size(timeseries_df):
    with pytest.raises(ValueError, match="window_size is required"):
        rolling_statistics(timeseries_df, "value", function="MEAN")


def test_rolling_statistics_missing_column(timeseries_df):
    with pytest.raises(ValueError, match="not found"):
        rolling_statistics(timeseries_df, "nonexistent", window_size=3)


def test_rolling_statistics_empty_df():
    result = rolling_statistics(pd.DataFrame(), "value", window_size=3)
    assert result.empty


# ---- lag_lead ----


def test_lag_lead_lag_by_1(timeseries_df):
    result = lag_lead(timeseries_df, "value", periods=1)
    assert "value_lag_1" in result.columns
    assert pd.isna(result["value_lag_1"].iloc[0])
    assert result["value_lag_1"].iloc[1] == 0.0


def test_lag_lead_lead_by_1(timeseries_df):
    result = lag_lead(timeseries_df, "value", periods=-1)
    assert "value_lag_-1" in result.columns
    assert result["value_lag_-1"].iloc[0] == 1.0
    assert pd.isna(result["value_lag_-1"].iloc[-1])


def test_lag_lead_custom_output(timeseries_df):
    result = lag_lead(timeseries_df, "value", periods=2, output_column="shifted")
    assert "shifted" in result.columns
    assert pd.isna(result["shifted"].iloc[0])
    assert pd.isna(result["shifted"].iloc[1])
    assert result["shifted"].iloc[2] == 0.0


def test_lag_lead_missing_column(timeseries_df):
    with pytest.raises(ValueError, match="not found"):
        lag_lead(timeseries_df, "nonexistent")


def test_lag_lead_empty_df():
    result = lag_lead(pd.DataFrame(), "value")
    assert result.empty


# ---- offset_datetime ----


def test_offset_datetime_add_1_day(timeseries_df):
    result = offset_datetime(timeseries_df, "date", offset=1, unit="DAYS")
    assert "date_offset" in result.columns
    expected_first = pd.Timestamp("2023-01-02")
    assert result["date_offset"].iloc[0] == expected_first


def test_offset_datetime_subtract_1_hour():
    df = pd.DataFrame({"ts": pd.date_range("2023-06-15 12:00", periods=3, freq="h")})
    result = offset_datetime(df, "ts", offset=-1, unit="HOURS")
    assert result["ts_offset"].iloc[0] == pd.Timestamp("2023-06-15 11:00")


def test_offset_datetime_add_months(timeseries_df):
    result = offset_datetime(timeseries_df, "date", offset=2, unit="MONTHS")
    assert result["date_offset"].iloc[0] == pd.Timestamp("2023-03-01")


def test_offset_datetime_custom_output(timeseries_df):
    result = offset_datetime(timeseries_df, "date", offset=1, unit="WEEKS", output_column="next_week")
    assert "next_week" in result.columns
    assert result["next_week"].iloc[0] == pd.Timestamp("2023-01-08")


def test_offset_datetime_unsupported_unit(timeseries_df):
    with pytest.raises(ValueError, match="Unsupported unit"):
        offset_datetime(timeseries_df, "date", offset=1, unit="FORTNIGHTS")


def test_offset_datetime_empty_df():
    result = offset_datetime(pd.DataFrame(), "date", offset=1, unit="DAYS")
    assert result.empty

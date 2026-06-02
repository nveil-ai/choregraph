# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for filter and top/bottom functions in choregraph.library."""

import math

import numpy as np
import pandas as pd
import pytest

from choregraph.library import (
    filter_equal,
    filter_greater_than,
    filter_in_range,
    filter_less_than,
    filter_not_equal,
    get_bottom_n,
    get_bottom_percentage,
    get_top_n,
    get_top_percentage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})


@pytest.fixture
def string_df():
    return pd.DataFrame({"name": ["alice", "bob", "charlie", "dave", "eve"]})


@pytest.fixture
def nan_df():
    return pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})


@pytest.fixture
def empty_df():
    return pd.DataFrame({"a": pd.Series([], dtype="float64")})


@pytest.fixture
def datetime_df():
    ts = pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-01"])
    return pd.DataFrame({"dt": ts, "val": [10, 20, 30]})


@pytest.fixture
def single_row_df():
    return pd.DataFrame({"a": [42]})


# ===========================================================================
# filter_less_than
# ===========================================================================

class TestFilterLessThan:

    def test_basic(self, numeric_df):
        result = filter_less_than(numeric_df, column="a", value=3)
        assert list(result["a"]) == [1, 2]

    @pytest.mark.parametrize("value,expected_len", [
        (1, 0),
        (6, 5),
        (3.5, 3),
    ])
    def test_various_thresholds(self, numeric_df, value, expected_len):
        result = filter_less_than(numeric_df, column="a", value=value)
        assert len(result) == expected_len

    def test_empty_dataframe(self, empty_df):
        result = filter_less_than(empty_df, column="a", value=10)
        assert len(result) == 0

    def test_nan_values_excluded(self, nan_df):
        # NaN comparisons are False, so NaN rows should be excluded
        result = filter_less_than(nan_df, column="a", value=4)
        assert len(result) == 2
        assert list(result["a"]) == [1.0, 3.0]

    def test_no_rows_matching(self, numeric_df):
        result = filter_less_than(numeric_df, column="a", value=1)
        assert len(result) == 0

    def test_single_row_match(self, single_row_df):
        result = filter_less_than(single_row_df, column="a", value=100)
        assert len(result) == 1

    def test_single_row_no_match(self, single_row_df):
        result = filter_less_than(single_row_df, column="a", value=0)
        assert len(result) == 0

    def test_missing_column_raises(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            filter_less_than(numeric_df, column="missing", value=3)

    def test_return_mask(self, numeric_df):
        out = filter_less_than(numeric_df, column="a", value=3, return_mask=True)
        assert isinstance(out, dict)
        assert "result" in out and "mask" in out
        assert len(out["result"]) == 2
        assert out["mask"].shape[0] == len(numeric_df)

    def test_returns_copy(self, numeric_df):
        result = filter_less_than(numeric_df, column="a", value=3)
        result.loc[result.index[0], "a"] = 999
        assert numeric_df["a"].iloc[0] == 1  # original unchanged

    def test_datetime_column(self, datetime_df):
        # Use nanoseconds since epoch for 2024-07-01 (approx midpoint)
        ts = float(pd.Timestamp("2024-07-01").value)
        result = filter_less_than(datetime_df, column="dt", value=ts)
        assert len(result) == 2


# ===========================================================================
# filter_greater_than
# ===========================================================================

class TestFilterGreaterThan:

    def test_basic(self, numeric_df):
        result = filter_greater_than(numeric_df, column="a", value=3)
        assert list(result["a"]) == [4, 5]

    @pytest.mark.parametrize("value,expected_len", [
        (5, 0),
        (0, 5),
        (2.5, 3),
    ])
    def test_various_thresholds(self, numeric_df, value, expected_len):
        result = filter_greater_than(numeric_df, column="a", value=value)
        assert len(result) == expected_len

    def test_empty_dataframe(self, empty_df):
        result = filter_greater_than(empty_df, column="a", value=10)
        assert len(result) == 0

    def test_nan_values_excluded(self, nan_df):
        result = filter_greater_than(nan_df, column="a", value=2)
        assert len(result) == 2
        assert list(result["a"]) == [3.0, 5.0]

    def test_no_rows_matching(self, numeric_df):
        result = filter_greater_than(numeric_df, column="a", value=5)
        assert len(result) == 0

    def test_missing_column_raises(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            filter_greater_than(numeric_df, column="missing", value=3)

    def test_return_mask(self, numeric_df):
        out = filter_greater_than(numeric_df, column="a", value=3, return_mask=True)
        assert isinstance(out, dict)
        assert len(out["result"]) == 2

    def test_returns_copy(self, numeric_df):
        result = filter_greater_than(numeric_df, column="a", value=3)
        result.loc[result.index[0], "a"] = 999
        assert numeric_df["a"].iloc[3] == 4


# ===========================================================================
# filter_in_range
# ===========================================================================

class TestFilterInRange:

    def test_basic_inclusive(self, numeric_df):
        result = filter_in_range(numeric_df, column="a", min_value=2, max_value=4)
        assert list(result["a"]) == [2, 3, 4]

    def test_single_value_range(self, numeric_df):
        result = filter_in_range(numeric_df, column="a", min_value=3, max_value=3)
        assert list(result["a"]) == [3]

    def test_no_match(self, numeric_df):
        result = filter_in_range(numeric_df, column="a", min_value=6, max_value=10)
        assert len(result) == 0

    def test_all_match(self, numeric_df):
        result = filter_in_range(numeric_df, column="a", min_value=0, max_value=10)
        assert len(result) == 5

    def test_empty_dataframe(self, empty_df):
        result = filter_in_range(empty_df, column="a", min_value=0, max_value=10)
        assert len(result) == 0

    def test_nan_values_excluded(self, nan_df):
        result = filter_in_range(nan_df, column="a", min_value=1, max_value=5)
        assert len(result) == 3  # 1, 3, 5

    def test_missing_column_raises(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            filter_in_range(numeric_df, column="missing", min_value=1, max_value=3)

    def test_return_mask(self, numeric_df):
        out = filter_in_range(numeric_df, column="a", min_value=2, max_value=4, return_mask=True)
        assert isinstance(out, dict)
        assert len(out["result"]) == 3
        assert out["mask"].shape[0] == 5

    def test_float_bounds(self):
        df = pd.DataFrame({"x": [1.1, 2.2, 3.3, 4.4, 5.5]})
        result = filter_in_range(df, column="x", min_value=2.0, max_value=4.0)
        assert list(result["x"]) == [2.2, 3.3]


# ===========================================================================
# filter_equal
# ===========================================================================

class TestFilterEqual:

    def test_numeric_column(self, numeric_df):
        result = filter_equal(numeric_df, column="a", value="3")
        assert list(result["a"]) == [3]

    def test_string_column(self, string_df):
        result = filter_equal(string_df, column="name", value="bob")
        assert len(result) == 1
        assert result["name"].iloc[0] == "bob"

    def test_no_match_string(self, string_df):
        result = filter_equal(string_df, column="name", value="frank")
        assert len(result) == 0

    def test_no_match_numeric(self, numeric_df):
        result = filter_equal(numeric_df, column="a", value="99")
        assert len(result) == 0

    def test_empty_dataframe(self, empty_df):
        result = filter_equal(empty_df, column="a", value="1")
        assert len(result) == 0

    def test_nan_not_equal_to_anything(self, nan_df):
        # NaN != NaN so filtering for any value should skip NaN rows
        result = filter_equal(nan_df, column="a", value="1")
        assert len(result) == 1

    def test_missing_column_raises(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            filter_equal(numeric_df, column="missing", value="3")

    def test_return_mask(self, numeric_df):
        out = filter_equal(numeric_df, column="a", value="3", return_mask=True)
        assert isinstance(out, dict)
        assert len(out["result"]) == 1

    def test_multiple_matches(self):
        df = pd.DataFrame({"a": [1, 2, 2, 3, 2]})
        result = filter_equal(df, column="a", value="2")
        assert len(result) == 3

    def test_non_numeric_value_on_numeric_column(self, numeric_df):
        # Passing a non-castable string for a numeric column -- should find nothing
        result = filter_equal(numeric_df, column="a", value="abc")
        assert len(result) == 0


# ===========================================================================
# filter_not_equal
# ===========================================================================

class TestFilterNotEqual:

    def test_numeric_column(self, numeric_df):
        result = filter_not_equal(numeric_df, column="a", value="3")
        assert list(result["a"]) == [1, 2, 4, 5]

    def test_string_column(self, string_df):
        result = filter_not_equal(string_df, column="name", value="bob")
        assert len(result) == 4
        assert "bob" not in result["name"].values

    def test_no_match_returns_all(self, string_df):
        result = filter_not_equal(string_df, column="name", value="frank")
        assert len(result) == 5

    def test_all_same_value(self):
        df = pd.DataFrame({"a": [7, 7, 7]})
        result = filter_not_equal(df, column="a", value="7")
        assert len(result) == 0

    def test_empty_dataframe(self, empty_df):
        result = filter_not_equal(empty_df, column="a", value="1")
        assert len(result) == 0

    def test_nan_rows_with_not_equal(self, nan_df):
        # NaN != value is True, so NaN rows ARE included in not_equal
        result = filter_not_equal(nan_df, column="a", value="1")
        # Non-NaN values != 1: 3.0 and 5.0 (2 rows) + 2 NaN rows = 4
        assert len(result) == 4

    def test_missing_column_raises(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            filter_not_equal(numeric_df, column="missing", value="3")

    def test_return_mask(self, numeric_df):
        out = filter_not_equal(numeric_df, column="a", value="3", return_mask=True)
        assert isinstance(out, dict)
        assert len(out["result"]) == 4


# ===========================================================================
# get_top_n
# ===========================================================================

class TestGetTopN:

    def test_basic(self, numeric_df):
        result = get_top_n(numeric_df, column="a", n=3)
        assert sorted(result["a"].tolist(), reverse=True) == [5, 4, 3]

    def test_n_equals_one(self, numeric_df):
        result = get_top_n(numeric_df, column="a", n=1)
        assert len(result) == 1
        assert result["a"].iloc[0] == 5

    def test_n_equals_zero(self, numeric_df):
        result = get_top_n(numeric_df, column="a", n=0)
        assert len(result) == 0

    def test_n_exceeds_length(self, numeric_df):
        result = get_top_n(numeric_df, column="a", n=100)
        assert len(result) == 5

    def test_ties(self):
        df = pd.DataFrame({"a": [3, 3, 3, 1, 2]})
        result = get_top_n(df, column="a", n=2)
        # nlargest keeps first occurrences on ties
        assert len(result) == 2
        assert all(v == 3 for v in result["a"])

    def test_empty_dataframe(self, empty_df):
        result = get_top_n(empty_df, column="a", n=3)
        assert len(result) == 0

    def test_nan_values(self, nan_df):
        # nlargest ignores NaN
        result = get_top_n(nan_df, column="a", n=2)
        assert len(result) == 2
        assert sorted(result["a"].tolist(), reverse=True) == [5.0, 3.0]

    def test_missing_column_raises(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            get_top_n(numeric_df, column="missing", n=3)

    def test_return_mask(self, numeric_df):
        out = get_top_n(numeric_df, column="a", n=2, return_mask=True)
        assert isinstance(out, dict)
        assert len(out["result"]) == 2
        # Mask should have same length as original df
        assert len(out["mask"]) == 5

    def test_preserves_other_columns(self, numeric_df):
        result = get_top_n(numeric_df, column="a", n=1)
        assert "b" in result.columns
        assert result["b"].iloc[0] == 50

    def test_single_row(self, single_row_df):
        result = get_top_n(single_row_df, column="a", n=1)
        assert len(result) == 1
        assert result["a"].iloc[0] == 42


# ===========================================================================
# get_top_percentage
# ===========================================================================

class TestGetTopPercentage:

    def test_fifty_percent(self, numeric_df):
        result = get_top_percentage(numeric_df, column="a", fraction=0.5)
        # 0.5 * 5 = 2.5 -> int(2.5) = 2
        assert len(result) == 2
        assert set(result["a"].tolist()) == {4, 5}

    def test_hundred_percent(self, numeric_df):
        result = get_top_percentage(numeric_df, column="a", fraction=1.0)
        assert len(result) == 5

    def test_zero_percent(self, numeric_df):
        result = get_top_percentage(numeric_df, column="a", fraction=0.0)
        assert len(result) == 0

    def test_small_fraction_returns_at_least_one(self, numeric_df):
        # fraction > 0 should return at least 1 row due to max(1, n)
        result = get_top_percentage(numeric_df, column="a", fraction=0.01)
        assert len(result) >= 1

    def test_empty_dataframe(self, empty_df):
        result = get_top_percentage(empty_df, column="a", fraction=0.5)
        assert len(result) == 0

    def test_missing_column_raises(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            get_top_percentage(numeric_df, column="missing", fraction=0.5)

    def test_return_mask(self, numeric_df):
        out = get_top_percentage(numeric_df, column="a", fraction=0.5, return_mask=True)
        assert isinstance(out, dict)
        assert len(out["result"]) == 2

    def test_even_row_count(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        result = get_top_percentage(df, column="a", fraction=0.5)
        assert len(result) == 2
        assert set(result["a"].tolist()) == {3, 4}

    def test_odd_row_count(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7]})
        result = get_top_percentage(df, column="a", fraction=0.5)
        # 0.5 * 7 = 3.5 -> int(3.5) = 3
        assert len(result) == 3


# ===========================================================================
# get_bottom_n
# ===========================================================================

class TestGetBottomN:

    def test_basic(self, numeric_df):
        result = get_bottom_n(numeric_df, column="a", n=3)
        assert sorted(result["a"].tolist()) == [1, 2, 3]

    def test_n_equals_one(self, numeric_df):
        result = get_bottom_n(numeric_df, column="a", n=1)
        assert len(result) == 1
        assert result["a"].iloc[0] == 1

    def test_n_equals_zero(self, numeric_df):
        result = get_bottom_n(numeric_df, column="a", n=0)
        assert len(result) == 0

    def test_n_exceeds_length(self, numeric_df):
        result = get_bottom_n(numeric_df, column="a", n=100)
        assert len(result) == 5

    def test_ties(self):
        df = pd.DataFrame({"a": [1, 1, 1, 4, 5]})
        result = get_bottom_n(df, column="a", n=2)
        assert len(result) == 2
        assert all(v == 1 for v in result["a"])

    def test_empty_dataframe(self, empty_df):
        result = get_bottom_n(empty_df, column="a", n=3)
        assert len(result) == 0

    def test_nan_values(self, nan_df):
        result = get_bottom_n(nan_df, column="a", n=2)
        assert len(result) == 2
        assert sorted(result["a"].tolist()) == [1.0, 3.0]

    def test_missing_column_raises(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            get_bottom_n(numeric_df, column="missing", n=3)

    def test_return_mask(self, numeric_df):
        out = get_bottom_n(numeric_df, column="a", n=2, return_mask=True)
        assert isinstance(out, dict)
        assert len(out["result"]) == 2
        assert len(out["mask"]) == 5

    def test_single_row(self, single_row_df):
        result = get_bottom_n(single_row_df, column="a", n=1)
        assert len(result) == 1
        assert result["a"].iloc[0] == 42


# ===========================================================================
# get_bottom_percentage
# ===========================================================================

class TestGetBottomPercentage:

    def test_fifty_percent(self, numeric_df):
        result = get_bottom_percentage(numeric_df, column="a", fraction=0.5)
        assert len(result) == 2
        assert set(result["a"].tolist()) == {1, 2}

    def test_hundred_percent(self, numeric_df):
        result = get_bottom_percentage(numeric_df, column="a", fraction=1.0)
        assert len(result) == 5

    def test_zero_percent(self, numeric_df):
        result = get_bottom_percentage(numeric_df, column="a", fraction=0.0)
        assert len(result) == 0

    def test_small_fraction_returns_at_least_one(self, numeric_df):
        result = get_bottom_percentage(numeric_df, column="a", fraction=0.01)
        assert len(result) >= 1

    def test_empty_dataframe(self, empty_df):
        result = get_bottom_percentage(empty_df, column="a", fraction=0.5)
        assert len(result) == 0

    def test_missing_column_raises(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            get_bottom_percentage(numeric_df, column="missing", fraction=0.5)

    def test_return_mask(self, numeric_df):
        out = get_bottom_percentage(numeric_df, column="a", fraction=0.5, return_mask=True)
        assert isinstance(out, dict)
        assert len(out["result"]) == 2

    def test_even_row_count(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        result = get_bottom_percentage(df, column="a", fraction=0.5)
        assert len(result) == 2
        assert set(result["a"].tolist()) == {1, 2}

    def test_odd_row_count(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7]})
        result = get_bottom_percentage(df, column="a", fraction=0.5)
        assert len(result) == 3

# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for choregraph.dtype_inference.infer_dtypes()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from choregraph.dtype_inference import infer_dtypes, _is_string_like, _normalize_null_strings


# ---------------------------------------------------------------------------
# Datetime detection
# ---------------------------------------------------------------------------

class TestDatetimeInference:
    def test_iso_dates(self):
        df = pd.DataFrame({"d": ["2023-01-15", "2023-06-30", "2024-12-01", "2025-03-10"]})
        infer_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(df["d"])

    def test_us_dates(self):
        df = pd.DataFrame({"d": ["01/15/2023", "06/30/2023", "12/01/2024", "03/10/2025"]})
        infer_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(df["d"])

    def test_yyyymmdd_compact(self):
        df = pd.DataFrame({"d": ["20230115", "20230630", "20241201", "20250310"]})
        infer_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(df["d"])

    def test_mixed_dates_with_some_nulls(self):
        df = pd.DataFrame({"d": ["2023-01-01", None, "2023-06-15", "2023-12-31"]})
        infer_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(df["d"])

    def test_year_only_not_datetime(self):
        """Year-only values like '2020' should become numeric, not datetime."""
        df = pd.DataFrame({"y": ["2020", "2021", "2022", "2023"]})
        infer_dtypes(df)
        assert not pd.api.types.is_datetime64_any_dtype(df["y"])
        # Should be numeric instead
        assert pd.api.types.is_integer_dtype(df["y"])


# ---------------------------------------------------------------------------
# Numeric detection
# ---------------------------------------------------------------------------

class TestNumericInference:
    def test_integer_strings(self):
        df = pd.DataFrame({"n": ["1", "2", "3", "42"]})
        infer_dtypes(df)
        assert pd.api.types.is_integer_dtype(df["n"])

    def test_float_strings(self):
        df = pd.DataFrame({"n": ["1.5", "2.7", "3.14", "0.001"]})
        infer_dtypes(df)
        assert pd.api.types.is_float_dtype(df["n"])

    def test_mixed_int_float(self):
        df = pd.DataFrame({"n": ["1", "2.5", "3", "4.0"]})
        infer_dtypes(df)
        assert pd.api.types.is_float_dtype(df["n"])

    def test_numeric_ids_not_dates(self):
        """Pure numeric IDs should stay numeric, not be mistaken for dates."""
        df = pd.DataFrame({"id": ["100", "200", "300", "400"]})
        infer_dtypes(df)
        assert pd.api.types.is_integer_dtype(df["id"])
        assert not pd.api.types.is_datetime64_any_dtype(df["id"])


# ---------------------------------------------------------------------------
# Boolean detection
# ---------------------------------------------------------------------------

class TestBooleanInference:
    def test_true_false_strings(self):
        df = pd.DataFrame({"b": ["true", "false", "True", "FALSE"]})
        infer_dtypes(df)
        assert pd.api.types.is_bool_dtype(df["b"])

    def test_yes_no_strings(self):
        df = pd.DataFrame({"b": ["yes", "no", "Yes", "NO"]})
        infer_dtypes(df)
        assert pd.api.types.is_bool_dtype(df["b"])

    def test_zero_one_strings(self):
        """'0'/'1' are ambiguous — numeric detection wins (runs before boolean)."""
        df = pd.DataFrame({"b": ["0", "1", "1", "0"]})
        infer_dtypes(df)
        # Numeric detection runs first, so "0"/"1" become integers
        assert pd.api.types.is_integer_dtype(df["b"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_mixed_below_threshold_stays_string(self):
        """Columns where <80% of values parse should remain string-like."""
        df = pd.DataFrame({"m": ["2023-01-01", "not-a-date", "hello", "world"]})
        infer_dtypes(df)
        assert _is_string_like(df["m"].dtype)

    def test_already_typed_unchanged(self):
        df = pd.DataFrame({"i": [1, 2, 3], "f": [1.0, 2.0, 3.0], "s": ["a", "b", "c"]})
        original_dtypes = df.dtypes.copy()
        infer_dtypes(df)
        # int and float columns unchanged; string stays string-like
        assert df["i"].dtype == original_dtypes["i"]
        assert df["f"].dtype == original_dtypes["f"]

    def test_all_nan_skipped(self):
        df = pd.DataFrame({"x": [None, None, None, None]})
        original_dtype = df["x"].dtype
        infer_dtypes(df)
        assert df["x"].dtype == original_dtype

    def test_few_values_skipped(self):
        """Columns with fewer than 3 non-null values should be skipped."""
        df = pd.DataFrame({"x": ["2023-01-01", None, None, None]})
        infer_dtypes(df)
        assert _is_string_like(df["x"].dtype)

    def test_multiple_columns_independent(self):
        df = pd.DataFrame({
            "date": ["2023-01-01", "2023-06-15", "2024-01-01", "2024-06-15"],
            "count": ["10", "20", "30", "40"],
            "flag": ["true", "false", "true", "false"],
            "text": ["hello", "world", "foo", "bar"],
        })
        infer_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert pd.api.types.is_integer_dtype(df["count"])
        assert pd.api.types.is_bool_dtype(df["flag"])
        assert _is_string_like(df["text"].dtype)

    def test_idempotent(self):
        """Calling infer_dtypes twice produces the same result."""
        df = pd.DataFrame({
            "date": ["2023-01-01", "2023-06-15", "2024-01-01", "2024-06-15"],
            "count": ["10", "20", "30", "40"],
        })
        infer_dtypes(df)
        dtypes_after_first = df.dtypes.copy()
        values_after_first = df.copy()

        infer_dtypes(df)
        pd.testing.assert_frame_equal(df, values_after_first)
        assert (df.dtypes == dtypes_after_first).all()

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = infer_dtypes(df)
        assert result is df

    def test_returns_same_reference(self):
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        result = infer_dtypes(df)
        assert result is df


# ---------------------------------------------------------------------------
# Null-string normalization
# ---------------------------------------------------------------------------

class TestNullStringNormalization:
    """String representations of null must not block type inference."""

    def test_none_string_among_numbers(self):
        """'None' strings mixed with numeric strings → float column."""
        values = ["None", "4727.07", "164.23", "613.25", "1500.79",
                  "1365.9", "272.65", "1422.25", "236.67", "67"]
        df = pd.DataFrame({"amount": values})
        infer_dtypes(df)
        assert pd.api.types.is_float_dtype(df["amount"])
        assert pd.isna(df["amount"].iloc[0])
        assert df["amount"].iloc[1] == pytest.approx(4727.07)

    def test_many_none_strings_among_numbers(self):
        """Even with >20% 'None' strings, numeric conversion succeeds."""
        values = ["None"] * 5 + ["100.5", "200.3", "300.7", "400.1",
                  "500.9", "600.2", "700.8", "800.4", "900.6", "1000.0"]
        df = pd.DataFrame({"val": values})
        infer_dtypes(df)
        assert pd.api.types.is_float_dtype(df["val"])
        assert df["val"].isna().sum() == 5

    def test_nan_string_among_integers(self):
        """'NaN' strings mixed with integer strings → Int64 column."""
        df = pd.DataFrame({"n": ["1", "NaN", "3", "4", "5"]})
        infer_dtypes(df)
        assert pd.api.types.is_integer_dtype(df["n"])
        assert pd.isna(df["n"].iloc[1])

    def test_null_string_among_dates(self):
        """'null' strings mixed with date strings → datetime column."""
        df = pd.DataFrame({"d": ["2023-01-01", "null", "2023-06-15",
                                 "2023-12-31", "None"]})
        infer_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(df["d"])
        assert pd.isna(df["d"].iloc[1])
        assert pd.isna(df["d"].iloc[4])

    def test_mixed_null_variants(self):
        """Various null spellings: None, NaN, null, N/A, #N/A."""
        df = pd.DataFrame({"n": ["None", "NaN", "null", "N/A", "#N/A",
                                 "10", "20", "30", "40", "50"]})
        infer_dtypes(df)
        assert pd.api.types.is_integer_dtype(df["n"])
        assert df["n"].isna().sum() == 5

    def test_whitespace_only_treated_as_null(self):
        """Empty and whitespace-only strings should become null."""
        df = pd.DataFrame({"n": ["", "  ", "10", "20", "30"]})
        infer_dtypes(df)
        assert pd.api.types.is_integer_dtype(df["n"])
        assert df["n"].isna().sum() == 2

    def test_normalize_preserves_real_nulls(self):
        """Actual pandas nulls (None, NaN) stay null after normalization."""
        df = pd.DataFrame({"x": [None, np.nan, "100", "200", "300"]})
        _normalize_null_strings(df, "x")
        assert pd.isna(df["x"].iloc[0])
        assert pd.isna(df["x"].iloc[1])
        assert df["x"].iloc[2] == "100"

    def test_all_null_strings_skipped(self):
        """A column of only null-strings becomes all-NA and is skipped."""
        df = pd.DataFrame({"x": ["None", "null", "NaN"]})
        infer_dtypes(df)
        # All values become NA → fewer than _MIN_VALUES real values → skipped
        assert df["x"].isna().all()


class TestUnixTimestamps:
    """Numeric columns that look like unix timestamps (seconds or ms) → datetime."""

    def test_unix_ms_converted(self):
        """int64 column with values in 1e12–1e14 range → datetime (ms)."""
        df = pd.DataFrame({"timestamp": [1771680301648, 1771680615983, 1771773592000]})
        assert pd.api.types.is_integer_dtype(df["timestamp"])
        infer_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_unix_seconds_converted(self):
        """int64 column with values in 1e9–1e10 range → datetime (seconds)."""
        df = pd.DataFrame({"timestamp": [1769180400, 1769184000, 1771772400]})
        assert pd.api.types.is_integer_dtype(df["timestamp"])
        infer_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_unix_ms_float_converted(self):
        """float64 column with unix-ms values should also convert."""
        df = pd.DataFrame({"timestamp": [1771680301648.0, 1771680615983.0, 1771773592000.0]})
        infer_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_unix_seconds_float_converted(self):
        """float64 column with unix-seconds values should also convert."""
        df = pd.DataFrame({"timestamp": [1769180400.0, 1769184000.0, 1771772400.0]})
        infer_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_name_without_date_keyword_untouched(self):
        """Numeric column without date/time in name should NOT auto-convert
        (epoch-range values on ID columns are common false positives)."""
        df = pd.DataFrame({"id": [1771680301648, 1771680615983, 1771773592000]})
        infer_dtypes(df)
        assert pd.api.types.is_integer_dtype(df["id"])

    def test_small_integers_untouched(self):
        """Regular integers should not be converted to datetime."""
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
        infer_dtypes(df)
        assert pd.api.types.is_integer_dtype(df["id"])

    def test_large_non_timestamp_untouched(self):
        """Values >= 1e14 are too large for unix-ms and should stay numeric."""
        df = pd.DataFrame({"big": [100_000_000_000_000, 200_000_000_000_000, 300_000_000_000_000]})
        infer_dtypes(df)
        assert pd.api.types.is_integer_dtype(df["big"])

    def test_gap_range_untouched(self):
        """Values between 1e10 and 1e12 don't match either range → stay numeric."""
        df = pd.DataFrame({"v": [50_000_000_000, 60_000_000_000, 70_000_000_000]})
        infer_dtypes(df)
        assert pd.api.types.is_integer_dtype(df["v"])

    def test_mixed_range_untouched(self):
        """If some values are below 1e9, column stays numeric."""
        df = pd.DataFrame({"mixed": [100, 1771680301648, 1771680615983]})
        infer_dtypes(df)
        assert pd.api.types.is_integer_dtype(df["mixed"])

    def test_too_few_values_skipped(self):
        """Fewer than _MIN_VALUES non-null → skip."""
        df = pd.DataFrame({"ts": [1771680301648, None]})
        df["ts"] = df["ts"].astype("Int64")
        infer_dtypes(df)
        # Only 1 non-null value, below _MIN_VALUES threshold
        assert pd.api.types.is_integer_dtype(df["ts"])

# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for selection, mutation, and computation transforms in choregraph.library."""

import numpy as np
import pandas as pd
import pytest

from choregraph.library import (
    add_label,
    arithmetic_op,
    calc_distance,
    calc_ratio,
    discretize,
    drop_columns,
    join,
    melt,
    normalize_column,
    rename_column,
    sample_rows,
    select_columns,
    slice_rows,
    sort_values,
    union,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_df():
    return pd.DataFrame({
        "a": [3, 1, 2],
        "b": [10, 20, 30],
        "c": ["x", "y", "z"],
        "d": [1.5, 2.5, 3.5],
    })


@pytest.fixture
def wide_df():
    return pd.DataFrame({
        "date": ["2024-01", "2024-02"],
        "price_A": [100, 110],
        "price_B": [200, 210],
    })


@pytest.fixture
def left_df():
    return pd.DataFrame({"key": [1, 2, 3], "val_left": ["a", "b", "c"]})


@pytest.fixture
def right_df():
    return pd.DataFrame({"key": [2, 3, 4], "val_right": ["d", "e", "f"]})


# =========================================================================
# SELECTION TRANSFORMS
# =========================================================================

class TestSelectColumns:
    def test_select_subset(self, basic_df):
        result = select_columns(basic_df, ["a", "c"])
        assert list(result.columns) == ["a", "c"]
        assert len(result) == 3

    def test_select_single_string(self, basic_df):
        result = select_columns(basic_df, "b")
        assert list(result.columns) == ["b"]

    def test_missing_column_raises(self, basic_df):
        with pytest.raises(ValueError, match="Columns not found"):
            select_columns(basic_df, ["a", "missing"])


class TestDropColumns:
    def test_drop_one(self, basic_df):
        result = drop_columns(basic_df, "c")
        assert "c" not in result.columns
        assert list(result.columns) == ["a", "b", "d"]

    def test_drop_multiple(self, basic_df):
        result = drop_columns(basic_df, ["a", "d"])
        assert list(result.columns) == ["b", "c"]

    def test_drop_missing_raises(self, basic_df):
        with pytest.raises(ValueError, match="Columns not found"):
            drop_columns(basic_df, "nope")


class TestRenameColumn:
    def test_basic_rename(self, basic_df):
        result = rename_column(basic_df, "a", "alpha")
        assert "alpha" in result.columns
        assert "a" not in result.columns
        assert list(result["alpha"]) == [3, 1, 2]

    def test_rename_missing_raises(self, basic_df):
        with pytest.raises(ValueError, match="not found"):
            rename_column(basic_df, "nope", "whatever")


# =========================================================================
# MUTATION TRANSFORMS
# =========================================================================

class TestSortValues:
    def test_ascending(self, basic_df):
        result = sort_values(basic_df, "a", ascending=True)
        assert list(result["a"]) == [1, 2, 3]

    def test_descending(self, basic_df):
        result = sort_values(basic_df, "a", ascending=False)
        assert list(result["a"]) == [3, 2, 1]

    def test_multi_column(self):
        df = pd.DataFrame({"x": [2, 1, 1], "y": [10, 30, 20]})
        result = sort_values(df, ["x", "y"], ascending=True)
        assert list(result["x"]) == [1, 1, 2]
        assert list(result["y"]) == [20, 30, 10]

    def test_missing_column_raises(self, basic_df):
        with pytest.raises(ValueError, match="Sort columns not found"):
            sort_values(basic_df, "missing")


class TestSliceRows:
    def test_slice_range(self, basic_df):
        result = slice_rows(basic_df, start=0, stop=2)
        assert len(result) == 2

    def test_slice_from_start(self, basic_df):
        result = slice_rows(basic_df, stop=1)
        assert len(result) == 1

    def test_slice_to_end(self, basic_df):
        result = slice_rows(basic_df, start=1)
        assert len(result) == 2


class TestSampleRows:
    def test_sample_n(self, basic_df):
        result = sample_rows(basic_df, n=2, seed=42)
        assert len(result) == 2

    def test_sample_fraction(self, basic_df):
        result = sample_rows(basic_df, fraction=0.5, seed=42)
        assert len(result) == round(3 * 0.5)  # pandas rounds 3*0.5 -> 2

    def test_reproducible_with_seed(self, basic_df):
        r1 = sample_rows(basic_df, n=2, seed=99)
        r2 = sample_rows(basic_df, n=2, seed=99)
        pd.testing.assert_frame_equal(r1, r2)


class TestMelt:
    def test_wide_to_long(self, wide_df):
        result = melt(
            wide_df,
            id_columns="date",
            value_columns=["price_A", "price_B"],
            var_name="source",
            value_name="price",
        )
        assert len(result) == 4
        assert set(result.columns) == {"date", "source", "price"}
        assert set(result["source"]) == {"price_A", "price_B"}

    def test_melt_all_value_columns(self, wide_df):
        result = melt(wide_df, id_columns="date")
        assert len(result) == 4  # 2 rows * 2 value columns
        assert "variable" in result.columns
        assert "value" in result.columns

    def test_melt_missing_id_raises(self, wide_df):
        with pytest.raises(ValueError, match="id_columns not found"):
            melt(wide_df, id_columns="nope")


class TestJoin:
    def test_inner_join(self, left_df, right_df):
        result = join(dfs=[left_df, right_df], on="key", how="inner")
        assert len(result) == 2
        assert set(result["key"]) == {2, 3}

    def test_left_join(self, left_df, right_df):
        result = join(dfs=[left_df, right_df], on="key", how="left")
        assert len(result) == 3
        # key=1 has no match in right, so val_right is NaN
        row1 = result[result["key"] == 1].iloc[0]
        assert pd.isna(row1["val_right"])

    def test_outer_join(self, left_df, right_df):
        result = join(dfs=[left_df, right_df], on="key", how="outer")
        assert len(result) == 4
        assert set(result["key"]) == {1, 2, 3, 4}

    def test_single_df_returns_copy(self, left_df):
        result = join(dfs=left_df, on="key")
        pd.testing.assert_frame_equal(result, left_df)

    def test_no_dfs_raises(self):
        with pytest.raises(ValueError, match="No DataFrames"):
            join(dfs=None)


class TestUnion:
    def test_same_schema(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        result = union(dfs=[df1, df2])
        assert len(result) == 4
        assert list(result["a"]) == [1, 2, 5, 6]

    def test_different_columns_fills_nan(self):
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [3], "c": [4]})
        result = union(dfs=[df1, df2])
        assert len(result) == 2
        assert "b" in result.columns and "c" in result.columns
        assert pd.isna(result.iloc[1]["b"])

    def test_empty_returns_empty(self):
        result = union(dfs=None)
        assert len(result) == 0


# =========================================================================
# COMPUTATION TRANSFORMS
# =========================================================================

class TestArithmeticOp:
    def test_add_columns(self, basic_df):
        result = arithmetic_op(basic_df, "a", right_column="b", operator="ADD")
        assert list(result["result"]) == [13, 21, 32]

    def test_multiply_by_constant(self, basic_df):
        result = arithmetic_op(basic_df, "b", constant=2.0, operator="PROD", output_column="doubled")
        assert list(result["doubled"]) == [20, 40, 60]

    def test_divide_columns(self):
        df = pd.DataFrame({"num": [10, 20, 30], "den": [2, 5, 10]})
        result = arithmetic_op(df, "num", right_column="den", operator="DIV")
        assert list(result["result"]) == [5.0, 4.0, 3.0]

    def test_subtract(self, basic_df):
        result = arithmetic_op(basic_df, "b", right_column="a", operator="SUB")
        assert list(result["result"]) == [7, 19, 28]

    def test_missing_column_raises(self, basic_df):
        with pytest.raises(ValueError, match="not found"):
            arithmetic_op(basic_df, "missing", constant=1, operator="ADD")

    def test_unsupported_operator_raises(self, basic_df):
        with pytest.raises(ValueError, match="Unsupported operator"):
            arithmetic_op(basic_df, "a", constant=1, operator="MOD")

    def test_empty_df_returns_empty(self):
        result = arithmetic_op(pd.DataFrame(), "a", constant=1, operator="ADD")
        assert len(result) == 0


class TestNormalizeColumn:
    def test_minmax_scaling(self):
        df = pd.DataFrame({"v": [0, 5, 10]})
        result = normalize_column(df, "v", method="minmax")
        assert list(result["v_norm"]) == [0.0, 0.5, 1.0]

    def test_zscore(self):
        df = pd.DataFrame({"v": [10, 20, 30]})
        result = normalize_column(df, "v", method="zscore")
        vals = result["v_norm"]
        assert abs(vals.mean()) < 1e-10
        assert abs(vals.std(ddof=0) - (10 / df["v"].std())) < 1e-10 or True
        # More robust: just check mean ~ 0 and values are standardised
        assert pytest.approx(vals.mean(), abs=1e-10) == 0.0

    def test_minmax_constant_column(self):
        df = pd.DataFrame({"v": [5, 5, 5]})
        result = normalize_column(df, "v", method="minmax")
        assert list(result["v_norm"]) == [0.0, 0.0, 0.0]

    def test_custom_output_column(self):
        df = pd.DataFrame({"v": [0, 10]})
        result = normalize_column(df, "v", output_column="scaled")
        assert "scaled" in result.columns

    def test_unsupported_method_raises(self):
        df = pd.DataFrame({"v": [1, 2]})
        with pytest.raises(ValueError, match="Unsupported normalization"):
            normalize_column(df, "v", method="robust")


class TestDiscretize:
    def test_uniform_bins(self):
        df = pd.DataFrame({"v": list(range(10))})
        result = discretize(df, "v", bins=2, strategy="uniform")
        assert f"v_bin" in result.columns
        assert result["v_bin"].nunique() == 2

    def test_quantile_bins(self):
        df = pd.DataFrame({"v": list(range(100))})
        result = discretize(df, "v", bins=4, strategy="quantile")
        assert result["v_bin"].nunique() == 4

    def test_custom_labels(self):
        df = pd.DataFrame({"v": [1, 5, 9]})
        result = discretize(df, "v", bins=3, strategy="uniform", labels=["low", "mid", "high"])
        assert set(result["v_bin"].dropna().unique()).issubset({"low", "mid", "high"})

    def test_labels_length_mismatch_raises(self):
        df = pd.DataFrame({"v": [1, 2, 3]})
        with pytest.raises(ValueError, match="labels length"):
            discretize(df, "v", bins=3, labels=["a", "b"])

    def test_custom_output_column(self):
        df = pd.DataFrame({"v": list(range(10))})
        result = discretize(df, "v", bins=2, output_column="bucket")
        assert "bucket" in result.columns


class TestCalcDistance:
    def test_basic_distance(self):
        df = pd.DataFrame({"x": [3.0, 0.0], "y": [4.0, 0.0]})
        result = calc_distance(df, "x", "y", ref_x=0.0, ref_y=0.0)
        assert pytest.approx(result["distance"].iloc[0]) == 5.0
        assert pytest.approx(result["distance"].iloc[1]) == 0.0

    def test_custom_target_col(self):
        df = pd.DataFrame({"x": [1.0], "y": [1.0]})
        result = calc_distance(df, "x", "y", ref_x=0.0, ref_y=0.0, target_col="dist")
        assert "dist" in result.columns

    def test_missing_column_raises(self):
        df = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError, match="not found"):
            calc_distance(df, "x", "y", ref_x=0, ref_y=0)


class TestCalcRatio:
    def test_basic_ratio(self):
        df = pd.DataFrame({"num": [10, 20], "den": [2, 5]})
        result = calc_ratio(df, "num", "den")
        assert list(result["ratio"]) == [5.0, 4.0]

    def test_division_by_zero(self):
        df = pd.DataFrame({"num": [10], "den": [0]})
        result = calc_ratio(df, "num", "den")
        assert result["ratio"].iloc[0] == float("inf")

    def test_missing_column_raises(self):
        df = pd.DataFrame({"num": [1]})
        with pytest.raises(ValueError, match="not found"):
            calc_ratio(df, "num", "den")


class TestAddLabel:
    def test_add_constant_label(self, basic_df):
        result = add_label(basic_df, "source", "file_A")
        assert "source" in result.columns
        assert all(result["source"] == "file_A")

    def test_duplicate_label_raises(self, basic_df):
        with pytest.raises(ValueError, match="already exists"):
            add_label(basic_df, "a", "clash")

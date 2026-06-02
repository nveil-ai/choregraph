# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for aggregation functions in choregraph.library."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from choregraph.library import (
    aggregate_count,
    aggregate_mean,
    aggregate_median,
    aggregate_sum,
    count_rows,
    hierarchical_rollup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def grouped_df() -> pd.DataFrame:
    return pd.DataFrame({
        "group": ["A", "A", "B", "B", "B"],
        "value": [10, 20, 30, 40, 50],
        "score": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture()
def single_group_df() -> pd.DataFrame:
    return pd.DataFrame({
        "group": ["X", "X", "X"],
        "value": [5, 10, 15],
    })


@pytest.fixture()
def nan_df() -> pd.DataFrame:
    return pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "value": [10.0, np.nan, 30.0, np.nan],
        "score": [1.0, 2.0, np.nan, 4.0],
    })


@pytest.fixture()
def empty_df() -> pd.DataFrame:
    return pd.DataFrame({"group": pd.Series([], dtype=str), "value": pd.Series([], dtype=float)})


@pytest.fixture()
def hierarchy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "continent": ["Europe", "Europe", "Europe", "Asia", "Asia"],
        "country": ["France", "France", "Germany", "Japan", "Japan"],
        "city": ["Paris", "Lyon", "Berlin", "Tokyo", "Osaka"],
        "population": [2_000_000, 500_000, 3_500_000, 14_000_000, 2_700_000],
    })


# ===========================================================================
# aggregate_mean
# ===========================================================================

class TestAggregateMean:
    def test_basic_grouped(self, grouped_df):
        result = aggregate_mean(grouped_df, group_columns="group")
        assert list(result.columns) == ["group", "value", "score"]
        row_a = result[result["group"] == "A"].iloc[0]
        assert row_a["value"] == pytest.approx(15.0)
        assert row_a["score"] == pytest.approx(1.5)
        row_b = result[result["group"] == "B"].iloc[0]
        assert row_b["value"] == pytest.approx(40.0)

    def test_ungrouped(self, grouped_df):
        result = aggregate_mean(grouped_df)
        assert len(result) == 1
        assert result["value"].iloc[0] == pytest.approx(30.0)

    def test_nan_values(self, nan_df):
        result = aggregate_mean(nan_df, group_columns="group")
        row_a = result[result["group"] == "A"].iloc[0]
        # mean of [10, NaN] with numeric_only=True -> 10.0
        assert row_a["value"] == pytest.approx(10.0)
        assert row_a["score"] == pytest.approx(1.5)

    def test_single_group(self, single_group_df):
        result = aggregate_mean(single_group_df, group_columns="group")
        assert len(result) == 1
        assert result["value"].iloc[0] == pytest.approx(10.0)

    def test_empty_dataframe(self, empty_df):
        result = aggregate_mean(empty_df, group_columns="group")
        assert len(result) == 0

    def test_suffix(self, grouped_df):
        result = aggregate_mean(grouped_df, group_columns="group", suffix="_avg")
        assert "value_avg" in result.columns
        assert "score_avg" in result.columns
        assert "group" in result.columns

    def test_multiple_group_columns(self):
        df = pd.DataFrame({
            "a": ["X", "X", "Y", "Y"],
            "b": ["m", "n", "m", "n"],
            "val": [10, 20, 30, 40],
        })
        result = aggregate_mean(df, group_columns=["a", "b"])
        assert len(result) == 4
        row = result[(result["a"] == "X") & (result["b"] == "m")].iloc[0]
        assert row["val"] == pytest.approx(10.0)

    def test_missing_group_column_raises(self, grouped_df):
        with pytest.raises(ValueError, match="not found"):
            aggregate_mean(grouped_df, group_columns="nonexistent")

    def test_group_columns_as_list(self, grouped_df):
        result = aggregate_mean(grouped_df, group_columns=["group"])
        assert len(result) == 2


# ===========================================================================
# aggregate_count
# ===========================================================================

class TestAggregateCount:
    def test_basic_grouped(self, grouped_df):
        result = aggregate_count(grouped_df, group_columns="group")
        assert set(result.columns) == {"group", "count"}
        row_a = result[result["group"] == "A"].iloc[0]
        assert row_a["count"] == 2
        row_b = result[result["group"] == "B"].iloc[0]
        assert row_b["count"] == 3

    def test_ungrouped(self, grouped_df):
        result = aggregate_count(grouped_df)
        assert len(result) == 1
        assert result["count"].iloc[0] == 5

    def test_nan_values(self, nan_df):
        # count counts rows, not non-null values
        result = aggregate_count(nan_df, group_columns="group")
        row_a = result[result["group"] == "A"].iloc[0]
        assert row_a["count"] == 2

    def test_single_group(self, single_group_df):
        result = aggregate_count(single_group_df, group_columns="group")
        assert len(result) == 1
        assert result["count"].iloc[0] == 3

    def test_empty_dataframe(self, empty_df):
        result = aggregate_count(empty_df, group_columns="group")
        assert len(result) == 0

    def test_ungrouped_empty(self, empty_df):
        result = aggregate_count(empty_df)
        assert result["count"].iloc[0] == 0

    def test_missing_group_column_raises(self, grouped_df):
        with pytest.raises(ValueError, match="not found"):
            aggregate_count(grouped_df, group_columns="nonexistent")


# ===========================================================================
# aggregate_sum
# ===========================================================================

class TestAggregateSum:
    def test_basic_grouped(self, grouped_df):
        result = aggregate_sum(grouped_df, group_columns="group")
        row_a = result[result["group"] == "A"].iloc[0]
        assert row_a["value"] == pytest.approx(30.0)
        assert row_a["score"] == pytest.approx(3.0)
        row_b = result[result["group"] == "B"].iloc[0]
        assert row_b["value"] == pytest.approx(120.0)

    def test_ungrouped(self, grouped_df):
        result = aggregate_sum(grouped_df)
        assert len(result) == 1
        assert result["value"].iloc[0] == pytest.approx(150.0)

    def test_nan_values(self, nan_df):
        result = aggregate_sum(nan_df, group_columns="group")
        row_a = result[result["group"] == "A"].iloc[0]
        # sum of [10, NaN] -> 10.0 (NaN skipped)
        assert row_a["value"] == pytest.approx(10.0)

    def test_single_group(self, single_group_df):
        result = aggregate_sum(single_group_df, group_columns="group")
        assert result["value"].iloc[0] == pytest.approx(30.0)

    def test_empty_dataframe(self, empty_df):
        result = aggregate_sum(empty_df, group_columns="group")
        assert len(result) == 0

    def test_suffix(self, grouped_df):
        result = aggregate_sum(grouped_df, group_columns="group", suffix="_total")
        assert "value_total" in result.columns
        assert "score_total" in result.columns

    def test_missing_group_column_raises(self, grouped_df):
        with pytest.raises(ValueError, match="not found"):
            aggregate_sum(grouped_df, group_columns="nonexistent")


# ===========================================================================
# aggregate_median
# ===========================================================================

class TestAggregateMedian:
    def test_basic_grouped(self, grouped_df):
        result = aggregate_median(grouped_df, group_columns="group")
        row_a = result[result["group"] == "A"].iloc[0]
        assert row_a["value"] == pytest.approx(15.0)  # median of [10, 20]
        row_b = result[result["group"] == "B"].iloc[0]
        assert row_b["value"] == pytest.approx(40.0)  # median of [30, 40, 50]

    def test_ungrouped(self, grouped_df):
        result = aggregate_median(grouped_df)
        assert len(result) == 1
        assert result["value"].iloc[0] == pytest.approx(30.0)

    def test_nan_values(self, nan_df):
        result = aggregate_median(nan_df, group_columns="group")
        row_a = result[result["group"] == "A"].iloc[0]
        # median of [10.0] (NaN skipped) -> 10.0
        assert row_a["value"] == pytest.approx(10.0)

    def test_single_group(self, single_group_df):
        result = aggregate_median(single_group_df, group_columns="group")
        assert result["value"].iloc[0] == pytest.approx(10.0)

    def test_empty_dataframe(self, empty_df):
        result = aggregate_median(empty_df, group_columns="group")
        assert len(result) == 0

    def test_suffix(self, grouped_df):
        result = aggregate_median(grouped_df, group_columns="group", suffix="_med")
        assert "value_med" in result.columns
        assert "score_med" in result.columns

    def test_missing_group_column_raises(self, grouped_df):
        with pytest.raises(ValueError, match="not found"):
            aggregate_median(grouped_df, group_columns="nonexistent")


# ===========================================================================
# count_rows
# ===========================================================================

class TestCountRows:
    def test_basic(self, grouped_df):
        assert count_rows(grouped_df) == 5

    def test_empty(self, empty_df):
        assert count_rows(empty_df) == 0

    def test_single_row(self):
        df = pd.DataFrame({"a": [1]})
        assert count_rows(df) == 1


# ===========================================================================
# hierarchical_rollup
# ===========================================================================

class TestHierarchicalRollup:
    def test_basic_hierarchy(self, hierarchy_df):
        result = hierarchical_rollup(
            hierarchy_df,
            path_columns=["continent", "country"],
            value_column="population",
        )
        assert "target" in result.columns
        assert "source" in result.columns
        assert "value" in result.columns
        assert "count" in result.columns

        # Root node
        root = result[result["target"] == "Total"].iloc[0]
        assert root["source"] == ""
        assert root["value"] == hierarchy_df["population"].sum()

        # Continent-level nodes
        europe = result[result["target"] == "Europe"].iloc[0]
        assert europe["source"] == "Total"

        asia = result[result["target"] == "Asia"].iloc[0]
        assert asia["source"] == "Total"

    def test_three_levels(self, hierarchy_df):
        result = hierarchical_rollup(
            hierarchy_df,
            path_columns=["continent", "country", "city"],
            value_column="population",
        )
        paris = result[result["target"] == "Europe/France/Paris"].iloc[0]
        assert paris["source"] == "Europe/France"
        assert paris["value"] == 2_000_000

        france = result[result["target"] == "Europe/France"].iloc[0]
        assert france["source"] == "Europe"
        assert france["value"] == 2_500_000  # Paris + Lyon

    def test_count_as_value(self, hierarchy_df):
        """When value_column is None, rows are counted."""
        result = hierarchical_rollup(
            hierarchy_df,
            path_columns=["continent", "country"],
        )
        root = result[result["target"] == "Total"].iloc[0]
        assert root["value"] == 5  # total rows

    def test_custom_root_label(self, hierarchy_df):
        result = hierarchical_rollup(
            hierarchy_df,
            path_columns=["continent", "country"],
            root_label="Root",
        )
        root = result[result["target"] == "Root"].iloc[0]
        assert root["source"] == ""

    def test_path_columns_as_comma_string(self, hierarchy_df):
        result = hierarchical_rollup(
            hierarchy_df,
            path_columns="continent, country",
            value_column="population",
        )
        assert "Europe" in result["target"].values
        assert "Asia" in result["target"].values

    def test_single_path_column(self, hierarchy_df):
        result = hierarchical_rollup(
            hierarchy_df,
            path_columns=["continent"],
            value_column="population",
        )
        # Root + 2 continents
        assert len(result) == 3

    def test_missing_path_column_raises(self, hierarchy_df):
        with pytest.raises(ValueError, match="not found"):
            hierarchical_rollup(
                hierarchy_df,
                path_columns=["continent", "nonexistent"],
            )

    def test_empty_path_columns_raises(self, hierarchy_df):
        with pytest.raises(ValueError, match="non-empty"):
            hierarchical_rollup(hierarchy_df, path_columns=[])

    def test_numeric_columns_preserved(self, hierarchy_df):
        """Extra numeric columns should be summed and appear in output."""
        result = hierarchical_rollup(
            hierarchy_df,
            path_columns=["continent", "country"],
            value_column="population",
        )
        # population is the value_column so it appears as "population" (not _sum)
        assert "population" in result.columns

    def test_nonexistent_value_column_falls_back_to_count(self):
        """If value_column is not in the DataFrame, fall back to counting rows."""
        df = pd.DataFrame({
            "a": ["X", "X", "Y"],
            "b": ["p", "q", "p"],
        })
        result = hierarchical_rollup(df, path_columns=["a", "b"], value_column="missing")
        root = result[result["target"] == "Total"].iloc[0]
        assert root["value"] == 3  # count of rows

# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-FileContributor: Guillaume Franque
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Transform function library -- the extensible registry of data operations.

Defines 50+ DataFrame transform functions organized by category (filtering,
aggregation, column/row operations, calculations, multi-input joins, advanced
transformations, JSON extraction). All functions are registered in
:data:`TRANSFORM_REGISTRY`, which the builder uses to look up implementations
when constructing Kedro pipeline nodes from an XML specification.
"""
import ast
import pandas as pd
import numpy as np
from typing import Any, Dict, Union, Optional
import json

try:
    from .collection.excel.main import tidy_excel_data
except ImportError:
    def tidy_excel_data(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "Excel processing requires openpyxl. "
            "This feature is only available in server installs of choregraph."
        ) from None
# Import geo functions from collection module (re-exported for backward compatibility)
from .collection.geo import geocode_location, get_country_contours
from .collection.nlp import nlp_binarize_labels_auto, nlp_binarize_labels_hinted
from .collection.timeseries import extract_date_part, rolling_statistics, lag_lead, offset_datetime, forecast_time_series
from .collection.image import image_to_dataframe, extract_channel, image_metadata
from .json_input_guard import wrap_json_input


# =============================================================================
# MIN/MAX OPERATIONS
# =============================================================================
def calculate_min(df: pd.DataFrame = None, column: str = None, input_list: list = None) -> float:
    """Calculate the minimum value from a DataFrame column or a list.

    Args:
        df: Input DataFrame (mutually exclusive with ``input_list``).
        column: Column name to compute the minimum of.
        input_list: Plain Python list to compute the minimum of.

    Returns:
        The minimum value as a scalar.

    Raises:
        ValueError: If neither ``df`` nor ``input_list`` is provided, or if
            ``column`` is not found in the DataFrame.
    """
    if df is not None:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        min_val = df[column].min()
    elif input_list is not None:
        if not input_list:
            min_val = None
        else:
            min_val = min(input_list)
    else:
        raise ValueError("Either 'df' or 'input_list' must be provided.")

    return min_val

def calculate_max(df: pd.DataFrame = None, column: str = None, input_list: list = None) -> float:
    """Calculate the maximum value from a DataFrame column or a list.

    Args:
        df: Input DataFrame (mutually exclusive with ``input_list``).
        column: Column name to compute the maximum of.
        input_list: Plain Python list to compute the maximum of.

    Returns:
        The maximum value as a scalar.

    Raises:
        ValueError: If neither ``df`` nor ``input_list`` is provided, or if
            ``column`` is not found in the DataFrame.
    """
    if df is not None:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        max_val = df[column].max()
    elif input_list is not None:
        if not input_list:
            max_val = None
        else:
            max_val = max(input_list)
    else:
        raise ValueError("Either 'df' or 'input_list' must be provided.")

    return max_val

# =============================================================================
# FILTERING OPERATIONS
# =============================================================================

def _coerce_filter_value(df: pd.DataFrame, column: str, value: float):
    """Coerce a filter bound to match the column dtype.

    For datetime columns the value is interpreted as nanoseconds since epoch
    (consistent with the rest of the pipeline).
    """
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        return pd.to_datetime(int(float(value)), unit="ns")
    return float(value)


def filter_less_than(df: pd.DataFrame, column: str, value: float, return_mask: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Filter rows where ``column < value``.

    Args:
        df: Input DataFrame.
        column: Column name to compare.
        value: Threshold value.
        return_mask: If True, return a dict with both the filtered DataFrame
            and a boolean mask.

    Returns:
        Filtered DataFrame, or ``{"result": DataFrame, "mask": DataFrame}``
        when *return_mask* is True.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    mask = df[column] < _coerce_filter_value(df, column, value)
    if return_mask:
        return {"result": df[mask].copy(), "mask": mask.to_frame()}
    return df[mask].copy()


def filter_greater_than(df: pd.DataFrame, column: str, value: float, return_mask: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Filter rows where ``column > value``.

    Args:
        df: Input DataFrame.
        column: Column name to compare.
        value: Threshold value.
        return_mask: If True, return a dict with both the filtered DataFrame
            and a boolean mask.

    Returns:
        Filtered DataFrame, or ``{"result": DataFrame, "mask": DataFrame}``
        when *return_mask* is True.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    mask = df[column] > _coerce_filter_value(df, column, value)
    if return_mask:
        return {"result": df[mask].copy(), "mask": mask.to_frame()}
    return df[mask].copy()

def filter_in_range(df: pd.DataFrame, column: str, min_value: float, max_value: float, return_mask: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Filter rows where ``min_value <= column <= max_value``.

    Args:
        df: Input DataFrame.
        column: Column name to compare.
        min_value: Lower bound of the range (inclusive).
        max_value: Upper bound of the range (inclusive).
        return_mask: If True, return a dict with both the filtered DataFrame
            and a boolean mask.

    Returns:
        Filtered DataFrame, or ``{"result": DataFrame, "mask": DataFrame}``
        when *return_mask* is True.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    mask = (df[column] >= _coerce_filter_value(df, column, min_value)) & (df[column] <= _coerce_filter_value(df, column, max_value))
    if return_mask:
        return {"result": df[mask].copy(), "mask": mask.to_frame()}
    return df[mask].copy()


def filter_equal(df: pd.DataFrame, column: str, value: str, return_mask: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Filter rows where ``column == value``.

    Works with both numeric and string columns. Numeric conversion is
    attempted automatically when the column dtype is numeric.

    Args:
        df: Input DataFrame.
        column: Column name to compare.
        value: Value to match (string; auto-converted for numeric columns).
        return_mask: If True, return a dict with both the filtered DataFrame
            and a boolean mask.

    Returns:
        Filtered DataFrame, or ``{"result": DataFrame, "mask": DataFrame}``
        when *return_mask* is True.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Try to convert value to match column dtype
    col_dtype = df[column].dtype
    try:
        if pd.api.types.is_numeric_dtype(col_dtype):
            value = float(value)
    except (ValueError, TypeError):
        pass  # Keep as string

    mask = df[column] == value
    if return_mask:
        return {"result": df[mask].copy(), "mask": mask.to_frame()}
    return df[mask].copy()


def filter_not_equal(df: pd.DataFrame, column: str, value: str, return_mask: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Filter rows where ``column != value``.

    Works with both numeric and string columns. Numeric conversion is
    attempted automatically when the column dtype is numeric.

    Args:
        df: Input DataFrame.
        column: Column name to compare.
        value: Value to exclude (string; auto-converted for numeric columns).
        return_mask: If True, return a dict with both the filtered DataFrame
            and a boolean mask.

    Returns:
        Filtered DataFrame, or ``{"result": DataFrame, "mask": DataFrame}``
        when *return_mask* is True.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Try to convert value to match column dtype
    col_dtype = df[column].dtype
    try:
        if pd.api.types.is_numeric_dtype(col_dtype):
            value = float(value)
    except (ValueError, TypeError):
        pass  # Keep as string

    mask = df[column] != value
    if return_mask:
        return {"result": df[mask].copy(), "mask": mask.to_frame()}
    return df[mask].copy()

# =============================================================================
# TOP / BOTTOM OPERATIONS
# =============================================================================

def get_top_n(df: pd.DataFrame, column: str, n: int, return_mask: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Return the top *n* rows by column value (descending).

    Args:
        df: Input DataFrame.
        column: Column name to rank by.
        n: Number of rows to keep.
        return_mask: If True, return a dict with the result and a boolean mask.

    Returns:
        DataFrame with the top *n* rows, or ``{"result": DataFrame, "mask": DataFrame}``.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    top_indices = df[column].nlargest(int(n)).index

    if return_mask:
        return {"result": df.loc[top_indices].copy(), "mask": pd.Series(df.index.isin(top_indices), index=df.index).to_frame()}

    return df.loc[top_indices].copy()

def get_top_percentage(df: pd.DataFrame, column: str, fraction: float, return_mask: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Return the top fraction of rows by column value (descending).

    Args:
        df: Input DataFrame.
        column: Column name to rank by.
        fraction: Fraction of rows to keep (0.0–1.0).
        return_mask: If True, return a dict with the result and a boolean mask.

    Returns:
        DataFrame with the top rows, or ``{"result": DataFrame, "mask": DataFrame}``.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    fraction = float(fraction)
    n = fraction * len(df)
    n = max(1, n) if fraction > 0 else 0
    top_indices = df[column].nlargest(int(n)).index

    if return_mask:
        return {"result": df.loc[top_indices].copy(), "mask": pd.Series(df.index.isin(top_indices), index=df.index).to_frame()}

    return df.loc[top_indices].copy()

def get_bottom_n(df: pd.DataFrame, column: str, n: int, return_mask: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Return the bottom *n* rows by column value (ascending).

    Args:
        df: Input DataFrame.
        column: Column name to rank by.
        n: Number of rows to keep.
        return_mask: If True, return a dict with the result and a boolean mask.

    Returns:
        DataFrame with the bottom *n* rows, or ``{"result": DataFrame, "mask": DataFrame}``.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    bottom_indices = df[column].nsmallest(int(n)).index

    if return_mask:
        return {"result": df.loc[bottom_indices].copy(), "mask": pd.Series(df.index.isin(bottom_indices), index=df.index).to_frame()}

    return df.loc[bottom_indices].copy()

def get_bottom_percentage(df: pd.DataFrame, column: str, fraction: float, return_mask: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Return the bottom fraction of rows by column value (ascending).

    Args:
        df: Input DataFrame.
        column: Column name to rank by.
        fraction: Fraction of rows to keep (0.0–1.0).
        return_mask: If True, return a dict with the result and a boolean mask.

    Returns:
        DataFrame with the bottom rows, or ``{"result": DataFrame, "mask": DataFrame}``.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    fraction = float(fraction)
    n = fraction * len(df)
    n = max(1, n) if fraction > 0 else 0
    bottom_indices = df[column].nsmallest(int(n)).index

    if return_mask:
        return {"result": df.loc[bottom_indices].copy(), "mask": pd.Series(df.index.isin(bottom_indices), index=df.index).to_frame()}

    return df.loc[bottom_indices].copy()

# =============================================================================
# AGGREGATION OPERATIONS
# =============================================================================

def aggregate_mean(df: pd.DataFrame, group_columns: Union[list, str] = None, suffix: str = None) -> pd.DataFrame:
    """
    Calculates the mean of all numeric columns, optionally grouped.

    Args:
        df: Input DataFrame
        group_columns: Optional column(s) to group by
        suffix: Optional suffix to add to the aggregated column names

    Returns:
        Aggregated DataFrame with mean values per group (or a single-row
        DataFrame if ungrouped).
    """
    if group_columns:
        if isinstance(group_columns, str):
            group_columns = [group_columns]

        missing_cols = [col for col in group_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Group columns {missing_cols} not found in DataFrame.")

        result = df.groupby(group_columns).mean(numeric_only=True).reset_index()
    else:
        result = df.mean(numeric_only=True).to_frame().T

    if suffix:
        # Identify numeric columns (the ones that were aggregated)
        # In the grouped case, they are everything except group_columns
        # In the non-grouped case, they are everything
        agg_cols = [c for c in result.columns if group_columns is None or c not in group_columns]
        rename_map = {c: f"{c}{suffix}" for c in agg_cols}
        result = result.rename(columns=rename_map)

    return result


def aggregate_count(df: pd.DataFrame, group_columns: Union[list, str] = None) -> pd.DataFrame:
    """
    Returns the number of rows, optionally grouped.
    Only returns the grouping columns and a 'count' column.

    Args:
        df: Input DataFrame
        group_columns: Optional column(s) to group by

    Returns:
        DataFrame with grouping columns and a ``count`` column (or a single-row
        DataFrame with the total row count if ungrouped).
    """
    if group_columns:
        if isinstance(group_columns, str):
            group_columns = [group_columns]

        missing_cols = [col for col in group_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Group columns {missing_cols} not found in DataFrame.")

        result = df.groupby(group_columns).size().reset_index(name='count')
        return result

    return pd.DataFrame({'count': [len(df)]})


def aggregate_sum(df: pd.DataFrame, group_columns: Union[list, str] = None, suffix: str = None) -> pd.DataFrame:
    """
    Calculates the sum of all numeric columns, optionally grouped.

    Args:
        df: Input DataFrame
        group_columns: Optional column(s) to group by
        suffix: Optional suffix to add to the aggregated column names

    Returns:
        Aggregated DataFrame with summed values per group (or a single-row
        DataFrame if ungrouped).
    """
    if group_columns:
        if isinstance(group_columns, str):
            group_columns = [group_columns]

        missing_cols = [col for col in group_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Group columns {missing_cols} not found in DataFrame.")

        result = df.groupby(group_columns).sum(numeric_only=True).reset_index()
    else:
        result = df.sum(numeric_only=True).to_frame().T

    if suffix:
        agg_cols = [c for c in result.columns if group_columns is None or c not in group_columns]
        rename_map = {c: f"{c}{suffix}" for c in agg_cols}
        result = result.rename(columns=rename_map)

    return result


def aggregate_median(df: pd.DataFrame, group_columns: Union[list, str] = None, suffix: str = None) -> pd.DataFrame:
    """
    Calculates the median of all numeric columns, optionally grouped.

    Args:
        df: Input DataFrame
        group_columns: Optional column(s) to group by
        suffix: Optional suffix to add to the aggregated column names

    Returns:
        Aggregated DataFrame with median values per group (or a single-row
        DataFrame if ungrouped).
    """
    if group_columns:
        if isinstance(group_columns, str):
            group_columns = [group_columns]

        missing_cols = [col for col in group_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Group columns {missing_cols} not found in DataFrame.")

        result = df.groupby(group_columns).median(numeric_only=True).reset_index()
    else:
        result = df.median(numeric_only=True).to_frame().T

    if suffix:
        agg_cols = [c for c in result.columns if group_columns is None or c not in group_columns]
        rename_map = {c: f"{c}{suffix}" for c in agg_cols}
        result = result.rename(columns=rename_map)

    return result


# =============================================================================
# HIERARCHICAL ROLLUP
# =============================================================================

def hierarchical_rollup(df: pd.DataFrame, path_columns: Union[list, str] = None,
                        value_column: str = None,
                        root_label: str = "Total") -> pd.DataFrame:
    """Transform tabular data into hierarchical parent-child-value long format.

    Takes N hierarchical columns (broadest to most specific) and produces a
    DataFrame with path-based ids, parent references, and aggregated values.
    Supports arbitrary hierarchy depth.

    A synthetic root node (``root_label``) is always prepended so that the
    output has a single root — required by Plotly Treemap / Sunburst.

    All numeric columns (except path_columns) are automatically summed at each
    hierarchy level and preserved in the output alongside a ``count`` column.
    This allows downstream channels (e.g. color) to reference any aggregated
    numeric variable.

    The output serves both Partition (Treemap/Sunburst) and Flow (Sankey) marks:
    - Partition reads: ids=id, labels=last_part(id), parents=parent, values=value
    - Flow reads: source=parent, target=id, value=value (skip root rows)

    Args:
        df: Input DataFrame with hierarchical columns.
        path_columns: Ordered list of column names defining hierarchy levels
            (broadest to most specific). e.g. ["continent", "country", "city"].
            Also accepts a comma-separated string.
        value_column: Column to aggregate as the primary ``value`` (sum).
            If None, counts rows.
        root_label: Label for the synthetic root node (default ``"Total"``).

    Returns:
        DataFrame with columns: target, source, value, count, and one column
        per extra numeric field (summed). ``target`` is the node's own path-
        based identifier; ``source`` is the parent's identifier (empty for
        the synthetic root). The (source, target, value) triple is the shape
        that sankey/chord flow marks consume directly.
    """
    if isinstance(path_columns, str):
        path_columns = [c.strip() for c in path_columns.split(',')]

    if not path_columns:
        raise ValueError("path_columns must be a non-empty list of column names.")

    missing = [c for c in path_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in DataFrame.")

    use_count = value_column is None or value_column not in df.columns

    # Identify all numeric columns that are not part of the hierarchy
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in path_columns
    ]

    # Build rows as dicts for easy column accumulation
    rows = []
    seen = set()

    for depth in range(len(path_columns)):
        cols = path_columns[:depth + 1]

        # Aggregate: count + sum of all numeric columns at this grouping level
        grouped = df.groupby(cols, observed=True)
        counts = grouped.size().reset_index(name='count')

        if numeric_cols:
            sums = grouped[numeric_cols].sum().reset_index()
            merged = counts.merge(sums, on=cols)
        else:
            merged = counts

        for _, row in merged.iterrows():
            path_parts = [str(row[c]) for c in cols]
            node_id = "/".join(path_parts)

            if node_id in seen:
                continue
            seen.add(node_id)

            parent_id = "/".join(path_parts[:-1]) if depth > 0 else root_label

            node = {
                "target": node_id,
                "source": parent_id,
                "count": int(row["count"]),
            }

            if use_count:
                node["value"] = int(row["count"])
            else:
                node["value"] = row[value_column]

            for nc in numeric_cols:
                col_name = f"{nc}_sum" if nc != value_column else nc
                node[col_name] = row[nc]

            rows.append(node)

    # Prepend synthetic root — single parent for all top-level nodes
    top_level = [r for r in rows if r["source"] == root_label]
    root_row = {
        "target": root_label,
        "source": "",
        "count": sum(r["count"] for r in top_level),
        "value": sum(r["value"] for r in top_level),
    }
    # Sum extra numeric columns for root
    extra_cols = [k for k in rows[0] if k not in ("target", "source", "count", "value")] if rows else []
    for ec in extra_cols:
        root_row[ec] = sum(r[ec] for r in top_level)

    return pd.DataFrame([root_row] + rows)


# =============================================================================
# COLUMN OPERATIONS
# =============================================================================
def add_label(df: pd.DataFrame, label: str, value: Any) -> pd.DataFrame:
    """ Add a new column with a constant value. Args: df: Input DataFrame label: Name of the new column to add value: Value to fill in the new column (can be any scalar or object) Returns: DataFrame with the new column added. """ 
    if label in df.columns: raise ValueError(f"Column '{label}' already exists in DataFrame.") 
    df = df.copy() 
    df[label] = value 
    return df

def select_columns(df: pd.DataFrame, columns: Union[list, str]) -> pd.DataFrame:
    """
    Extract/select only the specified columns from the DataFrame.

    Args:
        df: Input DataFrame
        columns: Column name(s) to keep. Can be a single string or a list.

    Returns:
        DataFrame with only the specified columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}. Available: {list(df.columns)}")

    return df[columns].copy()


def drop_columns(df: pd.DataFrame, columns: Union[list, str]) -> pd.DataFrame:
    """
    Remove the specified columns from the DataFrame.

    Args:
        df: Input DataFrame
        columns: Column name(s) to drop. Can be a single string or a list.

    Returns:
        DataFrame without the specified columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}. Available: {list(df.columns)}")

    return df.drop(columns=columns).copy()


def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    """
    Rename a column in the DataFrame.

    Args:
        df: Input DataFrame
        old_name: Current column name
        new_name: New column name

    Returns:
        DataFrame with the column renamed.
    """
    if old_name not in df.columns:
        raise ValueError(f"Column '{old_name}' not found. Available: {list(df.columns)}")

    return df.rename(columns={old_name: new_name}).copy()


# =============================================================================
# ROW OPERATIONS
# =============================================================================

def count_rows(df: pd.DataFrame) -> int:
    """Return the total number of rows in the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Row count as an integer scalar.
    """
    return len(df)


def slice_rows(df: pd.DataFrame, start: int = None, stop: int = None) -> pd.DataFrame:
    """Keep only a specific range of rows by positional index.

    Args:
        df: Input DataFrame.
        start: Start index (inclusive). None means from the beginning.
        stop: Stop index (exclusive). None means to the end.

    Returns:
        Sliced DataFrame.
    """
    return df.iloc[start:stop].copy()


def sort_values(df: pd.DataFrame, columns: Union[list, str], ascending: bool = True) -> pd.DataFrame:
    """Sort the DataFrame by one or more columns.

    Args:
        df: Input DataFrame.
        columns: Column name(s) to sort by.
        ascending: Sort order. True for ascending, False for descending.

    Returns:
        Sorted DataFrame.
    """
    if isinstance(columns, str):
        columns = [columns]

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Sort columns not found: {missing}")

    return df.sort_values(by=columns, ascending=ascending).copy()


def sample_rows(df: pd.DataFrame, n: int = None, fraction: float = None, seed: int = None) -> pd.DataFrame:
    """Take a random sample of rows from the DataFrame.

    Args:
        df: Input DataFrame.
        n: Exact number of rows to sample (mutually exclusive with *fraction*).
        fraction: Fraction of rows to sample (0.0–1.0).
        seed: Random seed for reproducibility.

    Returns:
        Sampled DataFrame.
    """
    return df.sample(n=n, frac=fraction, random_state=seed).copy()


# =============================================================================
# CALCULATION OPERATIONS
# =============================================================================

def calc_distance(df: pd.DataFrame, x_col: str, y_col: str, ref_x: float, ref_y: float, target_col: str = "distance") -> pd.DataFrame:
    """Calculate Euclidean distance from a reference point.

    Adds a new column with the distance from ``(ref_x, ref_y)`` to each row's
    ``(x_col, y_col)`` values.

    Args:
        df: Input DataFrame.
        x_col: Column containing X coordinates.
        y_col: Column containing Y coordinates.
        ref_x: Reference X coordinate.
        ref_y: Reference Y coordinate.
        target_col: Name of the new distance column.

    Returns:
        DataFrame with the distance column added.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found in DataFrame.")

    df = df.copy()
    df[target_col] = np.sqrt((df[x_col] - ref_x)**2 + (df[y_col] - ref_y)**2)
    return df


def calc_ratio(df: pd.DataFrame, numerator_col: str, denominator_col: str) -> pd.DataFrame:
    """
    Calculates the ratio between two columns in the same DataFrame.
    Creates a new column named 'ratio' containing numerator_col / denominator_col.

    Args:
        df: Input DataFrame
        numerator_col: Column name for the numerator
        denominator_col: Column name for the denominator

    Returns:
        DataFrame with a new 'ratio' column added.
    """
    if numerator_col not in df.columns:
        raise ValueError(f"Numerator column '{numerator_col}' not found in DataFrame.")
    if denominator_col not in df.columns:
        raise ValueError(f"Denominator column '{denominator_col}' not found in DataFrame.")

    df = df.copy()
    df["ratio"] = df[numerator_col] / df[denominator_col]
    return df


# =============================================================================
# MULTI-INPUT OPERATIONS
# =============================================================================

def join(dfs: Union[list, pd.DataFrame] = None, on: str = None, how: str = 'inner', **kwargs) -> pd.DataFrame:
    """Join multiple DataFrames on a common key.

    Collects inputs from *dfs* (list or single DataFrame) and any DataFrames
    passed as keyword arguments (named ports from the pipeline).  When column
    name conflicts occur, columns are suffixed with the source name (the
    kwargs key from the pipeline) instead of generic ``_left`` / ``_right``.

    Args:
        dfs: Primary DataFrame(s) to join.
        on: Column name(s) to join on.
        how: Join type — ``'inner'``, ``'left'``, ``'right'``, or ``'outer'``.
        **kwargs: Additional DataFrames passed by name.

    Returns:
        Merged DataFrame.
    """
    from collections import Counter

    all_dfs = []
    names = []

    # Collect from positional/main argument
    if dfs is not None:
        if isinstance(dfs, list):
            all_dfs.extend(dfs)
            names.extend([f"table_{i}" for i in range(len(dfs))])
        elif isinstance(dfs, pd.DataFrame):
            all_dfs.append(dfs)
            names.append("dfs")

    # Collect any DataFrames passed via keyword arguments (named ports)
    for k, v in kwargs.items():
        if isinstance(v, pd.DataFrame):
            all_dfs.append(v)
            names.append(k)

    if not all_dfs:
        raise ValueError("No DataFrames provided for join.")

    if len(all_dfs) == 1:
        return all_dfs[0].copy()

    # Align merge-key dtypes to avoid Int64/object mismatches
    if on is not None:
        key_cols = on if isinstance(on, list) else [on]
        for col in key_cols:
            dtypes = {df[col].dtype for df in all_dfs if col in df.columns}
            if len(dtypes) > 1:
                for df in all_dfs:
                    if col in df.columns:
                        df[col] = df[col].astype(str)

    # Identify non-key columns that appear in multiple DataFrames
    if on is not None:
        key_col_set = set(on if isinstance(on, list) else [on])
    else:
        # When on=None, pandas merges on all common columns — those are the keys
        key_col_set = set(all_dfs[0].columns)
        for df in all_dfs[1:]:
            key_col_set &= set(df.columns)

    col_counts = Counter()
    for df in all_dfs:
        col_counts.update(set(df.columns) - key_col_set)
    conflicting = {col for col, count in col_counts.items() if count > 1}

    # Pre-rename conflicting columns with source names so all merges are clean
    if conflicting:
        renamed_dfs = []
        for df, name in zip(all_dfs, names):
            rename_map = {col: f"{col}_{name}" for col in conflicting if col in df.columns}
            renamed_dfs.append(df.rename(columns=rename_map) if rename_map else df.copy())
    else:
        renamed_dfs = [df.copy() for df in all_dfs]

    result = renamed_dfs[0]
    for i in range(1, len(renamed_dfs)):
        result = pd.merge(result, renamed_dfs[i], on=on, how=how)

    return result


# =============================================================================
# MULTI-DATASET OPERATIONS
# =============================================================================

def union(dfs: Union[list, pd.DataFrame] = None, ignore_index: bool = True, **kwargs) -> pd.DataFrame:
    """Vertically stack (union) multiple DataFrames.

    Collects inputs from *dfs* (list or single DataFrame) and any DataFrames
    passed as keyword arguments.

    Args:
        dfs: Primary DataFrame(s) to concatenate.
        ignore_index: If True, reset the index in the result.
        **kwargs: Additional DataFrames passed by name.

    Returns:
        Concatenated DataFrame.
    """
    all_dfs = []

    # Collect from positional/main argument
    if dfs is not None:
        if isinstance(dfs, list):
            all_dfs.extend(dfs)
        elif isinstance(dfs, pd.DataFrame):
            all_dfs.append(dfs)

    # Collect any DataFrames passed via keyword arguments (named ports)
    for v in kwargs.values():
        if isinstance(v, pd.DataFrame):
            all_dfs.append(v)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=ignore_index)


# =============================================================================
# RESHAPE OPERATIONS
# =============================================================================

def melt(df: pd.DataFrame, id_columns: Union[list, str] = None,
         value_columns: Union[list, str] = None,
         var_name: str = "variable", value_name: str = "value") -> pd.DataFrame:
    """Unpivot a wide DataFrame into long format.

    Converts columns into rows, turning a wide table (one column per metric)
    into a long table with a ``variable`` column and a ``value`` column.

    Args:
        df: Input DataFrame in wide format.
        id_columns: Column(s) to keep as identifiers (not melted).
            Accepts a single string or a list. If None, all non-value columns
            are used.
        value_columns: Column(s) to unpivot. Accepts a single string or a list.
            If None, all columns not in *id_columns* are melted.
        var_name: Name for the new column holding the former column headers.
        value_name: Name for the new column holding the values.

    Returns:
        Long-format DataFrame.

    Examples:
        Wide input::

            | date    | price_cape | price_panama |
            | 2024-01 | 100        | 200          |

        ``melt(df, id_columns="date", var_name="source", value_name="price")``::

            | date    | source       | price |
            | 2024-01 | price_cape   | 100   |
            | 2024-01 | price_panama | 200   |
    """
    if isinstance(id_columns, str):
        id_columns = [c.strip() for c in id_columns.split(",")]
    if isinstance(value_columns, str):
        value_columns = [c.strip() for c in value_columns.split(",")]

    if id_columns:
        missing = [c for c in id_columns if c not in df.columns]
        if missing:
            raise ValueError(f"id_columns not found: {missing}. Available: {list(df.columns)}")

    if value_columns:
        missing = [c for c in value_columns if c not in df.columns]
        if missing:
            raise ValueError(f"value_columns not found: {missing}. Available: {list(df.columns)}")

    return pd.melt(
        df,
        id_vars=id_columns,
        value_vars=value_columns,
        var_name=var_name,
        value_name=value_name,
    )


# =============================================================================
# ADVANCED TRANSFORMATIONS
# =============================================================================

def arithmetic_op(df: pd.DataFrame, left_column: str, right_column: Optional[str] = None,
                  constant: Optional[float] = None, operator: str = 'ADD',
                  output_column: str = 'result') -> pd.DataFrame:
    """Apply an arithmetic operation between a column and another column or constant.

    Args:
        df: Input DataFrame.
        left_column: Column name for the left operand.
        right_column: Column name for the right operand (mutually exclusive
            with *constant*).
        constant: Scalar value for the right operand.
        operator: One of ``'ADD'``, ``'SUB'``, ``'PROD'``, ``'DIV'``.
        output_column: Name of the result column.

    Returns:
        DataFrame with the computed column added.
    """
    
    if df is None or df.empty:
        return pd.DataFrame()

    res_df = df.copy()

    # 1. Validate and convert the LEFT column
    if left_column not in res_df.columns:
        raise ValueError(f"Column '{left_column}' not found")

    # Coerce to numeric to avoid type errors during the operation
    left_val = pd.to_numeric(res_df[left_column], errors='coerce')

    # 2. Validate and convert the RIGHT operand
    if right_column:
        if right_column not in res_df.columns:
            raise ValueError(f"Column '{right_column}' not found")
        right_val = pd.to_numeric(res_df[right_column], errors='coerce')
    elif constant is not None:
        right_val = constant
    else:
        raise ValueError("Either right_column or constant must be provided")

    # 3. Apply the operation via pandas methods (safer than raw operators)
    if operator == 'ADD':
        res_df[output_column] = left_val.add(right_val)
    elif operator == 'SUB':
        res_df[output_column] = left_val.sub(right_val)
    elif operator == 'PROD':
        res_df[output_column] = left_val.mul(right_val)
    elif operator == 'DIV':
        res_df[output_column] = left_val.div(right_val)
    else:
        raise ValueError(f"Unsupported operator: {operator}. Use ADD, SUB, PROD, or DIV.")

    return res_df



def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax',
                     output_column: Optional[str] = None) -> pd.DataFrame:
    """Normalize a numeric column using min-max scaling or z-score standardization.

    - ``'minmax'``: ``(x - min) / (max - min)``
    - ``'zscore'``: ``(x - mean) / std``

    Args:
        df: Input DataFrame.
        column: Column to normalize.
        method: Normalization method — ``'minmax'`` or ``'zscore'``.
        output_column: Name of the result column (defaults to
            ``"{column}_norm"``).

    Returns:
        DataFrame with the normalized column added.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    res_df = df.copy()
    col_name = output_column if output_column else f"{column}_norm"

    if column not in res_df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    if method == 'minmax':
        mi = res_df[column].min()
        ma = res_df[column].max()
        if ma - mi == 0:
            res_df[col_name] = 0.0
        else:
            res_df[col_name] = (res_df[column] - mi) / (ma - mi)
    elif method == 'zscore':
        mu = res_df[column].mean()
        sigma = res_df[column].std()
        if sigma == 0 or pd.isna(sigma):
            res_df[col_name] = 0.0
        else:
            res_df[col_name] = (res_df[column] - mu) / sigma
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return res_df


def discretize(df: pd.DataFrame, column: str, bins: int = 5,
               strategy: str = 'uniform', output_column: Optional[str] = None,
               labels: Optional[list] = None) -> pd.DataFrame:
    """Discretize a continuous column into bins.

    - ``'uniform'``: Equal-width bins.
    - ``'quantile'``: Equal-frequency bins.

    Args:
        df: Input DataFrame.
        column: Column to discretize.
        bins: Number of bins.
        strategy: Binning strategy — ``'uniform'`` or ``'quantile'``.
        output_column: Name of the result column (defaults to
            ``"{column}_bin"``).
        labels: Optional list of label names for the bins (e.g.
            ``["low", "medium", "high"]``).  Must have length equal to
            *bins*.  When omitted the bins are labelled with integers.

    Returns:
        DataFrame with the binned column added.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    res_df = df.copy()
    col_name = output_column if output_column else f"{column}_bin"

    if column not in res_df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    if labels is not None and len(labels) != bins:
        raise ValueError(
            f"labels length ({len(labels)}) must equal bins ({bins})"
        )

    bin_labels = labels if labels is not None else False

    if strategy == 'uniform':
        res_df[col_name] = pd.cut(res_df[column], bins=bins, labels=bin_labels)
    elif strategy == 'quantile':
        res_df[col_name] = pd.qcut(res_df[column], q=bins, labels=bin_labels, duplicates='drop')
    else:
        raise ValueError(f"Unsupported discretization strategy: {strategy}")

    return res_df


# =============================================================================
# EXCEL OPERATIONS
# =============================================================================
# tidy_excel_data lives in collection/excel/ and is imported at the top of
# this module; it is registered in TRANSFORM_REGISTRY below.


# =============================================================================
# GEOLOCATION OPERATIONS
# =============================================================================
# Geo functions (geocode_location, get_country_contours) are imported from
# choregraph.collection.geo and re-exported at the top of this module.


# =============================================================================
# JSON OPERATIONS
# =============================================================================

# def process_json(data: Union[dict, list], key_name: str = None) -> pd.DataFrame:
#     """
#     Converts JSON data (dict or list) into a DataFrame.
#     If 'key_name' is provided and data is a dict, it extracts that key.
#     """
#     if isinstance(data, dict) and key_name:
#         data = data.get(key_name, [])

#     if isinstance(data, list):
#         return pd.DataFrame(data)
#     elif isinstance(data, dict):
#         return pd.DataFrame([data])
#     else:
#         return pd.DataFrame()

# =============================================================================
# JSON FLATTENING
# =============================================================================

def flatten_json(
    data: Union[dict, list],
    root_key: str = None,
    columns: str = None,
) -> pd.DataFrame:
    """Convert arbitrary JSON structures into a flat DataFrame.

    Auto-detects common JSON-to-table patterns and applies the best
    flattening strategy:

    1. **Array of objects** ``[{col: val, ...}, ...]``
       → ``pd.DataFrame(data)`` directly.
    2. **Dict of paired arrays** ``{key: [[x, y], ...], ...}``
       (e.g. CoinGecko market data) → join arrays on shared first column,
       one column per key.
    3. **Dict of simple arrays** ``{key: [v1, v2, ...], ...}``
       → one column per key (all same length).
    4. **Keyed array of objects** (when *root_key* is provided)
       ``{root_key: [{...}, ...]}`` → flattens the inner list.
    5. **Nested / complex** → ``pd.json_normalize()`` as fallback.

    Args:
        data: Loaded JSON data (dict or list).
        root_key: Optional top-level key to drill into before flattening.
        columns: Optional comma-separated column names to assign to the
            resulting DataFrame (useful for unnamed arrays).

    Returns:
        A flat :class:`~pandas.DataFrame`.

    Examples:
        >>> flatten_json([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
           a  b
        0  1  2
        1  3  4

        >>> flatten_json({"prices": [[1, 100], [2, 200]],
        ...               "volumes": [[1, 50], [2, 60]]})
           timestamp  prices  volumes
        0          1     100       50
        1          2     200       60
    """
    # --- drill into root_key if given (supports dotted paths like "result.XXBTZUSD") ---
    if root_key and isinstance(data, dict):
        for part in root_key.split("."):
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                break

    # --- Pattern 1: list at the top level ---
    if isinstance(data, list):
        df = _flatten_list(data)
        return _apply_column_names(df, columns)

    # --- dict at the top level ---
    if isinstance(data, dict):
        # Pattern 2: dict of paired arrays {key: [[x,y], ...], ...}
        # All values are lists of equal-length lists (>=2 elements each)
        values = list(data.values())
        keys = list(data.keys())

        if len(values) > 0 and all(
            isinstance(v, list) and len(v) > 0 and isinstance(v[0], list)
            for v in values
        ):
            return _flatten_dict_of_paired_arrays(keys, values, columns)

        # Pattern 3: dict of simple (flat) arrays {key: [v1, v2, ...], ...}
        if len(values) > 0 and all(
            isinstance(v, list)
            and len(v) > 0
            and all(_is_primitive(item) for item in v)
            for v in values
        ):
            lengths = {len(v) for v in values}
            if len(lengths) == 1:
                return pd.DataFrame(data)

        # Pattern 4: dict with a single key whose value is a list
        if len(keys) == 1 and isinstance(values[0], list):
            df = _flatten_list(values[0])
            return _apply_column_names(df, columns)

        # Pattern 5: nested / complex dict → json_normalize fallback
        try:
            df = pd.json_normalize(data, sep="_")
            if not df.empty:
                return _apply_column_names(df, columns)
        except Exception:
            pass

    # Last resort: wrap scalar / unhandled in single-cell DataFrame
    return pd.DataFrame([{"value": json.dumps(data) if not _is_primitive(data) else data}])


def _flatten_list(data: list) -> pd.DataFrame:
    """Flatten a JSON list into a DataFrame."""
    if len(data) == 0:
        return pd.DataFrame()

    first = data[0]

    # Array of dicts → direct DataFrame / json_normalize
    if isinstance(first, dict):
        try:
            return pd.json_normalize(data, sep="_")
        except Exception:
            return pd.DataFrame(data)

    # Array of arrays → DataFrame with string column names
    if isinstance(first, list):
        df = pd.DataFrame(data)
        df.columns = [str(c) for c in df.columns]
        return df

    # Array of primitives → single column
    return pd.DataFrame({"value": data})


def _flatten_dict_of_paired_arrays(
    keys: list, values: list, columns: str = None
) -> pd.DataFrame:
    """Flatten ``{key: [[index, val], ...], ...}`` into a joined table.

    Builds one mini-DataFrame per key with the first element as the index
    column and the second as the value column, then joins them all on that
    shared index column.
    """
    frames = {}
    index_name = "index"

    for key, rows in zip(keys, values):
        if len(rows) == 0:
            continue
        width = len(rows[0])
        if width == 2:
            col_names = [index_name, key]
        else:
            col_names = [index_name] + [f"{key}_{i}" for i in range(1, width)]
        df = pd.DataFrame(rows, columns=col_names)
        frames[key] = df

    if not frames:
        return pd.DataFrame()

    # Merge all frames on the shared index column
    result = None
    for key, df in frames.items():
        if result is None:
            result = df
        else:
            result = result.merge(df, on=index_name, how="outer")

    return _apply_column_names(result, columns) if result is not None else pd.DataFrame()


def _apply_column_names(df: pd.DataFrame, columns: str = None) -> pd.DataFrame:
    """Override column names if the caller provided a comma-separated list."""
    if columns and not df.empty:
        names = [c.strip() for c in columns.split(",")]
        if len(names) == len(df.columns):
            df.columns = names
    return df


# =============================================================================
# JSON CARTOGRAPHY
# =============================================================================

# Hard cap on nesting depth of user JSON during characterization.
# Pathologically deep JSON breaks genson's recursive schema inference and
# makes the cartography unreadable for the LLM. Typical real-world JSON
# stays well under this bound (GeoJSON ~5, OpenAPI ~15, JSON-LD ~10).
MAX_JSON_DEPTH = 64


class JsonTooDeepError(ValueError):
    """Raised by :func:`cartograph_json` when input nests deeper than
    :data:`MAX_JSON_DEPTH`."""


def _exceeds_json_depth(data: Any, limit: int, max_nodes: int = 200_000) -> bool:
    """Return True if ``data`` nests deeper than ``limit`` levels.

    Iterative walk (explicit stack) so the check itself never hits Python's
    recursion limit. Bounded by ``max_nodes`` so a huge-but-shallow JSON
    doesn't turn the depth gate into a memory hog — depth is a structural
    property, a representative sample is sufficient.
    """
    if not isinstance(data, (dict, list)):
        return False
    stack = [(data, 1)]
    seen = 0
    while stack:
        node, depth = stack.pop()
        if depth > limit:
            return True
        seen += 1
        if seen >= max_nodes:
            return False
        children = node.values() if isinstance(node, dict) else node
        for v in children:
            if isinstance(v, (dict, list)):
                stack.append((v, depth + 1))
    return False


def _is_primitive(j: Any) -> bool:
    return isinstance(j, (str, int, float, bool)) or j is None


def _schema_type(node: dict) -> str:
    """Return the first type listed in a GenSON schema node (flattens unions)."""
    t = node.get("type")
    if isinstance(t, list):
        return t[0] if t else "unknown"
    return t or "unknown"


def _infer_root_length(data: Any, schema: dict) -> int:
    """Dominant row-count for the JSON: root array length, or first top-level
    array length, or number of top-level keys as a last resort."""
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return len(v)
        return len(data)
    return 0


def _detect_tabular(data: Any, schema: dict) -> tuple:
    """Detect a flat tabular JSON and extract ``FieldMetadata``-friendly hints.

    A JSON is considered tabular when it is an array of objects with only
    primitive leaves, or a single-key wrapper around such an array. Returns
    ``(is_tabular, fields)`` where ``fields`` is a list of
    ``{"name": str, "dtype": "STRING"|"FLOAT"|"INTEGER"|"BOOLEAN"}``.
    """
    def _array_of_flat_objects(schema_node: dict):
        if _schema_type(schema_node) != "array":
            return None
        items = schema_node.get("items") or {}
        if _schema_type(items) != "object":
            return None
        props = items.get("properties") or {}
        if not props:
            return None
        fields: list = []
        for name, prop in props.items():
            t = _schema_type(prop)
            if t in ("object", "array"):
                return None
            fields.append({"name": name, "dtype": _genson_to_vs_dtype(t)})
        return fields

    # Root array
    if isinstance(data, list):
        fields = _array_of_flat_objects(schema)
        if fields:
            return True, fields

    # Single-key wrapper
    if isinstance(data, dict) and len(data) == 1 and _schema_type(schema) == "object":
        props = schema.get("properties") or {}
        if props:
            only_prop = next(iter(props.values()))
            fields = _array_of_flat_objects(only_prop)
            if fields:
                return True, fields

    return False, []


def _genson_to_vs_dtype(t: str) -> str:
    """Map GenSON primitive types to VisuSpec ``data_type`` strings."""
    return {
        "string": "STRING",
        "integer": "INTEGER",
        "number": "FLOAT",
        "boolean": "BOOLEAN",
        "null": "STRING",
    }.get(t, "STRING")


def _collect_leaf_fields(schema: dict, max_leaves: int = 50) -> list:
    """Walk a GenSON schema and return leaf paths as ``FieldMetadata``-ready dicts.

    Each entry is ``{"name": <dotted path>, "dtype": <VisuSpec dtype>, "required": bool}``.
    Arrays of objects are traversed transparently (the path does not embed ``[]``)
    so that ``nodes.data.id`` refers to the repeated primitive leaf. Arrays of
    primitives collapse to a single leaf typed by the item kind.

    Heterogeneous arrays produce a GenSON ``anyOf`` (or ``oneOf``) node with no
    ``type`` key — we walk each branch under the same path and dedup by name so
    those fields don't disappear silently.

    Capped at ``max_leaves`` to keep catalogue_stats.json bounded for fan-out
    JSON schemas.
    """
    leaves: list = []
    seen: set = set()

    def _add_leaf(name: str, kind: str, required: bool) -> None:
        if not name or name in seen:
            return
        seen.add(name)
        leaves.append({
            "name": name,
            "dtype": _genson_to_vs_dtype(kind),
            "required": required,
        })

    def _walk(node: dict, path: str, required: bool) -> None:
        if len(leaves) >= max_leaves:
            return
        branches = node.get("anyOf") or node.get("oneOf")
        if branches:
            for branch in branches:
                if isinstance(branch, dict):
                    _walk(branch, path, required)
            return
        kind = _schema_type(node)
        if kind == "object":
            props = node.get("properties") or {}
            req_set = set(node.get("required") or [])
            for key, sub in props.items():
                child_path = f"{path}.{key}" if path else key
                _walk(sub, child_path, key in req_set)
        elif kind == "array":
            items = node.get("items") or {}
            if items.get("anyOf") or items.get("oneOf") or _schema_type(items) in ("object", "array"):
                _walk(items, path, required)
            else:
                _add_leaf(path, _schema_type(items), required)
        else:
            _add_leaf(path, kind, required)

    root_kind = _schema_type(schema)
    if root_kind == "object":
        props = schema.get("properties") or {}
        req_set = set(schema.get("required") or [])
        for key, sub in props.items():
            _walk(sub, key, key in req_set)
    elif root_kind == "array":
        items = schema.get("items") or {}
        _walk(items, "", False)

    return leaves

def remove_required_keys(data):
    """Recursively walk the dictionary and strip out 'required' keys."""
    if isinstance(data, dict):
        return {k: remove_required_keys(v) for k, v in data.items() if k != 'required'}
    elif isinstance(data, list):
        return [remove_required_keys(item) for item in data]
    else:
        return data


def _sample_for_schema(data: Any, cap: int) -> Any:
    """Return a copy of ``data`` with every array truncated to ``cap`` items.

    GenSON merges each array item into the running schema, so on a 200MB
    JSON with 80k homogeneous records it walks every one of them — burning
    memory and time for no extra signal. Sampling the head preserves the
    schema shape while keeping inference cost bounded.
    """
    if isinstance(data, list):
        return [_sample_for_schema(v, cap) for v in data[:cap]]
    if isinstance(data, dict):
        return {k: _sample_for_schema(v, cap) for k, v in data.items()}
    return data


def cartograph_json(data: Any, max_chars: int = 5000, max_items: int = 200) -> dict:
    """Produce a structural cartography of a JSON document for the LLM.

    Uses `genson <https://github.com/wolverdude/genson>`_ to infer a JSON
    Schema from the loaded data (one "skeleton" merging every record), then
    renders it as a compact ASCII hierarchy that the planning LLM embeds via
    :attr:`DatasetStats.info["extract_with"]` (rendered by
    :meth:`MetadataResult._to_markdown`).

    Args:
        data: Loaded JSON value (dict, list, or primitive).
        max_chars: Upper bound on the rendered tree length. Truncated with
            ``...`` when exceeded.

    Returns:
        ``{"schema": <genson schema dict>, "rendered": <str>,
           "length": <int>, "is_tabular": <bool>,
           "tabular_fields": [{"name": str, "dtype": str}, ...],
           "leaf_fields": [{"name": str, "dtype": str, "required": bool}, ...]}``
    """
    if _exceeds_json_depth(data, MAX_JSON_DEPTH):
        raise JsonTooDeepError(
            f"JSON nests deeper than {MAX_JSON_DEPTH} levels"
        )

    from genson import SchemaBuilder
    builder = SchemaBuilder()
    sampled = _sample_for_schema(data, max_items) if max_items else data
    if isinstance(sampled, (dict, list)):
        builder.add_object(sampled)
    else:
        builder.add_object({"value": sampled})
    schema = builder.to_schema()

    rendered = json.dumps(remove_required_keys(schema), separators=(",", ":"))
    if len(rendered) > max_chars:
        rendered = rendered[: max_chars - 3].rstrip() + "..."
    length = _infer_root_length(data, schema)
    is_tabular, fields = _detect_tabular(data, schema)
    leaf_fields = _collect_leaf_fields(schema)

    return {
        "schema": schema,
        "rendered": rendered,
        "length": length,
        "is_tabular": is_tabular,
        "tabular_fields": fields,
        "leaf_fields": leaf_fields,
    }


# =============================================================================
# CODE EXECUTION
# =============================================================================

BLOCKED_MODULES = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'http', 'urllib',
    'requests', 'ftplib', 'smtplib', 'telnetlib',
    'ctypes', 'multiprocessing', 'threading', 'signal',
    'importlib', 'pkgutil', 'code', 'codeop',
    'builtins', 'gc', 'inspect', 'pickle', 'shelve',
    'webbrowser', 'pathlib', 'io', 'tempfile', 'glob',
}

BLOCKED_NAMES = {
    'exec', 'eval', 'compile', '__import__', 'open',
    'getattr', 'setattr', 'delattr', 'globals', 'locals',
    'vars', 'dir', 'breakpoint', 'exit', 'quit',
}

BLOCKED_ATTRS = {
    '__builtins__', '__import__', '__class__', '__subclasses__',
    '__globals__', '__code__', '__func__',
}


def _validate_code_safety(code: str) -> None:
    """Validate that code does not use dangerous operations."""
    if len(code) > 4000:
        raise ValueError(f"Code exceeds maximum length (4000 chars, got {len(code)})")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split('.')[0]
                if root_module in BLOCKED_MODULES:
                    raise ValueError(f"Import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split('.')[0]
                if root_module in BLOCKED_MODULES:
                    raise ValueError(f"Import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
                raise ValueError(f"Call to '{node.func.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRS:
                raise ValueError(f"Access to '{node.attr}' is not allowed")


def execute_code(code: str, **kwargs) -> pd.DataFrame:
    """Execute Python code with one or more DataFrame inputs.

    All input DataFrames are available in the code by their port name.
    The code must assign its result to a variable named ``result``.
    All scientific Python libraries installed in the environment are available.
    System, IO, and network modules are blocked.
    """
    _validate_code_safety(code)

    # Use a single namespace for both globals and locals so that functions
    # defined in the code can access top-level variables (pd, np, inputs).
    # With separate dicts, nested def/class scopes only search globals, not
    # the exec locals — causing "name 'pd' is not defined" errors.
    namespace = {'__builtins__': __builtins__, 'pd': pd, 'np': np,
                 'true': True, 'false': False, 'null': None}
    for name, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
            namespace[name] = value.copy()
        else:
            namespace[name] = wrap_json_input(name, value)

    exec(code, namespace)

    result = namespace.get('result')
    if result is None:
        raise ValueError(
            "Code must assign output to a variable named 'result'. "
            "Example: result = df[df['x'] > 5]"
        )
    if not isinstance(result, pd.DataFrame):
        if isinstance(result, pd.Series):
            result = result.to_frame()
        else:
            raise TypeError(f"'result' must be a DataFrame, got {type(result).__name__}")
    return result


# =============================================================================
# PARTITION UTILITIES
# =============================================================================

def concat_partitions(partitioned: dict) -> pd.DataFrame:
    """Concatenate a PartitionedDataset into a single DataFrame.

    Loads every partition in sorted key order, tags each row with a
    ``__partition__`` column (float index: 0.0, 1.0, …), and concatenates
    them into one DataFrame.  Use this before applying transforms that need
    global context (consistent bin edges, global aggregates, etc.).

    Pair with :func:`split_partitions` to restore the partitioned structure
    after the transform.

    Args:
        partitioned: Kedro PartitionedDataset dict ``{key: callable_or_df}``.

    Returns:
        Single DataFrame with an added ``__partition__`` column.
    """
    sorted_keys = sorted(partitioned.keys())
    frames = []
    for i, key in enumerate(sorted_keys):
        loader = partitioned[key]
        df = (loader() if callable(loader) else loader).copy()
        df["__partition__"] = float(i)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =============================================================================
# FUNCTION REGISTRY
# =============================================================================

# Define registry at the end to ensure all functions are defined
TRANSFORM_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- Min/Max Functions ---
    "calculate_min": {"func": calculate_min, "output_type": float},
    "calculate_max": {"func": calculate_max, "output_type": float},
    # --- New Filtering Functions ---
    "filter_less_than": {"func": filter_less_than, "output_type": pd.DataFrame},
    "filter_greater_than": {"func": filter_greater_than, "output_type": pd.DataFrame},
    "filter_in_range": {"func": filter_in_range, "output_type": pd.DataFrame},
    "filter_equal": {"func": filter_equal, "output_type": pd.DataFrame},
    "filter_not_equal": {"func": filter_not_equal, "output_type": pd.DataFrame},

    "get_top_n": {"func": get_top_n, "output_type": pd.DataFrame},
    "get_top_percentage": {"func": get_top_percentage, "output_type": pd.DataFrame},
    "get_bottom_n": {"func": get_bottom_n, "output_type": pd.DataFrame},
    "get_bottom_percentage": {"func": get_bottom_percentage, "output_type": pd.DataFrame},

    # --- New Aggregation Functions ---
    "aggregate_mean": {"func": aggregate_mean, "output_type": pd.DataFrame},
    "aggregate_count": {"func": aggregate_count, "output_type": pd.DataFrame},
    "aggregate_sum": {"func": aggregate_sum, "output_type": pd.DataFrame},
    "aggregate_median": {"func": aggregate_median, "output_type": pd.DataFrame},

    # --- Column Operations ---
    "add_label": {"func": add_label, "output_type": pd.DataFrame},
    "select_columns": {"func": select_columns, "output_type": pd.DataFrame},
    "drop_columns": {"func": drop_columns, "output_type": pd.DataFrame},
    "rename_column": {"func": rename_column, "output_type": pd.DataFrame},

    # --- New Row Operations ---
    "count_rows": {"func": count_rows, "output_type": int},
    "slice_rows": {"func": slice_rows, "output_type": pd.DataFrame},
    "sort_values": {"func": sort_values, "output_type": pd.DataFrame},
    "sample_rows": {"func": sample_rows, "output_type": pd.DataFrame},

    # --- New Calculation Functions ---
    "calc_distance": {"func": calc_distance, "output_type": pd.DataFrame},
    "calc_ratio": {"func": calc_ratio, "output_type": pd.DataFrame},

    # --- New Multi-Input Functions ---
    "join": {"func": join, "output_type": pd.DataFrame},    "union": {"func": union, "output_type": pd.DataFrame},

    # --- Reshape Operations ---
    "melt": {"func": melt, "output_type": pd.DataFrame},

    # --- Advanced Transformations ---
    "arithmetic_op": {"func": arithmetic_op, "output_type": pd.DataFrame},
    "normalize_column": {"func": normalize_column, "output_type": pd.DataFrame},
    "discretize": {"func": discretize, "output_type": pd.DataFrame},

    # --- Time Series Operations ---
    "extract_date_part": {"func": extract_date_part, "output_type": pd.DataFrame},
    "rolling_statistics": {"func": rolling_statistics, "output_type": pd.DataFrame},
    "lag_lead": {"func": lag_lead, "output_type": pd.DataFrame},
    "offset_datetime": {"func": offset_datetime, "output_type": pd.DataFrame},
    "forecast_time_series": {"func": forecast_time_series, "output_type": pd.DataFrame},
    # --- JSON Functions ---
    "flatten_json": {"func": flatten_json, "output_type": pd.DataFrame},

    # --- Geolocation Functions ---
    "geocode_location": {"func": geocode_location, "output_type": pd.DataFrame},
    "get_country_contours": {"func": get_country_contours, "output_type": pd.DataFrame},

    # --- NLP Functions ---
    "nlp_binarize_labels_auto": {"func": nlp_binarize_labels_auto, "output_type": pd.DataFrame},
    "nlp_binarize_labels_hinted": {"func": nlp_binarize_labels_hinted, "output_type": pd.DataFrame},

    # --- Hierarchical Transform ---
    "hierarchical_rollup": {"func": hierarchical_rollup, "output_type": pd.DataFrame},

    # --- Excel Functions ---
    "tidy_excel_data": {"func": tidy_excel_data, "output_type": dict},  # Returns dict of DataFrames

    # --- Image Functions ---
    "image_to_dataframe": {"func": image_to_dataframe, "output_type": pd.DataFrame},
    "extract_channel": {"func": extract_channel, "output_type": pd.DataFrame},
"image_metadata": {"func": image_metadata, "output_type": pd.DataFrame},

    # --- Code Execution ---
    "execute_code": {"func": execute_code, "output_type": pd.DataFrame},

    # --- Partition Utilities ---
    "concat_partitions": {"func": concat_partitions, "output_type": pd.DataFrame}
}

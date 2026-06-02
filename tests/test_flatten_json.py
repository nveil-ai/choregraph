# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for flatten_json transform."""
import json
import pytest
import pandas as pd

from choregraph.library import flatten_json, TRANSFORM_REGISTRY


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registered():
    assert "flatten_json" in TRANSFORM_REGISTRY
    assert TRANSFORM_REGISTRY["flatten_json"]["output_type"] is pd.DataFrame


# ---------------------------------------------------------------------------
# Pattern 1: array of objects
# ---------------------------------------------------------------------------

def test_array_of_objects():
    data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
    df = flatten_json(data)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 3
    assert df["a"].tolist() == [1, 2, 3]


def test_array_of_objects_nested():
    """Nested dicts should be flattened via json_normalize."""
    data = [
        {"id": 1, "info": {"name": "Alice", "age": 30}},
        {"id": 2, "info": {"name": "Bob", "age": 25}},
    ]
    df = flatten_json(data)
    assert "info_name" in df.columns
    assert "info_age" in df.columns
    assert len(df) == 2


# ---------------------------------------------------------------------------
# Pattern 2: dict of paired arrays (CoinGecko style)
# ---------------------------------------------------------------------------

def test_coingecko_style():
    """The exact pattern from the CoinGecko API."""
    data = {
        "prices": [[1000, 100.5], [2000, 200.3], [3000, 150.0]],
        "market_caps": [[1000, 9999], [2000, 8888], [3000, 7777]],
        "total_volumes": [[1000, 50], [2000, 60], [3000, 70]],
    }
    df = flatten_json(data)
    # Should have 4 columns: shared index + one per key
    assert len(df.columns) == 4
    assert "prices" in df.columns
    assert "market_caps" in df.columns
    assert "total_volumes" in df.columns
    assert len(df) == 3
    assert df["prices"].tolist() == [100.5, 200.3, 150.0]


def test_paired_arrays_unix_ms_stays_numeric():
    """Unix ms index values should stay numeric (DtypeInferenceHook handles conversion)."""
    data = {
        "prices": [[1771680301648, 68170.87], [1771680615983, 68215.58]],
    }
    df = flatten_json(data)
    assert "index" in df.columns
    assert pd.api.types.is_numeric_dtype(df["index"])


def test_paired_arrays_non_unix_stays_numeric():
    """Small index values should stay numeric."""
    data = {
        "values": [[1, 10], [2, 20], [3, 30]],
    }
    df = flatten_json(data)
    index_col = df.columns[0]
    assert pd.api.types.is_numeric_dtype(df[index_col])


# ---------------------------------------------------------------------------
# Pattern 3: dict of simple (flat) arrays
# ---------------------------------------------------------------------------

def test_dict_of_simple_arrays():
    data = {"x": [1, 2, 3], "y": [4, 5, 6]}
    df = flatten_json(data)
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 3
    assert df["x"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Pattern 4: root_key drill-down
# ---------------------------------------------------------------------------

def test_root_key():
    data = {"result": [{"a": 1}, {"a": 2}], "status": "ok"}
    df = flatten_json(data, root_key="result")
    assert list(df.columns) == ["a"]
    assert len(df) == 2


def test_root_key_dotted_path():
    """Dotted root_key should traverse nested dicts (e.g. Kraken API)."""
    data = {
        "error": [],
        "result": {
            "XXBTZUSD": [[1, "open1"], [2, "open2"]],
            "last": 999,
        },
    }
    df = flatten_json(data, root_key="result.XXBTZUSD")
    assert len(df) == 2
    assert len(df.columns) == 2


# ---------------------------------------------------------------------------
# Pattern 5: nested dict (json_normalize fallback)
# ---------------------------------------------------------------------------

def test_nested_dict_normalize():
    data = {"name": "test", "count": 42, "nested": {"x": 1}}
    df = flatten_json(data)
    assert not df.empty
    assert "name" in df.columns


# ---------------------------------------------------------------------------
# columns override
# ---------------------------------------------------------------------------

def test_columns_override():
    data = [[1, 100], [2, 200]]
    df = flatten_json(data, columns="time, value")
    assert list(df.columns) == ["time", "value"]
    assert len(df) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_list():
    df = flatten_json([])
    assert df.empty


def test_empty_dict():
    df = flatten_json({})
    assert not df.empty  # falls through to json_normalize or scalar wrap


def test_array_of_primitives():
    data = [10, 20, 30]
    df = flatten_json(data)
    assert len(df) == 3
    assert "value" in df.columns


def test_single_key_list_value():
    data = {"items": [{"x": 1}, {"x": 2}]}
    df = flatten_json(data)
    assert "x" in df.columns
    assert len(df) == 2


def test_real_coingecko_file(tmp_path):
    """Simulate loading a real CoinGecko JSON file."""
    raw = {
        "prices": [[1771680301648, 68170.87], [1771680615983, 68215.58]],
        "market_caps": [[1771680301648, 1.36e12], [1771680615983, 1.36e12]],
        "total_volumes": [[1771680301648, 4.5e10], [1771680615983, 4.5e10]],
    }
    json_path = tmp_path / "url_data.json"
    json_path.write_text(json.dumps(raw))

    with open(json_path) as f:
        data = json.load(f)

    df = flatten_json(data)
    assert "index" in df.columns
    assert "prices" in df.columns
    assert "market_caps" in df.columns
    assert "total_volumes" in df.columns
    assert len(df) == 2
    assert df["prices"].iloc[0] == pytest.approx(68170.87)

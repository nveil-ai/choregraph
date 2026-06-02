# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the JSON input guard used by execute_code."""
from __future__ import annotations

import pickle

import pandas as pd
import pytest

from choregraph.json_input_guard import (
    JsonDictInput,
    JsonListInput,
    wrap_json_input,
)
from choregraph.library import execute_code


# ---------------------------------------------------------------------------
# wrap_json_input
# ---------------------------------------------------------------------------

class TestWrapJsonInput:
    def test_wraps_list(self):
        wrapped = wrap_json_input("df_json", [{"a": 1}, {"a": 2}])
        assert isinstance(wrapped, JsonListInput)
        assert isinstance(wrapped, list)
        assert len(wrapped) == 2

    def test_wraps_dict(self):
        wrapped = wrap_json_input("payload", {"a": 1, "b": 2})
        assert isinstance(wrapped, JsonDictInput)
        assert isinstance(wrapped, dict)
        assert wrapped["a"] == 1

    def test_passes_through_dataframe(self):
        df = pd.DataFrame({"x": [1, 2]})
        assert wrap_json_input("df", df) is df

    def test_passes_through_scalars(self):
        assert wrap_json_input("n", 42) == 42
        assert wrap_json_input("s", "hello") == "hello"
        assert wrap_json_input("none", None) is None

    def test_idempotent(self):
        once = wrap_json_input("x", [{"a": 1}])
        twice = wrap_json_input("x", once)
        assert twice is once


# ---------------------------------------------------------------------------
# JsonListInput behaviour
# ---------------------------------------------------------------------------

class TestJsonListInput:
    def test_integer_indexing_works(self):
        proxy = JsonListInput("df_json", [{"a": 1}, {"a": 2}])
        assert proxy[0] == {"a": 1}
        assert proxy[-1] == {"a": 2}

    def test_slice_works(self):
        proxy = JsonListInput("df_json", [1, 2, 3, 4])
        assert list(proxy[1:3]) == [2, 3]

    def test_iteration_works(self):
        proxy = JsonListInput("df_json", [{"a": 1}, {"a": 2}])
        assert [item["a"] for item in proxy] == [1, 2]

    def test_string_indexing_raises_clear_error(self):
        proxy = JsonListInput("df_json", [{"name": "Bulbasaur"}])
        with pytest.raises(TypeError) as excinfo:
            _ = proxy["name"]
        msg = str(excinfo.value)
        assert "df_json" in msg
        assert "pd.DataFrame(df_json)" in msg
        assert "pd.json_normalize(df_json)" in msg

    def test_pd_dataframe_conversion_works(self):
        proxy = JsonListInput("df_json", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        df = pd.DataFrame(proxy)
        assert list(df.columns) == ["a", "b"]
        assert df["a"].tolist() == [1, 3]

    def test_pd_json_normalize_works(self):
        proxy = JsonListInput(
            "df_json", [{"name": {"french": "Bulbizarre"}, "id": 1}]
        )
        df = pd.json_normalize(proxy)
        assert "name.french" in df.columns

    def test_picklable(self):
        proxy = JsonListInput("df_json", [{"a": 1}])
        restored = pickle.loads(pickle.dumps(proxy))
        assert isinstance(restored, JsonListInput)
        assert restored[0] == {"a": 1}


# ---------------------------------------------------------------------------
# JsonDictInput behaviour
# ---------------------------------------------------------------------------

class TestJsonDictInput:
    def test_string_key_access_works(self):
        proxy = JsonDictInput("payload", {"name": "Alice", "age": 30})
        assert proxy["name"] == "Alice"
        assert proxy.get("age") == 30
        assert "name" in proxy

    def test_iteration_works(self):
        proxy = JsonDictInput("payload", {"a": 1, "b": 2})
        assert sorted(proxy.keys()) == ["a", "b"]

    def test_dataframe_attribute_raises(self):
        proxy = JsonDictInput("payload", {"a": 1})
        with pytest.raises(AttributeError) as excinfo:
            _ = proxy.iloc
        msg = str(excinfo.value)
        assert "payload" in msg
        assert "pd.DataFrame(payload)" in msg

    def test_unknown_attribute_still_raises(self):
        proxy = JsonDictInput("payload", {"a": 1})
        with pytest.raises(AttributeError):
            _ = proxy.nonexistent_attr


# ---------------------------------------------------------------------------
# Integration with execute_code
# ---------------------------------------------------------------------------

class TestExecuteCodeWithJson:
    def test_legitimate_json_to_dataframe(self):
        """The standard pattern: convert at the top, then operate."""
        code = (
            "df = pd.DataFrame(df_json)\n"
            "result = df[df['a'] > 1]\n"
        )
        result = execute_code(code=code, df_json=[{"a": 1}, {"a": 2}, {"a": 3}])
        assert list(result["a"]) == [2, 3]

    def test_misuse_raises_actionable_error(self):
        """LLM forgot to convert — should produce the helpful message."""
        code = "result = pd.DataFrame(df_json['rows'])"
        with pytest.raises(TypeError) as excinfo:
            execute_code(code=code, df_json=[{"a": 1}])
        assert "df_json" in str(excinfo.value)
        assert "Convert it at the top" in str(excinfo.value)

    def test_dict_misuse_raises_actionable_error(self):
        code = "result = payload.iloc[0]"
        with pytest.raises(AttributeError) as excinfo:
            execute_code(code=code, payload={"a": 1, "b": 2})
        assert "payload" in str(excinfo.value)

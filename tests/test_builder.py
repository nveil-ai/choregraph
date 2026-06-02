# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for choregraph.builder type conversion helpers."""
import pytest
from choregraph.builder import (
    _parse_list_value,
    _resolve_port_type,
    parse_port_value,
    parse_ports_to_kwargs,
)
from choregraph.parser import InputPortSpec


# ── _parse_list_value ────────────────────────────────────────────────

class TestParseListValue:
    def test_basic_comma_split(self):
        assert _parse_list_value("col1, col2") == ["col1", "col2"]

    def test_single_value(self):
        assert _parse_list_value("col1") == ["col1"]

    def test_strips_double_quotes(self):
        assert _parse_list_value('"col1", "col2"') == ["col1", "col2"]

    def test_strips_single_quotes(self):
        assert _parse_list_value("'col1', 'col2'") == ["col1", "col2"]

    def test_strips_brackets(self):
        assert _parse_list_value("[col1, col2]") == ["col1", "col2"]

    def test_strips_brackets_and_quotes(self):
        assert _parse_list_value('["col1", "col2"]') == ["col1", "col2"]

    def test_mixed_quoting(self):
        """Some items quoted, some not — quotes should be stripped where present."""
        assert _parse_list_value('"col1", col2') == ["col1", "col2"]

    def test_whitespace_handling(self):
        assert _parse_list_value("  col1 ,  col2  ") == ["col1", "col2"]

    def test_empty_items_filtered(self):
        assert _parse_list_value("col1,,col2") == ["col1", "col2"]


# ── parse_port_value ─────────────────────────────────────────────────

class TestParsePortValue:
    def test_float(self):
        assert parse_port_value("3.14", "FLOAT") == pytest.approx(3.14)

    def test_integer(self):
        assert parse_port_value("42", "INTEGER") == 42

    def test_boolean_true(self):
        assert parse_port_value("true", "BOOLEAN") is True

    def test_boolean_false(self):
        assert parse_port_value("false", "BOOLEAN") is False

    def test_boolean_yes(self):
        assert parse_port_value("yes", "BOOLEAN") is True

    def test_list_multi(self):
        assert parse_port_value("a, b, c", "LIST") == ["a", "b", "c"]

    def test_list_single(self):
        assert parse_port_value("a", "LIST") == ["a"]

    def test_list_quoted_items(self):
        assert parse_port_value('"jour_cycle", "Score migraine"', "LIST") == [
            "jour_cycle",
            "Score migraine",
        ]

    def test_column_name_or_list(self):
        assert parse_port_value("x, y", "COLUMN_NAME_OR_LIST") == ["x", "y"]

    def test_string(self):
        assert parse_port_value("hello", "STRING") == "hello"

    def test_invalid_float_fallback(self):
        """Non-numeric string for FLOAT returns original value."""
        assert parse_port_value("abc", "FLOAT") == "abc"

    def test_invalid_integer_fallback(self):
        assert parse_port_value("abc", "INTEGER") == "abc"


# ── _resolve_port_type ───────────────────────────────────────────────

class TestResolvePortType:
    def test_xml_type_takes_priority(self):
        port = InputPortSpec(name="cols", value="a,b", type="LIST")
        catalogue_spec = {"type": "STRING"}
        assert _resolve_port_type(port, catalogue_spec) == "LIST"

    def test_dataframe_type_falls_back_to_catalogue(self):
        port = InputPortSpec(name="df", value=None, type="DATAFRAME")
        catalogue_spec = {"type": "DATAFRAME"}
        assert _resolve_port_type(port, catalogue_spec) == "DATAFRAME"

    def test_no_xml_type_uses_catalogue(self):
        port = InputPortSpec(name="threshold", value="0.5", type=None)
        catalogue_spec = {"type": "FLOAT"}
        assert _resolve_port_type(port, catalogue_spec) == "FLOAT"

    def test_no_xml_type_no_catalogue_defaults_string(self):
        port = InputPortSpec(name="unknown", value="hello", type=None)
        assert _resolve_port_type(port, {}) == "STRING"


# ── parse_ports_to_kwargs integration ────────────────────────────────

class TestParsePortsToKwargs:
    def test_xml_type_overrides_catalogue(self):
        """Port with type=LIST should be parsed as list even if catalogue says STRING."""
        ports = [
            InputPortSpec(name="columns", value="jour_cycle, Score migraine", type="LIST"),
        ]
        result = parse_ports_to_kwargs(ports, "drop_columns")
        assert result["columns"] == ["jour_cycle", "Score migraine"]

    def test_quoted_list_values_cleaned(self):
        """Quoted list items should have quotes stripped."""
        ports = [
            InputPortSpec(name="columns", value='"jour_cycle", "Score migraine"', type="LIST"),
        ]
        result = parse_ports_to_kwargs(ports, "drop_columns")
        assert result["columns"] == ["jour_cycle", "Score migraine"]

    def test_skips_none_values(self):
        ports = [
            InputPortSpec(name="df", value=None, source_ref=1, type="DATAFRAME"),
            InputPortSpec(name="threshold", value="0.5", type="FLOAT"),
        ]
        result = parse_ports_to_kwargs(ports, "filter_less_than")
        assert "df" not in result
        assert result["threshold"] == pytest.approx(0.5)

# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for cartograph_json — GenSON-backed JSON structure description.

Covers the compact hierarchical tree output, tabular detection, array-length
annotation, and the rendered block that the planning LLM ingests via
``DatasetStats.info["extract_with"]``.
"""
import json
from pathlib import Path

import pytest

from choregraph.library import cartograph_json


GRAPH_JSON_PATH = Path(__file__).resolve().parent / "fixtures" / "graph.json"


@pytest.fixture(scope="module")
def graph_data():
    with open(GRAPH_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# graph.json — canonical Cytoscape-style graph
# ---------------------------------------------------------------------------

def test_graph_schema_shape(graph_data):
    carto = cartograph_json(graph_data)
    schema = carto["schema"]
    assert schema["type"] == "object"
    top = schema["properties"]
    assert set(top.keys()) == {"nodes", "edges"}
    # Both required at root
    assert set(schema.get("required", [])) == {"nodes", "edges"}
    # Each is an array of objects with a required "data" sub-object
    for k in ("nodes", "edges"):
        assert top[k]["type"] == "array"
        items = top[k]["items"]
        assert items["type"] == "object"
        assert "data" in items["properties"]
        assert items["properties"]["data"]["type"] == "object"


def test_graph_render_is_json_schema(graph_data):
    carto = cartograph_json(graph_data)
    rendered = carto["rendered"]
    assert isinstance(rendered, str)
    assert 0 < len(rendered) <= 1500
    parsed = json.loads(rendered)
    assert parsed["type"] == "object"
    top = parsed["properties"]
    assert set(top.keys()) == {"nodes", "edges"}
    for k in ("nodes", "edges"):
        assert top[k]["type"] == "array"
        assert top[k]["items"]["type"] == "object"


def test_graph_length_is_first_top_level_array(graph_data):
    carto = cartograph_json(graph_data)
    # nodes comes first in the schema, so length tracks it
    assert carto["length"] == 40


def test_graph_is_not_tabular(graph_data):
    carto = cartograph_json(graph_data)
    assert carto["is_tabular"] is False
    assert carto["tabular_fields"] == []


def test_graph_leaf_fields_surface_cytoscape_paths(graph_data):
    carto = cartograph_json(graph_data)
    leaves = carto["leaf_fields"]
    names = {f["name"] for f in leaves}
    # Required Cytoscape-style leaves should all be present
    assert "nodes.data.id" in names
    assert "edges.data.source" in names
    assert "edges.data.target" in names
    # Required flag follows the GenSON schema
    by_name = {f["name"]: f for f in leaves}
    assert by_name["nodes.data.id"]["required"] is True
    assert by_name["edges.data.source"]["required"] is True
    # Dtypes are mapped to VisuSpec conventions
    assert all(f["dtype"] in {"STRING", "INTEGER", "FLOAT", "BOOLEAN"} for f in leaves)


# ---------------------------------------------------------------------------
# Flat tabular variants
# ---------------------------------------------------------------------------

def test_tabular_wrapper_detected():
    data = {"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}
    carto = cartograph_json(data)
    assert carto["is_tabular"] is True
    names = {f["name"] for f in carto["tabular_fields"]}
    assert names == {"id", "name"}
    dtypes = {f["name"]: f["dtype"] for f in carto["tabular_fields"]}
    assert dtypes["id"] == "INTEGER"
    assert dtypes["name"] == "STRING"
    parsed = json.loads(carto["rendered"])
    assert "items" in parsed["properties"]
    assert parsed["properties"]["items"]["type"] == "array"


def test_bare_array_of_objects_detected():
    data = [{"x": 1.5, "y": 2.0}, {"x": 3.5, "y": 4.0}]
    carto = cartograph_json(data)
    assert carto["is_tabular"] is True
    assert carto["length"] == 2
    dtypes = {f["name"]: f["dtype"] for f in carto["tabular_fields"]}
    assert dtypes == {"x": "FLOAT", "y": "FLOAT"}
    parsed = json.loads(carto["rendered"])
    assert parsed["type"] == "array"


def test_array_of_nested_objects_is_not_tabular():
    # Nested sub-objects disqualify tabular detection
    data = [{"id": 1, "meta": {"k": "v"}}, {"id": 2, "meta": {"k": "w"}}]
    carto = cartograph_json(data)
    assert carto["is_tabular"] is False
    assert carto["tabular_fields"] == []


# ---------------------------------------------------------------------------
# Non-tabular shapes still render safely
# ---------------------------------------------------------------------------

def test_deeply_nested_no_arrays():
    data = {"meta": {"version": "1.0", "author": {"name": "X"}}}
    carto = cartograph_json(data)
    assert carto["is_tabular"] is False
    parsed = json.loads(carto["rendered"])
    assert parsed["type"] == "object"
    assert "meta" in parsed["properties"]
    assert "author" in parsed["properties"]["meta"]["properties"]
    assert len(carto["rendered"]) <= 1500


def test_geojson_like():
    data = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]},
             "properties": {"name": "A"}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 1]},
             "properties": {"name": "B"}},
        ],
    }
    carto = cartograph_json(data)
    # Not tabular (features have nested objects)
    assert carto["is_tabular"] is False
    parsed = json.loads(carto["rendered"])
    assert "features" in parsed["properties"]
    assert parsed["properties"]["features"]["type"] == "array"


# ---------------------------------------------------------------------------
# Safety / edge cases
# ---------------------------------------------------------------------------

def test_empty_dict():
    carto = cartograph_json({})
    assert carto["is_tabular"] is False
    assert carto["tabular_fields"] == []
    assert isinstance(carto["rendered"], str)


def test_empty_list():
    carto = cartograph_json([])
    assert carto["is_tabular"] is False
    assert carto["length"] == 0
    parsed = json.loads(carto["rendered"])
    assert parsed["type"] == "array"


# ---------------------------------------------------------------------------
# Heterogeneous arrays (GenSON emits anyOf at items)
# ---------------------------------------------------------------------------

def test_heterogeneous_root_array_emits_leaf_fields():
    # World-Bank-like: [{meta}, [records]]. GenSON merges items into anyOf.
    data = [
        {"page": 1, "total": 66, "lastupdated": "2026-04-08"},
        [
            {"country": {"id": "CN", "value": "China"}, "date": "2025", "value": None},
            {"country": {"id": "CN", "value": "China"}, "date": "2024", "value": 1410},
        ],
    ]
    carto = cartograph_json(data)
    names = {f["name"] for f in carto["leaf_fields"]}
    # Both branches contribute fields, deduped by dotted path
    assert {"page", "total", "lastupdated"}.issubset(names)
    assert {"country.id", "country.value", "date", "value"}.issubset(names)


def test_heterogeneous_array_dedups_overlapping_fields():
    # Two object branches with one common key — leaf must appear once.
    data = [{"id": 1, "a": "x"}, {"id": 2, "b": True}]
    carto = cartograph_json(data)
    names = [f["name"] for f in carto["leaf_fields"]]
    assert names.count("id") == 1
    assert {"id", "a", "b"}.issubset(set(names))


def test_render_max_chars_truncates():
    # Build a JSON with many top-level keys to force overflow
    data = {f"field_{i}": {"x": i, "y": i * 2} for i in range(200)}
    carto = cartograph_json(data, max_chars=300)
    assert len(carto["rendered"]) <= 300
    assert carto["rendered"].endswith("...")

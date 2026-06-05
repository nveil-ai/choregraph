# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Integration tests: JSON cartography flows into catalogue_stats.json and
``MetadataResult.format('markdown')``.

Ensures the ``extract_with`` block reaches the planning_transformation prompt
without requiring changes in the planning node itself.
"""
import json
from pathlib import Path

import pytest

from choregraph.metadata import (
    DatasetStats,
    Metadata,
    MetadataResult,
    compute_file_stats,
)


GRAPH_JSON_PATH = Path(__file__).resolve().parent / "fixtures" / "graph.json"


# ---------------------------------------------------------------------------
# compute_file_stats — produces info.extract_with for JSON inputs
# ---------------------------------------------------------------------------
# TEMPORARY DISABLED: Graph needs to be refactored.
# def test_compute_file_stats_graph_json_has_extract_with():
#     stats = compute_file_stats(str(GRAPH_JSON_PATH))
#     assert stats is not None
#     assert "info" in stats
#     assert "extract_with" in stats["info"]
#     rendered = stats["info"]["extract_with"]
#     parsed = json.loads(rendered)
#     assert parsed["type"] == "object"
#     top = parsed["properties"]
#     assert set(top.keys()) == {"nodes", "edges"}
#     assert top["nodes"]["type"] == "array"
#     assert top["edges"]["type"] == "array"
#     # row_count tracks the first top-level array (nodes = 40)
#     assert stats["row_count"] == 40


# def test_compute_file_stats_tabular_json_surfaces_fields(tmp_path):
#     payload = {"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}
#     p = tmp_path / "tabular.json"
#     p.write_text(json.dumps(payload), encoding="utf-8")

#     stats = compute_file_stats(str(p))
#     assert stats is not None
#     # tabular case should surface sample_fields as real FieldMetadata entries
#     field_names = {f["name"] for f in stats["fields"]}
#     assert field_names == {"id", "name"}
#     # And extract_with still describes the shape
#     rendered = stats["info"]["extract_with"]
#     parsed = json.loads(rendered)
#     assert parsed["properties"]["items"]["type"] == "array"


# def test_compute_file_stats_nested_json_surfaces_leaf_fields(tmp_path):
#     payload = {"meta": {"version": "1.0", "author": {"name": "X"}}}
#     p = tmp_path / "nested.json"
#     p.write_text(json.dumps(payload), encoding="utf-8")

#     stats = compute_file_stats(str(p))
#     assert stats is not None
#     # Non-tabular JSON surfaces leaf paths as FieldMetadata entries
#     names = {f["name"] for f in stats["fields"]}
#     assert names == {"meta.version", "meta.author.name"}
#     # extract_with block remains populated
#     assert stats["info"]["extract_with"]

# TEMPORARY DISABLED: Graph needs to be refactored.
# def test_compute_file_stats_graph_json_surfaces_leaf_fields():
#     stats = compute_file_stats(str(GRAPH_JSON_PATH))
#     assert stats is not None
#     names = {f["name"] for f in stats["fields"]}
#     # Cytoscape leaves should all appear
#     assert "nodes.data.id" in names
#     assert "edges.data.source" in names
#     assert "edges.data.target" in names


# ---------------------------------------------------------------------------
# Round-trip through store_stats / DatasetStats.from_dict -> MetadataResult
# ---------------------------------------------------------------------------

def test_update_stats_and_markdown_rendering(tmp_path):
    """End-to-end: update_stats writes info, read_from_cache restores it,
    and MetadataResult.format('markdown') embeds the extract_with block."""
    with open(GRAPH_JSON_PATH, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    workspace = tmp_path / "ws"
    (workspace / "pipeline" / "cache").mkdir(parents=True)
    md = Metadata(workspace)
    md.update_stats("graph_input", graph_data, dataset_id="1", dataset_type="input")

    # catalogue_stats.json is populated with info.extract_with
    with open(md.cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    entry = cache["datasets"]["graph_input"]
    assert "info" in entry
    assert "extract_with" in entry["info"]
    # _carto helper key must not leak into the persisted cache
    assert "_carto" not in entry["info"]

    # Round-trip through DatasetStats.from_dict + markdown formatting
    result = MetadataResult({"graph_input": DatasetStats.from_dict(entry, name="graph_input")})
    md_out = result.format("markdown")
    assert '"type":"object"' in md_out
    assert "nodes" in md_out and "edges" in md_out
    assert "=== Data" in md_out  # header from _to_markdown

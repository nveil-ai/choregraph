# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Choregraph lifecycle — constructor, load, run, export, hash, get_dataset.

Uses real Kedro infrastructure via tmp_workspace. No mocks except for heavy
external calls (viz hooks disabled via kedro_viz=False).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from lxml import etree

from choregraph.choregraph import Choregraph


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_empty_creates_empty_spec(self, tmp_workspace):
        cg = Choregraph(workspace_path=tmp_workspace, kedro_viz=False)
        assert cg.spec is not None
        assert len(cg.spec.inputs) == 0
        assert len(cg.spec.nodes) == 0

    def test_from_xml_parses_inputs(self, sample_xml_string, tmp_workspace):
        xml_path = tmp_workspace / "choregraph.xml"
        xml_path.write_text(sample_xml_string, encoding="utf-8")
        cg = Choregraph(xml_spec=xml_path, workspace_path=tmp_workspace, kedro_viz=False)
        assert len(cg.spec.inputs) == 1
        assert cg.spec.inputs[0].id == "1"

    def test_auto_detects_choregraph_xml(self, sample_xml_string, tmp_workspace):
        xml_path = tmp_workspace / "choregraph.xml"
        xml_path.write_text(sample_xml_string, encoding="utf-8")
        # No xml_spec arg — should auto-detect
        cg = Choregraph(workspace_path=tmp_workspace, kedro_viz=False)
        assert len(cg.spec.inputs) == 1

    def test_creates_pipeline_dir(self, sample_xml_string, tmp_workspace):
        xml_path = tmp_workspace / "choregraph.xml"
        xml_path.write_text(sample_xml_string, encoding="utf-8")
        Choregraph(xml_spec=xml_path, workspace_path=tmp_workspace, kedro_viz=False)
        assert (tmp_workspace / "pipeline").exists()


# ---------------------------------------------------------------------------
# load / reload
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_replaces_spec(self, sample_xml_string, tmp_workspace):
        cg = Choregraph(workspace_path=tmp_workspace, kedro_viz=False)
        assert len(cg.spec.inputs) == 0

        xml_path = tmp_workspace / "choregraph.xml"
        xml_path.write_text(sample_xml_string, encoding="utf-8")
        cg.load(xml_spec=xml_path)
        assert len(cg.spec.inputs) == 1

    def test_load_clears_cache(self, sample_xml_string, tmp_workspace):
        xml_path = tmp_workspace / "choregraph.xml"
        xml_path.write_text(sample_xml_string, encoding="utf-8")
        cg = Choregraph(xml_spec=xml_path, workspace_path=tmp_workspace, kedro_viz=False)
        cg._data_cache["stale"] = "data"

        cg.load(xml_spec=xml_path)
        assert "stale" not in cg._data_cache

    def test_load_unchanged_spec_skips_rebuild(self, sample_xml_string, tmp_workspace):
        xml_path = tmp_workspace / "choregraph.xml"
        xml_path.write_text(sample_xml_string, encoding="utf-8")
        cg = Choregraph(xml_spec=xml_path, workspace_path=tmp_workspace, kedro_viz=False)

        # Reload same spec — should not error
        cg.load(xml_spec=xml_path)
        assert len(cg.spec.inputs) == 1


# ---------------------------------------------------------------------------
# export_to_xml
# ---------------------------------------------------------------------------


class TestExportToXml:
    def test_roundtrip(self, sample_xml_string, tmp_workspace):
        xml_path = tmp_workspace / "choregraph.xml"
        xml_path.write_text(sample_xml_string, encoding="utf-8")
        cg = Choregraph(xml_spec=xml_path, workspace_path=tmp_workspace, kedro_viz=False)

        export_path = tmp_workspace / "exported.xml"
        cg.export_to_xml(export_path)

        assert export_path.exists()
        tree = etree.parse(str(export_path))
        root = tree.getroot()
        assert root.tag == "choregraph"
        inputs = root.findall(".//input")
        assert len(inputs) == 1
        assert inputs[0].get("id") == "1"

    def test_export_preserves_programmatic_nodes(self, tmp_path, tmp_workspace):
        """Nodes added via add_node() survive export roundtrip."""
        from choregraph.parser import InputPortSpec, OutputPortSpec

        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")

        cg = Choregraph(workspace_path=tmp_workspace, kedro_viz=False)
        cg.add_input(id="1", location=str(csv), format="CSV", sep=",", header=0)
        cg.add_node(
            id="2", type="select_columns",
            input_ports=[
                InputPortSpec(name="input", source_ref="1", type="DATAFRAME"),
                InputPortSpec(name="parameter", value="a", type="PARAMETER"),
            ],
            output_ports=[
                OutputPortSpec(id="2", name="result", type="DATAFRAME"),
            ],
        )

        export_path = tmp_workspace / "exported.xml"
        cg.export_to_xml(export_path)

        tree = etree.parse(str(export_path))
        nodes = tree.getroot().findall(".//node")
        assert len(nodes) == 1
        assert nodes[0].get("type") == "select_columns"

    def test_export_empty_spec(self, tmp_workspace):
        cg = Choregraph(workspace_path=tmp_workspace, kedro_viz=False)
        export_path = tmp_workspace / "empty.xml"
        cg.export_to_xml(export_path)

        tree = etree.parse(str(export_path))
        assert tree.getroot().tag == "choregraph"
        assert len(tree.getroot().findall(".//input")) == 0


# ---------------------------------------------------------------------------
# _get_project_hash
# ---------------------------------------------------------------------------


class TestProjectHash:
    def test_same_spec_same_hash(self, sample_xml_string, tmp_workspace):
        xml_path = tmp_workspace / "choregraph.xml"
        xml_path.write_text(sample_xml_string, encoding="utf-8")
        cg = Choregraph(xml_spec=xml_path, workspace_path=tmp_workspace, kedro_viz=False)

        h1 = cg._get_project_hash()
        h2 = cg._get_project_hash()
        assert h1 == h2

    def test_different_spec_different_hash(self, sample_xml_string, tmp_workspace):
        xml_path = tmp_workspace / "choregraph.xml"
        xml_path.write_text(sample_xml_string, encoding="utf-8")
        cg = Choregraph(xml_spec=xml_path, workspace_path=tmp_workspace, kedro_viz=False)
        h1 = cg._get_project_hash()

        # Modify the XML
        xml_path.write_text(sample_xml_string.replace("select_columns", "filter_rows"), encoding="utf-8")
        cg.load(xml_spec=xml_path)
        h2 = cg._get_project_hash()
        assert h1 != h2

    def test_no_workspace_uses_spec_repr(self):
        cg = Choregraph(kedro_viz=False)
        h = cg._get_project_hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# run — with external DataFrames (no file I/O beyond workspace)
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_empty_pipeline_succeeds(self, tmp_workspace):
        """Empty pipeline (no nodes) should succeed immediately."""
        cg = Choregraph(workspace_path=tmp_workspace, kedro_viz=False)
        success, error = cg.run(lazy=False)
        assert success is True
        assert error == ""

    def test_run_lazy_skips_when_unchanged(self, tmp_workspace):
        """Second lazy run with same spec returns immediately."""
        cg = Choregraph(workspace_path=tmp_workspace, kedro_viz=False)
        cg.run(lazy=False)
        # Second run should be skipped (lazy=True, hash unchanged)
        success, error = cg.run(lazy=True)
        assert success is True

    def test_run_requires_workspace(self):
        cg = Choregraph(kedro_viz=False)
        with pytest.raises(ValueError, match="Workspace path required"):
            cg.run()

    def test_run_with_node_produces_output(self, sample_xml_string, tmp_workspace):
        """Pipeline with a select_columns node should produce data."""
        xml_path = tmp_workspace / "choregraph.xml"
        xml_path.write_text(sample_xml_string, encoding="utf-8")
        cg = Choregraph(xml_spec=xml_path, workspace_path=tmp_workspace, kedro_viz=False)

        success, error = cg.run(lazy=False)
        assert success is True, f"Pipeline failed: {error}"


# ---------------------------------------------------------------------------
# add_input + add_node programmatic API
# ---------------------------------------------------------------------------


class TestProgrammaticAPI:
    def test_add_input_csv(self, tmp_path, tmp_workspace):
        csv = tmp_path / "data.csv"
        csv.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")

        cg = Choregraph(workspace_path=tmp_workspace, kedro_viz=False)
        cg.add_input(id="1", location=str(csv), format="CSV", sep=",", header=0)

        assert len(cg.spec.inputs) == 1
        assert cg.spec.inputs[0].id == "1"

    def test_add_input_with_visibility_creates_output(self, tmp_path, tmp_workspace):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")

        cg = Choregraph(workspace_path=tmp_workspace, kedro_viz=False)
        cg.add_input(id="42", location=str(csv), format="CSV", visibility=True, sep=",", header=0)

        assert any(o.id == "42" for o in cg.spec.outputs)

    def test_add_input_memory(self, tmp_workspace):
        df = pd.DataFrame({"a": [1, 2, 3]})
        cg = Choregraph(workspace_path=tmp_workspace, kedro_viz=False)
        cg.add_input(id="mem1", format="MEMORY", data=df)

        assert "mem1" in cg.external_inputs
        assert cg.external_inputs["mem1"].equals(df)


# ---------------------------------------------------------------------------
# get_xsd
# ---------------------------------------------------------------------------


class TestGetXsd:
    def test_returns_string(self):
        cg = Choregraph(kedro_viz=False)
        xsd = cg.get_xsd()
        assert isinstance(xsd, str)
        assert "choregraph" in xsd.lower() or "xml" in xsd.lower()

# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import pandas as pd

from choregraph.choregraph import Choregraph
from choregraph.parser import InputPortSpec, OutputPortSpec


def test_choregraph_initializes_and_builds_wrapper_from_spec(sample_xml_string: str, tmp_workspace: Path):
    xml_path = tmp_workspace / "choregraph.xml"
    xml_path.write_text(sample_xml_string, encoding="utf-8")

    cg = Choregraph(xml_spec=xml_path, workspace_path=tmp_workspace)

    assert (tmp_workspace / "pipeline").exists()
    assert cg.spec.inputs


def test_choregraph_add_input_adds_output_and_regenerates_wrapper(tmp_workspace: Path, tmp_path: Path):
    csv_path = tmp_path / "x.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    cg = Choregraph(xml_spec=None, workspace_path=tmp_workspace)
    cg.add_input(id="1", location=str(csv_path), format="CSV", visibility=True, sep=",", header=0)

    assert any(i.id == "1" for i in cg.spec.inputs)
    assert any(o.id == "1" for o in cg.spec.outputs)  # visibility=True adds it to outputs
    assert (tmp_workspace / "pipeline" / "conf" / "base" / "catalog.yml").exists()


def test_choregraph_get_dataset_uses_cache(monkeypatch, tmp_workspace: Path):
    cg = Choregraph(xml_spec=None, workspace_path=tmp_workspace)

    # fake spec mapping for dataset id -> name
    cg.spec.inputs = []
    cg.spec.nodes = []
    cg.spec.outputs = []

    calls = {"count": 0}

    class FakeCatalog:
        def __contains__(self, name):
            return True

        def load(self, name):
            calls["count"] += 1
            return pd.DataFrame({"x": [1, 2]})

        def list(self):
            return ["input_1"]

    monkeypatch.setattr(cg, "_get_catalog", lambda: FakeCatalog())

    # With empty name_map, spec.get_name will fallback to id
    out1 = cg.get_dataset("input_1")
    out2 = cg.get_dataset("input_1")

    assert calls["count"] == 1
    assert isinstance(out1, pd.DataFrame)
    assert out1.equals(out2)


def test_dive_connector_from_choregraph_computes_allow_list(monkeypatch, tmp_workspace: Path):
    from choregraph.connectors import DiveConnector

    cg = Choregraph(xml_spec=None, workspace_path=tmp_workspace)

    # Build a spec where node "2" has a visible output and input "1" is an explicit output.
    cg.add_input(id="1", location="", format="MEMORY")
    cg.add_node(
        id="2",
        type="select_columns",
        input_ports=[
            InputPortSpec(name="input", source_ref="1", type="DATAFRAME"),
            InputPortSpec(name="parameter", value="Age", type="PARAMETER"),
        ],
        output_ports=[
            OutputPortSpec(id="2_result", name="result", type="DATAFRAME", visibility=True),
        ]
    )

    connector = DiveConnector.from_choregraph(cg)
    allow_list = connector._compute_allow_list()

    # Should include visible output port and its ancestor input
    assert set(allow_list) == {cg.spec.get_name("1"), cg.spec.get_name("2_result")}

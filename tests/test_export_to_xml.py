# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

from lxml import etree

from choregraph.choregraph import Choregraph


def test_export_to_xml_writes_valid_structure(tmp_workspace: Path):
    cg = Choregraph(xml_spec=None, workspace_path=tmp_workspace)
    cg.add_input(id="1", location="/tmp/some.csv", format="CSV", sep=",", header=0)

    out_path = tmp_workspace / "out.xml"
    cg.export_to_xml(out_path)

    root = etree.parse(str(out_path)).getroot()
    assert root.tag == "choregraph"
    assert root.find("inputs") is not None
    assert root.find("pipeline") is not None

    # At least one input is present
    assert root.find("inputs").find("input") is not None

    # When only inputs are added (no nodes), the pipeline element will be empty
    # This is expected behavior - nodes are only added explicitly

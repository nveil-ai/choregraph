# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from choregraph.choregraph import Choregraph
from choregraph.connectors import CacheProxy
from choregraph.parser import InputPortSpec, OutputPortSpec


def test_cache_proxy_list_includes_inputs_and_node_outputs(monkeypatch, tmp_workspace):
    cg = Choregraph(xml_spec=None, workspace_path=tmp_workspace)

    # Build spec with one input + one node with visible output
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

    # Avoid real catalog loading
    class FakeCatalog:
        def list(self):
            return [cg.spec.get_name("1")]

    monkeypatch.setattr(cg, "_get_catalog", lambda: FakeCatalog())

    proxy = CacheProxy(cg)
    keys = set(proxy.list())

    assert cg.spec.get_name("1") in keys
    assert cg.spec.get_name("2_result") in keys


def test_cache_proxy_load_delegates_to_choregraph(monkeypatch, tmp_workspace):
    cg = Choregraph(xml_spec=None, workspace_path=tmp_workspace)

    called = {"name": None}

    def fake_get_dataset(name: str):
        called["name"] = name
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(cg, "get_dataset", fake_get_dataset)

    proxy = CacheProxy(cg)
    out = proxy.load("foo")
    assert called["name"] == "foo"
    assert isinstance(out, pd.DataFrame)

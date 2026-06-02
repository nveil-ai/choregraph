# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

from choregraph.parser import ChoregraphSpecParser, ChoregraphSpec


def test_parse_from_string_builds_inputs_nodes_outputs():
    xml = """<choregraph>
  <inputs>
    <input id="1" label="Input1" location="test.csv" format="CSV" header="true" fieldSeparator="," visibility="true"/>
  </inputs>
  <pipeline>
    <node id="2" label="Node2" type="filter_less_than">
      <inputPort name="df" sourceRef="1" type="DATAFRAME"/>
      <inputPort name="column" value="Age" type="STRING"/>
      <inputPort name="value" value="30" type="FLOAT"/>
      <outputPort id="10" name="result" type="DATAFRAME" visibility="true"/>
    </node>
  </pipeline>
</choregraph>"""
    spec = ChoregraphSpecParser.parse(xml)

    assert len(spec.inputs) == 1
    assert spec.inputs[0].id == "1"  # String ID
    assert spec.inputs[0].label == "Input1"
    assert spec.inputs[0].format == "CSV"
    assert spec.inputs[0].options["sep"] == ","
    assert spec.inputs[0].options["header"] == 0

    assert len(spec.nodes) == 1
    node = spec.nodes[0]
    assert node.id == "2"  # String ID
    assert node.label == "Node2"
    assert node.type == "filter_less_than"
    
    # Check input ports
    assert len(node.input_ports) == 3
    df_port = next(p for p in node.input_ports if p.name == "df")
    assert df_port.source_ref == 1  # Integer source_ref
    assert df_port.type == "DATAFRAME"
    
    column_port = next(p for p in node.input_ports if p.name == "column")
    assert column_port.value == "Age"
    
    # Check output ports
    assert len(node.output_ports) == 1
    assert node.output_ports[0].id == 10  # Integer ID
    assert node.output_ports[0].name == "result"
    assert node.output_ports[0].visibility is True

    # input marked visibility=true should be added as an output
    assert any(o.id == "1" for o in spec.outputs)


def test_parse_from_file_path(tmp_path: Path):
    sample_xml_string = """<choregraph>
  <inputs>
    <input id="1" label="Data" location="test.csv" format="CSV"/>
  </inputs>
  <pipeline>
    <node id="2" label="Process" type="filter_less_than">
      <inputPort name="df" sourceRef="1" type="DATAFRAME"/>
      <inputPort name="column" value="x" type="STRING"/>
      <inputPort name="value" value="10" type="FLOAT"/>
      <outputPort id="10" name="result" type="DATAFRAME" visibility="false"/>
    </node>
  </pipeline>
</choregraph>"""
    xml_path = tmp_path / "choregraph.xml"
    xml_path.write_text(sample_xml_string, encoding="utf-8")

    spec = ChoregraphSpecParser.parse(xml_path)
    assert isinstance(spec, ChoregraphSpec)
    assert len(spec.inputs) == 1
    assert len(spec.nodes) == 1


def test_get_name_maps_inputs_and_output_ports(tmp_path: Path):
    xml = """<choregraph>
  <inputs>
    <input id="10" label="My File" location="/tmp/my_file.csv" format="CSV" />
    <input id="11" label="Mem Input" location="" format="MEMORY" />
  </inputs>
  <pipeline>
    <node id="20" label="Add Value" type="filter_less_than">
      <inputPort name="df" sourceRef="10" type="DATAFRAME"/>
      <inputPort name="column" value="x" type="STRING"/>
      <inputPort name="value" value="5" type="FLOAT"/>
      <outputPort id="100" name="result" type="DATAFRAME" visibility="false"/>
    </node>
    <node id="21" label="Complex-Label (Test)" type="filter_greater_than">
      <inputPort name="df" sourceRef="100" type="DATAFRAME"/>
      <inputPort name="column" value="y" type="STRING"/>
      <inputPort name="value" value="5" type="FLOAT"/>
      <outputPort id="101" name="result" type="DATAFRAME" visibility="true"/>
    </node>
  </pipeline>
</choregraph>"""

    spec = ChoregraphSpecParser.parse(xml)

    # Input names based on labels (sanitized)
    assert spec.get_name("10") == "my_file"  
    assert spec.get_name("11") == "mem_input"
    
    # Output port names based on node labels + port name suffix
    assert spec.get_name(100) == "add_value_out"
    assert spec.get_name(101) == "complex_label_test_out"

    # Map should be cached and stable
    assert spec.get_name("10") == "my_file"


def test_output_port_visibility():
    """Test that output port visibility is correctly parsed."""
    xml = """<choregraph>
  <inputs>
    <input id="1" label="Data" location="test.csv" format="CSV"/>
  </inputs>
  <pipeline>
    <node id="2" label="Filter with mask" type="filter_greater_than">
      <inputPort name="df" sourceRef="1" type="DATAFRAME"/>
      <inputPort name="column" value="price" type="STRING"/>
      <inputPort name="value" value="100" type="FLOAT"/>
      <inputPort name="return_mask" value="true" type="BOOLEAN"/>
      <outputPort id="10" name="result" type="DATAFRAME" visibility="false"/>
      <outputPort id="11" name="mask" type="DATAFRAME" visibility="true"/>
    </node>
  </pipeline>
</choregraph>"""
    spec = ChoregraphSpecParser.parse(xml)
    
    node = spec.nodes[0]
    assert len(node.output_ports) == 2
    
    result_port = next(op for op in node.output_ports if op.name == "result")
    mask_port = next(op for op in node.output_ports if op.name == "mask")
    
    assert result_port.visibility is False
    assert mask_port.visibility is True
    assert result_port.id == 10  # Integer ID
    assert mask_port.id == 11  # Integer ID


def test_get_node_for_output_port():
    """Test finding the node that owns an output port."""
    xml = """<choregraph>
  <inputs>
    <input id="1" label="Data" location="test.csv" format="CSV"/>
  </inputs>
  <pipeline>
    <node id="2" label="TestNode" type="filter_less_than">
      <inputPort name="df" sourceRef="1" type="DATAFRAME"/>
      <inputPort name="column" value="x" type="STRING"/>
      <inputPort name="value" value="10" type="FLOAT"/>
      <outputPort id="10" name="result" type="DATAFRAME" visibility="true"/>
    </node>
  </pipeline>
</choregraph>"""
    spec = ChoregraphSpecParser.parse(xml)
    
    node = spec.get_node_for_output_port(10)  # Integer port ID
    assert node is not None
    assert node.id == "2"
    assert node.label == "TestNode"
    
    # Non-existent port should return None
    assert spec.get_node_for_output_port(99) is None


def test_sanitize_name_strips_accents():
    from choregraph.parser import _sanitize_name
    assert _sanitize_name("Salaire moyen par rôle") == "salaire_moyen_par_role"
    assert _sanitize_name("Données générales") == "donnees_generales"
    assert _sanitize_name("café_crème") == "cafe_creme"
    assert _sanitize_name("Complex-Label (Test)") == "complex_label_test"
    assert _sanitize_name("Hello World!") == "hello_world"
    assert _sanitize_name("a---b") == "a_b"


def test_get_name_strips_accents_from_labels():
    xml = """<choregraph>
  <inputs>
    <input id="1" label="Données RH" location="data.csv" format="CSV"/>
  </inputs>
  <pipeline>
    <node id="2" label="Calcul" type="aggregate_mean">
      <inputPort name="df" sourceRef="1" type="DATAFRAME"/>
      <outputPort id="10" label="Salaire moyen par rôle" name="result" type="DATAFRAME" visibility="true"/>
    </node>
  </pipeline>
</choregraph>"""
    spec = ChoregraphSpecParser.parse(xml)
    assert spec.get_name("1") == "donnees_rh"
    assert spec.get_name("10") == "salaire_moyen_par_role"

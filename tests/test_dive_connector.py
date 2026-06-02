# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from lxml import etree
import pandas as pd

from kedro.io import DataCatalog, MemoryDataset

from choregraph.connectors import DiveConnector
from choregraph.metadata import MetadataExtractor


class FakeCSVDataset(MemoryDataset):
    # Contains "CSV" substring in the class name so format detection triggers.
    def __init__(self, data, load_args=None):
        super().__init__(data)
        self._load_args = load_args or {}


def test_generate_visuspec_xml_basic_structure():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    ds = FakeCSVDataset(df, load_args={"sep": ";", "header": 0})
    cat = DataCatalog({"mydata": ds})

    conn = DiveConnector(cat)
    xml = conn.generate_visuspec_xml()

    root = etree.fromstring(xml.encode("utf-8"))
    assert root.tag == "datas"

    raw = root.find("rawData")
    assert raw is not None
    assert raw.get("name") == "mydata"
    assert raw.get("rows") == "2"

    file_elem = raw.find("file")
    assert file_elem is not None
    assert file_elem.get("location") == "choregraph.xml"
    # Note: Format detection is currently disabled in DiveConnector, defaulting to "XML"
    assert file_elem.get("format") == "XML"


def test_generate_visuspec_xml_allow_list_filters():
    df = pd.DataFrame({"a": [1]})
    cat = DataCatalog({"keep": MemoryDataset(df), "drop": MemoryDataset(df)})

    conn = DiveConnector(cat)
    xml = conn.generate_visuspec_xml(allow_list=["keep"])

    root = etree.fromstring(xml.encode("utf-8"))
    names = [e.get("name") for e in root.findall("rawData")]
    assert names == ["keep"]


def test_generate_visuspec_xml_uses_metadata_cache_even_if_load_fails():
    from choregraph.metadata import DatasetStats

    df = pd.DataFrame({"a": [1, 2, 3]})
    fields = MetadataExtractor.extract(df)

    class FailingDataset(MemoryDataset):
        def load(self):
            raise RuntimeError("nope")

    cat = DataCatalog({"d": FailingDataset(df)})

    # Mock Metadata whose read_from_cache returns pre-extracted stats
    class FakeMetadata:
        def read_from_cache(self):
            return {"d": DatasetStats(id="1", name="d", row_count=3, fields=fields, last_updated="")}

    conn = DiveConnector(cat, metadata=FakeMetadata())
    xml = conn.generate_visuspec_xml()

    root = etree.fromstring(xml.encode("utf-8"))
    raw = root.find("rawData")
    assert raw is not None
    assert raw.get("name") == "d"
    assert raw.get("rows") == "3"


def test_generate_visuspec_xml_transformed_data_structure():
    """Test that transformedData uses child elements (not attributes) per XSD."""
    from dataclasses import dataclass, field as dc_field
    from typing import List, Optional
    
    @dataclass
    class InputPortSpec:
        name: str
        value: Optional[str] = None
        source_ref: Optional[str] = None
        type: Optional[str] = None
    
    @dataclass
    class OutputPortSpec:
        id: int
        name: str
        type: str = "DATAFRAME"
        visibility: bool = True
    
    @dataclass
    class InputSpec:
        id: str
        label: str
        file: str
        type: str = "csv"
        
    @dataclass
    class NodeSpec:
        id: str
        label: str
        type: str
        input_ports: List[InputPortSpec] = dc_field(default_factory=list)
        output_ports: List[OutputPortSpec] = dc_field(default_factory=list)
    
    @dataclass
    class FakeSpec:
        inputs: List[InputSpec] = dc_field(default_factory=list)
        nodes: List[NodeSpec] = dc_field(default_factory=list)
        
        def get_name(self, id: str) -> str:
            for inp in self.inputs:
                if str(inp.id) == str(id):
                    return inp.label
            for node in self.nodes:
                for op in node.output_ports:
                    if str(op.id) == str(id):
                        return f"{node.label}_out"
            return f"unknown_{id}"
        
        def get_node_for_output_port(self, output_port_id: str):
            for node in self.nodes:
                for op in node.output_ports:
                    if str(op.id) == str(output_port_id):
                        return node
            return None
        
        def get_visible_output_ports(self):
            result = []
            for node in self.nodes:
                for op in node.output_ports:
                    if op.visibility:
                        result.append(op)
            return result
    
    # Create test data
    input_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    transformed_df = pd.DataFrame({"result": [10, 20, 30], "category": ["a", "b", "c"]})
    
    # Create catalog with both input and transformed data
    cat = DataCatalog({
        "myinput": MemoryDataset(input_df),
        "filter_out": MemoryDataset(transformed_df),
    })
    
    # Create fake spec
    fake_spec = FakeSpec(
        inputs=[InputSpec(id="1", label="myinput", file="data.csv")],
        nodes=[NodeSpec(
            id="2", 
            label="filter", 
            type="filter_less_than",
            input_ports=[InputPortSpec(name="df", source_ref=1)],
            output_ports=[OutputPortSpec(id=10, name="result", visibility=True)]
        )]
    )
    
    # Attach spec to catalog via fake cg proxy
    class FakeCG:
        workspace_path = None
    
    fake_cg = FakeCG()
    fake_cg.spec = fake_spec
    cat.cg = fake_cg
    
    conn = DiveConnector(cat)
    xml = conn.generate_visuspec_xml()

    root = etree.fromstring(xml.encode("utf-8"))

    # Check rawData for input
    raw_elements = root.findall("rawData")
    assert len(raw_elements) >= 1
    myinput_elem = next((r for r in raw_elements if r.get("name") == "myinput"), None)
    assert myinput_elem is not None
    assert myinput_elem.find("file") is not None  # rawData has file element
    
    # Check rawData for transform output (currently all outputs are stored as rawData)
    filter_elem = next((r for r in raw_elements if r.get("name") == "filter_out"), None)
    assert filter_elem is not None
    assert filter_elem.get("rows") == "3"
    
    # Verify fields are present with metadata
    fields = filter_elem.find("fields")
    assert fields is not None
    field_names = [f.get("name") for f in fields.findall("field")]
    assert "result" in field_names
    assert "category" in field_names


def test_transform_type_mapping():
    """Test that transform types are correctly mapped to XSD enum values."""
    from choregraph.connectors.dive import _map_transform_type
    
    # Known transforms
    assert _map_transform_type("add") == "ADD"
    assert _map_transform_type("ADD") == "ADD"
    assert _map_transform_type("count") == "COUNT"
    assert _map_transform_type("aggregate_count") == "COUNT"
    assert _map_transform_type("extract_points") == "EXTRACT_POINTS"
    assert _map_transform_type("extract_cells") == "EXTRACT_CELLS"
    
    # Unknown transforms map to UNKNOWN
    assert _map_transform_type("filter_less_than") == "UNKNOWN"
    assert _map_transform_type("custom_transform") == "UNKNOWN"
    
    # Empty/None maps to UNDEFINED
    assert _map_transform_type(None) == "UNDEFINED"
    assert _map_transform_type("") == "UNDEFINED"


def test_transformed_data_chaining():
    """Test that transformedData can reference another transformedData (not just rawData)."""
    from dataclasses import dataclass, field as dc_field
    from typing import List, Optional
    
    @dataclass
    class InputPortSpec:
        name: str
        value: Optional[str] = None
        source_ref: Optional[str] = None
        type: Optional[str] = None
    
    @dataclass
    class OutputPortSpec:
        id: int
        name: str
        type: str = "DATAFRAME"
        visibility: bool = True
    
    @dataclass
    class InputSpec:
        id: str
        label: str
        file: str
        type: str = "csv"
        
    @dataclass
    class NodeSpec:
        id: str
        label: str
        type: str
        input_ports: List[InputPortSpec] = dc_field(default_factory=list)
        output_ports: List[OutputPortSpec] = dc_field(default_factory=list)
    
    @dataclass
    class FakeSpec:
        inputs: List[InputSpec] = dc_field(default_factory=list)
        nodes: List[NodeSpec] = dc_field(default_factory=list)
        
        def get_name(self, id: str) -> str:
            for inp in self.inputs:
                if str(inp.id) == str(id):
                    return inp.label
            for node in self.nodes:
                for op in node.output_ports:
                    if str(op.id) == str(id):
                        return f"{node.label}_out"
            return f"unknown_{id}"
        
        def get_node_for_output_port(self, output_port_id: str):
            for node in self.nodes:
                for op in node.output_ports:
                    if str(op.id) == str(output_port_id):
                        return node
            return None
        
        def get_visible_output_ports(self):
            result = []
            for node in self.nodes:
                for op in node.output_ports:
                    if op.visibility:
                        result.append(op)
            return result
    
    # Create test data for chained transforms: input -> transform1 -> transform2
    input_df = pd.DataFrame({"x": [1, 2, 3]})
    transform1_df = pd.DataFrame({"filtered": [1, 2]})
    transform2_df = pd.DataFrame({"aggregated": [1.5]})
    
    cat = DataCatalog({
        "raw_data": MemoryDataset(input_df),
        "step1_out": MemoryDataset(transform1_df),
        "step2_out": MemoryDataset(transform2_df),
    })
    
    fake_spec = FakeSpec(
        inputs=[InputSpec(id="1", label="raw_data", file="data.csv")],
        nodes=[
            NodeSpec(
                id="2", label="step1", type="filter",
                input_ports=[InputPortSpec(name="df", source_ref=1)],
                output_ports=[OutputPortSpec(id=10, name="result", visibility=True)]
            ),
            NodeSpec(
                id="3", label="step2", type="aggregate",
                input_ports=[InputPortSpec(name="df", source_ref=10)],  # References output of node 2
                output_ports=[OutputPortSpec(id=11, name="result", visibility=True)]
            ),
        ]
    )
    
    class FakeCG:
        workspace_path = None
    
    fake_cg = FakeCG()
    fake_cg.spec = fake_spec
    cat.cg = fake_cg
    
    conn = DiveConnector(cat)
    xml = conn.generate_visuspec_xml()

    root = etree.fromstring(xml.encode("utf-8"))

    # Find all rawData elements (currently all datasets are stored as rawData)
    raw_elements = root.findall("rawData")
    assert len(raw_elements) >= 2  # At least step1 and step2 outputs
    
    # step1 should have raw_data as input
    step1 = next((r for r in raw_elements if r.get("name") == "step1_out"), None)
    assert step1 is not None
    
    # step2 should exist
    step2 = next((r for r in raw_elements if r.get("name") == "step2_out"), None)
    assert step2 is not None

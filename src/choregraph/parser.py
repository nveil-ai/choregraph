# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from lxml import etree


def _sanitize_name(text: str) -> str:
    """Normalize a label into a safe ASCII-only Kedro dataset name."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    clean = "".join(c if c.isascii() and (c.isalnum() or c == "_") else "_" for c in ascii_text.lower())
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


@dataclass
class InputSpec:
    """Defines an input data source."""
    id: str  # Unique ID (string for consistency with output port IDs)
    label: str
    location: str
    format: str
    options: dict = field(default_factory=dict)
    visibility: bool = False
    url: Optional[str] = None  # Origin URL for URL-based data sources


@dataclass
class InputPortSpec:
    """Defines an input port consumed by a transform node."""
    name: str
    value: Optional[str] = None        # Static parameter value
    source_ref: Optional[int] = None   # Reference to source entity ID (input ID or output port ID) - integer
    type: Optional[str] = None


@dataclass
class OutputPortSpec:
    """Defines an output port produced by a transform node."""
    id: int                            # Unique integer ID for this output (can be referenced by downstream nodes)
    name: str                          # Port name (e.g., "result", "mask")
    label: str = ""                    # Polished human-readable name (populated by AI) for Kedro graph display
    type: Optional[str] = None
    visibility: bool = False           # Whether this output is exposed for visualization


@dataclass
class NodeSpec:
    """Defines a transform node in the pipeline."""
    id: str  # Node ID (string for consistency)
    label: str
    type: str
    input_ports: List[InputPortSpec] = field(default_factory=list)
    output_ports: List[OutputPortSpec] = field(default_factory=list)


@dataclass
class OutputSpec:
    """Defines a pipeline output entry."""
    id: str
    as_name: Optional[str] = None


@dataclass
class ChoregraphSpec:
    """In-memory representation of a complete Choregraph pipeline specification.

    Holds inputs, transform nodes, and output declarations. Provides name
    resolution from XML IDs to sanitized Kedro dataset names via :meth:`get_name`.
    """
    inputs: List[InputSpec] = field(default_factory=list)
    nodes: List[NodeSpec] = field(default_factory=list)
    outputs: List[OutputSpec] = field(default_factory=list)
    
    # Unified map for ID -> Name
    _name_map: Dict[str, str] = field(default_factory=dict, repr=False)

    def select_by_tag(self, tag: str) -> 'ChoregraphSpec':
        """
        Returns a new ChoregraphSpec containing only objects corresponding to the tag.
        Supported tags: 'input' (for inputs), 'node' (for nodes).
        """
        new_spec = ChoregraphSpec()
        t = tag.lower()
        if t in ('input', 'inputs'):
            new_spec.inputs = list(self.inputs)
        elif t in ('node', 'nodes'):
            new_spec.nodes = list(self.nodes)
        return new_spec

    def select_by_attribute(self, attribute: str, value: Any) -> 'ChoregraphSpec':
        """
        Returns a new ChoregraphSpec containing objects (nodes or inputs) 
        where the specified attribute matches the given value.
        """
        new_spec = ChoregraphSpec()
        new_spec.inputs = [i for i in self.inputs if getattr(i, attribute, None) == value]
        new_spec.nodes = [n for n in self.nodes if getattr(n, attribute, None) == value]
        return new_spec

    def get_attribute(self, attribute: str) -> List[Any]:
        """
        Returns a list of values for the specified attribute from all objects 
        (inputs and nodes) in the spec.
        """
        values = []
        for inp in self.inputs:
            if hasattr(inp, attribute):
                values.append(getattr(inp, attribute))
        for node in self.nodes:
            if hasattr(node, attribute):
                values.append(getattr(node, attribute))
        return values

    def get_name(self, id: Union[int, str]) -> str:
        """
        Returns the clean, readable Kedro name for a given XML ID.
        Rebuilds the map if the ID is not found.
        Works for both input IDs and output port IDs.
        """
        id_str = str(id)
        
        if id_str not in self._name_map:
            self._name_map = {}
            seen_ids = set()
            
            # 1. Map Inputs
            for inp in self.inputs:
                sid = str(inp.id)
                if sid in seen_ids:
                    continue  # Avoid error on duplicate for mapping
                seen_ids.add(sid)
                
                try:
                    clean_label = _sanitize_name(inp.label)
                except Exception:
                    clean_label = f"input_{inp.id}"
                self._name_map[sid] = clean_label

            # 2. Map Output Ports (each has unique ID)
            for n in self.nodes:
                for op in n.output_ports:
                    sid = str(op.id)
                    if sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                    
                    try:
                        # Prefer the polished label if provided by AI
                        if op.label:
                            clean_label = _sanitize_name(op.label)
                        else:
                            # Fallback: Use node label + output port name
                            label_text = n.label if n.label else f"node_{n.id}"
                            clean_label = _sanitize_name(label_text)
                            # Append port name if not "result" to distinguish multiple outputs
                            if op.name != "result":
                                clean_label = f"{clean_label}_{op.name}"
                            else:
                                clean_label = f"{clean_label}_out"
                    except Exception:
                        clean_label = f"output_{op.id}"
                    self._name_map[sid] = clean_label

        return self._name_map.get(id_str, id_str)
    
    def select_by_attribute(self, attribute: str, value: Any) -> 'ChoregraphSpec':
        """
        Returns a new ChoregraphSpec containing objects (nodes or inputs) 
        where the specified attribute matches the given value.
        """
        new_spec = ChoregraphSpec()
        new_spec.inputs = [i for i in self.inputs if getattr(i, attribute, None) == value]
        new_spec.nodes = [n for n in self.nodes if getattr(n, attribute, None) == value]
        return new_spec
    
    def select_by_visibility(self) -> 'ChoregraphSpec':
        """
        Returns a new ChoregraphSpec containing objects (nodes or inputs) 
        where the specified attribute matches the given value.
        """
        new_spec = ChoregraphSpec()
        new_spec.inputs = [i for i in self.inputs if i.visibility]
        new_spec.nodes = [n for n in self.nodes if any(op.visibility for op in n.output_ports)]
        return new_spec
    
    
    def get_visible(self) -> List[Any]:
        """Returns all ports with visibility=True."""
        visible = []
        for i in self.inputs:
            if i.visibility:
                visible.append(i)
        for n in self.nodes:
            for op in n.output_ports:
                if op.visibility:
                    visible.append(op)
        return visible
    
    def get_output_port_by_id(self, port_id: str) -> Optional[OutputPortSpec]:
        """Find an output port by its ID."""
        for n in self.nodes:
            for op in n.output_ports:
                if str(op.id) == str(port_id):
                    return op
        return None
    
    def get_node_for_output_port(self, port_id: str) -> Optional['NodeSpec']:
        """Find the node that owns a given output port."""
        for n in self.nodes:
            for op in n.output_ports:
                if str(op.id) == str(port_id):
                    return n
        return None


class ChoregraphSpecParser:
    """Parses the Choregraph XML specification."""

    @staticmethod
    def parse(xml_spec: Union[str, Path]) -> ChoregraphSpec:
        """Parse an XML specification into a ChoregraphSpec.

        Args:
            xml_spec: Path to an XML file, or a raw XML string.

        Returns:
            Parsed ChoregraphSpec with inputs, nodes, and outputs populated.
        """
        if (isinstance(xml_spec, Path) and xml_spec.exists()) or (isinstance(xml_spec, str) and Path(xml_spec).exists()):
            tree = etree.parse(str(xml_spec))
            root = tree.getroot()
        else:
            root = etree.fromstring(xml_spec)

        spec = ChoregraphSpec()

        # Parse Inputs
        inputs_node = root.find("inputs")
        if inputs_node is not None:
            for input_node in inputs_node.findall("input"):
                metadata_keys = ["id", "location", "format", "name", "output", "visibility", "label", "url"]
                options = {k: v for k, v in input_node.attrib.items() if k not in metadata_keys}
                
                if "fieldSeparator" in options: options["sep"] = options.pop("fieldSeparator")
                if "skipLines" in options:
                    try: options["skiprows"] = int(options.pop("skipLines"))
                    except ValueError: pass
                if "skiprows" in options:
                    try: options["skiprows"] = int(options["skiprows"])
                    except ValueError: pass
                if "header" in options:
                    val = str(options["header"]).lower()
                    if val == "true": options["header"] = 0
                    elif val == "false": options["header"] = None
                    elif val.isdigit(): options["header"] = int(val)
                    elif val == "none": options["header"] = None
                if "recordSeparator" in options: options["lineterminator"] = options.pop("recordSeparator")
                
                # Look for 'visibility' first, then fallback to legacy 'output'
                visibility_attr = input_node.get("visibility") or input_node.get("output", "false")
                is_output = str(visibility_attr).lower() == "true"
                input_id = str(input_node.get("id"))  # Keep as string for consistency
                # Fallback for label if missing (legacy XML)
                input_label = input_node.get("label")
                if not input_label:
                    input_label = Path(input_node.get("location", "")).stem if input_node.get("location") else "Input"

                # Infer format from file extension when not explicitly set
                explicit_format = input_node.get("format")
                if explicit_format:
                    fmt = explicit_format.upper()
                else:
                    loc = input_node.get("location", "")
                    ext = Path(loc).suffix.lstrip(".").upper() if loc else ""
                    fmt = ext if ext else "CSV"

                spec.inputs.append(InputSpec(
                    id=input_id,
                    label=input_label,
                    location=input_node.get("location"),
                    format=fmt,
                    options=options,
                    visibility=is_output,
                    url=input_node.get("url"),
                ))
                
                if is_output:
                     if not any(o.id == str(input_id) for o in spec.outputs):
                          spec.outputs.append(OutputSpec(id=str(input_id)))

        # --- Parse pipeline (nodes container) ---
        pipeline_node = root.find("pipeline")
        
        if pipeline_node is not None:
            for node_elem in pipeline_node.findall("node"):
                raw_node_id = node_elem.get("id")
                node_id = str(raw_node_id)  # Keep as string for consistency
                
                node_type = node_elem.get("type")
                # Fallback for label if missing
                node_label = node_elem.get("label")
                if not node_label:
                    node_label = node_type

                # Parse input ports from <inputPort> elements (flattened structure)
                input_ports = []
                for port in node_elem.findall("inputPort"):
                    source_ref_str = port.get("sourceRef")
                    port_name = port.get("name")
                    port_type = port.get("type")

                    # Handle comma-separated sourceRef (e.g. "100, 101, 102")
                    # by expanding into one InputPortSpec per reference.
                    if source_ref_str and "," in source_ref_str:
                        for ref_token in source_ref_str.split(","):
                            ref_token = ref_token.strip()
                            if ref_token.isdigit():
                                input_ports.append(InputPortSpec(
                                    name=port_name,
                                    source_ref=int(ref_token),
                                    type=port_type,
                                ))
                    else:
                        source_ref = int(source_ref_str) if source_ref_str and source_ref_str.isdigit() else None
                        port_value = port.get("value")
                        if port_value is None and port.text and port.text.strip():
                            port_value = port.text.strip()
                        input_ports.append(InputPortSpec(
                            name=port_name,
                            value=port_value,
                            source_ref=source_ref,
                            type=port_type,
                        ))

                # Parse output ports from <outputPort> elements (flattened structure)
                output_ports = []
                for port in node_elem.findall("outputPort"):
                    visibility_attr = port.get("visibility", "false")
                    is_visible = str(visibility_attr).lower() == "true"
                    port_id_str = port.get("id")
                    port_id = int(port_id_str) if port_id_str else 0
                    output_ports.append(OutputPortSpec(
                        id=port_id,
                        name=port.get("name"),
                        type=port.get("type"),
                        visibility=is_visible,
                        label=port.get("label")  # Polished name from AI
                    ))

                spec.nodes.append(NodeSpec(
                    id=node_id,
                    label=node_label,
                    type=node_type,
                    input_ports=input_ports,
                    output_ports=output_ports
                ))

        return spec

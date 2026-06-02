# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pipeline builder -- converts a ChoregraphSpec into Kedro Pipeline objects.

Sits between the parser (which produces ChoregraphSpec) and the Kedro runner.
Resolves input/output port connections, applies XSD-based type conversion for
parameters, and produces Kedro Pipeline objects with correctly wired nodes.
"""
from functools import partial
from typing import List, Dict, Any, Tuple
from kedro.pipeline import Pipeline, node
from .library import TRANSFORM_REGISTRY
from .parser import ChoregraphSpec, InputPortSpec, _sanitize_name
from .xsd_catalogue_utils import load_function_catalogue_from_xsd
import logging

logger = logging.getLogger(__name__)


def _resolve_port_type(port: InputPortSpec, catalogue_port_spec: Dict[str, Any]) -> str:
    """Determine port type, prioritising the XML type attribute over catalogue.

    The XML ``type`` attribute (set by the LLM when it generates the graph)
    is the most authoritative source.  Fall back to the catalogue-derived
    type only when the port has no explicit type or it is ``DATAFRAME``
    (which signals a connected port, not a value port).
    """
    if port.type and port.type != "DATAFRAME":
        return port.type
    return catalogue_port_spec.get("type", "STRING")


def _parse_list_value(raw: str) -> list[str]:
    """Parse a comma-separated string into a list, stripping quotes and brackets.

    Handles all common LLM output formats:
      - ``col1, col2``
      - ``"col1", "col2"``
      - ``[col1, col2]``
      - ``['col1', 'col2']``
    """
    value = raw.strip()
    # Strip surrounding brackets
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    # Split on commas
    items = [v.strip() for v in value.split(",")] if "," in value else [value.strip()]
    # Strip surrounding quotes from each item
    cleaned = []
    for item in items:
        if not item:
            continue
        if (item.startswith('"') and item.endswith('"')) or \
           (item.startswith("'") and item.endswith("'")):
            item = item[1:-1]
        cleaned.append(item)
    return cleaned


def parse_port_value(raw_value: str, port_type: str):
    """Convert a raw string value to the appropriate Python type.

    Args:
        raw_value: The string value from the XML port.
        port_type: The resolved type (FLOAT, INTEGER, BOOLEAN, LIST, etc.).

    Returns:
        The converted Python value.
    """
    try:
        if port_type == "FLOAT":
            return float(raw_value)
        elif port_type == "INTEGER":
            return int(raw_value)
        elif port_type == "BOOLEAN":
            return raw_value.lower() in ("true", "1", "yes")
        elif port_type in ("LIST", "COLUMN_NAME_OR_LIST"):
            return _parse_list_value(raw_value)
        else:  # STRING, COLUMN_NAME, JSON, etc.
            return raw_value
    except (ValueError, AttributeError):
        return raw_value


def parse_ports_to_kwargs(input_ports: List[InputPortSpec], function_name: str) -> Dict[str, Any]:
    """Convert port values from strings to appropriate Python types.

    Uses the function catalogue extracted from the XSD to determine the
    expected type for each port (FLOAT, INTEGER, BOOLEAN, LIST, STRING).
    The XML ``type`` attribute on each port takes priority over the catalogue.

    Args:
        input_ports: List of input port specifications to convert.
        function_name: Transform function name for catalogue lookup.

    Returns:
        Dict mapping port names to their converted Python values.
    """
    # Load catalogue from XSD
    try:
        catalogue = load_function_catalogue_from_xsd()
    except Exception:
        # Fallback: return ports as-is
        return {p.name: p.value for p in input_ports if p.value is not None}

    function_spec = catalogue.get("functions", {}).get(function_name, {})
    input_ports_spec = function_spec.get("input_ports", {})

    kwargs = {}
    for port in input_ports:
        if port.value is None:
            continue

        catalogue_port_spec = input_ports_spec.get(port.name, {})
        port_type = _resolve_port_type(port, catalogue_port_spec)
        kwargs[port.name] = parse_port_value(port.value, port_type)

    return kwargs

def build_pipeline_from_spec(spec: ChoregraphSpec) -> Tuple[Pipeline, Dict[str, str]]:
    """Build a Kedro Pipeline from a ChoregraphSpec.
    
    Args:
        spec: The ChoregraphSpec defining the pipeline
        
    Returns:
        Tuple of (Pipeline, label_mapping) where label_mapping maps sanitized node names
        to their human-readable labels for display in Kedro Viz
    """
    nodes = []
    used_node_names = set()
    # Mapping from sanitized pretty_name to human-readable label
    label_mapping: Dict[str, str] = {}
    for n in spec.nodes:
        if n.type not in TRANSFORM_REGISTRY:
            continue
            
        func_entry = TRANSFORM_REGISTRY[n.type]
        base_func = func_entry["func"]
        
        # 1. Handle Parameters (input ports with 'value')
        kwargs = parse_ports_to_kwargs(n.input_ports, n.type)
        if kwargs:
            node_func = partial(base_func, **kwargs)
            node_func.__name__ = f"{base_func.__name__}_{n.id}"
        else:
            node_func = base_func
            
        # 2. Resolve Inputs (input ports with 'source_ref')
        # sourceRef now points to either an input ID or an output port ID
        # For execute_code nodes, use port names as kwargs keys (the code
        # references DataFrames by these names). For other nodes, use port
        # names as function parameter names.
        # When multiple ports share the same name (multi-input like join/union),
        # fall back to dataset names as kwargs keys.
        node_inputs = {}
        port_counts = {}
        for p in n.input_ports:
            if p.source_ref is not None:
                dataset_name = spec.get_name(p.source_ref)
                port_name = p.name
                if port_name in node_inputs:
                    if port_counts.get(port_name) == 0:
                        # First duplicate: rename the initial occurrence to its dataset name
                        first_dataset = node_inputs.pop(port_name)
                        node_inputs[first_dataset] = first_dataset
                    port_counts[port_name] = port_counts.get(port_name, 0) + 1
                    node_inputs[dataset_name] = dataset_name
                else:
                    node_inputs[port_name] = dataset_name
                    port_counts[port_name] = 0
        
        # 3. Resolve Outputs from output_ports
        # If output_ports are defined, use them; otherwise infer from function
        if n.output_ports:
            if len(n.output_ports) == 1:
                # Single output - use the output port ID to get the dataset name
                final_output = spec.get_name(n.output_ports[0].id)
            else:
                # Multiple outputs - create dict mapping port names to dataset names
                final_output = {}
                for op in n.output_ports:
                    # Use port name as key, dataset name from port ID as value
                    final_output[op.name] = spec.get_name(op.id)
        else:
            # Fallback: infer from function's return_mask parameter
            # This shouldn't happen as nodes should always have output_ports defined
            if kwargs.get("return_mask") is True:
                # Edge case: no output_ports but return_mask=True
                label_text = n.label if n.label else f"node_{n.id}"
                clean_label = _sanitize_name(label_text)
                final_output = {
                    "result": f"{clean_label}_result",
                    "mask": f"{clean_label}_mask"
                }
            else:
                # Use node label as fallback naming
                label_text = n.label if n.label else f"node_{n.id}"
                clean_label = _sanitize_name(label_text)
                final_output = f"{clean_label}_out"
        
        # Special handling for tidy_excel_data: use single output name for PartitionedDataset
        # PartitionedDataset handles Dict[str, DataFrame] internally, saving each key as {key}.parquet
        if n.type == "tidy_excel_data":
            # Use the base output name - PartitionedDataset will handle the dict
            base_name = spec.get_name(n.output_ports[0].id) if n.output_ports else f"excel_{n.id}"
            final_output = base_name
            logger.info(f"tidy_excel_data will output to PartitionedDataset: {base_name}")

        # 4. Create pretty node name (sanitized for Kedro) and store label mapping
        label_text = n.label if n.label else n.type
        sanitized_label = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in label_text)
        
        pretty_name = sanitized_label
        if pretty_name in used_node_names:
            pretty_name = f"{pretty_name}_{n.id}"
        
        # In case the ID-suffixed name is also taken (unlikely but theoretically possible)
        # we can loop, but usually ID is enough.
        base_pretty_name = pretty_name
        counter = 1
        while pretty_name in used_node_names:
            pretty_name = f"{base_pretty_name}_{counter}"
            counter += 1
            
        used_node_names.add(pretty_name)
        
        # Store the mapping from sanitized name to human-readable label
        # This will be used by Kedro Viz proxy to display proper labels
        label_mapping[pretty_name] = label_text

        nodes.append(
            node(
                func=node_func,
                inputs=node_inputs,
                outputs=final_output,
                name=pretty_name
            )
        )
        
    return Pipeline(nodes), label_mapping
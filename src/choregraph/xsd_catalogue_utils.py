# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Clément Baraille
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
XSD Catalogue Utilities

This module extracts function catalogue information from the TransformGraph.xsd file.
It replaces the JSON-based catalogue with XSD-based definitions.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from lxml import etree
import re

# Namespace for XSD
XS_NS = "{http://www.w3.org/2001/XMLSchema}"

# Cache for loaded catalogue
_catalogue_cache: Optional[Dict[str, Any]] = None
_xsd_mtime: Optional[float] = None


def _extract_port_type_from_complex_type(complex_type_name: str) -> str:
    """Extract the port type from a complex type name like 'StaticFloatPort' -> 'FLOAT'."""
    mapping = {
        "ConnectedDataFramePort": "DATAFRAME",
        "StaticColumnNamePort": "COLUMN_NAME",
        "StaticColumnNameOrListPort": "COLUMN_NAME_OR_LIST",
        "StaticListPort": "LIST",
        "StaticFloatPort": "FLOAT",
        "StaticIntegerPort": "INTEGER",
        "StaticBooleanPort": "BOOLEAN",
        "StaticStringPort": "STRING",
        # New mappings based on Primitive types in XSD
        "DataFrame": "DATAFRAME",
        "Image": "IMAGE",
        "String": "STRING",
        "Float": "FLOAT",
        "Integer": "INTEGER",
        "Boolean": "BOOLEAN",
        "List": "LIST",
        "Json": "JSON",
        "Dict": "JSON", # Alias if needed
    }
    return mapping.get(complex_type_name, "STRING")


def _is_connected_port(complex_type_name: str) -> bool:
    """Check if a port type represents a connected port (vs static)."""
    return complex_type_name.startswith("Connected") or complex_type_name in ("DataFrame", "Image")


def _extract_comment_before_element(element: etree._Element) -> Optional[str]:
    """Extract the XML comment immediately before an element."""
    prev = element.getprevious()
    while prev is not None:
        if isinstance(prev, etree._Comment):
            return prev.text.strip()
        # Skip whitespace-only text nodes
        if prev.tag is not None:
            break
        prev = prev.getprevious()
    return None


def _extract_function_from_complex_type(complex_type: etree._Element, root: etree._Element) -> Optional[Dict[str, Any]]:
    """
    Extract function metadata from a complexType element that defines a function.
    
    Returns a dict with:
    - description: from XML comment before the complexType
    - input_ports: dict of port definitions
    - output_ports: dict of port definitions
    - group: optional group name from appinfo
    """
    # Get function name from appinfo
    appinfo = complex_type.find(f".//{XS_NS}appinfo")
    if appinfo is None:
        return None
    
    function_elem = appinfo.find("function")
    if function_elem is None:
        return None
    
    func_name = function_elem.get("name")
    if not func_name:
        return None
    
    # Get description from comment before complexType
    description = _extract_comment_before_element(complex_type)
    
    # Get group if present
    group = function_elem.get("group")
    
    # Get notes if present
    notes_elem = appinfo.find("notes")
    notes = notes_elem.text if notes_elem is not None else None
    
    # Parse input ports from sequence
    input_ports = {}
    sequence = complex_type.find(f"{XS_NS}sequence")
    if sequence is not None:
        # Collect elements: direct children + those nested inside xs:choice
        all_elements = list(sequence.findall(f"{XS_NS}element"))
        for choice in sequence.findall(f"{XS_NS}choice"):
            all_elements.extend(choice.findall(f"{XS_NS}element"))
        for elem in all_elements:
            port_name = elem.get("name")
            port_type_ref = elem.get("type")
            min_occurs = elem.get("minOccurs", "1")
            max_occurs = elem.get("maxOccurs", "1")
            
            if port_name and port_type_ref:
                port_type = _extract_port_type_from_complex_type(port_type_ref)
                is_connected = _is_connected_port(port_type_ref)
                
                # Get port description from comment before element
                port_desc = _extract_comment_before_element(elem)
                
                # Check for semantic type definition in description
                # e.g. "The column ... (Semantic: COLUMN_NAME)"
                semantic_match = None
                if port_desc:
                    semantic_match = re.search(r"\(Semantic:\s*(\w+)\)", port_desc)
                
                if semantic_match:
                    port_type = semantic_match.group(1).strip()
                
                port_def = {
                    "type": port_type,
                    "description": port_desc or f"The {port_name} parameter",
                    "connection": "connected" if is_connected else "static",
                    "required": min_occurs != "0"
                }
                
                if max_occurs == "unbounded":
                    port_def["multiple"] = True
                
                input_ports[port_name] = port_def
    
    # Output ports - always a single DATAFRAME result for now
    output_ports = {
        "result": {
            "type": "DATAFRAME",
            "description": "The resulting DataFrame"
        }
    }
    
    result = {
        "description": description or f"Function {func_name}",
        "input_ports": input_ports,
        "output_ports": output_ports
    }
    
    if group:
        result["group"] = group
    if notes:
        result["notes"] = notes
    
    return {func_name: result}


def _extract_types_from_xsd(root: etree._Element) -> Dict[str, Any]:
    """Extract port type definitions from xsd."""
    types = {}
    
    # Find PortType simpleType
    for simple_type in root.findall(f"{XS_NS}simpleType"):
        name = simple_type.get("name")
        if name == "PortType":
            restriction = simple_type.find(f"{XS_NS}restriction")
            if restriction is not None:
                for enum in restriction.findall(f"{XS_NS}enumeration"):
                    type_name = enum.get("value")
                    if type_name:
                        types[type_name] = {
                            "description": f"A {type_name.lower().replace('_', ' ')} value"
                        }
    
    # Add specific descriptions
    type_descriptions = {
        "DATAFRAME": "A pandas DataFrame",
        "COLUMN_NAME": "A string representing a column name",
        "COLUMN_NAME_OR_LIST": "A column name or a list of column names",
        "LIST": "A list of values (comma-separated string)",
        "FLOAT": "A floating-point number",
        "INTEGER": "An integer number",
        "BOOLEAN": "A boolean value (true/false)",
        "STRING": "A string value",
        "JSON": "A JSON object or list",
    }
    
    for type_name, desc in type_descriptions.items():
        if type_name in types:
            types[type_name]["description"] = desc
        else:
            # Inject types that are used semantically but might not be in the enum
            types[type_name] = {"description": desc}
    
    return types


def load_function_catalogue_from_xsd(xsd_path: Path = None) -> Dict[str, Any]:
    """
    Parse the XSD and return a dictionary compatible with the old JSON format.
    
    Uses caching to avoid re-parsing if the file hasn't changed.
    
    Returns:
        {
            "functions": {
                "filter_less_than": {
                    "description": "...",
                    "input_ports": {...},
                    "output_ports": {...}
                }, ...
            },
            "types": {...}
        }
    """
    global _catalogue_cache, _xsd_mtime

    if xsd_path is None:
        xsd_path = Path(__file__).parent / "TransformGraph.xsd"

    # Check cache validity
    current_mtime = xsd_path.stat().st_mtime if xsd_path.exists() else None
    if _catalogue_cache is not None and _xsd_mtime == current_mtime:
        return _catalogue_cache

    # Parse XSD — file first (dev), embedded fallback (compiled)
    if xsd_path.exists():
        tree = etree.parse(str(xsd_path))
    else:
        from ._xsd_data import get_transformgraph_xsd
        tree = etree.fromstring(get_transformgraph_xsd().encode("utf-8"))
    root = tree if isinstance(tree, etree._Element) else tree.getroot()
    
    functions = {}
    
    # Find all complexType elements that define functions
    for complex_type in root.findall(f"{XS_NS}complexType"):
        func_data = _extract_function_from_complex_type(complex_type, root)
        if func_data:
            functions.update(func_data)
    
    # Extract types
    types = _extract_types_from_xsd(root)
    
    _catalogue_cache = {
        "functions": functions,
        "types": types
    }
    _xsd_mtime = current_mtime
    
    return _catalogue_cache


def get_function_spec(function_name: str, xsd_path: Path = None) -> Optional[Dict[str, Any]]:
    """
    Get the specification for a single function.
    
    Returns None if function not found.
    """
    catalogue = load_function_catalogue_from_xsd(xsd_path)
    return catalogue.get("functions", {}).get(function_name)


def get_port_type_info(port_type: str, xsd_path: Path = None) -> Optional[Dict[str, Any]]:
    """
    Get information about a port type.
    
    Returns None if type not found.
    """
    catalogue = load_function_catalogue_from_xsd(xsd_path)
    return catalogue.get("types", {}).get(port_type)


def list_functions(xsd_path: Path = None) -> list:
    """
    List all available transformation functions.
    """
    catalogue = load_function_catalogue_from_xsd(xsd_path)
    return list(catalogue.get("functions", {}).keys())


def generate_catalogue_text(xsd_path: Path = None) -> str:
    """
    Generate a human-readable text representation of the function catalogue.
    Useful for LLM prompts.
    """
    catalogue = load_function_catalogue_from_xsd(xsd_path)
    
    lines = ["# Available Transformation Functions\n"]
    
    for func_name, func_spec in catalogue.get("functions", {}).items():
        lines.append(f"## {func_name}")
        lines.append(f"{func_spec.get('description', '')}\n")
        
        # Input ports
        lines.append("  **Input Ports:**")
        for port_name, port_spec in func_spec.get("input_ports", {}).items():
            required = "required" if port_spec.get("required", True) else "optional"
            connection = port_spec.get("connection", "static")
            lines.append(f"  - `{port_name}` ({port_spec['type']}, {connection}, {required}): {port_spec.get('description', '')}")
        
        # Output ports
        lines.append("\n  **Output Ports:**")
        for port_name, port_spec in func_spec.get("output_ports", {}).items():
            lines.append(f"- `{port_name}` ({port_spec['type']}): {port_spec.get('description', '')}")
        
        lines.append("")
    
    return "\n".join(lines)


def clear_cache():
    """Clear the catalogue cache. Useful for testing."""
    global _catalogue_cache, _xsd_mtime
    _catalogue_cache = None
    _xsd_mtime = None

# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""DIVE VisuSpec connector -- exports pipeline data to DIVE XML format.

Translates Choregraph pipeline outputs into VisuSpec XML consumed by the DIVE
C++ visualization kernel. Handles metadata extraction, field statistics, and
XML generation/merging with existing specification files.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from lxml import etree
import pandas as pd
from kedro.io import DataCatalog
from ..metadata import MetadataExtractor, FieldMetadata

if TYPE_CHECKING:
    from ..metadata import Metadata
    from ..choregraph import Choregraph

logger = logging.getLogger(__name__)

# Mapping from choregraph transform types to XSD TransformType enum values
# XSD allowed values: ADD, COUNT, EXTRACT_POINTS, EXTRACT_CELLS, UNKNOWN, UNDEFINED
TRANSFORM_TYPE_MAPPING = {
    "add": "ADD",
    "count": "COUNT",
    "aggregate_count": "COUNT",
    "extract_points": "EXTRACT_POINTS",
    "extract_cells": "EXTRACT_CELLS",
    # All other transforms map to UNKNOWN
}


def _map_transform_type(transform_type: Optional[str]) -> str:
    """Map choregraph transform type to XSD TransformType enum value."""
    if not transform_type:
        return "UNDEFINED"
    normalized = transform_type.lower().strip()
    return TRANSFORM_TYPE_MAPPING.get(normalized, "UNKNOWN")


class CacheProxy:
    """Adapter making a Choregraph instance look like a Kedro DataCatalog.

    Used by :class:`DiveConnector` to load datasets transparently through
    Choregraph's cache layer.
    """

    def __init__(self, choregraph_instance: "Choregraph"):
        self.cg = choregraph_instance

    def load(self, name: str) -> Any:
        return self.cg.get_dataset(name)

    def list(self) -> List[str]:
        # 1. Get explicit datasets (e.g. Inputs defined in catalog.yml)
        keys = set(self.cg.list_data())

        # 2. FORCE ADD the "invisible" factory datasets (Output Ports)
        # Because they are pattern-matched, Kedro's list() ignores them.
        # We manually check the spec and add them here.
        for n in self.cg.spec.nodes:
            for op in n.output_ports:
                if op.visibility:
                    keys.add(self.cg.spec.get_name(op.id))

        # 3. Force Add Inputs (just for safety)
        for inp in self.cg.spec.inputs:
            keys.add(self.cg.spec.get_name(inp.id))

        # 4. Add Excel multi-table datasets from data cache
        for key in self.cg._data_cache.keys():
            if key not in keys:
                keys.add(key)

        return list(keys)

    def keys(self) -> List[str]:
        return self.list()

    def __contains__(self, name: str) -> bool:
        return name in self.list()

    def __iter__(self):
        return iter(self.list())

    def __getattr__(self, name: str):
        return getattr(self.cg._get_catalog(), name)


class DiveConnector:
    """
    Handles translation and export of Choregraph data to DIVE-compatible formats.
    """

    def __init__(self, catalog: DataCatalog, metadata: "Metadata" = None):
        """Initialize the DIVE connector.

        Args:
            catalog: Kedro DataCatalog (or :class:`CacheProxy`)
                for loading datasets.
            metadata: Metadata accessor for reading cached dataset stats
                from ``catalogue_stats.json``.
        """
        self.catalog = catalog
        self.metadata = metadata

    @classmethod
    def from_choregraph(cls, cg: "Choregraph") -> "DiveConnector":
        """Create a DiveConnector wired to a Choregraph instance.

        Args:
            cg: The Choregraph facade to read data and metadata from.
        """
        return cls(CacheProxy(cg), metadata=cg._datasets_metadata)

    # ------------------------------------------------------------------
    # Allow-list computation (transplanted from Choregraph)
    # ------------------------------------------------------------------

    def _compute_allow_list(self) -> Optional[List[str]]:
        """Compute which dataset names to include in VisuSpec XML.

        Collects visible output ports, their ancestors, and dynamic
        multi-table outputs from the workspace filesystem.

        Returns ``None`` when no spec is available (= include everything).
        """
        cg = getattr(self.catalog, "cg", None)
        spec = cg.spec if cg else None
        if spec is None:
            return None

        # 1. Collect visible output IDs + explicit outputs
        included_ids = set()
        for o in getattr(spec, "outputs", []):
            included_ids.add(str(o.id))
        for n in spec.nodes:
            for op in n.output_ports:
                if op.visibility:
                    included_ids.add(str(op.id))

        # 2. If no visible outputs, find terminal nodes
        if not included_ids and spec.nodes:
            referenced_ids = set()
            for n in spec.nodes:
                for port in n.input_ports:
                    if port.source_ref is not None:
                        referenced_ids.add(str(port.source_ref))
            for n in spec.nodes:
                for op in n.output_ports:
                    if str(op.id) not in referenced_ids:
                        included_ids.add(str(op.id))
            if not included_ids:
                for n in spec.nodes:
                    for op in n.output_ports:
                        included_ids.add(str(op.id))

        # 3. If no nodes at all, include visible inputs (or everything if none visible)
        if not included_ids:
            for inp in spec.inputs:
                if getattr(inp, "visibility", False):
                    included_ids.add(str(inp.id))
            if not included_ids:
                return None  # nothing qualifies → include everything
            return [spec.get_name(id) for id in included_ids]

        # 4. Recursively add ancestors
        queue = list(included_ids)
        visited = set()
        while queue:
            curr_id = str(queue.pop(0))
            if curr_id in visited:
                continue
            visited.add(curr_id)
            node = spec.get_node_for_output_port(curr_id)
            if node:
                for port in node.input_ports:
                    if port.source_ref is not None:
                        src_id = str(port.source_ref)
                        if src_id not in included_ids:
                            included_ids.add(src_id)
                            queue.append(src_id)

        outputs = [spec.get_name(id) for id in included_ids]

        # 5. Add multi-table outputs from workspace filesystem
        ws = cg.workspace_path if cg else None
        if ws:
            data_dir = Path(ws) / "pipeline" / "data"
            inputs_dir = data_dir / "inputs"
            multi_tables = []

            if inputs_dir.exists():
                for pf in inputs_dir.glob("*.parquet"):
                    multi_tables.append(pf.stem)

            if data_dir.exists():
                for pd_dir in data_dir.glob("*_partitioned"):
                    if pd_dir.is_dir():
                        base_name = pd_dir.name.replace("_partitioned", "")
                        for pf in pd_dir.glob("*.parquet"):
                            multi_tables.append(pf.stem)
                        outputs = [o for o in outputs if o != base_name]

            for table_name in multi_tables:
                if table_name not in outputs:
                    outputs.append(table_name)

        return outputs

    # ------------------------------------------------------------------
    # XML generation
    # ------------------------------------------------------------------

    def generate_visuspec_xml(self, allow_list: List[str] = None) -> str:
        """Generate VisuSpec XML containing dataset definitions and field metadata.

        Args:
            allow_list: Optional list of dataset names to include. If None,
                the list is computed automatically from the spec.

        Returns:
            XML string with a ``<datas>`` root element.
        """
        if allow_list is None:
            allow_list = self._compute_allow_list()

        root = etree.Element("datas")

        # 0. Preparation: Access ChoregraphSpec if available via Proxy
        cg = getattr(self.catalog, "cg", None)
        spec = cg.spec if cg else None

        # Default location for transformations/redirects
        default_location = "choregraph.xml"
        if cg and cg.workspace_path:
            default_location = str("./choregraph.xml")

        # Pre-load all cached stats once (read-only)
        all_stats = self.metadata.read_from_cache() if self.metadata else {}

        xml_id_map = {} # Maps dataset clean name to assigned XML ID
        next_xml_id = 1
        spec_datasets = set() # Track all names defined in spec (visible or not)

        # Formats where the viz renderer loads the file directly (not from catalog).
        # The file element gets the native format + actual path instead of XML.
        _DIRECT_FORMATS = {"MHD", "DICOM"}

        def add_data_element(name: str, spec_id: Any, is_input: bool = True,
                             source_id: str = None, transform_type: str = None,
                             input_format: str = None, input_location: str = None):
            nonlocal next_xml_id

            is_direct = input_format and input_format.upper() in _DIRECT_FORMATS

            # --- Common Metadata Extraction ---
            fields_metadata = None
            row_count = 0

            # Check cached stats from Metadata (catalogue_stats.json)
            # This works for all formats: CSV, images (proxy fields), MHD (proxy fields)
            stats = all_stats.get(name)
            if stats:
                fields_metadata = stats.fields
                row_count = stats.row_count
            elif not is_direct:
                # Fallback: load from catalog + extract (read-only, no write to JSON)
                try:
                    df = self.catalog.load(name)
                    if isinstance(df, pd.DataFrame):
                        row_count = len(df)
                        fields_metadata = MetadataExtractor.extract(df)
                    else:
                        # Generic object (JSON dict/list)
                        row_count = 1
                        fields_metadata = [FieldMetadata(id="1", name="file_content", data_type="RAW", distinct_count=1)]
                except Exception as e:
                    logger.warning(f"Failed to load {name} for metadata extraction: {e}")

            if fields_metadata is None:
                if is_direct:
                    # Direct formats without precomputed stats: leave empty,
                    # C++ parser will create proxy fields from the file header.
                    fields_metadata = []
                else:
                    # Fallback to generic metadata to avoid skipping the entry
                    fields_metadata = [FieldMetadata(id="1", name="unknown", data_type="STRING", distinct_count=0)]

            # --- Create Element ---
            if spec_id is not None:
                xml_id = str(spec_id)
            else:
                # Find valid next ID
                while str(next_xml_id) in xml_id_map.values():
                     next_xml_id += 1
                xml_id = str(next_xml_id)
                next_xml_id += 1

            xml_id_map[name] = xml_id

            elem = etree.SubElement(root, "rawData")
            elem.set("id", xml_id)
            elem.set("name", name)
            elem.set("rows", str(row_count))

            # --- Fields ---
            fields_elem = etree.SubElement(elem, "fields")
            for field in fields_metadata:
                f_elem = etree.SubElement(fields_elem, "field")
                f_elem.set("id", field.id)
                f_elem.set("name", field.name)
                f_elem.set("dataType", field.data_type)
                f_elem.set("unit", field.units)
                f_elem.set("distinctCount", str(field.distinct_count))
                f_elem.set("fieldMin", str(field.min_value) if field.min_value is not None else "0")
                f_elem.set("fieldMax", str(field.max_value) if field.max_value is not None else str(field.distinct_count))

            # --- File element ---
            file_elem = etree.SubElement(elem, "file")

            if is_direct:
                # Direct format: use actual file path and native format
                file_elem.set("format", input_format.upper())
                if input_location:
                    loc = input_location
                    if not Path(loc).is_absolute() and cg and cg.workspace_path:
                        loc = str((Path(cg.workspace_path) / loc).resolve())
                    file_elem.set("location", loc)
                else:
                    file_elem.set("location", default_location)
            else:
                file_elem.set("location", default_location)
                file_elem.set("format", "XML")

            return elem

        # 1. First Pass: Inputs (rawData)
        if spec:
            for inp in spec.inputs:
                clean_name = spec.get_name(inp.id)
                spec_datasets.add(clean_name)
                if not getattr(inp, "visibility", True):
                    continue  # Skip invisible inputs
                if allow_list is None or clean_name in allow_list:
                    add_data_element(
                        clean_name, inp.id, is_input=True,
                        input_format=getattr(inp, "format", None),
                        input_location=getattr(inp, "location", None),
                    )

        # 2. Second Pass: Output Ports (transformedData)
        if spec:
            for node in spec.nodes:
                for op in node.output_ports:
                    clean_name = spec.get_name(op.id)
                    spec_datasets.add(clean_name)
                    # Check output port visibility
                    if not getattr(op, "visibility", True):
                        continue  # Skip invisible output ports
                    if allow_list is None or clean_name in allow_list:
                        # Resolve originalDataId from first input port's source
                        source_xml_id = None
                        if node.input_ports:
                            source_ref = node.input_ports[0].source_ref
                            if source_ref:
                                source_clean_name = spec.get_name(source_ref)
                                source_xml_id = xml_id_map.get(source_clean_name)

                        add_data_element(clean_name, op.id, is_input=False, source_id=source_xml_id, transform_type=node.type)

        # 3. Fallback: Catch-all for datasets not in spec (legacy / stand-alone catalog)
        keys = []
        if hasattr(self.catalog, "list"): keys = self.catalog.list()
        elif hasattr(self.catalog, "keys"): keys = list(self.catalog.keys())

        for k in keys:
            if k not in xml_id_map and k not in spec_datasets:
                if allow_list is None or k in allow_list:
                    add_data_element(k, None, is_input=True)

        return etree.tostring(root, pretty_print=True).decode()

    def update_visuspec_xml(self, save_to_path: Union[str, Path], allow_list: List[str] = None) -> str:
        """Generate the VisuSpec XML datas block and merge it into an existing file."""
        if allow_list is None:
            allow_list = self._compute_allow_list()
        datas_xml_str = self.generate_visuspec_xml(allow_list=allow_list)
        if save_to_path:
            try:
                path = Path(save_to_path)

                if path.exists():
                    parser = etree.XMLParser(remove_blank_text=True)
                    tree = etree.parse(str(path), parser)
                    root = tree.getroot()

                    new_datas = etree.fromstring(datas_xml_str)
                    existing_datas = root.find("datas")

                    if existing_datas is not None:
                        existing_datas.getparent().replace(existing_datas, new_datas)
                    else:
                        root.insert(0, new_datas)

                    tree.write(str(path), pretty_print=True, xml_declaration=True, encoding="utf-8")
                    logger.info(f"Updated VisuSpec at {path}")
                else:
                    logger.warning(f"VisuSpec file does not exist: {path}")
            except Exception as e:
                logger.error(f"Failed to update VisuSpec: {e}")
                import traceback
                traceback.print_exc()
        return datas_xml_str

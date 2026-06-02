# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Guillaume Franque
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Choregraph file building utilities.

Convenience functions for creating / updating choregraph.xml and
specifications.xml using the native Choregraph API.  These are pure library
operations -- no DB, no auth, no network.  Any service with choregraph
installed can call them.
"""
from __future__ import annotations

import logging
import os
from lxml import etree
import zlib
from pathlib import Path
from typing import List, Optional
from .security import safe_path

logger = logging.getLogger(__name__)


def _stable_input_id(stem: str) -> str:
    """Deterministic 4-digit numeric ID from a filename stem.

    Uses CRC32 clamped to 1000–9999 so IDs stay short (LLM-friendly)
    while remaining stable across rebuilds — the same filename always
    gets the same ID regardless of link order or other files present.
    """
    return str(zlib.crc32(stem.encode()) % 9000 + 1000)


def build_choregraph_inputs(
    workspace_path: str,
    file_paths: list[str],
    existing_xml: str | None = None,
    temporal_info: dict | None = None,
) -> None:
    """Create or update choregraph.xml from file paths.

    If *existing_xml* is ``None``: creates from scratch (stable hash-based IDs
    derived from filename stems — same file always gets the same ID).
    If *existing_xml* provided: loads it, appends new files (``give_id()``),
    skips already-present files (by basename).  Preserves all existing IDs.

    Characterizes each file:
      CSV  -> ``characterize_csv()``
      JSON -> structure described via ``catalogue_stats.json`` (``extract_with``)
      Other -> format from extension

    Uses: ``Choregraph.add_input()``, ``give_id()``, ``export_to_xml()``

    Args:
        workspace_path: Root workspace directory containing data files.
        file_paths: Absolute paths to data files to add.
        existing_xml: Path to an existing ``choregraph.xml`` to append to,
            or ``None`` to create from scratch.
        temporal_info: Optional dict mapping primary file path to temporal
            metadata for ONE-input collections::

                { "/path/primary.csv": {
                    "all_paths": ["/path/primary.csv", "/path/companion1.csv", ...],
                    "time_mode": "index" | "time_based",
                    "time_delta": "PT60S",
                }}
    """
    from .choregraph import Choregraph
    from .loaders import characterize_csv

    choregraph_out = os.path.join(workspace_path, "choregraph.xml")
    sanitize_choregraph_out = safe_path(choregraph_out)

    if existing_xml and os.path.isfile(existing_xml):
        cg = Choregraph(xml_spec=existing_xml, workspace_path=workspace_path)
    else:
        cg = Choregraph(workspace_path=workspace_path)
        cg.reset_spec()

    # Collect existing input basenames to skip duplicates
    existing_basenames = set()
    for inp in cg.spec.inputs:
        if inp.location:
            existing_basenames.add(os.path.basename(inp.location))

    # Collect Excel filenames for tidy_excel_data file_context
    excel_filenames = [
        Path(p).stem
        for p in file_paths
        if p and Path(p).suffix.lower() in (".xlsx", ".xls", ".ods", ".xlsm")
    ]
    file_context = (
        "\n".join(f"- {name}" for name in excel_filenames)
        if len(excel_filenames) > 1
        else ""
    )

    # Track used IDs to detect (extremely unlikely) CRC32 collisions
    used_ids: set[str] = set()

    def _get_id(stem: str, suffix: str = "") -> str:
        """Return a stable ID for *stem*, falling back to give_id() for
        incremental appends (existing_xml is set)."""
        if existing_xml:
            return cg.give_id()
        candidate = _stable_input_id(stem + suffix)
        while candidate in used_ids:
            candidate = str(int(candidate) + 1)
        used_ids.add(candidate)
        return candidate

    # Build set of companion files that belong to temporal collections
    # (these are skipped — only the primary file creates an input)
    temporal_companions: set[str] = set()
    temporal_info = temporal_info or {}
    for primary_path, info in temporal_info.items():
        for p in info.get("all_paths", []):
            if p != primary_path:
                temporal_companions.add(os.path.basename(p))

    for file_idx, datafile_path in enumerate(file_paths):
        if datafile_path is None:
            continue

        datafile_path_str = str(datafile_path)
        sanitize_datafile_path = safe_path(datafile_path_str)
        fname = os.path.basename(datafile_path_str)

        # Skip if already present
        if fname in existing_basenames:
            logger.info(f"build_choregraph_inputs: '{fname}' already in choregraph, skipping")
            continue

        # Skip companion files of temporal collections (primary creates the input)
        if fname in temporal_companions:
            existing_basenames.add(fname)
            continue

        # DICOM series: directory of .dcm slices → single input
        if os.path.isdir(datafile_path_str):
            new_id = _get_id(Path(datafile_path_str).name)
            cg.add_input(
                id=new_id,
                location=datafile_path_str,
                format="DICOM",
                visibility=True,
            )
            existing_basenames.add(fname)
            continue

        # Build temporal kwargs if this file is a temporal collection primary
        ckw: dict = {}
        temporal_location = None  # override location for PartitionedDataset
        tinfo = temporal_info.get(datafile_path_str)
        if tinfo:
            all_paths = tinfo["all_paths"]
            ckw["temporalFiles"] = "|".join(os.path.basename(p) for p in all_paths)
            ckw["collectionTimeMode"] = tinfo.get("time_mode", "index")
            if tinfo.get("time_delta"):
                ckw["collectionTimeDelta"] = tinfo["time_delta"]
            # Location points to the primary file (consistent with non-temporal inputs)
            temporal_location = str(all_paths[0])
            for p in all_paths:
                existing_basenames.add(os.path.basename(p))

        _, ext = os.path.splitext(sanitize_datafile_path)
        ext = ext.lower()
        fmt = ext.lstrip(".").upper()
        options: dict = {}

        if ext == ".csv":
            csv_char = characterize_csv(sanitize_datafile_path)
            logger.info(f'build_choregraph_inputs: CSV characterization "{fname}": {csv_char}')
            if csv_char:
                options = {
                    "header": str(csv_char["header"]),
                    "fieldSeparator": str(csv_char["fieldSeparator"]),
                    "skipLines": str(csv_char["skipLines"]),
                }
            else:
                options = {"header": "True", "fieldSeparator": ",", "skipLines": "0"}
        elif ext == ".json":
            fmt = "JSON"
            # Structure description is carried via catalogue_stats.json ->
            # datasets.<name>.info.extract_with (populated by
            # Metadata._describe_json_structure at upload time).

        if ext in (".xlsx", ".xls", ".ods", ".xlsm"):
            fmt = "XLSX"
            stem = Path(datafile_path_str).stem
            input_id = _get_id(stem)
            cg.add_input(
                id=input_id,
                location=sanitize_datafile_path,
                format="XLSX",
                visibility=False,
                **options,
                **ckw,
            )

            # Add tidy_excel_data node
            try:
                from .parser import InputPortSpec, OutputPortSpec

                node_id = _get_id(stem, "_node")
                out_id = _get_id(stem, "_out")
                cg.add_node(
                    id=node_id,
                    type="tidy_excel_data",
                    input_ports=[
                        InputPortSpec(name="path_excel", value=sanitize_datafile_path),
                        InputPortSpec(name="file_context", value=file_context),
                    ],
                    output_ports=[
                        OutputPortSpec(
                            id=out_id,
                            name="result",
                            label="",
                            type="DICT",
                            visibility=True,
                        )
                    ],
                    label=f"Excel Tidy Data {stem}",
                )
            except Exception as e:
                logger.error(f"build_choregraph_inputs: tidy_excel_data node failed: {e}")
        else:
            # Skip companion files
            if fmt in ("ZRAW", "DCM"):
                continue

            is_json = fmt == "JSON"
            new_id = _get_id(Path(sanitize_datafile_path).stem)
            # For temporal collections, location is the subdirectory but
            # the label should be the primary file's stem (not the dir name)
            label = Path(sanitize_datafile_path).stem if temporal_location else None
            cg.add_input(
                id=new_id,
                location=temporal_location or sanitize_datafile_path,
                format=fmt,
                label=label,
                visibility=False if is_json else True,
                **options,
                **ckw,
            )

        existing_basenames.add(fname)

    cg.export_to_xml(choregraph_out)
    cg.close()


def remove_choregraph_inputs(
    workspace_path: str,
    filenames: list[str],
) -> list[str]:
    """Remove inputs by filename from existing choregraph.xml.

    Preserves all other IDs/nodes.

    Args:
        workspace_path: Root workspace directory.
        filenames: Basenames of files to remove (e.g. ``["airbnb.csv"]``).

    Returns:
        List of removed labels.
    """
    from .choregraph import Choregraph

    choregraph_xml = os.path.join(workspace_path, "choregraph.xml")
    sanitize_choregraph_xml = safe_path(choregraph_xml)
    if not os.path.isfile(sanitize_choregraph_xml):
        return []

    cg = Choregraph(xml_spec=sanitize_choregraph_xml, workspace_path=workspace_path)
    filenames_set = set(filenames)
    removed: list[str] = []

    ids_to_remove = []
    for inp in cg.spec.inputs:
        if inp.location and os.path.basename(inp.location) in filenames_set:
            ids_to_remove.append(
                (str(inp.id), inp.label or os.path.basename(inp.location))
            )

    for inp_id, inp_label in ids_to_remove:
        cg.remove_input(inp_id)
        removed.append(inp_label)
        logger.info(f"remove_choregraph_inputs: Removed '{inp_label}' (id={inp_id})")

    if removed:
        cg.export_to_xml(sanitize_choregraph_xml)

    cg.close()
    return removed


def count_choregraph_inputs(workspace_path: str) -> int:
    """Return the number of inputs in choregraph.xml (0 if missing)."""
    choregraph_xml = os.path.join(workspace_path, "choregraph.xml")
    sanitize_xml_path = safe_path(choregraph_xml)
    if not os.path.isfile(sanitize_xml_path):
        return 0
    try:
        tree = etree.parse(sanitize_xml_path)
        return len(tree.getroot().findall(".//input"))
    except etree.XMLSyntaxError:
        return 0


def create_specifications_xml(workspace_path: str) -> None:
    """Create an empty specifications.xml scaffold.

    Args:
        workspace_path: Root workspace directory where ``specifications.xml``
            will be written.
    """
    xml_filepath = os.path.join(workspace_path, "specifications.xml")
    sanitize_xml_path = safe_path(xml_filepath)

    root = etree.Element("visuSpec", name="UserFile")
    etree.SubElement(root, "coordinates").text = "CARTESIAN"
    etree.SubElement(root, "datas")

    color_palettes = etree.SubElement(root, "colorPalettes")
    etree.SubElement(color_palettes, "colorPalette", id="1", name="UNDEFINED")

    shape_palettes = etree.SubElement(root, "shapePalettes")
    shape_palette = etree.SubElement(shape_palettes, "shapePalette", id="1")
    etree.SubElement(shape_palette, "shape").text = "POINT"

    channels = etree.SubElement(root, "channels")
    etree.SubElement(channels, "undefinedChannels")
    etree.SubElement(channels, "numericChannels")
    etree.SubElement(channels, "colorChannels")
    etree.SubElement(channels, "shapeChannels")

    etree.SubElement(root, "marks")

    etree.SubElement(
        root,
        "space",
        xSpatialScaling="-1",
        ySpatialScaling="-1",
        zSpatialScaling="-1",
        sizeSpatialScaling="-1",
    )

    _indent(root)
    tree = etree.ElementTree(root)
    tree.write(sanitize_xml_path, encoding="utf-8", xml_declaration=True)


def extract_datasets_metadata(workspace_path: str) -> list[dict]:
    """Extract dataset metadata from specifications.xml.

    Pure XML parsing -- reads ``<rawData>`` elements.

    Args:
        workspace_path: Root workspace directory containing ``specifications.xml``.

    Returns:
        List of dataset dicts with ``data_id``, ``name``, ``rows``,
        ``filename``, and ``fields`` list.
    """
    sanitize_workspace_path = safe_path(workspace_path)
    xml_path = os.path.join(sanitize_workspace_path, "specifications.xml")
    datasets: list[dict] = []

    sanitize_xml_path = safe_path(xml_path)
    if not os.path.isfile(sanitize_xml_path):
        return datasets

    try:
        tree = etree.parse(sanitize_xml_path)
        root = tree.getroot()

        for raw_data in root.findall(".//rawData"):
            filename = ""
            file_element = raw_data.find("file")
            if file_element is not None:
                file_location = file_element.get("location", "")
                if file_location:
                    filename = os.path.basename(file_location)

            fields_metadata = []
            for field in raw_data.findall(".//field"):
                fields_metadata.append({
                    "field_id": field.get("id"),
                    "name": field.get("name"),
                    "data_type": field.get("dataType"),
                    "field_min": field.get("fieldMin"),
                    "field_max": field.get("fieldMax"),
                    "distinct_count": field.get("distinctCount"),
                })

            datasets.append({
                "data_id": raw_data.get("id"),
                "name": raw_data.get("name", ""),
                "rows": raw_data.get("rows", "0"),
                "filename": filename,
                "fields": fields_metadata,
            })
    except Exception as e:
        logger.error(f"extract_datasets_metadata error: {e}")

    return datasets


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _indent(elem, level: int = 0) -> None:
    """Format XML with proper indentation."""
    i = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

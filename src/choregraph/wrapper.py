# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Managed project builder -- generates Kedro project structure on disk.

Produces the complete Kedro project files (pyproject.toml, settings.py,
catalog.yml, pipeline_registry.py) in a ``pipeline/`` directory. The
generated project is the single source of truth for Kedro session execution.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict
import yaml
import kedro

from .parser import ChoregraphSpec, _sanitize_name
from .loaders import prepare_load_args
from .builder import _resolve_port_type, parse_port_value

logger = logging.getLogger(__name__)

# Scalar port types that should stay in memory (Kedro default MemoryDataset).
_SCALAR_TYPES = frozenset({"FLOAT", "INTEGER", "BOOLEAN", "STRING"})


def _catalog_suffix(port_type: str | None) -> str:
    """Return the Kedro catalog ``#suffix`` for a given port type.

    - DATAFRAME / None  → ``#parquet``
    - JSON / LIST       → ``#json``
    - Scalars / DICT    → ``""`` (no suffix — MemoryDataset or explicit entry)
    """
    if port_type in _SCALAR_TYPES or port_type == "DICT":
        return ""
    if port_type in ("JSON", "LIST"):
        return "#json"
    return "#parquet"


def _write_if_changed(path: Path, content: str) -> bool:
    """Write content to file only if it differs from existing content.
    
    This prevents unnecessary file modification timestamps that would
    trigger kedro viz --autoreload.
    
    Returns True if file was written, False if unchanged.
    """
    if path.exists():
        try:
            existing = path.read_text(encoding='utf-8')
            if existing == content:
                return False  # No change needed
        except Exception:
            pass  # File exists but couldn't read - write anyway
    
    path.write_text(content, encoding='utf-8')
    return True

def _kedro_viz_metadata(layer: str) -> dict:
    """Build kedro-viz metadata for a tabular catalog entry."""
    return {"kedro-viz": {"layer": layer, "preview_args": {"nrows": 10}}}


class ManagedProjectBuilder:
    """
    Generates a standard Kedro project structure from a ChoregraphSpec.
    This acts as the Single Source of Truth for the Kedro run.
    """

    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.wrapper_dir = self.workspace_path / "pipeline"

    def _resolve_input_path(self, location: str) -> str:
        """Resolve a relative input location to an absolute path.

        Search order:
        1. ``pipeline/inputs/{location}`` — temporal collection symlinks.
        2. Workspace root ``{workspace}/{location}`` — backward-compat for
           pre-existing symlinks at the old location.
        3. ``pipeline/data/inputs/{location}`` — promoted leaf datasets.
        """
        inputs_dir = (self.wrapper_dir / ".." / "inputs" / location).resolve()
        if inputs_dir.exists():
            return inputs_dir.as_posix()
        candidate = (self.wrapper_dir / ".." / location).resolve()
        if not candidate.exists():
            alt = (self.wrapper_dir / "data" / "inputs" / location).resolve()
            if alt.exists():
                candidate = alt
        return candidate.as_posix()

    def build(self, spec: ChoregraphSpec, transform_registry: Dict[str, Any], include_viz_hooks: bool = True):
        """Builds the Kedro project files.

        Args:
            spec: The choregraph specification.
            transform_registry: Transform function registry.
            include_viz_hooks: If True (web flow), include kedro-viz hooks
                in settings.py. If False (Toolkit mode), skip them.
        """
        self._ensure_directories()
        self._generate_pyproject()
        self._generate_settings(include_viz_hooks=include_viz_hooks)
        self._generate_catalog(spec, transform_registry)
        self._generate_pipeline_registry(spec, transform_registry)

    def _ensure_directories(self):
        self.wrapper_dir.mkdir(parents=True, exist_ok=True)
        (self.wrapper_dir / "conf" / "base").mkdir(parents=True, exist_ok=True)
        (self.wrapper_dir / "conf" / "local").mkdir(parents=True, exist_ok=True)
        (self.wrapper_dir / "cache").mkdir(parents=True, exist_ok=True)  # For catalogue_stats.json
        # Silence Kedro's verbose logging (applies to both web and Toolkit paths)
        logging_conf = self.wrapper_dir / "conf" / "logging.yml"
        if not logging_conf.exists():
            logging_conf.write_text(
                "version: 1\n"
                "disable_existing_loggers: False\n"
                "loggers:\n"
                "  kedro:\n"
                "    level: WARNING\n",
                encoding="utf-8",
            )
        if not (self.wrapper_dir / "cache" / "catalogue_stats.json").exists():
            (self.wrapper_dir / "cache" / "catalogue_stats.json").write_text('{"datasets": {}, "last_pipeline_run": ""}', encoding='utf-8')
        (self.wrapper_dir / "src" / "viz_wrapper").mkdir(parents=True, exist_ok=True)
        # Ensure __init__.py exists (only create if missing)
        init_path = self.wrapper_dir / "src" / "viz_wrapper" / "__init__.py"
        if not init_path.exists():
            init_path.touch()
        # Create empty parameters.yml to silence warnings (only if missing)
        params_path = self.wrapper_dir / "conf" / "base" / "parameters.yml"
        if not params_path.exists():
            params_path.touch()

    def _generate_pyproject(self):
        """Generate pyproject.toml with necessary dependencies."""
        content = f"""
[tool.kedro]
package_name = "viz_wrapper"
project_name = "Viz Wrapper"
kedro_init_version = "{kedro.__version__}"

[project]
name = "viz_wrapper"
version = "0.1.0"
description = "Auto-generated wrapper"
requires-python = ">=3.10"
dependencies = [
    "kedro", 
    "kedro-datasets", 
    "pandas", 
    "pyarrow", 
    "choregraph"
]

[tool.setuptools.packages.find]
where = ["src"]
namespaces = false
"""
        _write_if_changed(self.wrapper_dir / "pyproject.toml", content.strip())

    def _generate_settings(self, include_viz_hooks: bool = True):
        """Generate settings.py with conditional kedro-viz hooks.

        Args:
            include_viz_hooks: If True (default, web flow), include kedro-viz
                hooks for dataset stats and pipeline run status. If False
                (Toolkit mode), skip them to avoid slow imports.
        """
        if include_viz_hooks:
            hooks_block = """
try:
    from kedro_viz.integrations.kedro.hooks import DatasetStatsHook
    from kedro_viz.integrations.kedro.run_hooks import PipelineRunStatusHook
    HOOKS = (DatasetStatsHook(), PipelineRunStatusHook())
except ImportError:
    HOOKS = ()
"""
        else:
            hooks_block = """
HOOKS = ()
"""

        content = f"""
from kedro.config import OmegaConfigLoader

CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {{"base_env": "base", "default_run_env": "local"}}
{hooks_block}
# Workaround for pydantic-core bug: pd.NaT passes isinstance(x, datetime)
# but NaT.year returns nan (float), crashing Pydantic's JSON serializer.
# Replace NaT with None in preview output only.
try:
    import pandas as _pd
    from functools import wraps
    _NaTType = type(_pd.NaT)

    def _patch_preview_nat(cls):
        _orig = cls.preview
        @wraps(_orig)
        def preview(self, *args, **kwargs):
            result = _orig(self, *args, **kwargs)
            if isinstance(result, dict) and 'data' in result:
                result['data'] = [
                    [None if isinstance(v, _NaTType) else v for v in row]
                    for row in result['data']
                ]
            return result
        cls.preview = preview

    from kedro_datasets.pandas import ParquetDataset, CSVDataset
    _patch_preview_nat(ParquetDataset)
    _patch_preview_nat(CSVDataset)
except Exception:
    pass
"""
        _write_if_changed(self.wrapper_dir / "src" / "viz_wrapper" / "settings.py", content.strip())

    def _generate_catalog(self, spec: ChoregraphSpec, registry: Dict[str, Any]):
        """Generate conf/base/catalog.yml with PRETTY names and ABSOLUTE PATHS."""
        catalog = {}

        # Determine which output ports are consumed by downstream nodes
        # to assign "processing" vs "output" layers for kedro-viz
        consumed_refs = set()
        for n in spec.nodes:
            for p in n.input_ports:
                if p.source_ref is not None:
                    consumed_refs.add(str(p.source_ref))

        # 1. Inputs: Ask the Spec for the name
        for inp in spec.inputs:
            if not inp.location and inp.format != "MEMORY":
                continue
            clean_name = spec.get_name(inp.id)
            
            # Skip ZRAW files - they are raw data files referenced by MHD headers
            # and should not be loaded as standalone datasets
            if inp.format.upper() == "ZRAW":
                continue
                
            if inp.format.upper() in ("MHD", "DICOM"):
                continue

            if inp.format.upper() == "EDF":
                loc = inp.location
                if not Path(loc).is_absolute():
                    loc = str(self._resolve_input_path(loc))
                catalog[clean_name] = {
                    "type": "choregraph.datasets.edf.EDFDataset",
                    "filepath": loc,
                    "metadata": _kedro_viz_metadata("input"),
                }
                continue

            # Temporal collection: register as PartitionedDataset
            temporal_files = inp.options.get("temporalFiles")
            if temporal_files:
                fmt = inp.format.upper()
                # Map format to underlying Kedro dataset type
                if fmt in ("IMAGE", "PNG", "JPG", "JPEG", "TIFF", "TIF", "BMP", "WEBP", "GIF"):
                    inner_type = "pillow.ImageDataset"
                elif fmt == "JSON":
                    inner_type = "json.JSONDataset"
                elif fmt == "PARQUET":
                    inner_type = "pandas.ParquetDataset"
                else:
                    inner_type = f"pandas.{fmt}Dataset"

                # Location points to the primary file; directory is its parent
                loc = inp.location
                if not Path(loc).is_absolute():
                    loc = str(self._resolve_input_path(loc))
                shared_dir = Path(loc).parent

                # Create a per-input subdirectory with symlinks to only this
                # group's files.  Without this, multiple temporal groups in
                # the same parent dir (e.g. PhysiCell cells + microenvironment)
                # would all be loaded by each PartitionedDataset entry.
                tf_names = set(temporal_files.split("|"))
                group_dir = shared_dir / f"_group_{clean_name}"
                group_dir.mkdir(parents=True, exist_ok=True)
                for tf_name in tf_names:
                    src = shared_dir / tf_name
                    dst = group_dir / tf_name
                    if src.exists() and not dst.exists():
                        dst.symlink_to(src.resolve())
                partition_dir = str(group_dir)

                # Determine filename suffix from the first file
                first_file = temporal_files.split("|")[0]
                suffix = Path(first_file).suffix

                # Build inner dataset config with load_args (stripped of temporal keys)
                clean_opts = {k: v for k, v in inp.options.items()
                              if k not in ("temporalFiles", "collectionTimeMode", "collectionTimeDelta")}
                inner_config = {"type": inner_type}
                # loc IS the primary file path — use directly for CSV sniffing
                first_file_path = loc
                load_args = prepare_load_args(inp.format, first_file_path, clean_opts)
                if load_args:
                    inner_config["load_args"] = load_args

                catalog[clean_name] = {
                    "type": "partitions.PartitionedDataset",
                    "path": partition_dir,
                    "dataset": inner_config,
                    "filename_suffix": suffix,
                    "metadata": _kedro_viz_metadata("input"),
                }
                continue

            if inp.format.upper() in ("IMAGE", "PNG", "JPG", "JPEG", "TIFF", "TIF", "BMP", "WEBP", "GIF"):
                loc = inp.location
                if not Path(loc).is_absolute():
                    loc = self._resolve_input_path(loc)
                catalog[clean_name] = {
                    "type": "pillow.ImageDataset",
                    "filepath": loc,
                }
                continue

            if inp.format == "MEMORY":
                catalog[clean_name] = {
                    "type": "MemoryDataset",
                    "metadata": _kedro_viz_metadata("input"),
                }
            elif inp.format.upper() in ("XLSX", "XLS", "XLSM", "ODS"):
                xlsx_loc = Path(inp.location).with_suffix(".xlsx").as_posix()
                entry = {
                    "type": "pandas.ExcelDataset",
                    "filepath": xlsx_loc
                }
                # Handle relative paths
                if not Path(xlsx_loc).is_absolute():
                    entry["filepath"] = self._resolve_input_path(xlsx_loc)
                # Load ALL sheets (sheet_name=None returns dict) without header
                entry["load_args"] = {"sheet_name": None, "header": None}
                entry["metadata"] = _kedro_viz_metadata("input")
                catalog[clean_name] = entry
                
                # Check if a tidy_excel_data node is connected to this Excel input
                # If so, create a PartitionedDataset for multi-table output
                for n in spec.nodes:
                    if n.type == "tidy_excel_data":
                        # Check if this node's data input is connected to our Excel input
                        is_connected = any(
                            p.source_ref is not None and 
                            p.name == "data" and 
                            str(p.source_ref) == str(inp.id)
                            for p in n.input_ports
                        )
                        if is_connected:
                            base_name = spec.get_name(n.output_ports[0].id) if n.output_ports else f"excel_{n.id}"
                            data_dir_abs = (self.wrapper_dir / "data").resolve().as_posix()
                            partitioned_path = f"{data_dir_abs}/{base_name}_partitioned"
                            out_id = str(n.output_ports[0].id) if n.output_ports else None
                            excel_layer = "processing" if out_id and out_id in consumed_refs else "output"
                            catalog[base_name] = {
                                "type": "partitions.PartitionedDataset",
                                "path": partitioned_path,
                                "dataset": {
                                    "type": "pandas.ParquetDataset",
                                    "save_args": {"engine": "pyarrow"}
                                },
                                "filename_suffix": ".parquet",
                                "metadata": _kedro_viz_metadata(excel_layer),
                            }
                            break  # Only one tidy_excel_data node per input
            elif inp.format.upper() == "JSON":
                catalog[clean_name] = {
                    "type": "json.JSONDataset",
                    "filepath": inp.location
                }
                # Handle relative paths
                if not Path(inp.location).is_absolute():
                     catalog[clean_name]["filepath"] = self._resolve_input_path(inp.location)
                # Note: json.JSONDataset does not support load_args
            else:
                entry = {
                    "type": "pandas.ParquetDataset" if inp.format.upper() == "PARQUET" else f"pandas.{inp.format}Dataset",
                    "filepath": inp.location
                }
                
                # Handle relative paths for inputs too (optional but safer)
                if not Path(inp.location).is_absolute():
                     entry["filepath"] = self._resolve_input_path(inp.location)
                
                load_args = prepare_load_args(inp.format, inp.location, inp.options)
                if load_args: entry["load_args"] = load_args
                entry["metadata"] = _kedro_viz_metadata("input")

                catalog[clean_name] = entry

        # We calculate the absolute path to the wrapper's data directory
        data_dir_abs = (self.wrapper_dir / "data").resolve().as_posix()

        # Explicit entries for PartitionedDataset outputs (DICT type).
        # Scalars (FLOAT, INTEGER, …) have no catalog entry — Kedro keeps
        # them in memory by default, which is correct.
        for node in spec.nodes:
            for outport in node.output_ports:
                if outport.type == "DICT" or node.type == "tidy_excel_data":
                    output_name = spec.get_name(outport.id)
                    if output_name in catalog:
                        continue
                    layer = "processing" if str(outport.id) in consumed_refs else "output"
                    partitioned_path = f"{data_dir_abs}/{output_name}_partitioned"
                    catalog[output_name] = {
                        "type": "partitions.PartitionedDataset",
                        "path": partitioned_path,
                        "dataset": {
                            "type": "pandas.ParquetDataset",
                            "save_args": {"engine": "pyarrow"}
                        },
                        "filename_suffix": ".parquet",
                        "metadata": _kedro_viz_metadata(layer),
                    }

        # Typed factory patterns for pipeline outputs.
        # Pipeline registry appends #parquet / #json to output names so
        # each dataset resolves to the correct type. Datasets without a
        # #suffix (scalars, intermediates) stay as Kedro's default
        # MemoryDataset — no catch-all needed.
        catalog["{name}#parquet"] = {
            "type": "pandas.ParquetDataset",
            "filepath": f"{data_dir_abs}/{{name}}.parquet",
            "save_args": {"engine": "pyarrow"},
            "versioned": False,
            "metadata": _kedro_viz_metadata("output"),
        }
        catalog["{name}#json"] = {
            "type": "json.JSONDataset",
            "filepath": f"{data_dir_abs}/{{name}}.json",
        }

        catalog_content = yaml.dump(catalog, default_flow_style=False, sort_keys=False)
        _write_if_changed(self.wrapper_dir / "conf" / "base" / "catalog.yml", catalog_content)

    def _generate_pipeline_registry(self, spec: ChoregraphSpec, registry: Dict[str, Any]):
        """
        Generates a static pipeline_registry.py with the Kedro pipeline already defined.
        
        This approach:
        - Builds the pipeline at write time (no XML parsing at runtime)
        - Imports functions from choregraph.library
        - Uses functools.partial for parameters
        - No separate JSON files (pipeline_definition.json, label_mapping.json)
        """
        from .xsd_catalogue_utils import load_function_catalogue_from_xsd
        
        # Load function catalogue for type conversion
        try:
            catalogue = load_function_catalogue_from_xsd()
        except Exception:
            catalogue = {"functions": {}}
        
        # Collect which functions are actually used
        used_functions = set()

        # Generate node definitions
        node_defs = []
        used_node_names = set()
        label_mapping = {}  # kedro_name -> original_label

        # Build a map of entity ID → catalog dataset name (with #suffix).
        # Inputs keep their clean name (explicit catalog entries).
        # Output ports get a #suffix matching their factory pattern.
        _catalog_names = {}
        for inp in spec.inputs:
            _catalog_names[str(inp.id)] = spec.get_name(inp.id)
        for n in spec.nodes:
            for op in n.output_ports:
                clean = spec.get_name(op.id)
                # Nodes that return dicts (tidy_excel_data, etc.) must use clean
                # names to match their explicit PartitionedDataset catalog entry.
                if op.type == "DICT" or n.type == "tidy_excel_data":
                    suffix = ""
                else:
                    suffix = _catalog_suffix(op.type)
                _catalog_names[str(op.id)] = f"{clean}{suffix}" if suffix else clean

        # Collect input label mappings first
        for inp in spec.inputs:
            input_kedro_name = spec.get_name(inp.id)
            if inp.label and input_kedro_name != inp.label:
                label_mapping[input_kedro_name] = inp.label

        for n in spec.nodes:
            if n.type not in registry:
                continue
            
            used_functions.add(n.type)
            
            # 1. Parse kwargs from input ports with 'value'
            kwargs = {}
            function_spec = catalogue.get("functions", {}).get(n.type, {})
            input_ports_spec = function_spec.get("input_ports", {})

            for port in n.input_ports:
                if port.value is None:
                    continue
                catalogue_port_spec = input_ports_spec.get(port.name, {})
                port_type = _resolve_port_type(port, catalogue_port_spec)
                kwargs[port.name] = parse_port_value(port.value, port_type)
            
            # 2. Resolve inputs (input ports with 'source_ref')
            # When multiple ports share the same name (multi-input like join/union),
            # use dataset names as kwargs keys so functions can identify each source.
            node_inputs = {}
            port_counts = {}
            for p in n.input_ports:
                if p.source_ref is not None:
                    dataset_name = _catalog_names.get(str(p.source_ref), spec.get_name(p.source_ref))
                    port_name = p.name
                    if port_name in node_inputs:
                        if port_counts.get(port_name) == 0:
                            first_dataset = node_inputs.pop(port_name)
                            node_inputs[first_dataset] = first_dataset
                        port_counts[port_name] = port_counts.get(port_name, 0) + 1
                        node_inputs[dataset_name] = dataset_name
                    else:
                        node_inputs[port_name] = dataset_name
                        port_counts[port_name] = 0
            
            # 3. Resolve outputs from output_ports (with #suffix)
            if n.output_ports:
                if len(n.output_ports) == 1:
                    output_kedro_name = _catalog_names.get(str(n.output_ports[0].id), spec.get_name(n.output_ports[0].id))
                    final_output = repr(output_kedro_name)
                    # Track output label mapping (use clean name as key)
                    if n.output_ports[0].label and output_kedro_name != n.output_ports[0].label:
                        label_mapping[output_kedro_name] = n.output_ports[0].label
                else:
                    output_dict = {op.name: _catalog_names.get(str(op.id), spec.get_name(op.id)) for op in n.output_ports}
                    final_output = repr(output_dict)
                    # Track all output label mappings
                    for op in n.output_ports:
                        output_kedro_name = _catalog_names.get(str(op.id), spec.get_name(op.id))
                        if op.label and output_kedro_name != op.label:
                            label_mapping[output_kedro_name] = op.label
            else:
                label_text = n.label if n.label else f"node_{n.id}"
                clean_label = _sanitize_name(label_text)
                final_output = repr(f"{clean_label}_out")
            
            # 4. Create unique node name
            label_text = n.label if n.label else n.type
            sanitized_label = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in label_text)
            pretty_name = sanitized_label
            if pretty_name in used_node_names:
                pretty_name = f"{pretty_name}_{n.id}"
            base_pretty_name = pretty_name
            counter = 1
            while pretty_name in used_node_names:
                pretty_name = f"{base_pretty_name}_{counter}"
                counter += 1
            used_node_names.add(pretty_name)
            
            # Track label mapping: kedro_name -> original_label
            if n.label and pretty_name != n.label:
                label_mapping[pretty_name] = n.label
            
            # 5. Build the node definition string
            if kwargs:
                func_str = f"update_wrapper(partial({n.type}, **{repr(kwargs)}), {n.type})"
            else:
                func_str = n.type
            
            node_def = f"""    node(
        func={func_str},
        inputs={repr(node_inputs)},
        outputs={final_output},
        name={repr(pretty_name)}
    )"""
            node_defs.append(node_def)
        
        # Generate the imports
        func_imports = ", ".join(sorted(used_functions)) if used_functions else ""
        
        # Build the pipeline_registry.py content
        lines = [
            '"""Auto-generated Kedro pipeline registry. DO NOT EDIT MANUALLY."""',
            'from functools import partial, update_wrapper',
            'from kedro.pipeline import Pipeline, node',
        ]
        
        if func_imports:
            lines.append(f'from choregraph.library import {func_imports}')
        
        lines.append('')
        lines.append('')
        lines.append('def register_pipelines() -> dict[str, Pipeline]:')
        lines.append('    """Register the pipelines for this Kedro project."""')
        
        if node_defs:
            lines.append('    pipeline = Pipeline([')
            lines.append(',\n'.join(node_defs))
            lines.append('    ])')
        else:
            lines.append('    pipeline = Pipeline([])')
        
        lines.append('')
        lines.append('    return {')
        lines.append('        "__default__": pipeline,')
        lines.append('        "choregraph": pipeline')
        lines.append('    }')
        
        content = '\n'.join(lines)
        
        _write_if_changed(self.wrapper_dir / "src" / "viz_wrapper" / "pipeline_registry.py", content)
        
        # Write label_mapping.json for display purposes
        # This maps kedro names back to original labels from choregraph.xml
        import json
        label_mapping_content = json.dumps(label_mapping, indent=2, ensure_ascii=False)
        _write_if_changed(self.wrapper_dir / "label_mapping.json", label_mapping_content)
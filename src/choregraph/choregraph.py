# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-FileContributor: Guillaume Franque
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Choregraph facade -- the main entry point for pipeline lifecycle management.

This module provides the :class:`Choregraph` class which orchestrates XML spec
parsing, Kedro project generation, pipeline execution, data caching, and DIVE
VisuSpec export. It delegates to the parser, builder, wrapper, and connectors
modules internally.
"""
from __future__ import annotations
import os
import contextlib
import logging
import hashlib
import sys
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, List, Tuple
import pandas as pd
from lxml import etree

# Silence kedro's verbose startup logging before any kedro import.
# Kedro instantiates _ProjectLogging at import time of kedro.framework.project,
# which reconfigures logging via dictConfig. We create a minimal logging config
# file and point KEDRO_LOGGING_CONFIG to it to prevent the rich_logging fallback.
if "KEDRO_LOGGING_CONFIG" not in os.environ:
    import tempfile as _tf
    _log_cfg = Path(_tf.gettempdir()) / "kedro_silent_logging.yml"
    if not _log_cfg.exists():
        _log_cfg.write_text(
            "version: 1\n"
            "disable_existing_loggers: False\n"
            "loggers:\n"
            "  kedro:\n"
            "    level: WARNING\n",
            encoding="utf-8",
        )
    os.environ["KEDRO_LOGGING_CONFIG"] = str(_log_cfg)

from kedro.io import DataCatalog
from kedro.framework.session import KedroSession
from kedro.runner import SequentialRunner
from kedro.framework.startup import bootstrap_project

from .parser import ChoregraphSpecParser, InputSpec, NodeSpec, OutputSpec, ChoregraphSpec, InputPortSpec, OutputPortSpec
from .wrapper import ManagedProjectBuilder
from .library import TRANSFORM_REGISTRY
from .metadata import Metadata
from .security import safe_path
# from .xsd_catalogue_utils import generate_catalogue_text
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def pushd(new_dir):
    """
    Context manager to temporarily change the working directory.
    Restores the original directory automatically when exiting the 'with' block.
    """
    previous_dir = os.getcwd()
    os.chdir(safe_path(new_dir))
    try:
        yield
    finally:
        os.chdir(previous_dir)

from .connectors import CacheProxy as _CacheProxy  # backward compat for test imports


class Choregraph:
    """Main facade for Choregraph pipeline lifecycle management.

    Orchestrates XML spec parsing, Kedro project generation, pipeline execution,
    data caching, and DIVE VisuSpec export. Supports both programmatic pipeline
    construction (via :meth:`add_input` / :meth:`add_node`) and loading from XML.

    Can be used as a context manager::

        with Choregraph(xml_spec="pipeline.xml") as cg:
            cg.run()
            df = cg.get_dataset("my_output")
    """

    def __init__(self, xml_spec: Union[str, Path] = None, external_inputs: Dict[str, Any] = None, workspace_path: Union[str, Path] = None, kedro_viz: bool = True):
        self.workspace_path = Path(workspace_path) if workspace_path else None
        self.external_inputs = external_inputs or {}
        self.kedro_viz = kedro_viz
        self._listeners = []

        # RAM Caches (Read-Through)
        self._data_cache = {}      
        self._spec_hash = None
        self._last_run_hash = None
        
        # ID -> Name mapping (internal, independent of visuspec)
        self._id_name_map = {}
        
        # Catalog Cache (Performance)
        self._catalog_instance = None
        
        # Auto-detect existing spec
        if xml_spec is None and self.workspace_path:
            default_xml = self.workspace_path / "choregraph.xml"
            if default_xml.exists():
                logger.info(f"Auto-detected existing spec at {default_xml}")
                xml_spec = default_xml

        if xml_spec:
            self.spec = ChoregraphSpecParser.parse(xml_spec)
        else:
            self.spec = ChoregraphSpec()

        self.transform_registry = TRANSFORM_REGISTRY.copy()

        # We assume the wrapper exists or will be built on run/add
        if self.workspace_path and not (self.workspace_path / "pipeline").exists():
          self._ensure_wrapper()

    @property
    def _datasets_metadata(self) -> Optional[Metadata]:
        """Lazy-loaded Metadata dependent on workspace_path."""
        if self.workspace_path:
            return Metadata(self.workspace_path)
        return None

    @property
    def _data_dir(self) -> Path:
        """Root data directory inside the generated Kedro wrapper."""
        return self.workspace_path / "pipeline" / "data"

    @property
    def _inputs_dir(self) -> Path:
        """Directory where promoted / external input files are stored."""
        return self._data_dir / "inputs"

    def _get_project_hash(self) -> str:
        """Hash the choregraph.xml content + input file mtimes to detect changes."""
        s = ""

        # Hash the actual XML file content if it exists
        if self.workspace_path:
            xml_path = self.workspace_path / "choregraph.xml"
            if xml_path.exists():
                try:
                    s += xml_path.read_text(encoding='utf-8')
                except Exception:
                    s += str(self.spec)  # Fallback to object repr
            else:
                s += str(self.spec)
        else:
            s += str(self.spec)



        # Include inputs mtime
        for inp in self.spec.inputs:
            if inp.location:
                # Handle relative paths if workspace_path is set
                p = Path(inp.location)
                if not p.is_absolute() and self.workspace_path:
                    p = self.workspace_path / p

                if p.exists() and p.is_file():
                    s += str(p.stat().st_mtime_ns)

        hash = hashlib.sha256(s.encode()).hexdigest()
        return hash

    def _ensure_wrapper(self):
        """Regenerate the Kedro project files (catalog.yml, pipelines, etc)."""
        if not self.workspace_path: return
        builder = ManagedProjectBuilder(self.workspace_path)
        builder.build(self.spec, self.transform_registry, include_viz_hooks=self.kedro_viz)

    def _get_catalog(self) -> DataCatalog:
        """Load the catalog from the generated ``catalog.yml``.

        Reads the YAML directly instead of creating a KedroSession,
        avoiding the ~3.5s overhead of ``bootstrap_project()`` +
        ``KedroSession.create()``. The catalog config is simple
        generated YAML with no OmegaConf templating.
        """
        if self._catalog_instance:
            return self._catalog_instance

        import yaml
        catalog_path = safe_path(self.workspace_path / "pipeline" / "conf" / "base" / "catalog.yml")
        with open(catalog_path, encoding="utf-8") as f:
            conf_catalog = yaml.safe_load(f) or {}

        self._catalog_instance = DataCatalog.from_config(conf_catalog)
        return self._catalog_instance

    def get_xsd(self) -> str:
        """Get the XSD content as a string (bundled with the package)."""
        xsd_path = Path(__file__).parent / "TransformGraph.xsd"
        return xsd_path.read_text(encoding="utf-8")
    
    def run(self, lazy: bool = True) -> Tuple[bool, str]:
        """Execute the pipeline using a Kedro session.

        Generates Kedro project files, dumps external inputs to disk, and runs
        the pipeline via ``SequentialRunner``. Supports lazy evaluation — if the
        spec and input files haven't changed, cached results are returned.

        Args:
            lazy: If True, skip execution when the spec hash is unchanged.

        Returns:
            A tuple ``(success, error_message)`` where *success* is ``True``
            when the pipeline executed (or was skipped) without error, and
            *error_message* contains the failure description otherwise.

        Raises:
            ValueError: If ``workspace_path`` is not set.
        """
        if not self.workspace_path:
            raise ValueError("Workspace path required for execution.")

        current_hash = self._get_project_hash()
        # print(f"Current project hash: {current_hash}")
        # print(f"Last run hash: {self._last_run_hash}")
        # 1. Lazy Check
        if lazy and self._last_run_hash == current_hash:
            logger.info("Pipeline inputs/spec unchanged. Skipping run.")
            # self._emit("graph_update", {})
            self._emit("status", {"status": "completed"})
            return (True, "")

        # # 1b. Hash changed - purge cached parquet files to prevent stale data
        # data_dir = self.workspace_path / "pipeline" / "data"
        # if data_dir.exists():
        #     for parquet_file in data_dir.glob("*.parquet"):
        #         try:
        #             parquet_file.unlink()
        #             logger.info(f"Purged stale parquet: {parquet_file.name}")
        #         except Exception as e:
        #             logger.warning(f"Failed to delete {parquet_file}: {e}")

        self._emit("status", {"status": "running"})

        # 2. Sync Wrapper (Source of Truth)
        self._ensure_wrapper()

        # 3. Dump file-backed inputs to disk; collect MEMORY inputs for hook injection
        inputs_dir = self._inputs_dir
        inputs_dir.mkdir(parents=True, exist_ok=True)

        memory_datasets = {}  # {catalog_name: DataFrame} — injected via hook
        for input_id, data in self.external_inputs.items():
            inp = next((i for i in self.spec.inputs if str(i.id) == str(input_id)), None)
            is_memory = inp and getattr(inp, "format", "").upper() == "MEMORY"

            if is_memory and isinstance(data, pd.DataFrame):
                catalog_name = self.spec.get_name(input_id)
                memory_datasets[catalog_name] = data
            elif isinstance(data, pd.DataFrame):
                file_path = inputs_dir / f"{input_id}.parquet"
                data.to_parquet(file_path)
            elif isinstance(data, (dict, list)):
                file_path = safe_path(inputs_dir / f"{input_id}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
            else:
                logger.warning(f"Unsupported input data type for {input_id}: {type(data)}")

        # 4. HOT RELOAD FIX
        self._purge_wrapper_modules()

        # 5. Run Session
        if not self.spec.nodes:
            logger.info("Pipeline is empty (no nodes). Skipping Kedro run.")
            self._last_run_hash = current_hash
            # Invalidate catalog because wrapper might have changed inputs
            self._catalog_instance = None

            self._data_cache.clear()

            # No Kedro session means MetadataStatsHook won't fire.
            # Populate stats for each input so catalogue_stats.json is filled.
            # Also populate _data_cache for MEMORY inputs so get_dataset() works.
            #
            # Use cached stats when available: history switches reset the run
            # hash (spec changed) but the underlying input data is the same.
            # Only recompute for inputs missing from the cache or MEMORY inputs
            # (which can change between runs).
            metadata = self._datasets_metadata
            if metadata is not None:
                cached = metadata.read_from_cache()
                catalog = self._get_catalog()
                for inp in self.spec.inputs:
                    name = self.spec.get_name(inp.id)
                    fmt = getattr(inp, "format", "").upper()

                    # MHD/ZRAW: not in catalog — parse header directly
                    if fmt in ("MHD",):
                        if name not in cached or not getattr(cached[name], "fields", None):
                            loc = inp.location
                            if loc and not Path(loc).is_absolute() and self.workspace_path:
                                loc = str((Path(self.workspace_path) / loc).resolve())
                            fields = metadata._describe_mhd(loc)
                            metadata.store_stats(name, fields, 0, dataset_id=str(inp.id))
                        continue
                    if fmt == "ZRAW":
                        continue  # companion file, skip entirely

                    # MEMORY inputs: always recompute (data can change between runs)
                    if fmt == "MEMORY" and str(inp.id) in self.external_inputs:
                        data = self.external_inputs[str(inp.id)]
                        self._data_cache[name] = data
                        metadata.update_stats(name, data, dataset_id=str(inp.id))
                        continue

                    # Skip inputs that already have cached stats
                    has_cached = name in cached and getattr(cached[name], "fields", None)
                    if has_cached:
                        # Ensure partition field is present for temporal inputs
                        temporal_files = inp.options.get("temporalFiles")
                        if temporal_files:
                            n = len(temporal_files.split("|"))
                            label = "time" if inp.options.get("collectionTimeMode") else "partition"
                            metadata.add_partition_field(name, n, partition_label=label)
                        continue

                    try:
                        temporal_files = inp.options.get("temporalFiles")
                        if temporal_files:
                            n = len(temporal_files.split("|"))
                            # Load the PartitionedDataset ONCE and iterate
                            # partitions directly.  The old code called
                            # get_dataset(id, time=t) per timestep which
                            # triggered N redundant catalog.load() calls —
                            # each rescanning the directory and rebuilding
                            # the full lazy dict.
                            partitioned = catalog.load(name)
                            if isinstance(partitioned, dict) and partitioned and all(callable(v) for v in partitioned.values()):
                                dfs = []
                                for key in sorted(partitioned.keys()):
                                    try:
                                        df_t = partitioned[key]()
                                        if isinstance(df_t, pd.DataFrame):
                                            dfs.append(df_t)
                                    except Exception:
                                        pass
                                if dfs:
                                    full_df = pd.concat(dfs, ignore_index=True)
                                    metadata.update_stats(name, full_df, dataset_id=str(inp.id))
                            else:
                                metadata.update_stats(name, partitioned, dataset_id=str(inp.id))
                            label = "time" if inp.options.get("collectionTimeMode") else "partition"
                            metadata.add_partition_field(name, n, partition_label=label)
                        else:
                            data = self.get_dataset(str(inp.id))
                            metadata.update_stats(name, data, dataset_id=str(inp.id))
                    except Exception as e:
                        logger.warning(f"Could not extract stats for input '{name}': {e}")

            self._emit("graph_update", {})
            self._emit("status", {"status": "completed"})
            return (True, "")

        wrapper_path = self.workspace_path / "pipeline"

        
        with pushd(wrapper_path):
            try:
                # Bootstrap and settings.py (with hooks) are loaded here
                bootstrap_project(wrapper_path)
                
                with KedroSession.create(project_path=wrapper_path, env="local") as session:
                    # Inject in-memory datasets before pipeline runs
                    if memory_datasets:
                        from .hooks import DataInjectionHook
                        session._hook_manager.register(DataInjectionHook(memory_datasets))

                    # Clean dtypes on load so union/concat never mixes types
                    from .hooks import DtypeInferenceHook
                    session._hook_manager.register(DtypeInferenceHook())

                    # Register MetadataStatsHook to capture stats during execution
                    if self._datasets_metadata is not None:
                        from .hooks import MetadataStatsHook
                        stats_hook = MetadataStatsHook(self._datasets_metadata, self.spec)
                        session._hook_manager.register(stats_hook)

                    # Register ExecutionStatusHook for UI updates (if listeners registered)
                    # if self._listeners:
                    #     from .hooks import ExecutionStatusHook
                    #     excluded_nodes = set()
                    #     # for n in self.spec.nodes:
                        #     if not n.visibility:
                        #         label_text = n.label if n.label else n.type
                        #         sanitized_label = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in label_text)
                        #         excluded_nodes.add(sanitized_label)

                        # status_hook = ExecutionStatusHook(
                        #     on_update=lambda status: self._emit("node_status", status),
                        #     excluded_nodes=excluded_nodes
                        # )
                        # session._hook_manager.register(status_hook)
                    runner=SequentialRunner(is_async=True)
                    session.run(runner=runner)

            except Exception as e:
                logger.error(f"Kedro Run failed: {e}")
                self._emit("status", {"error": str(e)})
                return (False, str(e))


        # 8. Update State
        self._last_run_hash = current_hash

        self._emit("graph_update", {})
        self._emit("status", {"status": "completed"})
        
        # Clear data cache to prevent stale data (metadata is persisted in JSON).
        # Keep _catalog_instance alive — catalog.yml hasn't changed since
        # _ensure_wrapper() was called at the start of run().
        self._data_cache.clear()

        # Re-populate cache for MEMORY inputs — Kedro's MemoryDataset is
        # per-session, so data is lost when the session closes.
        for inp in self.spec.inputs:
            if getattr(inp, "format", "").upper() == "MEMORY":
                input_id = str(inp.id)
                if input_id in self.external_inputs:
                    name = self.spec.get_name(input_id)
                    self._data_cache[name] = self.external_inputs[input_id]

        return (True, "")

    def _rebuild_id_map(self):
        """Rebuild the internal ID -> name mapping from spec + dynamic files."""
        self._id_name_map.clear()
        
        # 1. From spec: inputs
        for inp in self.spec.inputs:
            self._id_name_map[str(inp.id)] = self.spec.get_name(inp.id)
        
        # 2. From spec: output ports
        for n in self.spec.nodes:
            for op in n.output_ports:
                self._id_name_map[str(op.id)] = self.spec.get_name(op.id)
        
        # 3. Dynamic datasets (parquet files in inputs/ not in spec)
        if self.workspace_path:
            if self._inputs_dir.exists():
                next_id = max((int(k) for k in self._id_name_map if k.isdigit()), default=0) + 1
                for f in self._inputs_dir.glob("*.parquet"):
                    name = f.stem
                    if name not in self._id_name_map.values():
                        self._id_name_map[str(next_id)] = name
                        next_id += 1

    # Factory suffixes used by _generate_catalog / _generate_pipeline_registry.
    _CATALOG_SUFFIXES = ("#parquet", "#json")

    def get_dataset(self, data_id: str, time: int | None = None) -> Any:
        """Load a dataset by ID.

        For ``PartitionedDataset`` entries (temporal collections, etc.):
        - ``time=None``: loads the first partition (representative).
        - ``time=N``: loads the Nth partition.

        Returns whatever the underlying dataset produces (DataFrame, Image, dict, etc.).
        """
        # 1. Rebuild mapping if empty (lazy init)
        if not self._id_name_map:
            self._rebuild_id_map()

        # 2. Resolve ID -> name
        real_name = self.spec.get_name(data_id)
        if real_name == data_id and data_id in self._id_name_map:
            real_name = self._id_name_map[data_id]

        # 3. Check cache
        cache_key = f"{real_name}:t{time}" if time is not None else real_name
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        # 4. Load via DataCatalog
        logger.info(f"Loading {real_name} (ID: {data_id}, time={time}) via catalog...")
        catalog = self._get_catalog()

        names_to_try = [real_name, f"params:{real_name}"]
        for sfx in self._CATALOG_SUFFIXES:
            names_to_try.append(f"{real_name}{sfx}")

        last_err = None
        for name in names_to_try:
            try:
                data = catalog.load(name)

                # Temporal PartitionedDataset: Dict[str, Callable] — load specific partition
                if time is not None and isinstance(data, dict) and data and all(callable(v) for v in data.values()):
                    sorted_keys = sorted(data.keys())
                    idx = max(0, min(time, len(sorted_keys) - 1))
                    result = data[sorted_keys[idx]]()
                    self._data_cache[cache_key] = result
                    return result

                # PartitionedDataset without time param: load first partition
                # (only for temporal inputs — other PartitionedDatasets like Excel
                # multi-sheet return the full dict for downstream pipeline handling)
                if isinstance(data, dict) and data and all(callable(v) for v in data.values()):
                    # Check if this input has temporalFiles (temporal collection)
                    inp = next((i for i in self.spec.inputs if self.spec.get_name(i.id) == real_name), None)
                    if inp and inp.options.get("temporalFiles"):
                        sorted_keys = sorted(data.keys())
                        result = data[sorted_keys[0]]()
                        self._data_cache[cache_key] = result
                        return result

                self._data_cache[cache_key] = data
                return data
            except Exception as e:
                last_err = e
                continue

        logger.error(f"Failed to load dataset {real_name}: {last_err}")
        raise last_err

    def list_data(self) -> List[str]:
        """List all available datasets including dynamically generated multi-table outputs."""
        cat = self._get_catalog()
        base_list = []
        if hasattr(cat, "list"):
            base_list = cat.list()
        else:
            base_list = list(cat)

        # Also include promoted/external inputs and partitioned outputs
        if self._inputs_dir.exists():
            for parquet_file in self._inputs_dir.glob("*.parquet"):
                dataset_name = parquet_file.stem
                if dataset_name not in base_list:
                    base_list.append(dataset_name)

        # Also check partitioned folders that haven't been processed yet
        if self._data_dir.exists():
            for partitioned_dir in self._data_dir.glob("*_partitioned"):
                if partitioned_dir.is_dir():
                    for parquet_file in partitioned_dir.glob("*.parquet"):
                        dataset_name = parquet_file.stem
                        if dataset_name not in base_list:
                            base_list.append(dataset_name)
        
        return base_list

    def get_id_for_name(self, name: str) -> Optional[str]:
        """Reverse lookup: get ID for a dataset name."""
        if not self._id_name_map:
            self._rebuild_id_map()
        return next((k for k, v in self._id_name_map.items() if v == name), None)




    def get_datasets_metadata(self) -> List[Dict[str, Any]]:
        """Get full datasets metadata from catalogue_stats.json.

        Delegates to :meth:`MetadataResult.to_api_format`.
        """
        if self._datasets_metadata is None:
            return []
        return self._datasets_metadata.read_from_cache().to_api_format()

    def update_from_spec(self, xml_spec: Union[str, Path]):
        """Replace the current pipeline specification by parsing new XML.

        Args:
            xml_spec: Path to an XML file or an XML string.
        """
        self.spec = ChoregraphSpecParser.parse(xml_spec)
        self._last_run_hash = None
        self._emit("graph_update", None)


    def export_to_xml(self, save_to_path: Union[str, Path]):
        """Serialize the current pipeline specification to an XML file.

        Args:
            save_to_path: Destination file path for the XML output.
        """
        from .versions import TRANSFORMGRAPH_SCHEMA_VERSION

        root = etree.Element("choregraph")
        root.set("schemaVersion", TRANSFORMGRAPH_SCHEMA_VERSION)
        inputs_elem = etree.SubElement(root, "inputs")
        pipeline_elem = etree.SubElement(root, "pipeline")

        for inp in self.spec.inputs:
            input_attrs = {
                "id": str(inp.id),
                "label": str(inp.label) if inp.label is not None else "",
                "location": str(inp.location) if inp.location is not None else "",
                "format": str(inp.format) if inp.format is not None else "",
                "visibility": "true" if inp.visibility else "false"
            }
            if inp.url:
                input_attrs["url"] = str(inp.url)
            for k, v in inp.options.items():
                input_attrs[str(k)] = str(v)
            etree.SubElement(inputs_elem, "input", **input_attrs)


        for node in self.spec.nodes:
            node_attrs = {
                "id": str(node.id),
                "label": str(node.label) if node.label is not None else "",
                "type": str(node.type) if node.type is not None else ""
            }

            node_elem = etree.SubElement(pipeline_elem, "node", **node_attrs)

            # Input ports (flattened - directly under node)
            for port in node.input_ports:
                port_attrs = {"name": str(port.name) if port.name is not None else ""}
                if port.source_ref is not None:
                    port_attrs["sourceRef"] = str(port.source_ref)
                if port.value is not None:
                    port_attrs["value"] = str(port.value)
                if port.type:
                    port_attrs["type"] = str(port.type)
                etree.SubElement(node_elem, "inputPort", **port_attrs)

            # Output ports (flattened - directly under node)
            for port in node.output_ports:
                port_attrs = {
                    "id": str(port.id),
                    "name": str(port.name) if port.name is not None else "",
                    "visibility": "true" if port.visibility else "false"
                }
                if port.type:
                    port_attrs["type"] = str(port.type)
                if port.label:
                    port_attrs["label"] = str(port.label)
                etree.SubElement(node_elem, "outputPort", **port_attrs)


        tree = etree.ElementTree(root)
        tree.write(str(save_to_path), pretty_print=True, xml_declaration=True, encoding="utf-8")

    def add_input(self, id: str, location: str = "", format: str = "CSV", label: str = None, visibility: bool = False, url: str = None, data=None,
                  **options):
        """Add an input data source.

        Args:
            id: Unique input ID (string).
            location: File path or URL. Not required for in-memory data.
            format: Data format (CSV, JSON, MEMORY, etc.).
                Set automatically to ``"MEMORY"`` when ``data`` is provided.
            label: Human-readable label (auto-generated if None).
            visibility: Whether input is visible in visualization.
            url: Origin URL for URL-based data sources.
            data: Optional in-memory data (pandas DataFrame, dict, or list).
                When provided, the input is stored in ``external_inputs``
                and ``format`` is set to ``"MEMORY"``. No disk file is needed —
                Kedro reads from ``pipeline/data/inputs/{id}.parquet``.
            **options: Additional format-specific options.
        """
        id_str = str(id)

        # Enforce uniqueness across entire namespace (check node IDs and output port IDs)
        for n in self.spec.nodes:
            if str(n.id) == id_str:
                raise ValueError(f"ID {id} is already used by a Node. IDs must be unique across all entities.")
            for op in n.output_ports:
                if str(op.id) == id_str:
                    raise ValueError(f"ID {id} is already used by an output port. IDs must be unique across all entities.")

        # In-memory data: store in external_inputs, set format to MEMORY
        if data is not None:
            self.external_inputs[id_str] = data
            format = "MEMORY"

        # Auto-generate label if missing
        if label is None:
            label = Path(location).stem if location else f"Input {id}"

        for inp in self.spec.inputs:
            if str(inp.id) == id_str:
                self.spec.inputs.remove(inp)
                break

        self.spec.inputs.append(InputSpec(
            id=id_str, label=label, location=location, format=format,
            visibility=visibility, url=url, options=options,
        ))
        if visibility:
            if not any(o.id == id_str for o in self.spec.outputs):
                self.spec.outputs.append(OutputSpec(id=id_str))

        self._last_run_hash = None
        if self.workspace_path:
            self._ensure_wrapper()
        # Invalidate catalog cache so new entries are picked up
        self._catalog_instance = None
        self._emit("graph", {})

    def add_node(self, id: str, type: str, input_ports: List[InputPortSpec], output_ports: List[OutputPortSpec] = None, label: str = None):
        """Add a node to the pipeline.

        Args:
            id: Unique node ID
            type: Transform function name
            input_ports: List of input port specifications
            output_ports: List of output port specifications (auto-generated if None)
            label: Human-readable label (auto-generated if None)
        """
        id_str = str(id)

        # Enforce uniqueness across entire namespace
        if any(str(inp.id) == id_str for inp in self.spec.inputs):
            raise ValueError(f"ID {id} is already used by an Input. IDs must be unique across all entities.")

        # Auto-generate label if missing
        if label is None:
            label = f"{type}_({id})"

        # Auto-generate output ports if not provided
        if output_ports is None:
            # Generate next available integer ID for output port
            existing_output_ids = set()
            for inp in self.spec.inputs:
                try:
                    existing_output_ids.add(int(inp.id))
                except ValueError:
                    pass
            for n in self.spec.nodes:
                for op in n.output_ports:
                    existing_output_ids.add(op.id)
            next_id = max(existing_output_ids, default=0) + 1
            # Default: single "result" output with visibility=False
            output_ports = [OutputPortSpec(id=next_id, name="result", type="DATAFRAME", visibility=False)]

        # Upsert: remove existing node with same ID
        self.remove_node(id)

        new_node_spec = NodeSpec(id=id_str, label=label, type=type, input_ports=input_ports, output_ports=output_ports)
        self.spec.nodes.append(new_node_spec)

        # Update outputs list for visible output ports
        for op in output_ports:
            if op.visibility:
                if not any(o.id == str(op.id) for o in self.spec.outputs):
                    self.spec.outputs.append(OutputSpec(id=str(op.id)))

        self._last_run_hash = None
        if self.workspace_path:
            self._ensure_wrapper()
        self._emit("graph", {})

    def remove_node(self, id: str):
        """Remove a node from the pipeline."""
        id_str = str(id)
        self.spec.nodes = [n for n in self.spec.nodes if str(n.id) != id_str]
        self._last_run_hash = None
        if self.workspace_path:
            self._ensure_wrapper()
        self._emit("graph", {})

    def remove_input(self, id: str):
        """Remove an input from the pipeline.
        
        This removes the input from both the inputs list and outputs list (if visible).
        It also triggers catalog regeneration to ensure catalog.yml is updated.
        
        Args:
            id: The input ID to remove
        """
        id_str = str(id)
        
        # Remove from inputs list
        self.spec.inputs = [inp for inp in self.spec.inputs if str(inp.id) != id_str]
        
        # Remove from outputs list if it was visible
        self.spec.outputs = [out for out in self.spec.outputs if str(out.id) != id_str]
        
        # Invalidate cache and regenerate catalog
        self._last_run_hash = None
        if self.workspace_path:
            self._ensure_wrapper()
        self._catalog_instance = None
        self._emit("graph", {})

    # def register_transform(self, name: str, func: callable, output_type: Any = None):
    #     self.transform_registry[name] = {"func": func, "output_type": output_type}
    #     self._last_run_hash = None
    #     if self.workspace_path: self._ensure_wrapper()
    #     self._emit("graph", {})

    def subscribe(self, callback: Callable[[str, Any], None]):
        """Register a listener for pipeline events.

        Args:
            callback: Function called with ``(event_type, payload)`` on each event.
                Event types include ``"status"``, ``"graph"``, ``"graph_update"``,
                and ``"node_status"``.
        """
        if callback not in self._listeners: self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[str, Any], None]):
        """Remove a previously registered event listener.

        Args:
            callback: The callback function to remove.
        """
        if callback in self._listeners: self._listeners.remove(callback)

    def _emit(self, event_type: str, payload: Any):
        for listener in self._listeners:
            try: listener(event_type, payload)
            except: pass

    @staticmethod
    def _purge_wrapper_modules():
        """Remove cached viz_wrapper modules and stale sys.path entries.

        Kedro's ``bootstrap_project`` adds the wrapper's ``src/`` directory to
        ``sys.path`` and imports ``viz_wrapper.*`` modules.  When switching
        between rooms/workspaces the old entries must be removed so that the
        next ``bootstrap_project`` call imports the *new* room's wrapper.
        """
        # 1. Remove cached viz_wrapper modules
        keys_to_remove = [k for k in sys.modules if k.startswith("viz_wrapper")]
        for k in keys_to_remove:
            del sys.modules[k]

        # 2. Remove stale pipeline/src paths from sys.path
        sys.path[:] = [p for p in sys.path if "pipeline" not in p]

    def close(self):
        """Release cached data and catalog resources."""
        self._data_cache.clear()
        self._catalog_instance = None
        self._purge_wrapper_modules()

    def reset_spec(self):
        """Reset the spec to an empty state, clearing all inputs, nodes, and outputs."""
        self.spec = ChoregraphSpec()
        self._data_cache.clear()
        if self._datasets_metadata is not None:
            self._datasets_metadata.clear()
        self._catalog_instance = None
        self._last_run_hash = None
        self._spec_hash = None
        if self.workspace_path:
            self._ensure_wrapper()
        self._emit("graph", {})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # COMPATIBILITY LAYER
    def load(self, xml_spec: Union[str, Path] = None, external_inputs: Dict[str, Any] = None, workspace_path: Union[str, Path] = None):
        """Load or reload a pipeline specification (compatibility layer).

        Re-parses the XML spec and regenerates Kedro project files if the spec
        content has changed since the last call.

        Args:
            xml_spec: Path to an XML file or an XML string.
            external_inputs: Dict mapping input IDs to in-memory data objects.
            workspace_path: Override the workspace directory.
        """
        old_workspace = self.workspace_path
        if workspace_path: self.workspace_path = Path(workspace_path)
        workspace_changed = (old_workspace != self.workspace_path)

        new_spec_content = ""
        if xml_spec:
            if isinstance(xml_spec, (str, Path)) and Path(xml_spec).exists():
                 new_spec_content = safe_path(xml_spec).read_text()
            else:
                 new_spec_content = str(xml_spec)

        new_hash = hashlib.sha256(new_spec_content.encode('utf-8')).hexdigest()
        spec_changed = (new_hash != getattr(self, '_spec_hash', None))

        if spec_changed or workspace_changed:
            # Purge cached Kedro wrapper modules to prevent stale imports
            # when switching between rooms/workspaces
            self._purge_wrapper_modules()

            self._spec_hash = new_hash
            if xml_spec: self.spec = ChoregraphSpecParser.parse(xml_spec)
            else: self.spec = ChoregraphSpec()

            # Clear all caches when spec or workspace changes to prevent stale data
            self._data_cache.clear()
            self._catalog_instance = None

            self._last_run_hash = None
            if self.workspace_path: self._ensure_wrapper()
            self._emit("graph", {})
        else:
            logger.info("Choregraph spec unchanged.")

        if external_inputs: self.external_inputs = external_inputs

    
    def get_inputs(self) -> List[Tuple[str, str]]:
        """Get list of input specifications as (id, name) tuples."""
        return [(str(inp.id), self.spec.get_name(inp.id)) for inp in self.spec.inputs]

    def get_visibles(self) -> List[Tuple[str, str]]:
        """Get list of datasets marked as visible (visibility=True) as (id, name) tuples."""
        result = []
        seen_ids = set()

        for inp in self.spec.inputs:
            if inp.visibility:
                inp_id = str(inp.id)
                if inp_id not in seen_ids:
                    seen_ids.add(inp_id)
                    result.append((inp_id, self.spec.get_name(inp.id)))

        for n in self.spec.nodes:
            for op in n.output_ports:
                if op.visibility:
                    op_id = str(op.id)
                    if op_id not in seen_ids:
                        seen_ids.add(op_id)
                        result.append((op_id, self.spec.get_name(op.id)))

        return result
    
    def get_leaves(self) -> List[Tuple[str, str]]:
        """Get list of terminal output ports (not consumed by any downstream node) as (id, name) tuples.
        
        Note: Only returns output ports from nodes, not inputs. If there are no nodes,
        returns an empty list (inputs are already inputs, they don't need promotion).
        """
        referenced_ids = set()
        
        for n in self.spec.nodes:
            for port in n.input_ports:
                if port.source_ref is not None:
                    referenced_ids.add(str(port.source_ref))
        
        leaf_ids = []
        for n in self.spec.nodes:
            for op in n.output_ports:
                if str(op.id) not in referenced_ids:
                    leaf_ids.append(str(op.id))
        
        return [(leaf_id, self.spec.get_name(leaf_id)) for leaf_id in leaf_ids]
    
    def find_node_for_output_port(self, output_port_id: str) -> Optional[NodeSpec]:
        """Find the node that owns a given output port ID.
        
        Args:
            output_port_id: The ID of the output port
            
        Returns:
            The NodeSpec containing this output port, or None if not found
        """
        port_id_str = str(output_port_id)
        for n in self.spec.nodes:
            for op in n.output_ports:
                if str(op.id) == port_id_str:
                    return n
        return None
    
    def give_id(self) -> str: 
        """
        Give the next available integer ID as a string. This can be used for generating unique IDs for nodes and ports.
        """
        existing_ids = set()
        for inp in self.spec.inputs:
            try:
                existing_ids.add(int(inp.id))
            except ValueError:
                pass
        for n in self.spec.nodes:
            try:
                existing_ids.add(int(n.id))
            except ValueError:
                pass
            for op in n.output_ports:
                try:
                    existing_ids.add(int(op.id))
                except ValueError:
                    pass
        next_id = max(existing_ids, default=0) + 1
        return str(next_id)

    # ------------------------------------------------------------------
    # Promotion helpers
    # ------------------------------------------------------------------

    def _find_output_file(self, name: str, ext: str) -> Optional[Path]:
        """Locate a dataset file across standard workspace locations.

        Searches ``inputs/``, the root ``data/`` dir, and ``*_partitioned/``
        folders, returning the first match or ``None``.
        """
        if not self.workspace_path:
            return None
        candidates = [
            self._inputs_dir / f"{name}.{ext}",
            self._data_dir / f"{name}.{ext}",
        ]
        for c in candidates:
            if c.exists():
                return c
        # Partitioned folders
        if self._data_dir.exists():
            for partitioned_dir in self._data_dir.glob("*_partitioned"):
                if partitioned_dir.is_dir():
                    f = partitioned_dir / f"{name}.{ext}"
                    if f.exists():
                        return f
        return None

    def _remove_orphaned_sources(self, node: NodeSpec):
        """Remove invisible source inputs consumed *only* by *node*.

        After promoting a node's outputs, the inputs it consumed may no longer
        be needed. This helper removes them **only** when:
        - the input is not visible (standalone user inputs are preserved), and
        - no other node references the same input.

        Inputs that were not produced for any node (i.e. user-provided inputs
        that happen to have no downstream consumers) are never touched.
        """
        consumed_ids = {
            str(p.source_ref) for p in node.input_ports if p.source_ref is not None
        }
        # Gather refs used by *other* nodes
        other_refs: set[str] = set()
        for other in self.spec.nodes:
            if str(other.id) == str(node.id):
                continue
            for p in other.input_ports:
                if p.source_ref is not None:
                    other_refs.add(str(p.source_ref))

        for input_id in consumed_ids - other_refs:
            for inp in self.spec.inputs:
                if str(inp.id) == input_id and not inp.visibility:
                    self.remove_input(str(inp.id))
                    logger.info(f"Removed orphaned source input '{inp.label}' (ID: {inp.id})")
                    break

    _NON_PROCESSABLE_FORMATS = {"XLSX", "XLS", "ODS", "XLSM"}

    def _remove_non_processable_inputs(self):
        """Remove inputs whose format is a raw container (Excel, etc.).

        These formats are never consumed directly by the visualization
        pipeline — they go through conversion nodes (``tidy_excel_data``)
        that produce processable outputs (PARQUET/CSV).
        After leaf promotion, any remaining input in these formats is
        orphaned and should be cleaned up.
        Note: JSON is NOT included — it stays as an input and gets
        processed by AI-generated transforms (e.g. ``flatten_json``).
        """
        to_remove = [
            inp for inp in self.spec.inputs
            if inp.format.upper() in self._NON_PROCESSABLE_FORMATS
        ]
        for inp in to_remove:
            self.remove_input(str(inp.id))
            logger.info(f"Removed non-processable input '{inp.label}' (format={inp.format}, ID: {inp.id})")

    def _promote_output(
        self,
        output_port_id: str = None,
        dataset_name: str = None,
        location: str = None,
        format: str = "PARQUET",
        visibility: bool = True,
        **options,
    ) -> str:
        """Convert an output port into an input, moving the file to ``inputs/``.

        Locates the dataset by *output_port_id* or *dataset_name*. When
        *location* is ``None`` the file is auto-detected via
        :meth:`_find_output_file`.

        Returns:
            The ID of the newly created input.
        """
        import shutil

        ext = format.lower()

        # --- resolve port id + name --------------------------------
        if output_port_id is not None:
            source_port_id = str(output_port_id)
            if not any(
                str(op.id) == source_port_id
                for n in self.spec.nodes for op in n.output_ports
            ):
                raise ValueError(f"Output port ID '{output_port_id}' not found in spec nodes.")
            resolved_name = self.spec.get_name(source_port_id)
        elif dataset_name is not None:
            source_port_id = None
            for n in self.spec.nodes:
                for op in n.output_ports:
                    if self.spec.get_name(op.id) == dataset_name:
                        source_port_id = str(op.id)
                        break
                if source_port_id:
                    break
            if not source_port_id:
                raise ValueError(f"Dataset '{dataset_name}' not found in spec outputs.")
            resolved_name = dataset_name
        else:
            raise ValueError("Either output_port_id or dataset_name must be provided.")

        # --- locate file -------------------------------------------
        if location is not None:
            source_location = Path(location)
        else:
            source_location = self._find_output_file(resolved_name, ext)

        if source_location is None:
            raise ValueError(f"Could not auto-detect location for dataset '{resolved_name}'.")

        # --- move to inputs/ if needed -----------------------------
        if self.workspace_path:
            self._inputs_dir.mkdir(parents=True, exist_ok=True)
            if source_location.parent != self._inputs_dir:
                target_path = self._inputs_dir / source_location.name
                shutil.move(str(source_location), str(target_path))
                final_location = str(target_path.absolute())
                logger.info(f"Moved {source_location.name} to inputs folder")
            else:
                final_location = str(source_location.absolute())
        else:
            final_location = str(source_location)

        new_id = self.give_id()
        self.add_input(id=new_id, location=final_location, format=format,
                       label=resolved_name, visibility=visibility, **options)
        return new_id

    # Keep old name as alias for backward compatibility
    convert_dataset_into_input = _promote_output

    def _promote_partitioned(self, output_port_id: str, resolved_name: str) -> List[Tuple[str, str]]:
        """Promote every parquet file in a ``{name}_partitioned/`` folder as an input.

        After successful promotion, orphaned invisible source inputs of the
        producing node are cleaned up via :meth:`_remove_orphaned_sources`.

        Returns:
            List of ``(id, name)`` tuples for the newly created inputs.
        """
        import shutil

        if not self.workspace_path:
            return []

        partitioned_dir = self._data_dir / f"{resolved_name}_partitioned"
        if not partitioned_dir.is_dir():
            return []

        parquet_files = list(partitioned_dir.glob("*.parquet"))
        if not parquet_files:
            return []

        self._inputs_dir.mkdir(parents=True, exist_ok=True)
        promoted: List[Tuple[str, str]] = []

        for parquet_file in parquet_files:
            label = parquet_file.stem
            already_input = any(
                inp.location and Path(inp.location).name == parquet_file.name
                for inp in self.spec.inputs
            )
            if already_input:
                continue

            target_path = self._inputs_dir / parquet_file.name
            shutil.move(str(parquet_file), str(target_path))
            new_id = self.give_id()
            self.add_input(id=new_id, location=str(target_path.absolute()),
                           format="PARQUET", label=label, visibility=True)
            promoted.append((new_id, label))
            logger.info(f"Promoted partitioned output '{label}' (ID: {new_id}) as input")

        # Clean up empty partitioned directory
        if partitioned_dir.exists() and not any(partitioned_dir.iterdir()):
            partitioned_dir.rmdir()

        # Generic orphan cleanup (replaces the old tidy_excel_data-specific block)
        if promoted:
            source_node = self.find_node_for_output_port(output_port_id)
            if source_node:
                self._remove_orphaned_sources(source_node)

        return promoted

    def promote_leaves(self, remove_source_nodes: bool = True) -> List[Tuple[str, str]]:
        """Promote all leaf outputs as inputs, optionally removing their source nodes.

        For each terminal output port (not consumed downstream):
        - Single-file outputs are promoted via :meth:`_promote_output`.
        - Partitioned outputs are promoted via :meth:`_promote_partitioned`.

        Nodes are only removed when **all** their outputs were successfully promoted.

        Returns:
            List of ``(id, name)`` tuples for the promoted inputs.
        """
        leaves = self.get_leaves()
        promoted: List[Tuple[str, str]] = []
        nodes_to_remove: set[str] = set()
        nodes_with_failures: set[str] = set()

        for leaf_id, leaf_name in leaves:
            source_node = self.find_node_for_output_port(leaf_id)
            source_node_id = str(source_node.id) if source_node else None
            success = False

            try:
                self._promote_output(output_port_id=leaf_id)
                promoted.append((leaf_id, leaf_name))
                success = True
            except ValueError:
                partitioned = self._promote_partitioned(leaf_id, leaf_name)
                if partitioned:
                    promoted.extend(partitioned)
                    success = True
                else:
                    logger.warning(f"Could not promote leaf '{leaf_name}' (ID: {leaf_id}): no files found")

            if source_node_id:
                (nodes_to_remove if success else nodes_with_failures).add(source_node_id)

        if remove_source_nodes:
            for node_id in nodes_to_remove - nodes_with_failures:
                try:
                    self.remove_node(node_id)
                    logger.info(f"Removed transformation node ID: {node_id}")
                except Exception as e:
                    logger.warning(f"Could not remove node {node_id}: {e}")

        # Clean up raw container inputs (Excel, JSON) whose conversion nodes
        # have been promoted and removed above.
        if promoted:
            self._remove_non_processable_inputs()

        return promoted

    # Keep old name as alias for backward compatibility
    promote_leaves_as_inputs = promote_leaves


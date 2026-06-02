# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Kedro execution hooks for pipeline status tracking.

Provides :class:`ExecutionStatusHook` which reports node-level execution status
(pending, running, completed, failed) via callbacks, enabling real-time UI
updates during pipeline runs.
"""
import logging
from typing import Any, Dict, Callable, Optional, TYPE_CHECKING
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node
import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .metadata import Metadata
    from .parser import ChoregraphSpec

# class VizEventsHook:
#     """
#     Kedro hook to collect execution events and save them to .viz/kedro_pipeline_events.json
#     This mimics the behavior of KedroSession for Kedro-Viz compatibility.
#     """
#     def __init__(self, project_path: Path):
#         self.events = []
#         self.project_path = project_path

#     @hook_impl
#     def before_pipeline_run(self, run_params: Dict[str, Any], pipeline: Any, catalog: Any):
#         self.events.append({
#             "event_type": "before_pipeline_run",
#             "run_id": run_params.get("run_id", "manual_run"),
#             "timestamp": time.time() * 1000
#         })

#     @hook_impl
#     def before_node_run(self, node: Node, catalog: Any, inputs: Dict[str, Any], is_async: bool, run_id: str):
#         self.events.append({
#             "event_type": "before_node_run",
#             "node_name": node.name,
#             "run_id": run_id,
#             "timestamp": time.time() * 1000
#         })

#     @hook_impl
#     def after_node_run(self, node: Node, catalog: Any, inputs: Dict[str, Any], outputs: Dict[str, Any], is_async: bool, run_id: str):
#         self.events.append({
#             "event_type": "after_node_run",
#             "node_name": node.name,
#             "run_id": run_id,
#             "timestamp": time.time() * 1000
#         })

#     @hook_impl
#     def on_node_error(self, error: Exception, node: Node, catalog: Any, inputs: Dict[str, Any], is_async: bool, run_id: str):
#         self.events.append({
#             "event_type": "on_node_error",
#             "node_name": node.name,
#             "error": str(error),
#             "run_id": run_id,
#             "timestamp": time.time() * 1000
#         })

#     @hook_impl
#     def after_pipeline_run(self, run_params: Dict[str, Any], run_result: Any, pipeline: Any, catalog: Any):
#         self.events.append({
#             "event_type": "after_pipeline_run",
#             "run_id": run_params.get("run_id", "manual_run"),
#             "timestamp": time.time() * 1000
#         })
#         self._write_events()

#     def _write_events(self):
#         if not self.project_path:
#             return
#         viz_dir = self.project_path / ".viz"
#         viz_dir.mkdir(exist_ok=True)
#         events_file = viz_dir / "kedro_pipeline_events.json"
#         try:
#             with open(events_file, "w") as f:
#                 json.dump(self.events, f, indent=2)
#         except Exception as e:
#             print(f"Failed to write kedro_pipeline_events.json: {e}")

class DataInjectionHook:
    """Inject in-memory DataFrames into the catalog before pipeline execution.

    Registered when inputs use ``format="MEMORY"`` (Toolkit path). The hook
    fills ``MemoryDataset`` catalog entries with actual data before any
    node runs. Uses the official ``before_pipeline_run`` hook point.
    """

    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        self.datasets = datasets

    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any], pipeline: Any, catalog: Any) -> None:
        for name, data in self.datasets.items():
            # Use save() to fill the existing MemoryDataset entry from catalog.yml
            # instead of catalog[name] = data which replaces and triggers a warning.
            try:
                catalog.save(name, data)
            except Exception:
                catalog[name] = data


class DtypeInferenceHook:
    """Run :func:`infer_dtypes` on every DataFrame loaded from the catalog.

    This ensures that object-typed columns carrying numeric/date strings are
    converted to their proper pandas dtype *before* they enter pipeline nodes.
    Without this, ``pd.concat`` (union) on DataFrames with inconsistent dtypes
    produces mixed-type object columns that pyarrow cannot serialize to Parquet.
    """

    @hook_impl
    def after_dataset_loaded(self, dataset_name: str, data: Any) -> None:
        if isinstance(data, pd.DataFrame):
            from .dtype_inference import infer_dtypes
            infer_dtypes(data)


class ExecutionStatusHook:
    """
    Kedro hook to track execution status of nodes and trigger a callback on updates.
    Status can be: 'pending', 'running', 'completed', 'failed'.
    """

    def __init__(self, on_update: Callable[[Dict[str, str]], None] = None, excluded_nodes: set[str] = None):
        """Initialize the execution status hook.

        Args:
            on_update: Callback invoked with the full status dict whenever a
                node's status changes.
            excluded_nodes: Set of node names to skip tracking for.
        """
        self.on_update = on_update
        self.node_status: Dict[str, str] = {}
        self.excluded_nodes = excluded_nodes or set()

    def _should_skip(self, node_name: str) -> bool:
        return node_name in self.excluded_nodes

    @hook_impl
    def before_pipeline_run(self, run_params: Dict[str, Any], pipeline: Any, catalog: Any):
        """Initialize all nodes to pending."""
        for node in pipeline.nodes:
            if not self._should_skip(node.name):
                self.node_status[node.name] = "pending"
        
        if self.on_update:
            self.on_update(self.node_status.copy())

    @hook_impl
    def before_node_run(self, node: Node, catalog: Any, inputs: Dict[str, Any], is_async: bool, run_id: str):
        """Mark node as running."""
        if self._should_skip(node.name):
            return
        self.node_status[node.name] = "running"
        if self.on_update:
            self.on_update(self.node_status.copy())

    @hook_impl
    def after_node_run(self, node: Node, catalog: Any, inputs: Dict[str, Any], outputs: Dict[str, Any], is_async: bool, run_id: str):
        """Mark node as completed."""
        if self._should_skip(node.name):
            return
        self.node_status[node.name] = "completed"
        if self.on_update:
            self.on_update(self.node_status.copy())

    @hook_impl
    def on_node_error(self, error: Exception, node: Node, catalog: Any, inputs: Dict[str, Any], is_async: bool, run_id: str):
        """Mark node as failed."""
        if self._should_skip(node.name):
            return
        self.node_status[node.name] = "failed"
        if self.on_update:
            self.on_update(self.node_status.copy())


class MetadataStatsHook:
    """
    Kedro hook to capture dataset statistics during pipeline execution.
    
    - after_node_run: Captures stats for inputs and outputs while DataFrames are in memory
    - after_pipeline_run: Saves all collected stats to catalogue_stats.json
    
    This avoids expensive reloading of datasets just for metadata extraction.
    """

    def __init__(self, metadata_manager: "Metadata", spec: "ChoregraphSpec"):
        self.manager = metadata_manager
        self.spec = spec
        self._processed_datasets: set = set()  # Track already processed to avoid duplicates

    @staticmethod
    def _strip_catalog_suffix(name: str) -> str:
        """Strip the ``#parquet`` / ``#json`` catalog factory suffix from a dataset name."""
        idx = name.find("#")
        return name[:idx] if idx >= 0 else name

    def _get_output_port_info(self, name: str):
        """Find the output port whose Kedro dataset name matches."""
        clean = self._strip_catalog_suffix(name)
        for node in self.spec.nodes:
            for op in node.output_ports:
                if self.spec.get_name(op.id) == clean:
                    return op
        return None

    def _is_input_dataset(self, name: str) -> bool:
        """Check if a dataset name is defined as an input in the spec."""
        clean = self._strip_catalog_suffix(name)
        for inp in self.spec.inputs:
            if self.spec.get_name(inp.id) == clean:
                return True
        return False

    def _resolve_dataset_id(self, name: str) -> Optional[str]:
        """Reverse-map a dataset name to its spec ID (input ID or output port ID)."""
        clean = self._strip_catalog_suffix(name)
        for inp in self.spec.inputs:
            if self.spec.get_name(inp.id) == clean:
                return str(inp.id)
        for node in self.spec.nodes:
            for op in node.output_ports:
                if self.spec.get_name(op.id) == clean:
                    return str(op.id)
        return None

    @hook_impl
    def after_node_run(self, node: Node, catalog: Any, inputs: Dict[str, Any], outputs: Dict[str, Any], is_async: bool, run_id: str):
        """Capture stats for inputs and outputs while data is in memory.
        
        Type and visibility are derived from the spec, not stored in the cache.
        We store stats for:
        - All inputs (regardless of visibility)
        - Only visible outputs (visibility=True in spec)
        """
        
        # Process INPUTS (all inputs defined in spec)
        for name, data in inputs.items():
            clean_name = self._strip_catalog_suffix(name)
            if clean_name in self._processed_datasets:
                continue

            if not self._is_input_dataset(name):
                # Intermediate output being used as input to next node - skip
                logger.debug(f"Skipping intermediate input: {name}")
                continue

            if isinstance(data, pd.DataFrame):
                logger.debug(f"Processing INPUT: {clean_name} (rows={len(data)})")
                self.manager.update_stats(clean_name, data, dataset_id=self._resolve_dataset_id(name), dataset_type="input")
                self._processed_datasets.add(clean_name)
            elif isinstance(data, (dict, list)):
                # JSON input (dict/list) — update_stats routes to cartograph_json.
                logger.debug(f"Processing JSON INPUT: {clean_name} ({type(data).__name__})")
                self.manager.update_stats(clean_name, data, dataset_id=self._resolve_dataset_id(name), dataset_type="input")
                self._processed_datasets.add(clean_name)
            else:
                logger.debug(f"Skipping non-tabular input: {name} (type={type(data).__name__})")

        # Process OUTPUTS (only visible ones based on spec)
        for name, data in outputs.items():
            clean_name = self._strip_catalog_suffix(name)
            if clean_name in self._processed_datasets:
                continue

            port = self._get_output_port_info(name)
            if not port or not port.visibility:
                continue

            if isinstance(data, pd.DataFrame):
                logger.debug(f"Processing visible OUTPUT: {clean_name} (rows={len(data)})")
                self.manager.update_stats(clean_name, data, dataset_id=str(port.id), dataset_type="output")
                self._processed_datasets.add(clean_name)
            elif isinstance(data, dict):
                values = list(data.values())
                if values and all(isinstance(v, pd.DataFrame) for v in values):
                    # Dict output (e.g. tidy_excel_data → PartitionedDataset).
                    # Each value is a DataFrame representing a table/sheet.
                    for key, df in data.items():
                        table_name = str(key)
                        logger.debug(f"Processing visible DICT OUTPUT partition: {table_name} (rows={len(df)})")
                        self.manager.update_stats(table_name, df, dataset_id=str(port.id), dataset_type="input")
                    self._processed_datasets.add(name)
                else:
                    # Plain JSON dict output — describe structure via update_stats.
                    logger.debug(f"Processing visible JSON OUTPUT: {clean_name}")
                    self.manager.update_stats(clean_name, data, dataset_id=str(port.id), dataset_type="output")
                    self._processed_datasets.add(clean_name)
            elif isinstance(data, list):
                logger.debug(f"Processing visible JSON OUTPUT: {clean_name} (list)")
                self.manager.update_stats(clean_name, data, dataset_id=str(port.id), dataset_type="output")
                self._processed_datasets.add(clean_name)
            else:
                logger.debug(f"Skipping non-tabular output: {name} (type={type(data).__name__})")

    @hook_impl
    def after_pipeline_run(self, run_params: Dict[str, Any], run_result: Any, pipeline: Any, catalog: Any):
        """Ensure all inputs are processed, even if not used in the pipeline run."""
        try:
            full_catalog = catalog.keys()
            for name in full_catalog:
                if name in self._processed_datasets:
                    continue
                
                # Process any remaining inputs not yet captured
                if self._is_input_dataset(name):
                    try:
                        data = catalog.load(name)
                        if isinstance(data, (pd.DataFrame, dict, list)):
                            size_desc = f"rows={len(data)}" if isinstance(data, pd.DataFrame) else type(data).__name__
                            print(f"[DEBUG]   Processing remaining INPUT: {name} ({size_desc})")
                            self.manager.update_stats(name, data, dataset_id=self._resolve_dataset_id(name), dataset_type="input")
                            self._processed_datasets.add(name)
                    except Exception as e:
                        print(f"[DEBUG] Warning: could not load unused input '{name}': {e}")
        except Exception as e:
            print(f"[DEBUG] Error processing unused inputs: {e}")

        
        # try:
        #     self.manager.save_to_cache()
        # except Exception as e:
        #     print(f"[DEBUG] ERROR saving metadata: {e}")
        #     raise


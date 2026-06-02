# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Guillaume Franque
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Kedro Viz server management and graph utilities.

Manages the Kedro Viz subprocess lifecycle (start, stop, port checking,
readiness polling) and provides custom CSS for UI styling. Also includes
a helper to generate dummy graph structures for file-only views.
"""
import subprocess
import asyncio
import os
import signal
import time
import socket
import logging
from pathlib import Path
import atexit
import shutil
import tempfile
from typing import Optional, Union
import httpx

from .security import safe_path, sanitize_filename

logger = logging.getLogger(__name__)


# Default custom CSS for Kedro Viz styling
DEFAULT_KEDRO_CUSTOM_CSS = """
.pipeline-sidebar {
    background: #1e1f2100;
    left: 50px;
}
.pipeline-toolbar {
    background: #1c1c1c17;
    backdrop-filter: blur(10px);
}

.pipeline-menu-button--settings{
    display: none;
}

.pipeline-global-toolbar {
    width: 50px;
    background: #0f0f11a8;
}

.pipeline-menu-button--large {
    width: 50px;
    height: 50px;

  display: flex;
  flex-direction: column;
  align-items: center;
  place-content: center;
}

.pipeline-menu-button--logo{
    display: none;
    height: 0px;
    margin: 0;
    padding: 0;
}
button[aria-label="View your pipeline"]{
height: 170px;
}
button[aria-label="View your workflow"]{
height: 170px;
}

.pipeline-global-routes-toolbar, .pipeline-global-control-toolbar {
    padding: 0em 0!important;
}
.pipeline-global-routes-toolbar a{
    text-decoration: none;
}
button[aria-label="View your pipeline"]::before {
    content: "FlowChart";
    color: white;
    writing-mode: vertical-lr;
    font-size: large;
    margin-bottom: 20px;
    display: block;
    place-self: center;
}

button[aria-label="View your workflow"]::before {
    content: "Workflow";
    color: white;
    writing-mode: vertical-lr;
    font-size: large;
    margin-bottom: 20px;
    display: block;
    place-self: center;
}
.kui-theme--light button[aria-label="View your workflow"]::before {
    color: black;
}
.kui-theme--light button[aria-label="View your pipeline"]::before {
    color: black;
}

.kedro .pipeline-warning{
    display: none;
}

.run-status-notification{
    left: 50% !important;
}
.pipeline-minimap-container{
    display:none;
}
.update-reminder-version-tag{
    display:none;
}
.pipeline-menu-button--deploy{
    display: none;
}

dt.pipeline-metadata__label:has(+ dd.pipeline-metadata__row[data-label="Run Command:"]) {
  display: none;
}

dd.pipeline-metadata__row[data-label="Run Command:"] {
  display: none;
}

dt.pipeline-metadata__label:has(+ dd.pipeline-metadata__row[data-label="Type:"]) {
  display: none;
}

dd.pipeline-metadata__row[data-label="Type:"] {
  display: none;
}

dt.pipeline-metadata__label:has(+ dd.pipeline-metadata__row[data-label="Dataset Type:"]) {
  display: none;
}

dd.pipeline-metadata__row[data-label="Dataset Type:"] {
  display: none;
}

dt.pipeline-metadata__label:has(+ dd.pipeline-metadata__row[data-label="File Path:"]) {
  display: none;
}

dd.pipeline-metadata__row[data-label="File Path:"] {
  display: none;
}
dt.pipeline-metadata__label:has(+ dd.pipeline-metadata__row[data-label="Error Log:"]) {
  display: none;
}

dd.pipeline-metadata__row[data-label="Error Log:"] {
  display: none;
}
.pipeline-metadata {
    max-width: 40%!important;
    background: linear-gradient(360deg, #1f84ba21, transparent);
    backdrop-filter: blur(14px);
}
#minimap-toggle-icon{
    display:none;
}
.pipeline-metadata__preview .scrollable-container{
    overflow: hidden!important;
}
.pipeline-metadata-modal__preview {
    scrollbar-width: thin;
    scrollbar-color: rgba(255, 255, 255, 0.12) transparent;
}
.pipeline-metadata-modal__preview::-webkit-scrollbar {
    width: 4px;
}
.pipeline-metadata-modal__preview::-webkit-scrollbar-track {
    background: transparent;
}
.pipeline-metadata-modal__preview::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 2px;
}
.pipeline-metadata-modal__preview::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.25);
}
.pipeline-flowchart .pipeline-node--task .pipeline-node__bg{
    fill: #34343421;
    stroke-width: 0px;
}
.feature-hints{
    display: none;
}
.feature-hints__highlightDot{
    display: none;
}
.pipeline-metadata-modal{
    left: 30px!important;
}
.run-status-dot {
    top: 155px!important;
}
.pipeline-flowchart .pipeline-node--data .pipeline-node__icon {
    // fill: #71e6ac!important;
    }
#root{
background: url(/gradient_bg_light.webp)!important;
background-size: cover;
}
.kedro-pipeline{
    background: #09090940 !important;
    backdrop-filter: blur(40px);
}
.shareable-url-timestamp{
    display: none;
}
.pipeline-menu-button--theme{
    display: none;
}
.pipeline-menu-button--settings{
    display: none;
}
.pipeline-layer {
    fill: #0103078c!important;
}

"""


class KedroVizServer:
    """Manages a Kedro Viz server instance."""

    def __init__(self, port: int = 4141, custom_css: Optional[str] = None):
        self.port = port
        self.process = None
        self._atexit_registered = False
        self._ready = False
        self._custom_css = custom_css or DEFAULT_KEDRO_CUSTOM_CSS
        self._project_path: Optional[Path] = None
        self._stable_wrapper: Optional[Path] = None
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=20)
        self._client = httpx.AsyncClient(limits=limits, verify=False)

    def _is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    async def _is_http_ready(self) -> bool:
        """Check if the Kedro Viz HTTP server is actually responding."""
        try:
            # Try to fetch the main API endpoint
            req = await self._client.request(
                "GET", f"http://localhost:{self.port}/api/main", timeout=2
            )
            return req.status_code == 200
        except (httpx.RequestError):
            return False

    async def is_ready(self) -> bool:
        """Check if the Kedro Viz server is ready to accept connections."""
        return self._ready and await self._is_http_ready()

    async def wait_until_ready(
        self, timeout: float = 30.0, poll_interval: float = 0.5
    ) -> bool:
        """Wait until the server is ready or timeout is reached.

        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: Time between checks in seconds

        Returns:
            True if server is ready, False if timeout reached
        """
        start = time.time()
        while time.time() - start < timeout:
            # First check if port is bound (fast check)
            if self._is_port_in_use(self.port):
                # Then check if HTTP server is actually responding
                if await self._is_http_ready():
                    self._ready = True
                    return True
            await asyncio.sleep(poll_interval)
        return False

    def get_custom_css(self) -> str:
        """Get the custom CSS for Kedro Viz styling."""
        return self._custom_css

    def set_custom_css(self, css: str):
        """Set custom CSS for Kedro Viz styling."""
        self._custom_css = css

    def _find_pids_on_port(self, port: int) -> list[int]:
        """Find PIDs listening on a port by scanning /proc/net/tcp (Linux)."""
        hex_port = f"{port:04X}"
        inodes = set()
        try:
            with open("/proc/net/tcp") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    local = parts[1]  # e.g. "00000000:102D"
                    if local.endswith(f":{hex_port}"):
                        inodes.add(parts[9])
        except (FileNotFoundError, PermissionError):
            return []

        pids = []
        if not inodes:
            return pids
        try:
            for pid_dir in Path("/proc").iterdir():
                if not pid_dir.name.isdigit():
                    continue
                fd_dir = pid_dir / "fd"
                try:
                    for fd in fd_dir.iterdir():
                        try:
                            target = os.readlink(str(fd))
                            for inode in inodes:
                                if f"socket:[{inode}]" == target:
                                    pids.append(int(pid_dir.name))
                                    break
                        except (OSError, ValueError):
                            continue
                except (PermissionError, FileNotFoundError):
                    continue
        except Exception:
            pass
        return pids

    def _kill_process_on_port(self, port: int):
        """Force-kill any process listening on the given port."""
        pids = self._find_pids_on_port(port)
        my_pid = os.getpid()
        for pid in pids:
            if pid == my_pid:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # -----------------------------------------------------------------
    # Stable wrapper: real directory that persists across room switches.
    # Config files are copied here; autoreload detects the overwrites.
    # catalog.yml already uses absolute paths to room data, so no
    # data symlinks are needed.
    # -----------------------------------------------------------------

    def _ensure_stable_wrapper(self) -> Path:
        """Get or create the stable wrapper directory."""
        if self._stable_wrapper is None:
            self._stable_wrapper = (
                Path(tempfile.gettempdir()) / f"nveil_kedro_viz_{self.port}"
            )
            self._stable_wrapper.mkdir(parents=True, exist_ok=True)
        return self._stable_wrapper

    # Files synced from room pipeline to stable wrapper for kedro-viz.
    # Includes settings.py (registers DatasetStatsHook for previews)
    # and __init__.py (required for package import).
    _WRAPPER_FILES = [
        "pyproject.toml",
        "conf/base/catalog.yml",
        "conf/base/parameters.yml",
        "src/viz_wrapper/__init__.py",
        "src/viz_wrapper/settings.py",
        "src/viz_wrapper/pipeline_registry.py",
        "label_mapping.json",
    ]

    def _sync_wrapper(self, source_wrapper: Path):
        """Copy config files from a room's pipeline to the stable wrapper.

        Only small text config files are copied (see ``_WRAPPER_FILES``).
        The ``cache/`` directory is symlinked to the room's cache so that
        per-node hook writes (e.g. ``catalogue_stats.json``) are detected
        by the autoreload file-watcher without any extra sync step.

        Config files are force-written (overwrite regardless of content) so
        that the autoreload watcher picks up the mtime change on room switch.
        """
        stable = self._ensure_stable_wrapper()

        # Ensure directory skeleton exists (cache handled separately via symlink)
        for d in ("conf/base", "conf/local", "src/viz_wrapper"):
            (stable / d).mkdir(parents=True, exist_ok=True)

        # --- Symlink cache/ to room's cache directory ---
        source_cache = source_wrapper / "cache"
        stable_cache = stable / "cache"

        # Ensure the source cache directory exists
        source_cache.mkdir(parents=True, exist_ok=True)

        # Update symlink if it doesn't point to the right room
        needs_link = True
        if stable_cache.is_symlink():
            try:
                if stable_cache.resolve() == source_cache.resolve():
                    needs_link = False
                else:
                    stable_cache.unlink()
            except (OSError, ValueError):
                stable_cache.unlink(missing_ok=True)
        elif stable_cache.exists():
            # Real directory from a previous run — remove it
            shutil.rmtree(stable_cache)

        if needs_link:
            stable_cache.symlink_to(source_cache.resolve(), target_is_directory=True)

        # --- Symlink .viz/ to room's .viz directory ---
        source_viz = source_wrapper / ".viz"
        stable_viz = stable / ".viz"
        source_viz.mkdir(parents=True, exist_ok=True)

        needs_viz_link = True
        if stable_viz.is_symlink():
            try:
                if stable_viz.resolve() == source_viz.resolve():
                    needs_viz_link = False
                else:
                    stable_viz.unlink()
            except (OSError, ValueError):
                stable_viz.unlink(missing_ok=True)
        elif stable_viz.exists():
            shutil.rmtree(stable_viz)

        if needs_viz_link:
            stable_viz.symlink_to(source_viz.resolve(), target_is_directory=True)

        # --- Copy config files ---
        for rel in self._WRAPPER_FILES:
            src = source_wrapper / rel
            dst = stable / rel
            if src.exists():
                shutil.copy(src, dst)   # copy WITHOUT preserving source mtime
                dst.touch()             # force fresh mtime for watchfiles/inotify

    def switch_project(self, new_wrapper_path: Union[str, Path]):
        """Sync files from a new room's pipeline to trigger autoreload.
        
        Call :meth:`wait_for_switch` afterwards to wait for the server to
        restart with the new data.
        """
        self._sync_wrapper(safe_path(new_wrapper_path))
        self._ready = False

    def trigger_reload(self):
        """Touch a config file in the stable wrapper to trigger autoreload.

        Call this after a pipeline run to ensure kedro-viz picks up
        new output files for previews.
        """
        if self._stable_wrapper and self._stable_wrapper.exists():
            target = self._stable_wrapper / "conf" / "base" / "catalog.yml"
            if target.exists():
                target.touch()

    async def wait_for_switch(self, timeout: float = 20.0) -> bool:
        """Wait for autoreload to restart the server after a switch_project().

        Phase 1: wait for the old child to die (port becomes free).
        Phase 2: wait for the new child to become ready.
        """
        start = time.time()

        # Phase 1 — old child should die within ~2s of file change
        for _ in range(6):
            await asyncio.sleep(0.3)
            if not self._is_port_in_use(self.port):
                break

        # Phase 2 — new child binds and responds
        remaining = max(timeout - (time.time() - start), 5.0)
        return await self.wait_until_ready(timeout=remaining, poll_interval=0.3)

    # -----------------------------------------------------------------

    def start(self, project_path: Union[str, Path]) -> bool:
        """Start the Kedro Viz server.

        Copies the source pipeline into a stable directory and launches
        ``kedro viz run --autoreload`` from there.  Subsequent room switches
        only need :meth:`switch_project` (no process restart from our side).

        Args:
            project_path: Path to the room's pipeline directory.

        Returns:
            True if server started successfully or is already running, False on failure.
        """
        project_path = safe_path(project_path)

        # Sync source wrapper into the stable directory
        self._sync_wrapper(project_path)
        stable = self._ensure_stable_wrapper()

        if self.process:
            # Already running from the stable dir — nothing to do
            if self._project_path == stable:
                self._ready = True
                return True
            # Different stable path (shouldn't happen) — stop first
            logger.info(f"Project path changed, restarting Kedro Viz on port {self.port}")
            self.stop()

        # Kill any orphan process holding the port (non-blocking — just SIGKILL
        # and proceed; the new server will bind with SO_REUSEADDR).
        if self._is_port_in_use(self.port):
            logger.warning(f"Port {self.port} occupied, killing orphan processes...")
            self._kill_process_on_port(self.port)
            # Brief non-blocking wait — just enough for the kernel to clean up
            time.sleep(0.3)

        # Check if kedro viz is available
        if not shutil.which("kedro"):
            print("Kedro CLI not found. Cannot start Kedro Viz.")
            return False

        if not stable.exists():
            print(f"Stable wrapper path {stable} does not exist.")
            return False

        self._project_path = stable

        cmd = [
            "kedro",
            "viz",
            "run",
            "--port",
            str(self.port),
            "--no-browser",
            "--host",
            "0.0.0.0",
            "--include-hooks",
            "--autoreload",
        ]

        try:
            # start_new_session=True creates a process group so stop() can
            # kill the entire tree (kedro + uvicorn workers) at once.
            self.process = subprocess.Popen(
                cmd,
                cwd=str(stable),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # Register cleanup on exit
            if not self._atexit_registered:
                atexit.register(self.stop)
                self._atexit_registered = True
            return True
        except Exception as e:
            logger.error(f"Failed to start Kedro Viz: {e}")
            self.process = None
            return False

    def stop(self):
        if self.process:
            logger.info("Stopping Kedro Viz server...")
            self._ready = False
            pid = self.process.pid
            try:
                # Kill the entire process group (kedro + all child workers)
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                # Process already dead or no group — try direct kill
                try:
                    self.process.kill()
                except Exception:
                    pass
            try:
                self.process.wait(timeout=2)
            except Exception:
                pass
            self.process = None

        # Last resort: kill any remaining process holding the port
        if self._is_port_in_use(self.port):
            self._kill_process_on_port(self.port)

        self._project_path = None

        # Clean up stable wrapper directory
        if self._stable_wrapper and self._stable_wrapper.exists():
            try:
                # Unlink symlinks first to avoid deleting the room's
                # actual data via rmtree
                for link_name in ("cache", ".viz"):
                    link = self._stable_wrapper / link_name
                    if link.is_symlink():
                        link.unlink()
                shutil.rmtree(self._stable_wrapper)
            except Exception:
                pass
            self._stable_wrapper = None

        # Unregister atexit handler to avoid double cleanup
        if self._atexit_registered:
            try:
                atexit.unregister(self.stop)
            except Exception:
                pass
            self._atexit_registered = False

    def get_url(self):
        # Assuming localhost for now, as the browser will access it via port forwarding
        return f"http://localhost:{self.port}"


def generate_dummy_graph(file_paths: list[str]) -> dict:
    """
    Generates a dummy Kedro Viz graph structure for a list of files.
    Each file becomes a data node. No edges or tasks.
    """
    nodes = []

    for fpath in file_paths:
        # Normalize the path to prevent directory traversal
        fname = Path(fpath).name
        # Sanitize the filename to remove any dangerous characters
        fname = sanitize_filename(fname)
        # Create a deterministic but unique-ish ID based on filename
        node_id = f"data_{fname.replace('.', '_').replace('-', '_')}"

        nodes.append(
            {
                "id": node_id,
                "name": fname,
                "tags": [],
                "pipelines": ["choregraph", "__default__"],
                "type": "data",
                "modular_pipelines": None,
                "node_extras": {
                    "stats": {"rows": 0, "columns": 0, "file_size": 0},
                    "styles": None,
                },
                "layer": None,
                "dataset_type": "pandas.csv_dataset.CSVDataset",
            }
        )

    return {
        "data": {
            "nodes": nodes,
            "edges": [],
            "layers": [],
            "tags": [],
            "pipelines": [
                {"id": "__default__", "name": "__default__"},
                {"id": "choregraph", "name": "choregraph"},
            ],
            "modular_pipelines": {
                "__root__": {
                    "id": "__root__",
                    "name": "__root__",
                    "inputs": [],
                    "outputs": [],
                    "children": [],
                }
            },
            "selected_pipeline": "__default__",
        }
    }

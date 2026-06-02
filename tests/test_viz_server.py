# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from choregraph.viz import KedroVizServer


def test_viz_server_get_url_default():
    s = KedroVizServer(port=8123)
    assert s.get_url() == "http://localhost:8123"


def test_viz_server_start_returns_if_port_in_use(monkeypatch, tmp_path: Path):
    s = KedroVizServer(port=8123)

    monkeypatch.setattr(s, "_is_port_in_use", lambda port: True)
    monkeypatch.setattr(s, "_kill_process_on_port", lambda port: None)

    import choregraph.viz as viz_mod

    monkeypatch.setattr(viz_mod.shutil, "which", lambda _: "kedro")
    fake_proc = MagicMock()
    monkeypatch.setattr(viz_mod.subprocess, "Popen", lambda *a, **kw: fake_proc)

    # Port in use only kills orphans, does NOT prevent startup
    try:
        s.start(project_path=tmp_path)
        assert s.process is not None
    finally:
        s.process = None


def test_viz_server_start_no_kedro_cli(monkeypatch, tmp_path: Path):
    s = KedroVizServer(port=8123)

    monkeypatch.setattr(s, "_is_port_in_use", lambda port: False)

    import choregraph.viz as viz_mod

    monkeypatch.setattr(viz_mod.shutil, "which", lambda _: None)

    s.start(project_path=tmp_path)
    assert s.process is None


def test_viz_server_start_missing_project_path(monkeypatch, tmp_path: Path):
    s = KedroVizServer(port=8123)
    monkeypatch.setattr(s, "_is_port_in_use", lambda port: False)

    import choregraph.viz as viz_mod

    monkeypatch.setattr(viz_mod.shutil, "which", lambda _: "kedro")
    fake_proc = MagicMock()
    monkeypatch.setattr(viz_mod.subprocess, "Popen", lambda *a, **kw: fake_proc)

    # _sync_wrapper creates the dir, so process DOES start
    try:
        s.start(project_path=tmp_path / "does-not-exist")
        assert s.process is not None
    finally:
        s.process = None

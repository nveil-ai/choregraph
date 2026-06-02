# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Kedro dataset for EDF/EDF+ files.

Returns a ``pyedflib.EdfReader`` so pipeline nodes can access signals
and annotations using the standard pyedflib API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset
from kedro.io.core import DatasetError


class EDFDataset(AbstractDataset):
    """Read-only Kedro dataset wrapping an EDF/EDF+ file.

    ``_load()`` returns a ``pyedflib.EdfReader`` instance. The caller
    is responsible for closing it (or using it as a context manager
    if pyedflib supports that in the future).
    """

    def __init__(self, filepath: str, metadata: dict[str, Any] | None = None):
        self._filepath = Path(filepath)
        self._metadata = metadata or {}

    def _load(self) -> Any:
        import pyedflib
        return pyedflib.EdfReader(str(self._filepath))

    def _save(self, data: Any) -> None:
        raise DatasetError("EDFDataset is read-only")

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}

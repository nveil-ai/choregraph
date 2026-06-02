# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Optional dependency helpers for choregraph.

Provides a context manager that converts bare ``ImportError`` exceptions
from missing optional packages into user-friendly messages with install
instructions, and a chain-walker used by the Toolkit to intercept pipeline errors.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

# Maps a fragment of the missing module name → the pip command the user needs.
_HINTS: dict[str, str] = {
    "geopandas":     "pip install 'nveil[extra]'",
    "geonamescache": "pip install 'nveil[extra]'",
    "sklearn":       "pip install 'nveil[extra]'",
    "statsmodels":   "pip install 'nveil[extra]'",
    "PIL":           "pip install 'nveil[extra]'",
    "Pillow":        "pip install 'nveil[extra]'",
    "vtk":           "pip install 'nveil[extra]'",
    "vtkmodules":    "pip install 'nveil[extra]'",
    "langdetect":    "pip install 'nveil[extra]'",
    "simplemma":     "pip install 'nveil[extra]'",
    "unidecode":     "pip install 'nveil[extra]'",
    "rapidfuzz":     "pip install 'nveil[extra]'",
    "openpyxl":      "pip install 'choregraph[server]'",
    "pyexcel":       "pip install 'choregraph[server]'",
}


def _friendly(e: ImportError) -> ImportError:
    """Return a new ImportError with a helpful install hint."""
    msg = str(e)
    hint = next(
        (h for pkg, h in _HINTS.items() if pkg in msg),
        "pip install 'nveil[extra]'",
    )
    missing = msg.removeprefix("No module named ").strip("'\"")
    return ImportError(
        f"\n[nveil] Missing optional dependency: {missing}\n"
        f"        Install it with:  {hint}\n"
    )


@contextmanager
def optional_dep() -> Iterator[None]:
    """Context manager: converts ImportError at optional import sites.

    Usage::

        with optional_dep():
            import geopandas as gpd
    """
    try:
        yield
    except ImportError as e:
        raise _friendly(e) from None


def find_import_error(exc: BaseException) -> ImportError | None:
    """Walk the exception cause chain and return the first ImportError found.

    Kedro wraps node exceptions in its own types; this unwraps them so the
    Toolkit can detect a missing-dep failure and avoid pointless retries.
    """
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, ImportError):
            return e
        e = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
    return None

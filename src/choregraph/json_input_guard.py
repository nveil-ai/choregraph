# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Runtime guard for JSON inputs consumed by execute_code nodes.

A JSON input port delivers a raw Python ``list`` or ``dict`` to user code.
The LLM that writes the code sometimes forgets the conversion step
(``pd.DataFrame(...)`` or ``pd.json_normalize(...)``) and indexes the
value directly with a string — e.g. ``df_json['col']`` — which produces
the cryptic CPython error::

    TypeError: list indices must be integers or slices, not str

This module wraps JSON values in lightweight ``list`` / ``dict``
subclasses that detect the misuse and raise a self-explanatory error.
The retry loop in ``planning_transformation_node`` injects the message
into the next prompt, so the second LLM attempt converges on a correct
conversion.
"""
from __future__ import annotations

from typing import Any


_DATAFRAME_ONLY_ATTRS = frozenset({
    "iloc", "loc", "at", "iat",
    "columns", "dtypes", "shape", "index", "values",
    "head", "tail", "describe", "info",
    "groupby", "merge", "join", "pivot", "melt",
    "drop", "rename", "apply", "assign",
    "query", "sort_values", "reset_index", "set_index",
})


def _conversion_hint(name: str) -> str:
    return (
        f"Input '{name}' is a raw JSON value (not a DataFrame). "
        f"Convert it at the top of your code before any pandas-style access:\n"
        f"    df = pd.DataFrame({name})            # flat list of dicts\n"
        f"    df = pd.json_normalize({name})        # nested objects"
    )


class JsonListInput(list):
    """list subclass that flags pandas-style string indexing."""

    __slots__ = ("_port_name",)

    def __init__(self, name: str, data):
        super().__init__(data)
        self._port_name = name

    def __getitem__(self, key):
        if isinstance(key, str):
            raise TypeError(
                f"Cannot index JSON input '{self._port_name}' with string key {key!r}. "
                + _conversion_hint(self._port_name)
            )
        return super().__getitem__(key)

    def __reduce__(self):
        # Preserve picklability if Kedro ever serializes the value.
        return (self.__class__, (self._port_name, list(self)))


class JsonDictInput(dict):
    """dict subclass that flags DataFrame-only attribute access."""

    __slots__ = ("_port_name",)

    def __init__(self, name: str, data):
        super().__init__(data)
        self._port_name = name

    def __getattr__(self, attr):
        if attr in _DATAFRAME_ONLY_ATTRS:
            raise AttributeError(
                f"JSON input '{self._port_name}' is a Python dict, not a DataFrame. "
                + _conversion_hint(self._port_name)
            )
        raise AttributeError(f"'dict' object has no attribute {attr!r}")

    def __reduce__(self):
        return (self.__class__, (self._port_name, dict(self)))


def wrap_json_input(name: str, value: Any) -> Any:
    """Wrap a raw JSON value (``list`` or ``dict``) in a guard proxy.

    Returns *value* unchanged for any other type (including DataFrames,
    numpy arrays, scalars). Idempotent — already-wrapped values are
    returned untouched.
    """
    if isinstance(value, (JsonListInput, JsonDictInput)):
        return value
    # Plain list/dict only — subclasses authored elsewhere keep their behaviour.
    if type(value) is list:
        return JsonListInput(name, value)
    if type(value) is dict:
        return JsonDictInput(name, value)
    return value

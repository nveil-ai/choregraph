# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Automatic dtype inference for DataFrame columns.

Refines pandas ``object`` / ``StringDtype`` columns to their correct dtype
(datetime, numeric, boolean) by sampling values and applying heuristic
detection. Also converts numeric columns that look like Unix millisecond
timestamps to datetime. Designed to be called once from
:meth:`MetadataExtractor.extract` so that downstream VisuSpec classification
and visualization receive accurate types.
"""

import re
import pandas as pd
import numpy as np

# Minimum non-null values required to attempt inference
_MIN_VALUES = 3
# Sample size for heuristic checks
_SAMPLE_SIZE = 50
# Success threshold (fraction of parseable values)
_THRESHOLD = 0.80

_BOOL_VALUES = {"true", "false", "0", "1", "yes", "no"}

# Common string representations of missing values that pandas does NOT
# recognise as null (they survive ``dropna()`` as regular strings).
_NULL_STRINGS = frozenset({
    "none", "nan", "null", "n/a", "na", "#n/a", "#na", "#value!",
    "missing", "undefined", "<na>",
})

# 8-digit numeric strings that could be YYYYMMDD dates
_YYYYMMDD_RE = re.compile(r"^\d{8}$")

# Pattern to extract the first two numeric components of a date string
_DMY_RE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.]")


def _is_string_like(dtype) -> bool:
    """Return True if dtype is object or a pandas StringDtype."""
    if dtype == object:
        return True
    return isinstance(dtype, pd.StringDtype)


def _normalize_null_strings(df: pd.DataFrame, col: str) -> None:
    """Replace common string representations of null with real ``pd.NA``.

    Only touches cells that are currently non-null strings (actual pandas
    nulls are left untouched). Empty / whitespace-only strings are also
    converted.  Operates in-place.
    """
    notna_mask = df[col].notna()
    if not notna_mask.any():
        return
    lowered = df.loc[notna_mask, col].astype(str).str.strip().str.lower()
    is_null_str = lowered.isin(_NULL_STRINGS) | (lowered == "")
    if is_null_str.any():
        df.loc[is_null_str[is_null_str].index, col] = pd.NA


def _is_yyyymmdd_candidate(values: pd.Series) -> bool:
    """Check if numeric-looking values are plausible YYYYMMDD dates."""
    str_vals = values.astype(str).str.strip()
    matches = str_vals.str.match(_YYYYMMDD_RE)
    if matches.sum() < len(values) * _THRESHOLD:
        return False
    # Check that the 8-digit numbers are in a plausible date range
    try:
        nums = pd.to_numeric(str_vals[matches], errors="coerce").dropna()
        if nums.empty:
            return False
        # Year between 1900 and 2100, month 01-12, day 01-31
        years = nums // 10000
        months = (nums % 10000) // 100
        days = nums % 100
        valid = (
            (years >= 1900) & (years <= 2100)
            & (months >= 1) & (months <= 12)
            & (days >= 1) & (days <= 31)
        )
        return valid.mean() >= _THRESHOLD
    except Exception:
        return False


def _looks_numeric(values: pd.Series) -> bool:
    """Return True if >80% of non-null values look like plain numbers."""
    str_vals = values.dropna().astype(str).str.strip()
    numeric = pd.to_numeric(str_vals, errors="coerce")
    return numeric.notna().mean() > _THRESHOLD


def infer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Refine column dtypes to their correct pandas dtype.

    **Pass 1 — object-typed columns**: samples up to 50 non-null values and
    attempts conversion in order: datetime, numeric, boolean.

    **Pass 2 — numeric columns**: detects Unix millisecond timestamps
    (values in the 1e12–1e14 range, i.e. ~2001–5138) and converts to
    datetime.

    Columns are converted in-place; already-typed columns are skipped.

    Args:
        df: DataFrame to refine (mutated in-place).

    Returns:
        The same DataFrame reference (for chaining convenience).
    """
    for col in df.columns:
        if not _is_string_like(df[col].dtype):
            continue

        # Turn "None", "NaN", "null", … into real NA so they don't
        # pollute the sample and drag the success rate below threshold.
        _normalize_null_strings(df, col)

        non_null = df[col].dropna()
        if len(non_null) < _MIN_VALUES:
            continue

        sample = non_null.sample(n=min(_SAMPLE_SIZE, len(non_null)), random_state=0)

        # --- Datetime detection ---
        if _try_datetime(df, col, sample):
            continue

        # --- Numeric detection ---
        if _try_numeric(df, col, sample):
            continue

        # --- Boolean detection ---
        _try_boolean(df, col, non_null)

    # Pass 2: detect unix timestamps (seconds or milliseconds) in numeric columns
    for col in df.columns:
        if _try_unix_timestamp(df, col):
            continue

    return df


def _detect_dayfirst(values: pd.Series) -> bool:
    """Return True if dates appear to use day-first format (DD/MM/...).

    Scans up to 500 values looking for an unambiguous signal: a first
    component > 12 means day-first, a second component > 12 means
    month-first.  Falls back to False (pandas default) when ambiguous.
    """
    str_vals = values.astype(str).str.strip()
    for val in str_vals.head(500):
        m = _DMY_RE.match(val)
        if m:
            first, second = int(m.group(1)), int(m.group(2))
            if first > 12:
                return True
            if second > 12:
                return False
    return False


def _try_datetime(df: pd.DataFrame, col: str, sample: pd.Series) -> bool:
    """Attempt to convert a column to datetime. Returns True if converted."""
    # Pre-filter: if values look mostly numeric, only allow YYYYMMDD pattern
    if _looks_numeric(sample):
        if _is_yyyymmdd_candidate(sample):
            try:
                converted = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
                success = converted.notna().sum() / df[col].notna().sum()
                if success >= _THRESHOLD:
                    df[col] = converted
                    return True
            except Exception:
                pass
        # Numeric-looking but not YYYYMMDD → skip datetime entirely
        return False

    # General datetime parsing.  ``utc=True`` is required on both the
    # sample check and the final conversion — without it, pandas raises
    # ``ValueError: Mixed timezones detected`` on ISO strings carrying a
    # timezone offset (e.g. ``2025-01-28T22:56:46.631+01:00``), which
    # would silently fall through to string.
    try:
        import warnings
        dayfirst = _detect_dayfirst(df[col].dropna())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (UserWarning, FutureWarning))
            coerced = pd.to_datetime(sample, dayfirst=dayfirst, utc=True, errors="coerce")
        # Reject when >= 80% of successful parses land outside the
        # plausible data range.  pandas 3.0 will happily parse strings
        # like ``"09h00"`` as year 0001 AD, which we want to keep as
        # raw strings.  A realistic data column lands between 1900 and
        # 2100.
        ok = coerced.dropna()
        if len(ok) > 0:
            realistic = ((ok.dt.year >= 1900) & (ok.dt.year <= 2100)).mean()
            if realistic < _THRESHOLD:
                return False
        success = coerced.notna().sum() / len(sample)
        if success >= _THRESHOLD:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", (UserWarning, FutureWarning))
                converted = pd.to_datetime(
                    df[col], dayfirst=dayfirst, utc=True, errors="coerce",
                )
            # Normalize to tz-naive (UTC) for downstream consistency
            df[col] = converted.dt.tz_localize(None)
            return True
    except Exception:
        pass
    return False


def _try_numeric(df: pd.DataFrame, col: str, sample: pd.Series) -> bool:
    """Attempt to convert a column to numeric. Returns True if converted."""
    try:
        coerced = pd.to_numeric(sample, errors="coerce")
        success = coerced.notna().sum() / len(sample)
        if success >= _THRESHOLD:
            full = pd.to_numeric(df[col], errors="coerce")
            # Use Int64 if all non-null values are whole numbers
            non_null = full.dropna()
            if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
                df[col] = full.astype("Int64")  # nullable integer
            else:
                df[col] = full
            return True
    except Exception:
        pass
    return False


def _try_boolean(df: pd.DataFrame, col: str, non_null: pd.Series) -> bool:
    """Attempt to convert a column to boolean. Returns True if converted."""
    lowered = non_null.astype(str).str.strip().str.lower()
    if lowered.isin(_BOOL_VALUES).all():
        mapping = {"true": True, "yes": True, "1": True,
                   "false": False, "no": False, "0": False}
        df[col] = df[col].map(lambda v: mapping.get(str(v).strip().lower(), v) if pd.notna(v) else v)
        df[col] = df[col].astype("boolean")  # nullable boolean
        return True
    return False


# Unix timestamp ranges (non-overlapping):
#   seconds:      1e9  – 1e10  (~2001 to ~2286)
#   milliseconds: 1e12 – 1e14  (~2001 to ~5138)
_UNIX_S_MIN = 1_000_000_000       # 1e9
_UNIX_S_MAX = 10_000_000_000      # 1e10
_UNIX_MS_MIN = 1_000_000_000_000  # 1e12
_UNIX_MS_MAX = 100_000_000_000_000  # 1e14


_DATE_TIME_NAME_RE = re.compile(r"date|time|timestamp|datetime", re.IGNORECASE)


def _try_unix_timestamp(df: pd.DataFrame, col: str) -> bool:
    """Convert a numeric column to datetime if its values are unix timestamps.

    Only triggers when the column name contains a date/time keyword
    (case-insensitive: "date", "time", "timestamp", "datetime") AND
    all non-null values fall within a known epoch range.

    Detects both **seconds** (1e9–1e10) and **milliseconds** (1e12–1e14).

    Returns True if converted.
    """
    if not _DATE_TIME_NAME_RE.search(str(col)):
        return False

    if not pd.api.types.is_numeric_dtype(df[col].dtype):
        return False

    non_null = df[col].dropna()
    if len(non_null) < _MIN_VALUES:
        return False

    lo, hi = non_null.min(), non_null.max()

    # Determine unit based on value range
    if lo >= _UNIX_S_MIN and hi < _UNIX_S_MAX:
        unit = "s"
    elif lo >= _UNIX_MS_MIN and hi < _UNIX_MS_MAX:
        unit = "ms"
    else:
        return False

    try:
        df[col] = pd.to_datetime(non_null, unit=unit, errors="coerce")
        return True
    except Exception:
        return False

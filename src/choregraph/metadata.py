# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-FileContributor: Guillaume Franque
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Metadata extraction and caching for Choregraph datasets.
Provides MetadataExtractor for DataFrame analysis and Metadata for persistence.
"""
from dataclasses import dataclass, asdict
from typing import Any, List, Optional, Dict, Tuple, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
from collections import UserDict
import pandas as pd
import numpy as np
import json
import re

from .security import safe_path

if TYPE_CHECKING:
    from .parser import ChoregraphSpec

# Nanosecond multipliers for pandas Timestamp resolutions.
_NS_MULT = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}



def _timestamp_to_ns(ts: pd.Timestamp) -> int:
    """Convert a pandas Timestamp to nanoseconds since epoch.

    Handles pandas 2.x Timestamps that may use second, millisecond,
    or microsecond resolution internally — ``.value`` would overflow
    for dates far from epoch when the native unit is coarser than ns.
    """
    unit = getattr(ts, "unit", "ns")
    raw = ts.asm8.view("i8")  # int64 in the Timestamp's native unit
    return int(raw) * _NS_MULT.get(unit, 1)


def _safe_nunique(series: pd.Series) -> int:
    """Count distinct values, tolerating cells that aren't hashable.

    Flattening JSON inputs commonly produces columns whose cells are
    ``list`` or ``dict`` (e.g. Reddit's ``all_awardings``). Pandas hashes
    values to detect duplicates, so ``nunique()`` raises ``TypeError`` on
    those columns. Falls back to a string representation so the field
    metadata stays populated instead of crashing the pipeline.
    """
    try:
        return series.nunique()
    except TypeError:
        return series.dropna().astype(str).nunique()


def _safe_unique(series: pd.Series) -> list:
    """Return distinct values as a list, tolerating unhashable cells."""
    try:
        return series.unique().tolist()
    except TypeError:
        return series.dropna().astype(str).unique().tolist()


@dataclass
class FieldMetadata:
    """Metadata for a single DataFrame column.

    Attributes:
        id: Sequential field identifier (string).
        name: Column name from the DataFrame.
        data_type: One of INTEGER, FLOAT, DATETIME, STRING, BOOLEAN, OBJECT.
        min_value: Minimum value (numeric/datetime columns only).
        max_value: Maximum value (numeric/datetime columns only).
        is_unique: Whether all values in the column are unique.
        units: Unit label (default ``"UNITLESS"``).
        distinct_count: Number of distinct values (-1 if unknown).
        uniques: Comma-separated string of unique values (categorical fields).
    """
    id: str
    name: str
    data_type: str
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    is_unique: bool = False
    units: str = "UNITLESS"
    distinct_count: int = -1
    uniques: str = ""
    info: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: dict) -> "FieldMetadata":
        min_val = d.get("min_value")
        max_val = d.get("max_value")
        if min_val == "":
            min_val = None
        if max_val == "":
            max_val = None
        return cls(
            id=str(d.get("id", "0")),
            name=d.get("name", "unknown"),
            data_type=d.get("data_type", "STRING"),
            min_value=min_val,
            max_value=max_val,
            is_unique=d.get("is_unique", False),
            units=d.get("units", "UNITLESS"),
            distinct_count=d.get("distinct_count", -1),
            uniques=d.get("uniques", ""),
            info=d.get("info"),
        )


@dataclass
class DatasetStats:
    """Complete stats for a single dataset."""
    id: str
    name: str
    row_count: int
    fields: List[FieldMetadata]
    last_updated: str = ""  # ISO format datetime
    info: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: dict, name: str = None) -> "DatasetStats":
        ds_name = name or d.get("name", "dataset")
        return cls(
            id=str(d.get("id") or ds_name),
            name=ds_name,
            row_count=d.get("row_count", 0),
            fields=[FieldMetadata.from_dict(f) for f in d.get("fields", [])],
            last_updated=d.get("last_updated", ""),
            info=d.get("info"),
        )


class MetadataResult(UserDict):
    """
    Wrapper around the dict of DatasetStats to allow formatting methods.
    Behaves exactly like a Dict[str, DatasetStats], but adds .format().
    """

    def format(self, format_type: str = "markdown", user_message: str = "", detailed: bool = True) -> str:
        """
        Format the metadata collection into a string representation.
        
        Args:
            format_type: "markdown", "json"
            user_message: Filter fields based on user query context.
            detailed: Include all stats columns (min/max/uniques).
        """
        if format_type == "markdown":
            return self._to_markdown(user_message, detailed)
        elif format_type == "json":
             # Use asdict to convert dataclasses
             data_dict = {k: asdict(v) for k, v in self.data.items()}
             return json.dumps(data_dict, default=str, indent=2)
        else:
            raise ValueError(f"Unknown format: {format_type}")



    def to_api_format(self) -> list:
        """Convert to list-of-dicts format for the viz API (metadata.json).

        Returns:
            List of dataset metadata dicts with keys: data_id, name, rows, fields.
        """
        result = []
        for name, stats in self.data.items():
            fields_list = []
            for f in stats.fields:
                field_dict = {
                    "field_id": f.id,
                    "name": f.name,
                    "data_type": f.data_type,
                    "field_min": str(f.min_value) if f.min_value is not None else "0",
                    "field_max": str(f.max_value) if f.max_value is not None else str(f.distinct_count or -1),
                    "distinct_count": str(f.distinct_count or -1)
                }
                if f.uniques:
                    field_dict["uniques"] = f.uniques
                fields_list.append(field_dict)

            result.append({
                "data_id": str(stats.id) if stats.id and str(stats.id) != "0" else name,
                "name": name,
                "rows": str(stats.row_count),
                "fields": fields_list
            })
        return result

    @classmethod
    def from_datasets(cls, datasets: list) -> "MetadataResult":
        """Build a MetadataResult from a list of dataset dicts.

        Uses the same ``DatasetStats.from_dict()`` / ``FieldMetadata.from_dict()``
        deserialization as ``Metadata.read_from_cache()``, so both the
        workspace-based web flow and the stateless API produce identical objects.

        Args:
            datasets: List of dicts with keys matching ``DatasetStats`` fields
                (``id``, ``name``, ``row_count``, ``fields``).
        """
        result = {}
        for ds in datasets:
            stats = DatasetStats.from_dict(ds)
            result[stats.name] = stats
        return cls(result)

    def _to_markdown(self, user_message: str, detailed: bool) -> str:
        MAX_FIELD_NUMBER = 100
        output_lines = []

        target_datasets = self.data.values()


        if not target_datasets:
            return ""

        total_fields = sum(len(d.fields) for d in target_datasets)
        should_filter = total_fields > MAX_FIELD_NUMBER and not detailed and user_message
        user_message_norm = user_message.lower() if should_filter else ""

        for stats in target_datasets:
            header_parts = [f"dataId={stats.id}", stats.name]
            # Detect partitioned dataset via __partition__ virtual field
            partition_field = next((f for f in stats.fields if f.name == "__partition__"), None)
            if partition_field:
                n = partition_field.distinct_count if partition_field.distinct_count > 0 else "?"
                label = partition_field.units if partition_field.units else "partition"
                header_parts.append(f"partitioned: {n} {label}s")
            output_lines.append(f"=== Data ({' ; '.join(header_parts)}) ===")

            if stats.info and stats.info.get("extract_with"):
                output_lines.append(stats.info["extract_with"])

            if partition_field:
                output_lines.append(f"This dataset has {n} partitions indexed by the __partition__ field (represents {label}). Use __partition__ on the time channel to animate across partitions.")

            if detailed:
                output_lines.append("| ID | NAME | TYPE | MIN | MAX | DISTINCT_COUNT | UNIQUES_VALUES |")
                output_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            else:
                output_lines.append("| ID | NAME | TYPE |")
                output_lines.append("| --- | --- | --- |")

            for field in stats.fields:
                field_name = field.name
                
                if should_filter:
                    pattern = r"\\b" + re.escape(field_name.lower()) + r"\\b"
                    if not re.search(pattern, user_message_norm):
                        continue
                
                field_id = field.id
                data_type = field.data_type
                
                if detailed:
                    f_min = field.min_value if field.min_value is not None else ""
                    f_max = field.max_value if field.max_value is not None else ""
                    d_count = field.distinct_count if field.distinct_count != -1 else ""
                    uniques = str(field.uniques)[:150] + "...]" if len(str(field.uniques)) > 150 else str(field.uniques)

                    output_lines.append(f"{field_id} | {field_name} | {data_type} | {f_min} | {f_max} | {d_count} | {uniques}")
                else:
                    output_lines.append(f"| {field_id} | {field_name} | {data_type} |")
            
            output_lines.append("") 
             
        return "\n".join(output_lines).strip()


class MetadataExtractor:
    """
    Analyzes a pandas DataFrame to extract metadata.
    Optimized for performance on large datasets.
    """
    
    LARGE_DATASET_THRESHOLD = 1_000_000
    HIGH_UNIQUE_COUNT_THRESHOLD = 15

    @staticmethod
    def _map_dtype(dtype: np.dtype) -> str:
        """Map a numpy/pandas dtype to a VisuSpec type string.

        Args:
            dtype: The numpy dtype to classify.

        Returns:
            One of ``"INTEGER"``, ``"FLOAT"``, ``"DATETIME"``, ``"STRING"``,
            ``"BOOLEAN"``, or ``"OBJECT"``.
        """
        if pd.api.types.is_integer_dtype(dtype):
            return "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            return "FLOAT"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "DATETIME"
        elif pd.api.types.is_string_dtype(dtype):
            return "STRING"
        elif pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        # elif pd.api.types.is_object_dtype(dtype):
            # return "OBJECT"
        else:
            print(f"[DEBUG] Unknown dtype encountered: {dtype}, defaulting to STRING")
            return "STRING"

    @classmethod
    def extract(cls, df: pd.DataFrame) -> List[FieldMetadata]:
        """Extract field-level metadata from a DataFrame.

        Args:
            df: Input DataFrame to analyze.

        Returns:
            List of :class:`FieldMetadata` objects, one per column.
        """
        from .dtype_inference import infer_dtypes
        infer_dtypes(df)

        metadata = []
        field_id = 1
        row_count = len(df)
        is_large = row_count > cls.LARGE_DATASET_THRESHOLD

        if is_large:
            print(f"[DEBUG] Large dataset detected (>{cls.LARGE_DATASET_THRESHOLD} rows), optimizations enabled")

        for column in df.columns:
            if str(column).startswith("Unnamed:"): #skip unnamed columns
                continue

            series = df[column]
            dtype = cls._map_dtype(series.dtype)
            
            distinct_count = -1
            if not is_large or dtype == "STRING":
                distinct_count = _safe_nunique(series)
            
            field_meta = FieldMetadata(
                id=str(field_id),
                name=str(column),
                data_type=dtype,
                distinct_count=distinct_count
            )

            if dtype in ("INTEGER", "FLOAT"):
                v_min = series.min()
                v_max = series.max()
                if pd.notna(v_min):
                    field_meta.min_value = v_min
                if pd.notna(v_max):
                    field_meta.max_value = v_max
            elif dtype == "DATETIME":
                m = series.min()
                M = series.max()
                if pd.notna(m):
                    field_meta.min_value = _timestamp_to_ns(m)
                if pd.notna(M):
                    field_meta.max_value = _timestamp_to_ns(M)
            elif dtype in ("OBJECT", "STRING"):
                distinct_count = _safe_nunique(series)
                field_meta.distinct_count = distinct_count
                uniques = _safe_unique(series)
                if distinct_count < cls.HIGH_UNIQUE_COUNT_THRESHOLD:
                    uniques_str = str(uniques)
                else:
                    uniques_str = str(uniques[:cls.HIGH_UNIQUE_COUNT_THRESHOLD-3])[:-1] + " ... " + str(uniques[-3:])[1:]
                field_meta.uniques = uniques_str

            if distinct_count == row_count:
                field_meta.is_unique = True
            elif not is_large:
                try:
                    if series.is_unique:
                        field_meta.is_unique = True
                except TypeError:
                    pass

            metadata.append(field_meta)
            field_id += 1

        return metadata


class Metadata:
    """
    Centralized manager for dataset metadata.
    Reads directly from catalogue_stats.json without in-memory caching.
    """

    def __init__(self, workspace_path: Path):
        self.cache_path = workspace_path / "pipeline" / "cache" / "catalogue_stats.json"

    def update_stats(self, name: str, df, dataset_id: str = None, dataset_type: str = "input"):
        """
        Calculate and store stats for a dataset.

        Args:
            name: Dataset name (Kedro catalog key)
            df: The data to analyze (DataFrame, dict, or list)
            dataset_id: Optional spec ID (input ID or output port ID)
            dataset_type: "input" or "output"
        """

        if isinstance(df, pd.DataFrame):
            row_count = len(df)
            fields = MetadataExtractor.extract(df)
            self.store_stats(name, fields, row_count, dataset_id=dataset_id, dataset_type=dataset_type)
        elif isinstance(df, (dict, list)):
            fields, dataset_info = self._describe_json_structure(df)
            carto = dataset_info.pop("_carto", None) if dataset_info else None
            row_count = carto.get("length", 0) if carto else (
                len(df) if isinstance(df, list) else len(df.keys())
            )
            self.store_stats(
                name, fields, row_count,
                dataset_id=dataset_id, dataset_type=dataset_type,
                dataset_info=dataset_info,
            )
        else:
            # Check for PIL Image (loaded by pillow.ImageDataset)
            try:
                from PIL import Image as PILImage
                if isinstance(df, PILImage.Image):
                    fields = self._describe_image(df)
                    self.store_stats(name, fields, 1, dataset_id=dataset_id, dataset_type=dataset_type)
                    return
            except ImportError:
                pass
            return

    @staticmethod
    def _describe_image(image) -> List[FieldMetadata]:
        """Describe a PIL Image using proxy fields matching MHD convention.

        Creates virtual position_x, position_y, and color_value fields so
        that the LLM and ASP can understand the data structure without
        actually loading pixel data.  The renderer loads the image directly.
        """
        import numpy as np
        try:
            w, h = image.size
            arr = np.array(image)
            return [
                FieldMetadata(
                    id="1", name="position_x", data_type="FLOAT",
                    min_value=0.0, max_value=float(w),
                    units="UNITLESS", distinct_count=w,
                ),
                FieldMetadata(
                    id="2", name="position_y", data_type="FLOAT",
                    min_value=0.0, max_value=float(h),
                    units="UNITLESS", distinct_count=h,
                ),
                FieldMetadata(
                    id="3", name="color_value", data_type="FLOAT",
                    min_value=float(arr.min()), max_value=float(arr.max()),
                    units="UNITLESS", distinct_count=int(arr.max() - arr.min() + 1),
                ),
            ]
        except Exception:
            return [
                FieldMetadata(
                    id="1", name="position_x", data_type="FLOAT",
                    min_value=0.0, max_value=1.0,
                    units="UNITLESS", distinct_count=1,
                ),
                FieldMetadata(
                    id="2", name="position_y", data_type="FLOAT",
                    min_value=0.0, max_value=1.0,
                    units="UNITLESS", distinct_count=1,
                ),
                FieldMetadata(
                    id="3", name="color_value", data_type="FLOAT",
                    min_value=0.0, max_value=255.0,
                    units="UNITLESS", distinct_count=256,
                ),
            ]

    @staticmethod
    def _describe_dicom(dicom_path: str) -> List[FieldMetadata]:
        """Describe a DICOM file (or directory of slices) using proxy fields.

        Uses pydicom to read metadata from the first slice without loading
        pixel data.  For a directory, counts .dcm files to determine depth.
        """
        import os
        from pathlib import Path

        p = Path(dicom_path)
        w, h, d = 256, 256, 1
        bits_stored = 16

        try:
            import pydicom

            if p.is_dir():
                dcm_files = sorted(f for f in p.iterdir() if f.suffix.lower() == ".dcm")
                d = max(len(dcm_files), 1)
                first = dcm_files[0] if dcm_files else None
            else:
                first = p
                # Check for sibling .dcm files (DICOM series stored together)
                siblings = [f for f in p.parent.iterdir() if f.suffix.lower() == ".dcm"]
                d = max(len(siblings), 1)

            if first:
                ds = pydicom.dcmread(str(first), stop_before_pixels=True)
                w = int(getattr(ds, "Columns", 256))
                h = int(getattr(ds, "Rows", 256))
                bits_stored = int(getattr(ds, "BitsStored", 16))
                # NumberOfFrames for multi-frame single-file DICOM
                if d == 1:
                    d = int(getattr(ds, "NumberOfFrames", 1))
        except Exception:
            pass

        color_max = float((1 << bits_stored) - 1)
        total_voxels = w * h * d

        return [
            FieldMetadata(
                id="1", name="ID", data_type="INTEGER",
                min_value=1, max_value=total_voxels,
                units="UNITLESS", distinct_count=total_voxels,
            ),
            FieldMetadata(
                id="2", name="position_x", data_type="FLOAT",
                min_value=0.0, max_value=float(w),
                units="mm", distinct_count=w,
            ),
            FieldMetadata(
                id="3", name="position_y", data_type="FLOAT",
                min_value=0.0, max_value=float(h),
                units="mm", distinct_count=h,
            ),
            FieldMetadata(
                id="4", name="position_z", data_type="FLOAT",
                min_value=0.0, max_value=float(d),
                units="mm", distinct_count=d,
            ),
            FieldMetadata(
                id="5", name="color_value", data_type="FLOAT",
                min_value=0.0, max_value=color_max,
                units="UNITLESS", distinct_count=min(int(color_max) + 1, total_voxels),
            ),
        ]

    @staticmethod
    def _describe_mhd(mhd_path: str) -> List[FieldMetadata]:
        """Describe an MHD volume file using proxy fields.

        Parses the MHD text header to extract dimensions and element type,
        then creates proxy fields (position_x/y/z, color_value) with correct
        min/max so the C++ parser does NOT need to call setFieldsFromVolume()
        (which would trigger database summarization on an empty table).
        """
        dim_size = [1, 1, 1]
        element_type = "MET_UCHAR"

        # MHD element type → max color value
        _ELEMENT_MAX = {
            "MET_UCHAR": 255,
            "MET_CHAR": 127,
            "MET_USHORT": 65535,
            "MET_SHORT": 32767,
            "MET_UINT": 4294967295,
            "MET_INT": 2147483647,
            "MET_FLOAT": 1.0,
            "MET_DOUBLE": 1.0,
        }

        try:
            with open(safe_path(mhd_path), "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key == "DimSize":
                        parts = value.split()
                        dim_size = [int(p) for p in parts[:3]]
                        while len(dim_size) < 3:
                            dim_size.append(1)
                    elif key == "ElementType":
                        element_type = value.upper()
        except Exception:
            pass

        w, h, d = dim_size
        color_max = float(_ELEMENT_MAX.get(element_type, 255))
        total_voxels = w * h * d

        return [
            FieldMetadata(
                id="1", name="ID", data_type="INTEGER",
                min_value=1, max_value=total_voxels,
                units="UNITLESS", distinct_count=total_voxels,
            ),
            FieldMetadata(
                id="2", name="position_x", data_type="FLOAT",
                min_value=0.0, max_value=float(w),
                units="mm", distinct_count=w,
            ),
            FieldMetadata(
                id="3", name="position_y", data_type="FLOAT",
                min_value=0.0, max_value=float(h),
                units="mm", distinct_count=h,
            ),
            FieldMetadata(
                id="4", name="position_z", data_type="FLOAT",
                min_value=0.0, max_value=float(d),
                units="mm", distinct_count=d,
            ),
            FieldMetadata(
                id="5", name="color_value", data_type="FLOAT",
                min_value=0.0, max_value=color_max,
                units="UNITLESS", distinct_count=min(int(color_max) + 1, total_voxels),
            ),
        ]

    @staticmethod
    def _describe_json_fallback(data) -> Tuple[List[FieldMetadata], Dict[str, Any]]:
        """Best-effort metadata when :func:`cartograph_json` raises.

        Keeps the ``@`` picker and LLM prompt useful by surfacing top-level
        keys (or list-item keys) as fields, plus a truncated ``json.dumps``
        of the first record as ``extract_with``.
        """
        sample = data[0] if isinstance(data, list) and data else data
        keys = list(sample.keys()) if isinstance(sample, dict) else []
        fields = [
            FieldMetadata(id=str(i + 1), name=k, data_type="STRING")
            for i, k in enumerate(keys)
        ]
        try:
            preview = json.dumps(sample, default=str)[:1500]
        except Exception:
            preview = ""
        return fields, {"extract_with": preview}

    @staticmethod
    def _describe_json_structure(data) -> Tuple[List[FieldMetadata], Dict[str, Any]]:
        """Describe the structure of a JSON object for catalogue_stats.

        Returns ``(fields, dataset_info)``:
        - ``fields``: populated only when the JSON is a flat array of objects
          (or a single-key wrapper around one) so min/max/dtype stay
          exploitable downstream; empty otherwise.
        - ``dataset_info``: ``{"extract_with": <rendered tree>}`` for the
          planning LLM, produced by :func:`cartograph_json` (GenSON-based).
          The raw carto dict is also stashed under ``_carto`` (private key,
          stripped by callers) to avoid recomputing it.
        """
        from .library import cartograph_json, JsonTooDeepError
        try:
            carto = cartograph_json(data)
        except JsonTooDeepError as e:
            print(f"[WARNING] {e} — using fallback JSON description")
            return Metadata._describe_json_fallback(data)
        except Exception as e:
            import traceback
            print(f"[ERROR] cartograph_json failed: {e!r}")
            traceback.print_exc()
            return Metadata._describe_json_fallback(data)
        dataset_info: Dict[str, Any] = {
            "extract_with": carto["rendered"],
            "_carto": carto,
        }
        fields: List[FieldMetadata] = []
        if carto.get("is_tabular"):
            length = carto.get("length", -1)
            for idx, f in enumerate(carto.get("tabular_fields", [])):
                fields.append(FieldMetadata(
                    id=str(idx + 1),
                    name=f["name"],
                    data_type=str(f.get("dtype", "STRING")).upper(),
                    distinct_count=length,
                ))
        else:
            # Non-tabular JSON: surface leaf paths so the frontend '@' picker
            # and downstream planners can reference them. Stats are unknown
            # (no sampling of leaf values here), so min/max/distinct stay blank.
            for idx, f in enumerate(carto.get("leaf_fields", [])):
                fields.append(FieldMetadata(
                    id=str(idx + 1),
                    name=f["name"],
                    data_type=str(f.get("dtype", "STRING")).upper(),
                ))
        return fields, dataset_info

    def store_stats(self, name: str, fields: List[FieldMetadata], row_count: int, dataset_id: str = None, dataset_type: str = "input", dataset_info: Optional[Dict[str, Any]] = None):
        """
        Store pre-extracted stats for a dataset directly to JSON file.

        Args:
            name: Dataset name (Kedro catalog key)
            fields: Pre-extracted field metadata list
            row_count: Number of rows in the dataset
            dataset_id: Optional XML ID of the dataset
            dataset_type: "input" or "output"
            dataset_info: Optional structural description (e.g. JSON cartography)
                stored under the ``info`` key and rendered by
                :meth:`MetadataResult._to_markdown` via ``info["extract_with"]``.
        """
        # Load existing data from JSON
        existing_data = {"datasets": {}, "last_pipeline_run": ""}
        if self.cache_path.exists():
            try:
                with open(safe_path(self.cache_path), "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception as e:
                print(f"[WARNING] Warning: could not read existing cache: {e}")

        # Ensure expected keys exist (handles legacy files initialized with "{}")
        existing_data.setdefault("datasets", {})
        existing_data.setdefault("last_pipeline_run", "")

        # Convert fields to serializable format
        fields_data = []
        for f in fields:
            field_dict = {
                "id": f.id,
                "name": f.name,
                "data_type": f.data_type,
                "min_value": f.min_value,
                "max_value": f.max_value,
                "is_unique": f.is_unique,
                "distinct_count": f.distinct_count,
                "uniques": f.uniques
            }
            fields_data.append(field_dict)
        
        # Remove any existing entry with the same dataset_id (under a different name)
        # to avoid duplicate IDs confusing downstream consumers
        if dataset_id is not None:
            to_remove = [
                existing_name for existing_name, existing_entry in existing_data["datasets"].items()
                if existing_entry.get("id") == dataset_id and existing_name != name
            ]
            for existing_name in to_remove:
                del existing_data["datasets"][existing_name]

        # Add/update the dataset entry
        entry: Dict[str, Any] = {
            "id": dataset_id,
            "type": dataset_type,
            "row_count": row_count,
            "fields": fields_data,
            "last_updated": datetime.now().isoformat()
        }
        if dataset_info:
            entry["info"] = dataset_info
        existing_data["datasets"][name] = entry
        existing_data["last_pipeline_run"] = datetime.now().isoformat()
        
        # Write back to JSON
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(safe_path(self.cache_path), "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, default=str)
        except Exception as e:
            print(f"[ERROR] ERROR writing stats for '{name}': {e}")
            raise

    
    def write_raw_cache(self, json_string: str) -> None:
        """Write a raw JSON string directly to catalogue_stats.json.

        Used by the API flow: the Toolkit sends the pre-built catalogue_stats
        and the server writes it as-is.
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(safe_path(self.cache_path), "w", encoding="utf-8") as f:
            f.write(json_string)

    def read_from_cache(self, dataset_ids: Optional[List[str]] = None) -> MetadataResult:
        """
        Load stats directly from catalogue_stats.json.

        Args:
            dataset_ids: If provided, only retrieves metadata for these specific dataset IDs.
                         Accepts a single string or a list of strings.
                         If None, retrieves all datasets.

        Returns:
            MetadataResult (smart dict of dataset name -> DatasetStats)
        """
        if isinstance(dataset_ids, str):
            dataset_ids = [dataset_ids]

        
        if not self.cache_path.exists():
            print(f"[WARNING] Cache file does not exist, returning empty result")
            return MetadataResult({})
        
        try:
            with open(safe_path(self.cache_path), "r", encoding="utf-8") as f:
                data = json.load(f)
            
            datasets = data.get("datasets", {})
            result = {}

            for name, entry in datasets.items():
                entry_id = str(entry.get("id") or "0")
                if dataset_ids is not None and entry_id not in dataset_ids:
                    continue
                result[name] = DatasetStats.from_dict(entry, name=name)
            
            return MetadataResult(result)
            
        except Exception as e:
            print(f"[ERROR] ERROR reading catalogue_stats.json: {e}")
            return MetadataResult({})

    def clear(self):
        """Clear the JSON file on disk."""
        
        if self.cache_path.exists():
            try:
                self.cache_path.unlink()
            except Exception as e:
                print(f"[DEBUG] Error deleting cache file: {e}")
        else:
            print(f"[DEBUG] Cache file does not exist, nothing to clear")

    def get(self, name: str) -> Optional[DatasetStats]:
        """Get stats for a specific dataset directly from JSON."""
        result = self.read_from_cache()
        return result.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if a dataset exists in the JSON file."""
        return self.get(name) is not None

    def __len__(self) -> int:
        """Return the number of datasets in the JSON file."""
        all_datasets = self.read_from_cache()
        return len(all_datasets)

    def remove_datasets(self, names: List[str]) -> int:
        """Remove datasets from catalogue_stats.json by name.

        Args:
            names: Dataset names (filename stems) to remove.

        Returns:
            Number of datasets actually removed.
        """
        if not self.cache_path.exists() or not names:
            return 0
        try:
            with open(safe_path(self.cache_path), "r", encoding="utf-8") as f:
                catalogue = json.load(f)
            removed = 0
            datasets = catalogue.get("datasets", {})
            for name in names:
                if datasets.pop(name, None) is not None:
                    removed += 1
            if removed:
                with open(safe_path(self.cache_path), "w", encoding="utf-8") as f:
                    json.dump(catalogue, f, indent=2, default=str, ensure_ascii=False)
            return removed
        except Exception:
            return 0

    def add_partition_field(self, dataset_name: str, n_partitions: int,
                           partition_label: str = "partition"):
        """Add virtual ``__partition__`` field to a partitioned dataset's metadata.

        The field doesn't exist in the actual data files — it represents
        the index of each partition (file) in the dataset.

        Args:
            dataset_name: Name of the dataset in the catalogue.
            n_partitions: Number of partitions.
            partition_label: Semantic label (e.g. "time", "sheet", "slice").
        """
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                catalogue = json.load(f)
            ds = catalogue.get("datasets", {}).get(dataset_name)
            if not ds:
                return
            fields = ds.setdefault("fields", [])
            if any(f.get("name") == "__partition__" for f in fields):
                return
            next_id = str(max((int(f.get("id", 0)) for f in fields), default=0) + 1)
            fields.append({
                "id": next_id, "name": "__partition__", "data_type": "FLOAT",
                "min_value": 0.0, "max_value": float(n_partitions - 1), "distinct_count": n_partitions,
                "units": partition_label,
            })
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(catalogue, f, indent=2, default=str, ensure_ascii=False)
        except Exception:
            pass

    def merge_datasets(self, entries: Dict[str, dict]) -> int:
        """Merge pre-computed dataset entries into catalogue_stats.json.

        Each entry should follow the catalogue_stats schema::

            {
                "row_count": int,
                "fields": [{"id", "name", "data_type", ...}],
                "type": "input",
                ...
            }

        Args:
            entries: Dict of dataset_name -> stats dict.

        Returns:
            Number of datasets merged.
        """
        if not entries:
            return 0
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue = {"datasets": {}, "last_pipeline_run": ""}
        if self.cache_path.exists():
            try:
                with open(safe_path(self.cache_path), "r", encoding="utf-8") as f:
                    catalogue = json.load(f)
            except Exception:
                pass
        catalogue.setdefault("datasets", {})
        merged = 0
        for name, entry in entries.items():
            # Validate minimal structure
            if "fields" in entry:
                catalogue["datasets"][name] = entry
                merged += 1
        if merged:
            with open(safe_path(self.cache_path), "w", encoding="utf-8") as f:
                json.dump(catalogue, f, indent=2, default=str, ensure_ascii=False)
        return merged


def _read_tabular(file_path: str, nrows: Optional[int] = 50000):
    """Read a tabular file (CSV/TSV/Parquet) into a DataFrame.

    Returns ``None`` if the file cannot be read.
    """
    import os
    import pandas as _pd

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in (".csv", ".tsv"):
            from .loaders import characterize_csv
            csv_char = characterize_csv(file_path)
            read_kwargs: dict = {}
            if nrows is not None:
                read_kwargs["nrows"] = nrows
            if csv_char.get("fieldSeparator"):
                read_kwargs["sep"] = csv_char["fieldSeparator"]
            if csv_char.get("header") is False:
                read_kwargs["header"] = None
            return _pd.read_csv(file_path, **read_kwargs)
        elif ext == ".parquet":
            return _pd.read_parquet(file_path)
    except Exception:
        return None
    return None


def _describe_edf(edf_path: str) -> tuple:
    """Describe an EDF/EDF+ file from its header (no signal data loaded).

    Returns:
        Tuple of (fields, dataset_info) where fields is a list of
        FieldMetadata (one per signal channel) and dataset_info is a dict
        with recording-level metadata. dataset_info includes a
        ``_total_samples`` key (sum of all channel samples) used as
        row_count by compute_file_stats, removed before serialization.
    """
    import pyedflib

    reader = pyedflib.EdfReader(str(edf_path))
    try:
        n_signals = reader.signals_in_file
        labels = reader.getSignalLabels()
        start_dt = reader.getStartdatetime()
        duration = reader.getFileDuration()
        file_type_code = reader.filetype

        file_type_map = {0: "EDF", 1: "EDF+C", 2: "EDF+D", 3: "BDF", 4: "BDF+C", 5: "BDF+D"}
        file_type = file_type_map.get(file_type_code, "EDF")

        annotations = reader.readAnnotations()
        annotation_count = len(annotations[0]) if annotations and len(annotations[0]) > 0 else 0

        fields = []
        total_samples = 0
        seen_labels: dict = {}
        n_samples_arr = reader.getNSamples()

        for i in range(n_signals):
            label = labels[i].strip()
            if label == "EDF Annotations":
                continue

            if label in seen_labels:
                seen_labels[label] += 1
                label = f"{label}_{seen_labels[label]}"
            else:
                seen_labels[label] = 0

            sample_rate = reader.getSampleFrequency(i)
            phys_min = reader.getPhysicalMinimum(i)
            phys_max = reader.getPhysicalMaximum(i)
            unit = reader.getPhysicalDimension(i).strip()
            transducer = reader.getTransducer(i).strip()
            prefilter = reader.getPrefilter(i).strip()
            n_samples = n_samples_arr[i]
            total_samples += n_samples

            channel_info = {"sampling_rate_hz": sample_rate, "channel_index": i}
            if transducer:
                channel_info["transducer"] = transducer
            if prefilter:
                channel_info["prefiltering"] = prefilter

            fields.append(FieldMetadata(
                id=str(i + 1),
                name=label,
                data_type="FLOAT",
                min_value=phys_min,
                max_value=phys_max,
                units=unit or "UNITLESS",
                distinct_count=int(n_samples),
                info=channel_info,
            ))

        dataset_info = {
            "reader": "pyedflib.EdfReader",
            "extract_with": "This is an EDF file. The input variable is a pyedflib.EdfReader object, NOT a DataFrame. Use it directly: input_var.getSignalLabels(), input_var.readSignal(channel_index), input_var.getSampleFrequency(channel_index), input_var.getNSamples()[channel_index]. For partial reads use input_var.readSignal(channel_index, start_sample, n_samples) to load only the needed window instead of the full channel. Build a DataFrame from the extracted signals and assign to result.",
            "recording_start": start_dt.isoformat(),
            "duration_seconds": duration,
            "file_type": file_type,
            "annotation_count": annotation_count,
            "_total_samples": total_samples,
        }

        patient_info = getattr(reader, "getPatientName", lambda: "")()
        if patient_info:
            dataset_info["patient_info"] = patient_info

        return fields, dataset_info
    finally:
        reader.close()


def compute_file_stats(file_paths: "str | List[str]") -> Optional[dict]:
    """Compute metadata for one or more data files.

    Accepts a single path or a list of paths.  When multiple paths are
    given (e.g. all CSVs in a temporal group), tabular files are
    aggregated so that min/max/distinct reflect the full range across
    the group.  Non-tabular formats only use the first path.

    Supports CSV, TSV, Parquet, JSON, images (PNG/JPG/TIFF/BMP/WEBP/GIF),
    and MHD volumes. Returns a stats dict in the same format as
    ``catalogue_stats.json`` dataset entries, or ``None`` if the file type
    is unsupported.

    This is a standalone function — no Kedro, no workspace, no DB needed.

    Args:
        file_paths: Absolute path to a file, or a list of paths to
            aggregate over.

    Returns:
        Dict with ``row_count``, ``fields`` list, ``type``, ``last_updated``
        keys, or ``None`` if unsupported.
    """
    import os
    import pandas as _pd

    paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
    if not paths:
        return None

    primary = paths[0]
    ext = os.path.splitext(primary)[1].lower()

    fields: Optional[List[FieldMetadata]] = None
    row_count = 0
    dataset_info: Optional[Dict[str, Any]] = None

    if ext in (".csv", ".tsv", ".parquet"):
        try:
            dfs = []
            for p in paths:
                df = _read_tabular(p)
                if df is not None:
                    dfs.append(df)
            if not dfs:
                return None
            df = _pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
            fields = MetadataExtractor.extract(df)
            row_count = len(df)
        except Exception:
            return None

    elif ext == ".json":
        try:
            with open(primary, "r", encoding="utf-8") as jf:
                json_data = json.load(jf)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARNING] Failed to parse JSON {primary}: {e}")
            return None
        fields, dataset_info = Metadata._describe_json_structure(json_data)
        carto = dataset_info.pop("_carto", None) if dataset_info else None
        row_count = carto.get("length", 0) if carto else (
            len(json_data) if isinstance(json_data, list)
            else len(json_data.keys()) if isinstance(json_data, dict) else 0
        )

    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"):
        try:
            from PIL import Image as PILImage
            img = PILImage.open(primary)
            fields = Metadata._describe_image(img)
            row_count = 1
        except Exception:
            return None

    elif ext == ".mhd":
        try:
            fields = Metadata._describe_mhd(primary)
            row_count = 1
        except Exception:
            return None

    elif ext == ".dcm":
        try:
            fields = Metadata._describe_dicom(primary)
            row_count = 1
        except Exception:
            return None

    elif ext == ".edf":
        try:
            fields, dataset_info = _describe_edf(primary)
            row_count = dataset_info.pop("_total_samples", 0)
        except Exception:
            return None

    else:
        return None

    if fields is None:
        return None

    result = {
        "row_count": row_count,
        "fields": [
            {
                "id": f.id,
                "name": f.name,
                "data_type": f.data_type,
                "min_value": f.min_value,
                "max_value": f.max_value,
                "is_unique": f.is_unique,
                "units": f.units,
                "distinct_count": f.distinct_count,
                "uniques": f.uniques,
                **({"info": f.info} if f.info else {}),
            }
            for f in fields
        ],
        "type": "input",
        "last_updated": datetime.now().isoformat(),
    }
    if dataset_info:
        result["info"] = dataset_info
    return result
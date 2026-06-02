# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pandas as pd

from choregraph.metadata import MetadataExtractor


def test_metadata_extract_maps_types_and_stats():
    df = pd.DataFrame(
        {
            "i": [1, 2, 3],
            "f": [1.0, 2.5, 3.5],
            "s": ["a", "b", "b"],
            "t": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02"]),
        }
    )

    metas = MetadataExtractor.extract(df)
    by_name = {m.name: m for m in metas}

    assert by_name["i"].data_type == "INTEGER"
    assert by_name["i"].min_value == 1
    assert by_name["i"].max_value == 3

    assert by_name["f"].data_type == "FLOAT"
    assert by_name["f"].min_value == 1.0
    assert by_name["f"].max_value == 3.5

    assert by_name["s"].data_type == "STRING"
    assert by_name["s"].distinct_count == 2

    assert by_name["t"].data_type == "DATETIME"
    # DATETIME min/max are int64 nanoseconds since epoch.
    assert isinstance(by_name["t"].min_value, int)
    assert isinstance(by_name["t"].max_value, int)
    # 2020-01-01 == 1577836800 seconds == 1577836800000000000 ns
    assert by_name["t"].min_value == 1577836800000000000


def test_metadata_skips_unnamed_columns():
    df = pd.DataFrame({"Unnamed: 0": [1, 2], "real": [3, 4]})
    metas = MetadataExtractor.extract(df)
    assert [m.name for m in metas] == ["real"]


def test_metadata_large_dataset_threshold_skips_numeric_distinct(monkeypatch):
    # Make the threshold tiny so we can test the large-path without huge data.
    monkeypatch.setattr(MetadataExtractor, "LARGE_DATASET_THRESHOLD", 3)

    df = pd.DataFrame(
        {
            "num": [1, 2, 2, 3],
            "cat": ["a", "b", "b", "c"],
        }
    )

    metas = MetadataExtractor.extract(df)
    by_name = {m.name: m for m in metas}

    # For large datasets numeric distinct_count is skipped
    assert by_name["num"].distinct_count == -1
    # But strings still compute distinct_count
    assert by_name["cat"].distinct_count == 3


# ---------------------------------------------------------------------------
# Tests for Metadata persistence (store / read / clear)
# ---------------------------------------------------------------------------

from choregraph.metadata import (
    Metadata,
    DatasetStats,
    FieldMetadata,
    MetadataResult,
    compute_file_stats,
)


def test_metadata_store_and_read_back(tmp_path):
    meta = Metadata(tmp_path)
    fields = [
        FieldMetadata(id="1", name="col_a", data_type="INTEGER", min_value=0, max_value=100),
        FieldMetadata(id="2", name="col_b", data_type="STRING", distinct_count=5),
    ]
    meta.store_stats("test_dataset", fields, row_count=50, dataset_id="ds1")

    result = meta.read_from_cache()
    assert "test_dataset" in result
    stats = result["test_dataset"]
    assert stats.row_count == 50
    assert len(stats.fields) == 2
    assert stats.fields[0].name == "col_a"
    assert stats.fields[0].min_value == 0
    assert stats.fields[1].name == "col_b"


def test_metadata_read_by_dataset_id(tmp_path):
    meta = Metadata(tmp_path)
    fields_a = [FieldMetadata(id="1", name="x", data_type="FLOAT")]
    fields_b = [FieldMetadata(id="1", name="y", data_type="INTEGER")]
    meta.store_stats("alpha", fields_a, row_count=10, dataset_id="id_a")
    meta.store_stats("beta", fields_b, row_count=20, dataset_id="id_b")

    filtered = meta.read_from_cache(dataset_ids=["id_a"])
    assert "alpha" in filtered
    assert "beta" not in filtered


def test_metadata_clear(tmp_path):
    meta = Metadata(tmp_path)
    fields = [FieldMetadata(id="1", name="x", data_type="FLOAT")]
    meta.store_stats("ds", fields, row_count=5)
    assert len(meta) == 1

    meta.clear()
    assert len(meta) == 0


def test_metadata_remove_datasets(tmp_path):
    meta = Metadata(tmp_path)
    for name in ("a", "b", "c"):
        meta.store_stats(name, [FieldMetadata(id="1", name="x", data_type="FLOAT")], row_count=1)

    removed = meta.remove_datasets(["a", "c"])
    assert removed == 2
    result = meta.read_from_cache()
    assert "a" not in result
    assert "b" in result
    assert "c" not in result


def test_metadata_contains_and_len(tmp_path):
    meta = Metadata(tmp_path)
    meta.store_stats("ds1", [FieldMetadata(id="1", name="x", data_type="FLOAT")], row_count=1)
    meta.store_stats("ds2", [FieldMetadata(id="1", name="y", data_type="INTEGER")], row_count=2)

    assert "ds1" in meta
    assert "ds3" not in meta
    assert len(meta) == 2


def test_metadata_get_returns_none_for_missing(tmp_path):
    meta = Metadata(tmp_path)
    assert meta.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Tests for DatasetStats construction
# ---------------------------------------------------------------------------


def test_dataset_stats_from_dict():
    d = {
        "id": "42",
        "row_count": 100,
        "fields": [
            {"id": "1", "name": "col1", "data_type": "INTEGER", "min_value": 0, "max_value": 99},
            {"id": "2", "name": "col2", "data_type": "STRING", "distinct_count": 10, "uniques": "['a','b']"},
        ],
        "last_updated": "2024-01-01T00:00:00",
    }
    stats = DatasetStats.from_dict(d, name="my_data")
    assert stats.name == "my_data"
    assert stats.id == "42"
    assert stats.row_count == 100
    assert len(stats.fields) == 2
    assert stats.fields[0].min_value == 0
    assert stats.fields[1].uniques == "['a','b']"


def test_dataset_stats_from_dict_defaults():
    stats = DatasetStats.from_dict({})
    assert stats.row_count == 0
    assert stats.fields == []
    assert stats.name == "dataset"


def test_field_metadata_from_dict_empty_minmax():
    fm = FieldMetadata.from_dict({"id": "1", "name": "x", "data_type": "FLOAT", "min_value": "", "max_value": ""})
    assert fm.min_value is None
    assert fm.max_value is None


# ---------------------------------------------------------------------------
# Tests for MetadataResult
# ---------------------------------------------------------------------------


def test_metadata_result_from_datasets():
    datasets = [
        {"id": "1", "name": "ds_one", "row_count": 10, "fields": [
            {"id": "1", "name": "a", "data_type": "INTEGER"}
        ]},
        {"id": "2", "name": "ds_two", "row_count": 20, "fields": []},
    ]
    mr = MetadataResult.from_datasets(datasets)
    assert "ds_one" in mr
    assert "ds_two" in mr
    assert mr["ds_one"].row_count == 10


def test_metadata_result_format_json():
    stats = DatasetStats(id="1", name="test", row_count=5, fields=[
        FieldMetadata(id="1", name="x", data_type="FLOAT", min_value=0.0, max_value=1.0),
    ])
    mr = MetadataResult({"test": stats})
    import json
    output = mr.format("json")
    parsed = json.loads(output)
    assert "test" in parsed
    assert parsed["test"]["row_count"] == 5


# ---------------------------------------------------------------------------
# Tests for compute_file_stats
# ---------------------------------------------------------------------------


def test_compute_file_stats_single_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,value\nAlice,10\nBob,20\nCharlie,30\n", encoding="utf-8")

    result = compute_file_stats(str(csv_path))
    assert result is not None
    assert result["row_count"] == 3
    assert len(result["fields"]) == 2
    field_names = [f["name"] for f in result["fields"]]
    assert "name" in field_names
    assert "value" in field_names


def test_compute_file_stats_multiple_csvs(tmp_path):
    p1 = tmp_path / "part1.csv"
    p2 = tmp_path / "part2.csv"
    p1.write_text("x,y\n1,10\n2,20\n", encoding="utf-8")
    p2.write_text("x,y\n3,30\n4,40\n", encoding="utf-8")

    result = compute_file_stats([str(p1), str(p2)])
    assert result is not None
    assert result["row_count"] == 4


def test_compute_file_stats_unsupported_extension(tmp_path):
    txt = tmp_path / "data.xyz"
    txt.write_text("hello", encoding="utf-8")
    assert compute_file_stats(str(txt)) is None


def test_compute_file_stats_null_and_mixed_types(tmp_path):
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text("a,b\n1,hello\n,world\n3,\n", encoding="utf-8")

    result = compute_file_stats(str(csv_path))
    assert result is not None
    # Row count may vary depending on header detection; just verify it parsed
    assert result["row_count"] >= 3
    assert len(result["fields"]) == 2

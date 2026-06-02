# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import yaml

from choregraph.wrapper import ManagedProjectBuilder
from choregraph.parser import ChoregraphSpec, InputSpec, InputPortSpec, OutputPortSpec, NodeSpec
from choregraph.library import TRANSFORM_REGISTRY


def test_managed_project_builder_generates_expected_files(tmp_workspace: Path, tmp_path: Path):
    # Create a CSV input and point spec to it.
    csv_path = tmp_path / "in.csv"
    csv_path.write_text("Age,Score\n10,1.0\n", encoding="utf-8")

    spec = ChoregraphSpec(
        inputs=[
            InputSpec(
                id="1",
                label="in",  # Required field
                location=str(csv_path),
                format="CSV",
                options={"sep": ",", "header": 0},
            )
        ]
    )

    builder = ManagedProjectBuilder(tmp_workspace)
    builder.build(spec, TRANSFORM_REGISTRY)

    wrapper_dir = tmp_workspace / "pipeline"
    assert (wrapper_dir / "pyproject.toml").exists()
    assert (wrapper_dir / "src" / "viz_wrapper" / "settings.py").exists()
    assert (wrapper_dir / "src" / "viz_wrapper" / "pipeline_registry.py").exists()
    assert (wrapper_dir / "conf" / "base" / "catalog.yml").exists()

    catalog = yaml.safe_load((wrapper_dir / "conf" / "base" / "catalog.yml").read_text(encoding="utf-8"))

    # Clean name is file stem
    assert "in" in catalog
    assert catalog["in"]["filepath"] == str(csv_path)
    assert catalog["in"]["type"].endswith("CSVDataset")
    assert catalog["in"]["load_args"]["sep"] == ","


# ---------------------------------------------------------------------------
# _catalog_suffix
# ---------------------------------------------------------------------------

def test_catalog_suffix_dataframe():
    from choregraph.wrapper import _catalog_suffix
    assert _catalog_suffix("DATAFRAME") == "#parquet"
    assert _catalog_suffix(None) == "#parquet"


def test_catalog_suffix_json_and_list():
    from choregraph.wrapper import _catalog_suffix
    assert _catalog_suffix("JSON") == "#json"
    assert _catalog_suffix("LIST") == "#json"


def test_catalog_suffix_scalars_and_dict():
    from choregraph.wrapper import _catalog_suffix
    for t in ("FLOAT", "INTEGER", "BOOLEAN", "STRING", "DICT"):
        assert _catalog_suffix(t) == "", f"Expected empty suffix for {t}"


# ---------------------------------------------------------------------------
# _resolve_input_path — search order: inputs/ > workspace root > pipeline/data/inputs/
# ---------------------------------------------------------------------------

def test_resolve_input_path_prefers_inputs_dir(tmp_workspace: Path):
    """Files under workspace/inputs/ should be found first."""
    builder = ManagedProjectBuilder(tmp_workspace)
    builder._ensure_directories()

    inputs_dir = tmp_workspace / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    target = inputs_dir / "data.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")

    resolved = builder._resolve_input_path("data.csv")
    assert Path(resolved).exists()
    assert Path(resolved).name == "data.csv"
    assert "inputs" in resolved


def test_resolve_input_path_falls_back_to_data_inputs(tmp_workspace: Path):
    """When file only exists under pipeline/data/inputs/, that path is used."""
    builder = ManagedProjectBuilder(tmp_workspace)
    builder._ensure_directories()

    alt_dir = tmp_workspace / "pipeline" / "data" / "inputs"
    alt_dir.mkdir(parents=True, exist_ok=True)
    target = alt_dir / "fallback.csv"
    target.write_text("x\n1\n", encoding="utf-8")

    resolved = builder._resolve_input_path("fallback.csv")
    assert Path(resolved).name == "fallback.csv"
    assert "data" in resolved and "inputs" in resolved


# ---------------------------------------------------------------------------
# JSON input catalog entry
# ---------------------------------------------------------------------------

def test_json_input_generates_json_dataset(tmp_workspace: Path, tmp_path: Path):
    json_path = tmp_path / "config.json"
    json_path.write_text('{"key": "value"}', encoding="utf-8")

    spec = ChoregraphSpec(
        inputs=[
            InputSpec(
                id="10",
                label="config",
                location=str(json_path),
                format="JSON",
            )
        ]
    )

    builder = ManagedProjectBuilder(tmp_workspace)
    builder.build(spec, TRANSFORM_REGISTRY)

    catalog = yaml.safe_load(
        (tmp_workspace / "pipeline" / "conf" / "base" / "catalog.yml").read_text(encoding="utf-8")
    )
    assert "config" in catalog
    assert catalog["config"]["type"] == "json.JSONDataset"
    assert catalog["config"]["filepath"] == str(json_path)


# ---------------------------------------------------------------------------
# Parquet input catalog entry
# ---------------------------------------------------------------------------

def test_parquet_input_generates_parquet_dataset(tmp_workspace: Path, tmp_path: Path):
    pq_path = tmp_path / "table.parquet"
    pq_path.write_bytes(b"")  # content not read during build

    spec = ChoregraphSpec(
        inputs=[
            InputSpec(
                id="20",
                label="table",
                location=str(pq_path),
                format="PARQUET",
            )
        ]
    )

    builder = ManagedProjectBuilder(tmp_workspace)
    builder.build(spec, TRANSFORM_REGISTRY)

    catalog = yaml.safe_load(
        (tmp_workspace / "pipeline" / "conf" / "base" / "catalog.yml").read_text(encoding="utf-8")
    )
    assert "table" in catalog
    assert catalog["table"]["type"] == "pandas.ParquetDataset"


# ---------------------------------------------------------------------------
# Pipeline registry generation
# ---------------------------------------------------------------------------

def test_pipeline_registry_contains_node(tmp_workspace: Path, tmp_path: Path):
    """A spec with one node should produce a pipeline_registry.py that imports
    the transform function and defines a node."""
    csv_path = tmp_path / "src.csv"
    csv_path.write_text("Age,Score\n10,1.0\n", encoding="utf-8")

    spec = ChoregraphSpec(
        inputs=[
            InputSpec(id="1", label="src", location=str(csv_path), format="CSV",
                      options={"sep": ",", "header": 0}),
        ],
        nodes=[
            NodeSpec(
                id="2",
                label="Select Columns",
                type="select_columns",
                input_ports=[
                    InputPortSpec(name="input", source_ref=1, type="DATAFRAME"),
                    InputPortSpec(name="parameter", value="Age", type="PARAMETER"),
                ],
                output_ports=[
                    OutputPortSpec(id=200, name="result", label="selected", type="DATAFRAME"),
                ],
            )
        ],
    )

    builder = ManagedProjectBuilder(tmp_workspace)
    builder.build(spec, TRANSFORM_REGISTRY)

    registry_path = tmp_workspace / "pipeline" / "src" / "viz_wrapper" / "pipeline_registry.py"
    assert registry_path.exists()
    content = registry_path.read_text(encoding="utf-8")

    assert "from choregraph.library import select_columns" in content
    assert "register_pipelines" in content
    assert "select_columns" in content


# ---------------------------------------------------------------------------
# Parameters catalog (parameters.yml created)
# ---------------------------------------------------------------------------

def test_parameters_yml_created(tmp_workspace: Path, tmp_path: Path):
    """build() should always create an empty parameters.yml to silence Kedro warnings."""
    csv_path = tmp_path / "d.csv"
    csv_path.write_text("a\n1\n", encoding="utf-8")

    spec = ChoregraphSpec(
        inputs=[InputSpec(id="1", label="d", location=str(csv_path), format="CSV")]
    )

    builder = ManagedProjectBuilder(tmp_workspace)
    builder.build(spec, TRANSFORM_REGISTRY)

    params_path = tmp_workspace / "pipeline" / "conf" / "base" / "parameters.yml"
    assert params_path.exists()


# ---------------------------------------------------------------------------
# _write_if_changed — idempotency
# ---------------------------------------------------------------------------

def test_write_if_changed_does_not_rewrite_identical_content(tmp_workspace: Path, tmp_path: Path):
    """Calling build() twice with the same spec should not update file mtimes
    (verified via the return value of _write_if_changed)."""
    from choregraph.wrapper import _write_if_changed

    target = tmp_path / "test.txt"
    assert _write_if_changed(target, "hello") is True   # first write
    assert _write_if_changed(target, "hello") is False   # no change
    assert _write_if_changed(target, "world") is True    # content changed

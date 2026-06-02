# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for choregraph.file_builder module."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import pytest

from choregraph.file_builder import (
    _stable_input_id,
    build_choregraph_inputs,
    count_choregraph_inputs,
    create_specifications_xml,
    extract_datasets_metadata,
    remove_choregraph_inputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    """Return a temporary workspace directory path as a string."""
    return str(tmp_path)


@pytest.fixture
def sample_csv(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("name,age,score\nAlice,30,85.5\nBob,25,92.1\nCharlie,35,78.3\n")
    return csv


@pytest.fixture
def second_csv(tmp_path):
    csv = tmp_path / "other.csv"
    csv.write_text("city,population\nParis,2161000\nLyon,516092\n")
    return csv


@pytest.fixture
def specs_xml_with_data(tmp_path):
    """Create a specifications.xml containing rawData elements for metadata extraction."""
    xml_path = tmp_path / "specifications.xml"
    xml_path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<visuSpec name="Test">\n'
        '  <datas>\n'
        '    <rawData id="1" name="cities" rows="3">\n'
        '      <file location="/data/cities.csv" />\n'
        '      <fields>\n'
        '        <field id="1" name="city" dataType="STRING" fieldMin="" fieldMax="" distinctCount="3" />\n'
        '        <field id="2" name="pop" dataType="NUMERIC" fieldMin="100" fieldMax="9999" distinctCount="3" />\n'
        '      </fields>\n'
        '    </rawData>\n'
        '    <rawData id="2" name="temps" rows="5">\n'
        '      <file location="/data/temps.csv" />\n'
        '      <fields>\n'
        '        <field id="3" name="date" dataType="DATE" fieldMin="" fieldMax="" distinctCount="5" />\n'
        '      </fields>\n'
        '    </rawData>\n'
        '  </datas>\n'
        '</visuSpec>\n'
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Tests for _stable_input_id
# ---------------------------------------------------------------------------

class TestStableInputId:
    def test_deterministic(self):
        """Same stem always produces the same ID."""
        assert _stable_input_id("hello") == _stable_input_id("hello")

    def test_different_stems_differ(self):
        """Different stems produce different IDs."""
        assert _stable_input_id("alpha") != _stable_input_id("beta")

    def test_returns_string_of_digits(self):
        id_val = _stable_input_id("test_file")
        assert id_val.isdigit()

    def test_range_1000_to_9999(self):
        """ID is always in the 4-digit range 1000-9999."""
        for stem in ["a", "file", "long_name_here", "x" * 200, ""]:
            val = int(_stable_input_id(stem))
            assert 1000 <= val <= 9999, f"ID {val} out of range for stem={stem!r}"


# ---------------------------------------------------------------------------
# Tests for build_choregraph_inputs
# ---------------------------------------------------------------------------

class TestBuildChoregraphInputs:
    def test_creates_choregraph_xml(self, workspace, sample_csv):
        """Building inputs creates a choregraph.xml file."""
        build_choregraph_inputs(workspace, [str(sample_csv)])
        assert os.path.isfile(os.path.join(workspace, "choregraph.xml"))

    def test_xml_has_input_element(self, workspace, sample_csv):
        """The generated XML contains an <input> with the correct location."""
        build_choregraph_inputs(workspace, [str(sample_csv)])
        tree = ET.parse(os.path.join(workspace, "choregraph.xml"))
        inputs = tree.getroot().findall(".//input")
        assert len(inputs) == 1
        assert inputs[0].get("location") == str(sample_csv)
        assert inputs[0].get("format") == "CSV"

    def test_multiple_files(self, workspace, sample_csv, second_csv):
        """Multiple file paths produce multiple input elements."""
        build_choregraph_inputs(workspace, [str(sample_csv), str(second_csv)])
        tree = ET.parse(os.path.join(workspace, "choregraph.xml"))
        inputs = tree.getroot().findall(".//input")
        assert len(inputs) == 2

    def test_idempotent_same_files(self, workspace, sample_csv):
        """Calling twice with existing_xml set does not duplicate inputs."""
        build_choregraph_inputs(workspace, [str(sample_csv)])
        choregraph_xml = os.path.join(workspace, "choregraph.xml")
        # Second call: pass existing_xml so it loads the previous file
        build_choregraph_inputs(
            workspace, [str(sample_csv)], existing_xml=choregraph_xml
        )
        tree = ET.parse(choregraph_xml)
        inputs = tree.getroot().findall(".//input")
        assert len(inputs) == 1

    def test_appends_new_files(self, workspace, sample_csv, second_csv):
        """Calling with existing_xml and a new file appends without losing existing."""
        build_choregraph_inputs(workspace, [str(sample_csv)])
        choregraph_xml = os.path.join(workspace, "choregraph.xml")
        build_choregraph_inputs(
            workspace, [str(second_csv)], existing_xml=choregraph_xml
        )
        tree = ET.parse(choregraph_xml)
        inputs = tree.getroot().findall(".//input")
        assert len(inputs) == 2
        locations = {inp.get("location") for inp in inputs}
        assert str(sample_csv) in locations
        assert str(second_csv) in locations

    def test_csv_options_populated(self, workspace, sample_csv):
        """CSV inputs should have header/fieldSeparator/skipLines options."""
        build_choregraph_inputs(workspace, [str(sample_csv)])
        tree = ET.parse(os.path.join(workspace, "choregraph.xml"))
        inp = tree.getroot().find(".//input")
        assert inp.get("header") is not None
        assert inp.get("fieldSeparator") is not None

    def test_none_paths_skipped(self, workspace, sample_csv):
        """None entries in file_paths are silently ignored."""
        build_choregraph_inputs(workspace, [None, str(sample_csv), None])
        tree = ET.parse(os.path.join(workspace, "choregraph.xml"))
        inputs = tree.getroot().findall(".//input")
        assert len(inputs) == 1


# ---------------------------------------------------------------------------
# Tests for remove_choregraph_inputs
# ---------------------------------------------------------------------------

class TestRemoveChoregraphInputs:
    def test_removes_specified_input(self, workspace, sample_csv, second_csv):
        """Removing a file drops its input from the XML."""
        build_choregraph_inputs(workspace, [str(sample_csv), str(second_csv)])
        removed = remove_choregraph_inputs(workspace, [sample_csv.name])
        assert len(removed) == 1

        tree = ET.parse(os.path.join(workspace, "choregraph.xml"))
        inputs = tree.getroot().findall(".//input")
        assert len(inputs) == 1
        assert inputs[0].get("location") == str(second_csv)

    def test_preserves_remaining_inputs(self, workspace, sample_csv, second_csv):
        """After removing one input, the other remains intact."""
        build_choregraph_inputs(workspace, [str(sample_csv), str(second_csv)])
        remove_choregraph_inputs(workspace, [second_csv.name])
        tree = ET.parse(os.path.join(workspace, "choregraph.xml"))
        inputs = tree.getroot().findall(".//input")
        assert len(inputs) == 1
        assert inputs[0].get("location") == str(sample_csv)

    def test_remove_nonexistent_no_error(self, workspace, sample_csv):
        """Removing a filename that is not in the XML does not raise."""
        build_choregraph_inputs(workspace, [str(sample_csv)])
        removed = remove_choregraph_inputs(workspace, ["nonexistent.csv"])
        assert removed == []

    def test_remove_no_xml_file(self, workspace):
        """Removing from a workspace with no choregraph.xml returns empty list."""
        removed = remove_choregraph_inputs(workspace, ["anything.csv"])
        assert removed == []


# ---------------------------------------------------------------------------
# Tests for count_choregraph_inputs
# ---------------------------------------------------------------------------

class TestCountChoregraphInputs:
    def test_count_after_build(self, workspace, sample_csv, second_csv):
        build_choregraph_inputs(workspace, [str(sample_csv), str(second_csv)])
        assert count_choregraph_inputs(workspace) == 2

    def test_count_missing_xml(self, workspace):
        assert count_choregraph_inputs(workspace) == 0


# ---------------------------------------------------------------------------
# Tests for create_specifications_xml
# ---------------------------------------------------------------------------

class TestCreateSpecificationsXml:
    def test_creates_file(self, workspace):
        create_specifications_xml(workspace)
        assert os.path.isfile(os.path.join(workspace, "specifications.xml"))

    def test_valid_xml_structure(self, workspace):
        create_specifications_xml(workspace)
        tree = ET.parse(os.path.join(workspace, "specifications.xml"))
        root = tree.getroot()
        assert root.tag == "visuSpec"
        assert root.get("name") == "UserFile"
        assert root.find("coordinates") is not None
        assert root.find("coordinates").text == "CARTESIAN"
        assert root.find("datas") is not None
        assert root.find("colorPalettes") is not None
        assert root.find("shapePalettes") is not None
        assert root.find("channels") is not None
        assert root.find("marks") is not None
        assert root.find("space") is not None


# ---------------------------------------------------------------------------
# Tests for extract_datasets_metadata
# ---------------------------------------------------------------------------

class TestExtractDatasetsMetadata:
    def test_returns_datasets(self, specs_xml_with_data):
        datasets = extract_datasets_metadata(str(specs_xml_with_data))
        assert len(datasets) == 2

    def test_dataset_fields(self, specs_xml_with_data):
        datasets = extract_datasets_metadata(str(specs_xml_with_data))
        cities = next(d for d in datasets if d["name"] == "cities")
        assert cities["data_id"] == "1"
        assert cities["rows"] == "3"
        assert cities["filename"] == "cities.csv"
        assert len(cities["fields"]) == 2
        assert cities["fields"][0]["name"] == "city"
        assert cities["fields"][1]["data_type"] == "NUMERIC"

    def test_missing_specifications_xml(self, workspace):
        """Returns empty list when specifications.xml does not exist."""
        datasets = extract_datasets_metadata(workspace)
        assert datasets == []

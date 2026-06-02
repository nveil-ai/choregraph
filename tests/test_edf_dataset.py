# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for EDF/EDF+ dataset and metadata extraction."""
from __future__ import annotations

import pytest
from pathlib import Path

pyedflib = pytest.importorskip("pyedflib")


def _make_edf(path: Path, channels: list[dict], duration_sec: int = 10,
              annotations: list[tuple] | None = None, edf_plus: bool = True):
    """Create a synthetic EDF/EDF+ file for testing.

    Args:
        path: Output file path.
        channels: List of dicts with keys: label, rate_hz, unit, phys_min, phys_max.
        duration_sec: Recording duration in seconds.
        annotations: List of (onset_sec, duration_sec, text) tuples.
        edf_plus: Write EDF+ (True) or plain EDF (False).
    """
    import numpy as np

    file_type = pyedflib.FILETYPE_EDFPLUS if edf_plus else pyedflib.FILETYPE_EDF
    writer = pyedflib.EdfWriter(str(path), len(channels), file_type=file_type)
    try:
        for i, ch in enumerate(channels):
            writer.setSignalHeader(i, {
                "label": ch["label"],
                "dimension": ch.get("unit", "uV"),
                "sample_frequency": ch["rate_hz"],
                "physical_min": ch.get("phys_min", -1000.0),
                "physical_max": ch.get("phys_max", 1000.0),
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": ch.get("transducer", ""),
                "prefilter": ch.get("prefilter", ""),
            })

        if annotations:
            for onset, dur, text in annotations:
                writer.writeAnnotation(onset, dur, text)

        for _ in range(duration_sec):
            signals = []
            for ch in channels:
                n = ch["rate_hz"]
                signal = np.random.uniform(
                    ch.get("phys_min", -1000.0),
                    ch.get("phys_max", 1000.0),
                    n,
                )
                signals.append(signal)
            writer.writeSamples(signals)
    finally:
        writer.close()


@pytest.fixture
def simple_edf(tmp_path):
    path = tmp_path / "test.edf"
    channels = [
        {"label": "EEG Fp1", "rate_hz": 256, "unit": "uV", "phys_min": -3200.0, "phys_max": 3200.0},
        {"label": "EEG Fp2", "rate_hz": 256, "unit": "uV", "phys_min": -3200.0, "phys_max": 3200.0},
    ]
    _make_edf(path, channels, duration_sec=5)
    return path


@pytest.fixture
def multi_rate_edf(tmp_path):
    path = tmp_path / "multi_rate.edf"
    channels = [
        {"label": "EEG Fp1", "rate_hz": 256, "unit": "uV", "phys_min": -3200.0, "phys_max": 3200.0},
        {"label": "ECG", "rate_hz": 512, "unit": "mV", "phys_min": -5.0, "phys_max": 5.0},
        {"label": "Resp", "rate_hz": 1, "unit": "BPM", "phys_min": 0.0, "phys_max": 100.0},
    ]
    _make_edf(path, channels, duration_sec=5)
    return path


@pytest.fixture
def annotated_edf(tmp_path):
    path = tmp_path / "annotated.edf"
    channels = [
        {"label": "EEG Fp1", "rate_hz": 256, "unit": "uV"},
    ]
    annotations = [
        (0.0, 0.0, "Recording start"),
        (2.5, 1.0, "Event A"),
        (4.0, 0.0, "Event B"),
    ]
    _make_edf(path, channels, duration_sec=5, annotations=annotations)
    return path


class TestEDFDataset:
    def test_load_returns_reader(self, simple_edf):
        from choregraph.datasets.edf import EDFDataset
        ds = EDFDataset(filepath=str(simple_edf))
        reader = ds._load()
        try:
            assert reader.signals_in_file >= 2
            labels = reader.getSignalLabels()
            assert "EEG Fp1" in labels[0]
        finally:
            reader.close()

    def test_save_raises(self, simple_edf):
        from choregraph.datasets.edf import EDFDataset
        from kedro.io.core import DatasetError
        ds = EDFDataset(filepath=str(simple_edf))
        with pytest.raises(DatasetError):
            ds._save(None)

    def test_describe(self, simple_edf):
        from choregraph.datasets.edf import EDFDataset
        ds = EDFDataset(filepath=str(simple_edf))
        desc = ds._describe()
        assert "filepath" in desc

    def test_read_signal(self, simple_edf):
        from choregraph.datasets.edf import EDFDataset
        ds = EDFDataset(filepath=str(simple_edf))
        reader = ds._load()
        try:
            signal = reader.readSignal(0)
            assert len(signal) == 256 * 5
        finally:
            reader.close()


class TestDescribeEdf:
    def test_basic_fields(self, simple_edf):
        from choregraph.metadata import _describe_edf
        fields, info = _describe_edf(str(simple_edf))
        assert len(fields) == 2
        assert fields[0].name == "EEG Fp1"
        assert fields[0].units == "uV"
        assert fields[0].data_type == "FLOAT"
        assert fields[0].info["sampling_rate_hz"] == 256
        assert fields[0].info["channel_index"] == 0
        assert info["duration_seconds"] == 5

    def test_multi_rate(self, multi_rate_edf):
        from choregraph.metadata import _describe_edf
        fields, info = _describe_edf(str(multi_rate_edf))
        assert len(fields) == 3
        rates = {f.name: f.info["sampling_rate_hz"] for f in fields}
        assert rates["EEG Fp1"] == 256
        assert rates["ECG"] == 512
        assert rates["Resp"] == 1
        units = {f.name: f.units for f in fields}
        assert units["ECG"] == "mV"
        assert units["Resp"] == "BPM"

    def test_annotations_counted(self, annotated_edf):
        from choregraph.metadata import _describe_edf
        fields, info = _describe_edf(str(annotated_edf))
        assert info["annotation_count"] >= 3

    def test_distinct_count_is_sample_count(self, simple_edf):
        from choregraph.metadata import _describe_edf
        fields, _ = _describe_edf(str(simple_edf))
        assert fields[0].distinct_count == 256 * 5


class TestComputeFileStatsEdf:
    def test_returns_stats(self, simple_edf):
        from choregraph.metadata import compute_file_stats
        result = compute_file_stats(str(simple_edf))
        assert result is not None
        assert result["row_count"] > 0
        assert len(result["fields"]) == 2
        assert result["fields"][0]["info"]["sampling_rate_hz"] == 256
        assert "info" in result
        assert "duration_seconds" in result["info"]

    def test_units_preserved(self, multi_rate_edf):
        from choregraph.metadata import compute_file_stats
        result = compute_file_stats(str(multi_rate_edf))
        units_by_name = {f["name"]: f.get("units", "UNITLESS") for f in result["fields"]}
        assert units_by_name["ECG"] == "mV"

    def test_no_annotation_channel_in_fields(self, annotated_edf):
        from choregraph.metadata import compute_file_stats
        result = compute_file_stats(str(annotated_edf))
        names = [f["name"] for f in result["fields"]]
        assert "EDF Annotations" not in names


class TestDuplicateLabels:
    def test_dedup(self, tmp_path):
        path = tmp_path / "dup.edf"
        channels = [
            {"label": "EEG", "rate_hz": 256, "unit": "uV"},
            {"label": "EEG", "rate_hz": 256, "unit": "uV"},
        ]
        _make_edf(path, channels, duration_sec=2)
        from choregraph.metadata import _describe_edf
        fields, _ = _describe_edf(str(path))
        names = [f.name for f in fields]
        assert len(set(names)) == len(names)

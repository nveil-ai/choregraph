# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the URL fetcher module."""

from __future__ import annotations

import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from unittest.mock import patch

import pytest

from choregraph.fetcher import (
    _filename_from_content_disposition,
    _filename_from_url,
    _infer_extension,
    fetch_inputs,
    fetch_url,
)
from choregraph.parser import InputSpec


# ---------------------------------------------------------------------------
# Helper: lightweight HTTP server for tests
# ---------------------------------------------------------------------------

class _CSVHandler(SimpleHTTPRequestHandler):
    """Serve a small CSV payload from any GET request."""

    CSV_BODY = b"name,value\nalice,10\nbob,20\n"

    def do_GET(self):
        self.send_response(200)
        if self.path.endswith(".json"):
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'[{"a":1}]')
        else:
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", 'attachment; filename="served.csv"')
            self.end_headers()
            self.wfile.write(self.CSV_BODY)

    def log_message(self, *args):
        pass  # silence logs


@pytest.fixture(scope="module")
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _CSVHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestFilenameFromContentDisposition:
    def test_quoted(self):
        assert _filename_from_content_disposition('attachment; filename="data.csv"') == "data.csv"

    def test_unquoted(self):
        assert _filename_from_content_disposition("attachment; filename=report.xlsx") == "report.xlsx"

    def test_utf8(self):
        assert _filename_from_content_disposition("attachment; filename*=UTF-8''donn%C3%A9es.csv") == "données.csv"

    def test_none(self):
        assert _filename_from_content_disposition(None) is None
        assert _filename_from_content_disposition("") is None


class TestFilenameFromUrl:
    def test_simple(self):
        assert _filename_from_url("https://example.com/data/stocks.csv") == "stocks.csv"

    def test_no_extension(self):
        assert _filename_from_url("https://api.example.com/data") is None

    def test_trailing_slash(self):
        assert _filename_from_url("https://example.com/files/report.json/") == "report.json"


class TestInferExtension:
    def test_csv(self):
        assert _infer_extension("text/csv") == ".csv"
        assert _infer_extension("text/csv; charset=utf-8") == ".csv"

    def test_json(self):
        assert _infer_extension("application/json") == ".json"

    def test_default(self):
        assert _infer_extension(None) == ".csv"
        assert _infer_extension("application/octet-stream") == ".csv"


# ---------------------------------------------------------------------------
# Integration tests: fetch_url
# ---------------------------------------------------------------------------


class TestFetchUrl:
    def test_download_csv(self, http_server, tmp_path):
        path, fmt = fetch_url(f"{http_server}/data.csv", tmp_path)
        assert path.exists()
        assert fmt == "CSV"
        content = path.read_text()
        assert "alice" in content

    def test_filename_from_content_disposition(self, http_server, tmp_path):
        path, fmt = fetch_url(f"{http_server}/anything", tmp_path)
        # Content-Disposition says "served.csv"
        assert path.name == "served.csv"

    def test_custom_filename(self, http_server, tmp_path):
        path, fmt = fetch_url(f"{http_server}/data.csv", tmp_path, filename="custom.csv")
        assert path.name == "custom.csv"

    def test_size_limit(self, http_server, tmp_path):
        with patch("choregraph.fetcher.MAX_DOWNLOAD_SIZE", 10):
            with pytest.raises(ValueError, match="limit"):
                fetch_url(f"{http_server}/data.csv", tmp_path)


# ---------------------------------------------------------------------------
# Integration tests: fetch_inputs
# ---------------------------------------------------------------------------


class TestFetchInputs:
    def test_fetches_url_inputs(self, http_server, tmp_path):
        inputs = [
            InputSpec(id="1", label="test", location="", format="CSV", url=f"{http_server}/data.csv"),
            InputSpec(id="2", label="local", location="/some/path.csv", format="CSV"),
        ]
        count = fetch_inputs(inputs, tmp_path)
        assert count == 1
        # location should be updated
        assert inputs[0].location != ""
        assert Path(inputs[0].location).exists()
        # non-URL input unchanged
        assert inputs[1].location == "/some/path.csv"

    def test_skips_non_url(self, tmp_path):
        inputs = [
            InputSpec(id="1", label="local", location="/path.csv", format="CSV"),
        ]
        count = fetch_inputs(inputs, tmp_path)
        assert count == 0


# ---------------------------------------------------------------------------
# XML round-trip: url attribute
# ---------------------------------------------------------------------------


class TestInputSpecUrlXml:
    def test_url_round_trip(self, tmp_path):
        from choregraph import Choregraph

        cg = Choregraph(workspace_path=tmp_path)
        cg.add_input(id="1", location="/tmp/data.csv", format="CSV", url="https://example.com/data.csv", visibility=True)

        xml_path = tmp_path / "choregraph.xml"
        cg.export_to_xml(xml_path)

        # Re-parse
        from choregraph.parser import ChoregraphSpecParser
        spec = ChoregraphSpecParser.parse(xml_path)
        assert len(spec.inputs) == 1
        assert spec.inputs[0].url == "https://example.com/data.csv"
        assert spec.inputs[0].location == "/tmp/data.csv"
        cg.close()

    def test_no_url_attribute(self, tmp_path):
        from choregraph import Choregraph

        cg = Choregraph(workspace_path=tmp_path)
        cg.add_input(id="1", location="/tmp/data.csv", format="CSV", visibility=True)

        xml_path = tmp_path / "choregraph.xml"
        cg.export_to_xml(xml_path)

        from choregraph.parser import ChoregraphSpecParser
        spec = ChoregraphSpecParser.parse(xml_path)
        assert spec.inputs[0].url is None
        cg.close()

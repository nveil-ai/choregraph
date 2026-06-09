# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import csv
from pathlib import Path

from choregraph.loaders import sniff_csv_options, prepare_load_args


def test_sniff_csv_options_detects_sep_and_header(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")

    opts = sniff_csv_options(str(csv_path))
    # csv.Sniffer is heuristic; delimiter should be detected for this simple case
    assert opts.get("sep") == ";"
    # header detection should report header present
    assert opts.get("header") == 0


def test_sniff_csv_options_nonexistent_file_returns_empty(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    assert sniff_csv_options(str(missing)) == {}


def test_prepare_load_args_respects_existing_options(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    args = prepare_load_args("CSV", str(csv_path), options={"sep": "|", "header": None})
    assert args["sep"] == "|"
    assert args["header"] is None


def test_prepare_load_args_sniffs_when_sep_missing(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    args = prepare_load_args("CSV", str(csv_path), options={})
    assert args.get("sep") == ","
    # header is commonly detected
    assert "header" in args


# ---------------------------------------------------------------------------
# Tests for split_csv_line
# ---------------------------------------------------------------------------

from choregraph.loaders import split_csv_line, detect_skip_lines_smart, characterize_csv


def test_split_csv_line_comma():
    assert split_csv_line("a,b,c", ",") == ["a", "b", "c"]


def test_split_csv_line_tab():
    assert split_csv_line("x\ty\tz", "\t") == ["x", "y", "z"]


def test_split_csv_line_semicolon():
    assert split_csv_line("1;2;3", ";") == ["1", "2", "3"]


def test_split_csv_line_quoted_field():
    assert split_csv_line('"hello, world",b,c', ",") == ["hello, world", "b", "c"]


# ---------------------------------------------------------------------------
# Tests for detect_skip_lines_smart
# ---------------------------------------------------------------------------


def test_detect_skip_lines_smart_no_preamble():
    lines = ["a,b,c\n", "1,2,3\n", "4,5,6\n", "7,8,9\n"]
    assert detect_skip_lines_smart(lines, [","]) == 0


def test_detect_skip_lines_smart_with_comment_lines():
    # Comment lines contain commas too, so the function sees them as valid CSV.
    # The header "a,b,c" has 3 fields; comment lines have only 1 field (no commas
    # splitting them meaningfully). The function finds the first consistent run.
    lines = [
        "# This is a comment\n",
        "# Another comment\n",
        "a,b,c\n",
        "1,2,3\n",
        "4,5,6\n",
        "7,8,9\n",
        "10,11,12\n",
    ]
    # Comment lines happen to contain commas (in "This is a comment"), so
    # detect_skip_lines_smart may count them as 1-field rows depending on
    # parsing. The important thing: returned skip >= 2 (skips comments).
    result = detect_skip_lines_smart(lines, [","])
    assert result >= 2


def test_detect_skip_lines_smart_semicolon():
    # "header info" has no semicolons so it's a 1-field row.
    # The function finds where consistent multi-field rows begin.
    lines = [
        "header info\n",
        "a;b;c\n",
        "1;2;3\n",
        "4;5;6\n",
        "7;8;9\n",
        "10;11;12\n",
        "13;14;15\n",
    ]
    result = detect_skip_lines_smart(lines, [";"])
    assert result >= 1


def test_detect_skip_lines_smart_tab_separated():
    lines = [
        "a\tb\tc\n",
        "1\t2\t3\n",
        "4\t5\t6\n",
        "7\t8\t9\n",
    ]
    assert detect_skip_lines_smart(lines, ["\t"]) == 0


# ---------------------------------------------------------------------------
# Tests for characterize_csv
# ---------------------------------------------------------------------------


def test_characterize_csv_comma_separated(tmp_path: Path):
    csv_path = tmp_path / "comma.csv"
    csv_path.write_text("name,age,city\nAlice,30,Paris\nBob,25,London\n", encoding="utf-8")

    result = characterize_csv(str(csv_path))
    assert result["fieldSeparator"] == ","
    assert result["header"] is True
    assert result["skipLines"] == 0


def test_characterize_csv_tab_separated(tmp_path: Path):
    csv_path = tmp_path / "tabs.csv"
    csv_path.write_text("name\tage\tcity\nAlice\t30\tParis\nBob\t25\tLondon\n", encoding="utf-8")

    result = characterize_csv(str(csv_path))
    assert result["fieldSeparator"] == "\t"
    assert result["header"] is True


def test_characterize_csv_semicolon_separated(tmp_path: Path):
    csv_path = tmp_path / "semi.csv"
    csv_path.write_text("name;age;city\nAlice;30;Paris\nBob;25;London\n", encoding="utf-8")

    result = characterize_csv(str(csv_path))
    assert result["fieldSeparator"] == ";"
    assert result["header"] is True


def test_characterize_csv_with_skip_lines(tmp_path: Path):
    csv_path = tmp_path / "skip.csv"
    content = "# comment line 1\n# comment line 2\nname,age\nAlice,30\nBob,25\nCharlie,35\nDave,40\n"
    csv_path.write_text(content, encoding="utf-8")

    result = characterize_csv(str(csv_path))
    # characterize_csv detects preamble and removes it (modified=True, skipLines reset to 0)
    assert result["fieldSeparator"] == ","
    # After skip-line removal the file starts at data; header detection is heuristic
    assert result["skipLines"] == 0


def test_characterize_csv_nonexistent_file(tmp_path: Path):
    result = characterize_csv(str(tmp_path / "nope.csv"))
    # Should return fallback
    assert result["header"] is True
    assert result["fieldSeparator"] == ","


# ---------------------------------------------------------------------------
# Tests for LLM fallback in characterize_csv
# ---------------------------------------------------------------------------

import pytest
from unittest.mock import patch, MagicMock
from choregraph.loaders import _llm_characterize_csv, set_csv_llm_delegate


@pytest.fixture(autouse=True)
def _reset_csv_llm_delegate():
    """Ensure no registered CSV LLM delegate leaks between tests.

    characterize_csv delegates its LLM step to whatever was registered via
    set_csv_llm_delegate (file_service registers an ai_service HTTP call).
    Tests register a MagicMock instead; reset it around every test.
    """
    set_csv_llm_delegate(None)
    yield
    set_csv_llm_delegate(None)


def test_llm_fallback_called_when_sniffer_fails(tmp_path: Path):
    """When csv.Sniffer cannot detect the dialect, the LLM fallback is tried."""
    # Create a file with a tilde separator — Sniffer restricted to usual_del will fail.
    csv_path = tmp_path / "tilde.csv"
    csv_path.write_text("name~age~city\nAlice~30~Paris\nBob~25~London\n", encoding="utf-8")

    mock_delegate = MagicMock(return_value={
        "header": True,
        "fieldSeparator": "~",
        "recordSeparator": "\n",
        "skipLines": 0,
        "modified": False,
    })
    set_csv_llm_delegate(mock_delegate)
    result = characterize_csv(str(csv_path))

    mock_delegate.assert_called_once()
    assert result["fieldSeparator"] == "~"
    assert result["header"] is True


def test_llm_fallback_called_when_unusual_separator_detected(tmp_path: Path):
    """When Sniffer succeeds but detects a non-usual separator, the LLM is consulted."""
    # csv.Sniffer may detect '#' if we don't restrict to usual_del, but in the
    # code we DO restrict, so Sniffer would fail. Let's simulate via patching.
    csv_path = tmp_path / "unusual.csv"
    csv_path.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")

    # Patch Sniffer to return a non-usual delimiter to exercise the unusual-sep branch.
    original_sniff = csv.Sniffer.sniff

    def patched_sniff(self, sample, delimiters=None):
        dialect = original_sniff(self, sample, delimiters=[",", ";", "\t", "|", "~"])
        dialect.delimiter = "~"
        return dialect

    mock_delegate = MagicMock(return_value={
        "header": True,
        "fieldSeparator": ",",
        "recordSeparator": "\n",
        "skipLines": 0,
        "modified": False,
    })
    with patch.object(csv.Sniffer, "sniff", patched_sniff):
        set_csv_llm_delegate(mock_delegate)
        result = characterize_csv(str(csv_path))

    mock_delegate.assert_called_once()
    assert result["fieldSeparator"] == ","


def test_safe_fallback_when_both_sniffer_and_llm_fail(tmp_path: Path):
    """When Sniffer fails and the LLM also fails, safe defaults are returned."""
    csv_path = tmp_path / "weird.csv"
    csv_path.write_text("name~age~city\nAlice~30~Paris\nBob~25~London\n", encoding="utf-8")

    mock_delegate = MagicMock(return_value=None)
    set_csv_llm_delegate(mock_delegate)
    result = characterize_csv(str(csv_path))

    mock_delegate.assert_called_once()
    # Should return safe fallback values
    assert result["header"] is True
    assert result["fieldSeparator"] == ","
    assert result["recordSeparator"] == "\n"


def test_llm_not_called_for_usual_separator(tmp_path: Path):
    """When Sniffer detects a usual separator, the LLM is NOT called."""
    csv_path = tmp_path / "normal.csv"
    csv_path.write_text("name,age,city\nAlice,30,Paris\nBob,25,London\n", encoding="utf-8")

    mock_delegate = MagicMock()
    set_csv_llm_delegate(mock_delegate)
    result = characterize_csv(str(csv_path))

    mock_delegate.assert_not_called()
    assert result["fieldSeparator"] == ","


def test_llm_characterize_csv_no_api_key():
    """_llm_characterize_csv returns None when no provider is configured."""
    with patch("choregraph.loaders.select_provider", return_value=None):
        result = _llm_characterize_csv(["a,b,c\n", "1,2,3\n"])
    assert result is None


def test_llm_characterize_csv_import_error():
    """_llm_characterize_csv returns None when langchain is not installed."""
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "langchain" in name:
            raise ImportError("mocked")
        return original_import(name, *args, **kwargs)

    fake_sel = {
        "provider": "google_genai",
        "api_key": "fake-key",
        "base_url": None,
        "model_override": None,
    }
    with patch("choregraph.loaders.select_provider", return_value=fake_sel):
        with patch("builtins.__import__", side_effect=mock_import):
            result = _llm_characterize_csv(["a,b,c\n", "1,2,3\n"])

    assert result is None


def test_llm_characterize_csv_explicit_provider_skips_select():
    """A caller-supplied provider config bypasses select_provider entirely."""
    with patch("choregraph.loaders.select_provider") as mock_select, patch(
        "choregraph.loaders.build_chat_model", side_effect=RuntimeError("no network")
    ):
        result = _llm_characterize_csv(
            ["a,b,c\n", "1,2,3\n"],
            provider="google_genai",
            api_key="fake-key",
        )
    mock_select.assert_not_called()
    assert result is None


# ---------------------------------------------------------------------------
# Tests for consistency-check LLM fallback (condition 3)
# ---------------------------------------------------------------------------

from choregraph.loaders import _check_column_consistency, _find_best_separator


def test_check_column_consistency_consistent():
    lines = ["a,b,c\n", "1,2,3\n", "4,5,6\n", "7,8,9\n"]
    assert _check_column_consistency(lines, ",") is True


def test_check_column_consistency_inconsistent():
    # Column counts: 3, 1, 5, 3 — differences exceed tolerance of 1
    lines = ["a,b,c\n", "1\n", "4,5,6,7,8\n", "8,9,10\n"]
    assert _check_column_consistency(lines, ",") is False


def test_check_column_consistency_too_few_rows():
    # Fewer than min_rows (3) → passes optimistically.
    lines = ["a,b,c\n", "1,2,3\n"]
    assert _check_column_consistency(lines, ",") is True


def test_llm_fallback_on_inconsistent_columns(tmp_path: Path):
    """When Sniffer detects a usual separator but column counts are inconsistent
    AND no alternative separator gives consistent columns, LLM is called."""
    csv_path = tmp_path / "embedded_json.csv"
    csv_path.write_text(
        'id;name;data\n1;Alice;"{"a":1,"b":2}"\n2;Bob;"{"c":3}"\n3;Eve;"{"d":4,"e":5,"f":6}"\n',
        encoding="utf-8",
    )

    mock_delegate = MagicMock(return_value={
        "header": True,
        "fieldSeparator": ";",
        "recordSeparator": "\n",
        "skipLines": 0,
        "modified": False,
    })
    with patch("choregraph.loaders._check_column_consistency", return_value=False):
        with patch("choregraph.loaders._find_best_separator", return_value=None):
            set_csv_llm_delegate(mock_delegate)
            result = characterize_csv(str(csv_path))

    mock_delegate.assert_called_once()
    assert result["fieldSeparator"] == ";"


def test_no_llm_when_columns_are_consistent(tmp_path: Path):
    """When Sniffer succeeds with a usual separator and columns are consistent,
    the LLM is NOT called."""
    csv_path = tmp_path / "clean.csv"
    csv_path.write_text("a;b;c\n1;2;3\n4;5;6\n7;8;9\n", encoding="utf-8")

    mock_delegate = MagicMock()
    set_csv_llm_delegate(mock_delegate)
    result = characterize_csv(str(csv_path))

    mock_delegate.assert_not_called()
    assert result["fieldSeparator"] == ";"


def test_consistency_fallback_keeps_sniffer_when_llm_fails(tmp_path: Path):
    """When consistency check fails, no alt separator found, and LLM also fails,
    keep the Sniffer result."""
    csv_path = tmp_path / "noisy.csv"
    csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    mock_delegate = MagicMock(return_value=None)
    with patch("choregraph.loaders._check_column_consistency", return_value=False):
        with patch("choregraph.loaders._find_best_separator", return_value=None):
            set_csv_llm_delegate(mock_delegate)
            result = characterize_csv(str(csv_path))

    mock_delegate.assert_called_once()
    # Sniffer result kept as best-effort
    assert result["fieldSeparator"] == ","


# ---------------------------------------------------------------------------
# Tests for _find_best_separator
# ---------------------------------------------------------------------------


def test_find_best_separator_finds_semicolon():
    """When comma gives inconsistent counts but semicolon is consistent."""
    lines = [
        'id;name;data\n',
        '1;Alice;"{"a":1,"b":2}"\n',
        '2;Bob;"{"c":3}"\n',
        '3;Eve;"{"d":4}"\n',
    ]
    result = _find_best_separator(lines, [";", ",", "\t", "|"], exclude=",")
    assert result == ";"


def test_find_best_separator_returns_none_when_no_good_candidate():
    """When no alternative separator produces consistent columns, return None."""
    lines = ["abc\n", "def\n", "ghi\n"]
    result = _find_best_separator(lines, [";", ",", "\t", "|"], exclude=",")
    assert result is None


def test_re_sniff_picks_alt_separator_without_llm(tmp_path: Path):
    """Integration: when Sniffer picks wrong separator, the alt-separator
    heuristic fixes it without needing a LLM call."""
    # Semicolon-separated file with comma-heavy JSON in fields.
    csv_path = tmp_path / "carburants.csv"
    csv_path.write_text(
        'id;ville;prix\n'
        '1;Paris;"{"a":1,"b":2,"c":3}"\n'
        '2;Lyon;"{"d":4,"e":5,"f":6}"\n'
        '3;Nice;"{"g":7,"h":8,"i":9}"\n',
        encoding="utf-8",
    )

    mock_delegate = MagicMock()
    set_csv_llm_delegate(mock_delegate)
    result = characterize_csv(str(csv_path))

    # LLM should NOT be called — alt separator heuristic should suffice.
    mock_delegate.assert_not_called()
    assert result["fieldSeparator"] == ";"

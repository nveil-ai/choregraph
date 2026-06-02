# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Guillaume Franque
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data loading utilities for CSV sniffing, characterization, and catalog configuration.

Provides CSV dialect detection (separator, header), full heuristic CSV
characterization (skip-line detection, field separator, record separator),
and prepares ``load_args`` dictionaries for Kedro ``catalog.yml`` generation.

When heuristic detection fails or produces an unusual delimiter, an
optional LLM fallback (Google Gemini) is attempted before returning
hard-coded defaults.
"""
import csv
import io
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .security import safe_path
# Provider plumbing (model names, native kwargs, boot-order detection and
# credentials) lives in `choregraph.llm_config`, which reads the ai_service
# provider yaml files — the single source of truth shared with Excel tidying
# and the ai_service. CSV characterization uses the cheap `minimal:` profile.
from .llm_config import build_chat_model, select_provider

logger = logging.getLogger(__name__)

# Timeout for a single LLM characterization call (seconds).
LLM_CSV_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Pydantic schema for LLM structured output
# ---------------------------------------------------------------------------


class LLMCsvCharacterization(BaseModel):
    """Structured output returned by the LLM for CSV characterization."""

    header: bool = Field(description="True if the first data row is a header.")
    fieldSeparator: str = Field(description="Single-character field delimiter (e.g. ',', ';', '\\t', '|', '~').")
    recordSeparator: str = Field(description="Record (line) separator, typically '\\n' or '\\r\\n'.")
    skipLines: int = Field(description="Number of non-data preamble lines to skip before the header/data.")


# ---------------------------------------------------------------------------
# LLM helpers (optional dependency — gracefully degrades)
# ---------------------------------------------------------------------------


def _llm_characterize_csv(sample_lines: List[str]) -> Optional[dict]:
    """Ask an LLM to characterize a CSV file from its first lines.

    Auto-detects the provider by walking the stack's boot order over the
    configured env vars (see :func:`choregraph.llm_config.select_provider`)
    — the same selection the ai_service uses — and runs its cheap
    ``minimal:`` model from the provider yaml.

    Returns a dict compatible with :func:`characterize_csv` on success,
    or ``None`` when no provider is configured or the call fails.
    """
    sel = select_provider()
    if sel is None:
        logger.warning(
            "_llm_characterize_csv: no LLM provider configured — skipping LLM fallback"
        )
        return None
    provider = sel["provider"]

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        logger.warning("_llm_characterize_csv: langchain not installed — skipping LLM fallback")
        return None

    system_prompt = (
        "You are a CSV file analyst. Given the first lines of a CSV file, "
        "determine:\n"
        "- **fieldSeparator**: the single character used to separate fields "
        "  (common examples: ',', ';', '\\t', '|', '~', but it can be anything).\n"
        "- **recordSeparator**: the line ending ('\\n' or '\\r\\n').\n"
        "- **header**: whether the first data row is a column-name header.\n"
        "- **skipLines**: how many non-data preamble lines appear before the "
        "  header or first data row.\n\n"
        "Return ONLY the JSON object matching the schema."
    )

    # Send at most the first 20 lines to keep the prompt small.
    raw_sample = "".join(sample_lines[:20])
    user_prompt = f"Here are the first lines of the CSV file:\n\n```\n{raw_sample}\n```"

    try:
        # Model name + provider-native kwargs come from the provider yaml's
        # `minimal:` profile (single source of truth). ValueError if it can't
        # be resolved (no configs) → caught below, LLM fallback skipped.
        llm = build_chat_model(
            provider,
            sel["api_key"],
            "minimal",
            base_url=sel["base_url"],
            model_override=sel["model_override"],
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # Try `json_schema` first (decoder-enforced schema → higher quality);
        # fall back to `function_calling` if it fails (some routes, quantized
        # open-weights models, or schemas with unions don't tolerate json_schema).
        response: Optional[LLMCsvCharacterization] = None
        last_exc: Optional[Exception] = None
        for method in ("json_schema", "function_calling"):
            try:
                llm_structured = llm.with_structured_output(
                    schema=LLMCsvCharacterization, method=method
                )
                pool = ThreadPoolExecutor(max_workers=1)
                future = pool.submit(llm_structured.invoke, messages)
                try:
                    response = future.result(timeout=LLM_CSV_TIMEOUT)
                    pool.shutdown(wait=False)
                except (TimeoutError, Exception):
                    future.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                if response is not None:
                    break
            except Exception as e:
                last_exc = e
                logger.warning(f"_llm_characterize_csv: method={method} failed: {e}")
                continue

        if response is None:
            if last_exc:
                raise last_exc
            return None

        logger.info(f"_llm_characterize_csv: LLM response: {response}")
        return {
            "header": response.header,
            "fieldSeparator": response.fieldSeparator,
            "recordSeparator": response.recordSeparator,
            "skipLines": response.skipLines,
            "modified": False,
        }
    except Exception as exc:
        logger.warning(f"_llm_characterize_csv: LLM call failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# CSV characterization (moved from ai_service/file_service/characterization.py)
# ---------------------------------------------------------------------------


def split_csv_line(line: str, delimiter: str) -> List[str]:
    """Parse a CSV line with the given delimiter, considering quotes."""
    reader = csv.reader(io.StringIO(line), delimiter=delimiter, quotechar='"')
    return next(reader)


def detect_skip_lines_smart(
    lines: List[str],
    delimiter_list: List[str],
    tolerance: int = 1,
    min_ok_lines: int = 2,
) -> int:
    """Detect start of data by parsing full CSV logic (handling multiline quotes).

    Tries each delimiter in *delimiter_list* and returns the index of the first
    line that begins a consistent run of rows with similar column counts.

    Args:
        lines: Lines read from the CSV file (first ~50 lines).
        delimiter_list: Delimiters to try (e.g. ``[",", ";", "\\t", "|"]``).
        tolerance: Allowed difference in column count between consecutive rows.
        min_ok_lines: Minimum consecutive consistent rows to confirm data start.

    Returns:
        Number of lines to skip (0 if data starts at the beginning).
    """
    full_content = "\n".join(lines)

    for delimiter in delimiter_list:
        try:
            reader = csv.reader(full_content.splitlines(keepends=True), delimiter=delimiter)
            parsed_rows = list(reader)

            if not parsed_rows:
                continue

            for i, row in enumerate(parsed_rows):
                current_len = len(row)
                if current_len <= 1:
                    continue

                ok = 0
                for j in range(i + 1, min(i + 6, len(parsed_rows))):
                    next_row_len = len(parsed_rows[j])
                    if abs(next_row_len - current_len) <= tolerance:
                        ok += 1

                if ok >= min_ok_lines:
                    return i
        except Exception:
            continue

    return 0


def remove_skiplines(filepath: str, skip_lines: int, buffer_size: int = 8192) -> bool:
    """Delete the first *skip_lines* lines from a file using buffered I/O.

    Args:
        filepath: Path to the file to modify.
        skip_lines: Number of lines to remove from the top.
        buffer_size: Read buffer size in bytes.

    Returns:
        ``True`` if the file was modified, ``False`` otherwise.
    """
    if skip_lines <= 0:
        return False

    temp_file = str(filepath) + ".tmp"
    try:
        with open(safe_path(filepath), "rb") as f_in:
            lines_skipped = 0
            while lines_skipped < skip_lines:
                line = f_in.readline()
                if not line:
                    return False
                lines_skipped += 1

            start_pos = f_in.tell()
            with open(safe_path(temp_file), "wb") as f_out:
                f_in.seek(start_pos)
                while True:
                    chunk = f_in.read(buffer_size)
                    if not chunk:
                        break
                    f_out.write(chunk)

        os.replace(temp_file, filepath)
        return True
    except Exception as e:
        logger.error(f"Error removing skip lines: {e}")
        if os.path.exists(temp_file):
            os.remove(safe_path(temp_file))
        return False


def _find_best_separator(
    lines: List[str], candidates: List[str], exclude: str
) -> Optional[str]:
    """Try each *candidate* separator and return the one producing the best column consistency.

    "Best" means: most rows with an identical column count, preferring a
    higher column count (which indicates actual structural splitting rather
    than treating the whole line as a single field).

    Returns ``None`` if no candidate improves on the current (excluded)
    separator.
    """
    best_sep: Optional[str] = None
    best_score: tuple = (0, 0)  # (consistent_row_count, col_count)

    for sep in candidates:
        if sep == exclude:
            continue
        try:
            reader = csv.reader(lines, delimiter=sep, quotechar='"')
            col_counts = [len(row) for row in reader if row]
        except Exception:
            continue

        if not col_counts:
            continue

        reference = col_counts[0]
        if reference <= 1:
            continue  # Separator not present in the data.

        consistent = sum(1 for c in col_counts if c == reference)
        score = (consistent, reference)
        if score > best_score:
            best_score = score
            best_sep = sep

    # Only accept if we found a separator that gives all rows the same count.
    if best_sep and best_score[0] == len([l for l in lines if l.strip()]):
        return best_sep
    return None


def _check_column_consistency(
    lines: List[str], delimiter: str, tolerance: int = 1, min_rows: int = 3
) -> bool:
    """Return ``True`` if *lines* parsed with *delimiter* have consistent column counts.

    Parses the lines with :mod:`csv.reader` (honouring quoting) and checks
    that every row has roughly the same number of fields.  A discrepancy
    larger than *tolerance* for **any** row makes the check fail.

    At least *min_rows* must be available for the check to be meaningful;
    if fewer rows are parsed the check passes optimistically.
    """
    try:
        reader = csv.reader(lines, delimiter=delimiter, quotechar='"')
        col_counts = [len(row) for row in reader if row]
    except Exception:
        return True  # Can't parse → be optimistic, don't trigger LLM needlessly.

    if len(col_counts) < min_rows:
        return True

    reference = col_counts[0]
    logger.debug("_check_column_consistency: column counts = %s, reference = %d", col_counts, reference)
    return all(abs(c - reference) <= tolerance for c in col_counts[1:])


def characterize_csv(filepath: str) -> dict:
    """Full heuristic CSV characterization with optional LLM fallback.

    Detects field separator, record separator, header presence, and
    non-data preamble lines.  When the ``csv.Sniffer`` heuristic fails,
    detects a separator outside the usual set ``[; , \\t |]``, **or**
    produces inconsistent column counts across rows, the function
    attempts to characterize the file via an LLM call (Google Gemini).
    If the LLM is unavailable or also fails, safe fallback values are
    returned.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Dict with keys ``header``, ``fieldSeparator``, ``recordSeparator``,
        ``skipLines``, ``modified``.
    """
    result = {
        "header": False,
        "fieldSeparator": None,
        "recordSeparator": None,
        "skipLines": 0,
        "modified": False,
    }
    fallback = {
        "header": True,
        "fieldSeparator": ",",
        "recordSeparator": "\n",
        "skipLines": 0,
        "modified": False,
    }

    is_built_from_excel = bool(
        re.search(r"sheet\d+_table\d+\.csv", str(filepath), re.IGNORECASE)
    )

    usual_del = [";", ",", "\t", "|"]
    sample_size = 10240
    sample_lines: List[str] = []
    newlines = None
    logger.debug("characterize_csv: reading sample from %s", filepath)
    try:
        with open(safe_path(filepath), "r", encoding="utf-8", errors="replace", newline=None) as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                sample_lines.append(line)
            f.seek(0)
            f.read(min(sample_size, os.path.getsize(filepath)))
            newlines = f.newlines
    except Exception as e:
        logger.warning(f"characterize_csv: could not read {filepath}: {e}")
        return fallback

    logger.debug("characterize_csv: detecting skip lines")
    result["skipLines"] = detect_skip_lines_smart(sample_lines, usual_del)
    useful_lines = sample_lines[
        result["skipLines"]: min(10 + result["skipLines"], len(sample_lines))
    ]

    try:
        sample = "".join(useful_lines)
        if is_built_from_excel:
            logger.debug("characterize_csv: Excel-origin file — forcing comma delimiter")
            result["fieldSeparator"] = ","
            csv.Sniffer().sniff(sample, delimiters=[","])
            result["header"] = csv.Sniffer().has_header(sample)
        else:
            logger.debug("characterize_csv: sniffing with usual delimiters")
            dialect = csv.Sniffer().sniff(sample, delimiters=usual_del)
            result["fieldSeparator"] = dialect.delimiter
            result["header"] = csv.Sniffer().has_header(sample)

        if isinstance(newlines, str):
            result["recordSeparator"] = newlines.encode().decode("unicode_escape")
        else:
            result["recordSeparator"] = dialect.lineterminator.encode().decode(
                "unicode_escape"
            )
    except Exception:
        # Heuristic failed entirely — try LLM fallback before returning defaults.
        logger.debug("characterize_csv: Sniffer failed — attempting LLM fallback")
        llm_result = _llm_characterize_csv(sample_lines)
        if llm_result is not None:
            logger.info("characterize_csv: using LLM characterization result")
            result = llm_result
        else:
            return fallback
        # Result already came from LLM — skip the Sniffer post-checks below.
        sniffer_succeeded = False
    else:
        sniffer_succeeded = True

    # If the Sniffer succeeded but detected an unusual separator, double-check
    # with the LLM (it may understand uncommon formats better).
    if sniffer_succeeded and result["fieldSeparator"] and result["fieldSeparator"] not in usual_del:
        logger.debug(
            "characterize_csv: unusual separator %r detected — attempting LLM fallback",
            result["fieldSeparator"],
        )
        llm_result = _llm_characterize_csv(sample_lines)
        if llm_result is not None:
            logger.info("characterize_csv: using LLM characterization for unusual separator")
            result = llm_result
        # Otherwise keep the Sniffer result as-is.

    # If Sniffer succeeded with a usual separator, verify consistency: parse
    # the useful lines with the detected separator and check that the column
    # count is stable across rows.  Inconsistency often means the true
    # separator was mis-identified (e.g. CSV with embedded JSON full of commas
    # being detected as comma-separated when it is actually semicolon-separated).
    elif sniffer_succeeded and result["fieldSeparator"]:
        if not _check_column_consistency(useful_lines, result["fieldSeparator"]):
            logger.info(
                "characterize_csv: inconsistent column counts with separator %r "
                "— trying alternative separators",
                result["fieldSeparator"],
            )
            # --- Heuristic: try the other usual separators first (no LLM needed) ---
            alt_sep = _find_best_separator(
                useful_lines, usual_del, exclude=result["fieldSeparator"]
            )
            if alt_sep:
                logger.info(
                    "characterize_csv: alternative separator %r gives consistent columns",
                    alt_sep,
                )
                result["fieldSeparator"] = alt_sep
                result["header"] = csv.Sniffer().has_header(
                    "".join(useful_lines)
                )
            else:
                # No usual separator works — try the LLM as last resort.
                logger.info(
                    "characterize_csv: no alternative separator found — attempting LLM fallback"
                )
                llm_result = _llm_characterize_csv(sample_lines)
                if llm_result is not None:
                    logger.info(
                        "characterize_csv: using LLM characterization after consistency check failure"
                    )
                    result = llm_result
                # Otherwise keep the Sniffer result as-is (best-effort).

    if result["skipLines"] > 0:
        modified = remove_skiplines(filepath, result["skipLines"])
        if modified:
            result["modified"] = True
            result["skipLines"] = 0

    return result

def sniff_csv_options(filepath: str) -> Dict[str, Any]:
    """Detect CSV separator and header row from a file sample.

    Reads the first 2048 bytes of the file and uses :mod:`csv.Sniffer` to
    detect the dialect.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Dict with detected ``sep`` and ``header`` values (may be empty if
        detection fails).
    """
    options = {}
    path = Path(filepath)
    if not path.exists():
        return options

    try:
        with open(safe_path(path), 'r', encoding='utf-8', errors='replace') as f:
            sample = f.read(2048)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample)
                options['sep'] = dialect.delimiter
                if sniffer.has_header(sample):
                    options['header'] = 0
                else:
                    options['header'] = None
            except csv.Error:
                pass
    except Exception as e:
        logger.warning(f"Error sniffing CSV {filepath}: {e}")
        
    return options

def prepare_load_args(fmt: str, location: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """Prepare ``load_args`` for Kedro catalog.yml generation.

    Merges user-provided options with auto-detected CSV settings.

    Args:
        fmt: Data format string (e.g. ``"CSV"``, ``"JSON"``).
        location: File path for auto-detection.
        options: User-provided load options (take precedence over sniffed values).

    Returns:
        Dict of load arguments suitable for ``catalog.yml``.
    """
    load_args = (options or {}).copy()
    # Strip metadata attributes that are stored in options but should NOT
    # be passed to pandas/kedro as load_args.
    for k in ("temporalFiles", "collectionTimeMode", "collectionTimeDelta"):
        load_args.pop(k, None)
    if fmt.upper() == "CSV":
        if "sep" not in load_args:
            sniffed = sniff_csv_options(location)
            if "sep" in sniffed: load_args["sep"] = sniffed["sep"]
            if "header" in sniffed: load_args["header"] = sniffed["header"]
    return load_args
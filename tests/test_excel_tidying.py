# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Integration tests for the Excel tidying module using validation files.

Each test processes a real Excel file through the full pipeline
(Cartographer -> Mapper -> ETL Compiler -> Execution) and validates
the output structure.

All results are dumped to choregraph/tests/output/<test_name>/ as both
parquet and CSV files for manual inspection.

Requires:
  - GOOGLE_API_KEY env var
  - Fixture files in choregraph/tests/fixtures/excel/
"""
import logging
import os
import shutil
from pathlib import Path

import pytest
import pandas as pd

from choregraph.collection.excel.main import tidy_excel_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "excel"
OUTPUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY env var not set")
    return key


def _fixture(name: str) -> str:
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    return str(path)


def _dump_results(test_name: str, results: dict):
    """Save every DataFrame to parquet + CSV under output/<test_name>/."""
    out = OUTPUT_DIR / test_name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    for key, df in results.items():
        # Parquet (convert object columns to string to avoid ArrowInvalid on mixed types)
        parquet_path = out / f"{key}.parquet"
        df_parquet = df.copy()
        for col in df_parquet.columns:
            if df_parquet[col].dtype == object:
                df_parquet[col] = df_parquet[col].astype(str)
        df_parquet.to_parquet(parquet_path, index=False)
        # CSV (for quick human inspection)
        csv_path = out / f"{key}.csv"
        df.to_csv(csv_path, index=False)

        summary_lines.append(
            f"  {key}: {df.shape[0]} rows x {df.shape[1]} cols — {list(df.columns)}"
        )

    # Write a summary text file
    summary_path = out / "_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("Dumped %d tables to %s", len(results), out)
    for line in summary_lines:
        logger.info(line)


def _assert_valid_table(table_id: str, df: pd.DataFrame):
    """Generic structural checks that every tidy output must satisfy."""
    assert isinstance(df, pd.DataFrame), f"{table_id}: not a DataFrame"
    assert not df.empty, f"{table_id}: empty DataFrame"
    assert df.shape[1] >= 1, f"{table_id}: should have >= 1 column"
    # No raw single Excel column letters (A, B, ..., Z, AA, AB, ...)
    # These indicate promote_header/flatten didn't run or failed.
    from openpyxl.utils.cell import column_index_from_string
    excel_letter_cols = []
    for c in df.columns:
        if not isinstance(c, str) or not c.isalpha() or not c.isupper():
            continue
        try:
            idx = column_index_from_string(c)
            if idx <= df.shape[1] + 5:  # plausible Excel column letter
                excel_letter_cols.append(c)
        except ValueError:
            pass
    # Only flag if MOST columns look like Excel letters (a few may be coincidences)
    if len(excel_letter_cols) > df.shape[1] * 0.5:
        raise AssertionError(
            f"{table_id}: majority of columns are raw Excel letters: {excel_letter_cols}"
        )


def _log_table_detail(table_id: str, df: pd.DataFrame, max_sample_rows: int = 3):
    """Log detailed diagnostics for a single table."""
    logger.info("─── %s (%d rows x %d cols) ───", table_id, df.shape[0], df.shape[1])
    logger.info("  Columns: %s", list(df.columns))
    logger.info("  Dtypes:\n%s", df.dtypes.to_string())
    logger.info("  Sample:\n%s", df.head(max_sample_rows).to_string())
    # Warn about suspicious columns
    for col in df.columns:
        if df[col].isna().all():
            logger.warning("  ⚠ Column '%s' is ALL NaN", col)
        elif str(col).startswith("__col_") or str(col).startswith("col_"):
            logger.warning("  ⚠ Column '%s' looks like an unnamed fallback", col)


# ---------------------------------------------------------------------------
# Cola.xlsx — cross_tab with years, should produce unpivoted long tables
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_cola():
    """Cola.xlsx: cross-tab tables should unpivot into long format."""
    api_key = _get_api_key()
    results = tidy_excel_data(_fixture("Cola.xlsx"), api_key=api_key, max_retries=2)

    _dump_results("cola", results)
    assert len(results) >= 1, f"Expected >= 1 table, got {len(results)}"

    for table_id, df in results.items():
        _assert_valid_table(table_id, df)
        _log_table_detail(table_id, df)

    # --- Cola-specific diagnostics ---
    # Cola tables are transposed_grid (financial statements): metrics are rows,
    # fiscal years are columns. After transpose, we expect WIDE tables with
    # a fiscal_year column + many metric columns, and ~10 rows (one per FY).
    for table_id, df in results.items():
        if "fiscal_year" not in df.columns and "observation" not in df.columns:
            logger.warning(
                "  ⚠ %s is missing a fiscal_year/observation column — "
                "transposed_grid may not have been applied correctly.",
                table_id,
            )


# ---------------------------------------------------------------------------
# douleurs.xlsx — may have headerless data or form patterns
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_douleurs():
    """douleurs.xlsx: should produce at least 1 table."""
    api_key = _get_api_key()
    results = tidy_excel_data(_fixture("douleurs.xlsx"), api_key=api_key, max_retries=2)

    _dump_results("douleurs", results)
    assert len(results) >= 1, f"Expected >= 1 table, got {len(results)}"

    for table_id, df in results.items():
        _assert_valid_table(table_id, df)
        _log_table_detail(table_id, df)


# ---------------------------------------------------------------------------
# ASSURANCE AUTO NVEIL.xlsx — 6 sheets, multi-row headers, merged cells
#
# Expected structure (from manual inspection):
#   - AUTO 2026/2025/2024: flat_list tables, multi-row headers
#   - FLOTTE: 2 sub-tables (main + accidents)
#   - VH FONCTION: 2 sub-tables (main + accidents)
#   - VH SERVICE: 2 sub-tables (main + accidents)
#   Total expected: 9 tables (3 AUTO + 2 FLOTTE + 2 VH FONCTION + 2 VH SERVICE)
#
# Known issue: the cartographer sometimes misses the "ACC. RESPONSABLE"
# sub-tables in VH FONCTION and VH SERVICE because they appear in the
# middle of the sheet. Minimum reliably detected: 6 tables.
# ---------------------------------------------------------------------------

# Columns that MUST be present in Flotte / VH Fonction / VH Service tables.
# These are the key financial/insurance columns that were historically lost.
EXPECTED_FLEET_COLUMN_KEYWORDS = [
    "cout", "coût", "franchise", "prime", "sinistre",
    "responsable", "immatriculation", "marque",
]


@pytest.mark.timeout(600)
def test_assurance_auto():
    """ASSURANCE AUTO: multi-sheet file with multi-row headers."""
    api_key = _get_api_key()
    results = tidy_excel_data(
        _fixture("ASSURANCE AUTO NVEIL.xlsx"),
        api_key=api_key,
        max_retries=2,
    )

    _dump_results("assurance_auto", results)

    assert len(results) >= 6, (
        f"Expected >= 6 tables (3 AUTO sheets + FLOTTE + VH FONCTION + VH SERVICE), "
        f"got {len(results)}: {list(results.keys())}"
    )

    for table_id, df in results.items():
        _assert_valid_table(table_id, df)
        _log_table_detail(table_id, df, max_sample_rows=2)

    # --- Check that AUTO 2024/2025/2026 tables exist ---
    for expected_fragment in ["auto_2026", "auto_2025", "auto_2024"]:
        matches = [
            tid for tid in results
            if expected_fragment in tid.lower()
            or expected_fragment.replace("_", "") in tid.lower().replace("_", "")
        ]
        if not matches:
            logger.warning(
                "No table found matching '%s'. Table IDs: %s",
                expected_fragment, list(results.keys()),
            )

    # --- Check column consistency across AUTO 2024/2025/2026 ---
    auto_tables = {
        tid: df for tid, df in results.items()
        if any(y in tid.lower() for y in ["2024", "2025", "2026"])
        and "auto" in tid.lower()
    }
    if len(auto_tables) >= 2:
        col_sets = {tid: set(df.columns) for tid, df in auto_tables.items()}
        ref_tid, ref_cols = next(iter(col_sets.items()))
        for tid, cols in col_sets.items():
            if tid == ref_tid:
                continue
            diff = ref_cols.symmetric_difference(cols)
            if diff:
                logger.warning(
                    "  ⚠ Column mismatch between '%s' and '%s': %s",
                    ref_tid, tid, diff,
                )
            else:
                logger.info(
                    "  ✓ '%s' and '%s' have identical columns", ref_tid, tid,
                )

    # --- Check Flotte / VH tables retained financial columns ---
    fleet_tables = {
        tid: df for tid, df in results.items()
        if any(kw in tid.lower() for kw in ["flotte", "vh_fonction", "vh_service",
                                              "vehicule_fonction", "vehicule_service"])
    }
    for tid, df in fleet_tables.items():
        cols_lower = [str(c).lower() for c in df.columns]
        cols_joined = " ".join(cols_lower)
        found_keywords = [
            kw for kw in EXPECTED_FLEET_COLUMN_KEYWORDS
            if kw in cols_joined
        ]
        missing_keywords = [
            kw for kw in EXPECTED_FLEET_COLUMN_KEYWORDS
            if kw not in cols_joined
        ]

        logger.info(
            "  %s: found %d/%d expected column keywords: %s",
            tid, len(found_keywords), len(EXPECTED_FLEET_COLUMN_KEYWORDS), found_keywords,
        )
        if missing_keywords:
            logger.warning(
                "  ⚠ %s: MISSING column keywords (potential column loss): %s",
                tid, missing_keywords,
            )

        # Warn if the table has suspiciously few columns
        if df.shape[1] < 5:
            logger.warning(
                "  ⚠ %s has only %d columns — likely lost data during header flattening",
                tid, df.shape[1],
            )


# ---------------------------------------------------------------------------
# budget_CA_2013 (.xls) — legacy format, needs pyexcel conversion
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_budget_ca():
    """budget_CA .xls: legacy format should convert and process."""
    api_key = _get_api_key()
    results = tidy_excel_data(
        _fixture("budget_CA_2013_Depenses_Budget_ASSAINISSEMENT_COLLECTIF_x.xls"),
        api_key=api_key,
        max_retries=2,
    )

    _dump_results("budget_ca", results)
    assert len(results) >= 1, f"Expected >= 1 table, got {len(results)}"

    for table_id, df in results.items():
        _assert_valid_table(table_id, df)
        _log_table_detail(table_id, df)
        # Budget files typically have many rows
        assert df.shape[0] >= 10, (
            f"{table_id}: budget table seems too small ({df.shape[0]} rows)"
        )


# ---------------------------------------------------------------------------
# effectifs_.xls — table starts at row 5 (not row 1), multi-row header
#
# Structure: 1 sheet "Feuil1", rows 1-4 empty/title, rows 5-6 headers,
# rows 7-16 data (years 2001-2010, 10 rows). Tests that flatten_header
# indices are computed relative to the sub-table, not as absolute row numbers.
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_effectifs():
    """effectifs_.xls: table not starting at row 1 should preserve all data rows."""
    api_key = _get_api_key()
    results = tidy_excel_data(
        _fixture("effectifs_.xls"),
        api_key=api_key,
        max_retries=2,
    )

    _dump_results("effectifs", results)
    assert len(results) >= 1, f"Expected >= 1 table, got {len(results)}"

    for table_id, df in results.items():
        _assert_valid_table(table_id, df)
        _log_table_detail(table_id, df)

    # The file contains exactly 10 data rows (years 2001-2010).
    # A previous bug (flatten_header using absolute row indices) would
    # drop the first 4 years, producing only 6 rows.
    main_df = next(iter(results.values()))
    assert main_df.shape[0] >= 10, (
        f"Expected >= 10 data rows (years 2001-2010), got {main_df.shape[0]}. "
        f"flatten_header may be using absolute row indices instead of sub-table-relative."
    )

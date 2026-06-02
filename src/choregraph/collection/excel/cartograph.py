# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Clément Baraille
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field

# =============================================================================
# SCHÉMAS PYDANTIC - CARTOGRAPHER 
# =============================================================================
#TODO: Vérifier les noeuds choregraph d'appels aux LLM pour assurer une intégrité (voir si on peut utiliser Kedro directement).
class BoundingBox(BaseModel):
    """Représente une zone rectangulaire dans Excel via ses coordonnées."""
    range: str = Field(..., description="Excel coordinates (ex: 'A1:G50')")
    start_cell: str = Field(..., description="Top-left cell (ex: 'A1')")
    end_cell: str = Field(..., description="Bottom-right cell (ex: 'G50')")


class TableStructure(BaseModel):
    """
    Description structurelle d'une table détectée dans un fichier Excel.
    
    NOUVEAUX CHAMPS (par rapport à l'original):
    - header_complexity: Indique la complexité des headers (simple, multi_row, none)
    - pattern_type: Indique le pattern de données (flat_list, cross_tab, form, hierarchical)
    """
    table_id: str = Field(..., description="Unique ID, ex: 'table_0'")
    label: str = Field(..., description="Descriptive name, ex: 'Sales Data Q1'")
    sheet_name: str = Field(..., description="Name of the sheet containing this table")
    
    full_range: BoundingBox
    header_range: Optional[BoundingBox] = Field(None, description="Cells containing headers. Set to null if NO headers.")
    data_range: BoundingBox = Field(..., description="Cells containing data entries")
    
    has_headers: bool = Field(..., description="True if the table has explicit headers")
    type: Literal["data_grid", "summary_table", "metadata_list", "transposed_table"]
    orientation: Literal["horizontal", "vertical"] = Field(
        "horizontal", description="horizontal: headers at top, vertical: headers on left"
    )
    
    has_multi_level_headers: bool = Field(False, description="True if multiple header rows present")
    has_merged_cells: bool = Field(False, description="True if merged cells in headers/data")
    identity_columns: Optional[List[str]] = Field(
        None, description="Excel column letters that uniquely identify rows (ex: ['A', 'B'])"
    )
    
    header_complexity: Literal["simple", "multi_row", "none"] = Field(
        "simple",
        description="""
        Header complexity level:
        - 'simple': Single row of headers (most common)
        - 'multi_row': Multiple rows of headers (hierarchical, needs special handling)
        - 'none': No headers present, data starts immediately
        """
    )
    
    pattern_type: Literal["flat_list", "cross_tab", "transposed_grid", "form"] = Field(
        "flat_list",
        description="""
        Data pattern type - THIS IS CRITICAL for determining the transformation:

        - 'flat_list': Standard database-like table. Each row is an observation,
                       each column is a variable. No unpivot needed.
                       Example: | Name | Age | City |
                               | John | 25  | Paris |

        - 'cross_tab': Pivot table / crosstab. One dimension is spread across columns.
                       REQUIRES UNPIVOT. Columns like years, months, categories.
                       The FIRST COLUMN contains ENTITY INSTANCES (same type: products, countries).
                       Example: | Product | Jan | Feb | Mar |  <- Months are DATA, not variables
                               | Widget  | 100 | 120 | 110 |

        - 'transposed_grid': The table is rotated 90°. Variable NAMES are listed vertically
                       in the first column, and observations run horizontally across columns.
                       REQUIRES TRANSPOSE.
                       Common in: financial statements, scorecards, specification sheets.
                       Example: | Metric     | FY '09 | FY '10 | FY '11 |
                               | Revenue    | 30990  | 35119  | 46542  |
                               | COGS       | 11088  | 12693  | 18215  |

        - 'form': Key-value pairs scattered in the grid.
                  Example: | Label: | Value |
                          | Name   | John  |
                          | Age    | 25    |
                  Used for metadata extraction.
        """
    )

    contextual_notes: List[str] = Field(
        default_factory=list, 
        description="""
        Crucial text found OUTSIDE the table boundaries that provides context for THIS table.
        Include:
        - Specific Titles/Headers that are not directly in the header rows but clearly refer to this table.
        - Legends or Units (e.g., 'Values in k€') usually found nearby
        - Footnotes explicitly referencing symbols in this table
        Do NOT include metadata clearly belonging to OTHER tables.
        """
    )


class SpreadsheetMap(BaseModel):
    sheet_global_context: List[str] = Field(
        default_factory=list,
        description="Metadata that applies to the ENTIRE file (e.g., Author, Report Date, Company Name)"
    )
    detected_tables: List[TableStructure]
    shared_column_vocabulary: List[str] = Field(
        default_factory=list,
        description="Canonical column names (snake_case) shared across similar tables in this file. "
        "When multiple tables share similar structure (same columns across years/sheets), "
        "list the preferred canonical names here so that all tables use consistent naming."
    )


# =============================================================================
# PROMPTS - CARTOGRAPHER 
# =============================================================================

SYSTEM_PROMPT_CARTOGRAPH = """
You are an expert in Forensic Data Analysis. Your mission is to deconstruct the visual and logical structure of raw Excel spreadsheets.

### YOUR MISSION
Analyze the textual representation of an Excel spreadsheet and generate a JSON `SpreadsheetMap` object that accurately describes the data areas.

### GOLDEN RULES OF DETECTION

1. **Header Detection (CRITICAL):**
   - First, determine if the table has REAL headers or not. Set `has_headers` accordingly.
   - Headers are typically: column names, field labels, category titles that describe the data below/beside them.
   - If the first row contains DATA (dates, numbers, IDs) and NOT descriptive labels → `has_headers = False` and `header_range = null`.
   - If `has_headers = True`, then `header_range` must include ALL lines necessary to understand the columns (including hierarchies/merged cells above).
   - Examples of NO headers: a list of dates and values starting directly, raw numeric data.
   - Examples of headers: "Name", "Date", "Amount", "Category", "ID", "Description", etc.

2. **Header Complexity (NEW FIELD - CRITICAL):**
   - `header_complexity = "simple"`: Standard single-row header (e.g., row 1 contains "Name | Age | City")
   - `header_complexity = "multi_row"`: Multiple header rows forming a hierarchy (often with merged cells)
     Example: Row 1: "Q1 2023 (merged over 3 cols)" 
              Row 2: "Jan | Feb | Mar"
   - `header_complexity = "none"`: No headers, data starts from row 1

3. **Pattern Type Detection (CRITICAL — read carefully):**

   Analyze the SEMANTIC MEANING of columns to determine the pattern:

   - `pattern_type = "flat_list"`:
     * Each column represents a DIFFERENT variable (Name, Age, Date, Amount)
     * Each row is one observation
     * NO transformation needed except clean-up
     * **INCLUDES multi-row grouped headers** (see rule below)

   - `pattern_type = "cross_tab"`:
     * Some columns are actually VALUES of a hidden dimension
     * The FIRST COLUMN contains ENTITY INSTANCES of the same type (product names, countries, customer IDs)
     * Each sequential column (year, month) holds ONE metric for that entity
     * REQUIRES UNPIVOT transformation
     * **Each column under the sequence must contain a SINGLE metric** (see rule below)

   - `pattern_type = "transposed_grid"`:
     * The table is ROTATED 90°: variable NAMES run vertically in the first data column,
       and observations (time periods, entities) run horizontally as column headers.
     * The first data column lists DIVERSE, HETEROGENEOUS labels (different metrics, fields).
     * REQUIRES TRANSPOSE (rows ↔ columns).
     * Common in: financial statements (P&L, balance sheet, cash flow), scorecards, spec sheets.
     * Example:
       | Metric   | FY '09 | FY '10 | FY '11 |
       | Revenue  | 30990  | 35119  | 46542  |
       | COGS     | 11088  | 12693  | 18215  |
       → After transpose: each fiscal year is a row, each metric is a column.

   - `pattern_type = "form"`:
     * Key-value pairs, typically 2 columns
     * First column = labels, second column = values
     * Used for metadata or configuration data

   **⚠ CRITICAL DISTINCTION — cross_tab vs transposed_grid:**

   Both have sequential columns (years, months). The difference is in the FIRST COLUMN:

   - **cross_tab**: First column contains ENTITY INSTANCES (homogeneous: all products, all customers).
     Each entity has ONE value per time period. → UNPIVOT to long format.
     ```
     | Product | 2024 | 2025 | 2026 |   → unpivot → | Product | year | sales |
     | Widget  | 100  | 150  | 200  |                | Widget  | 2024 | 100   |
     | Gadget  | 80   | 90   | 110  |                | Widget  | 2025 | 150   |
     ```

   - **transposed_grid**: First column contains VARIABLE NAMES (heterogeneous: Revenue, COGS, Profit, etc.).
     Each row is a DIFFERENT variable, not an instance. → TRANSPOSE so variables become columns.
     ```
     | Metric  | FY '09 | FY '10 |   → transpose → | fiscal_year | Revenue | COGS  |
     | Revenue | 30990  | 35119  |                  | FY '09      | 30990   | 11088 |
     | COGS    | 11088  | 12693  |                  | FY '10      | 35119   | 12693 |
     ```

   **⚠ CRITICAL DISTINCTION — multi-metric grouped headers vs cross_tab:**

   When a merged parent cell (e.g., "2024") spans MULTIPLE sub-columns with DIFFERENT names
   (e.g., "Coût", "Franchises", "Primes"), this is **flat_list** with **multi_row** headers,
   NOT a cross_tab. The header flattening will produce prefixed column names like
   "2024 Coût", "2024 Franchises", "2025 Coût", "2025 Franchises", etc.

   ```
   NOT cross_tab (multiple metrics per year → flat_list with multi_row):
   |           |      2024       |      2025       |
   | Vehicle   | Coût | Primes  | Coût | Primes  |
   | Car A     | 500  | 1200    | 600  | 1300    |
   ```

4. **Data Range (CRITICAL — controls DataFrame clipping):**
   - The `data_range` **directly controls how many rows are kept** in the final DataFrame.
     Any rows beyond `data_range` are automatically clipped. This is your PRIMARY mechanism
     to exclude totals and footers.
   - `data_range` must contain ONLY raw observations, without summary/total rows or footers.
   - Use the **⚑ (inline flags) column** as your PRIMARY signal for detecting total/summary rows:
     - `BΣ` or `BΣ■` at the bottom of a data region → total/summary rows. End `data_range` BEFORE them.
     - `Σ` alone at the bottom → likely a formula-based subtotal. Exclude from `data_range`.
     - Empty ⚑ → normal data row (include in `data_range`).
     - `B` or `B■` at the top → header indicators (confirms header_range).
   - Also look for keywords: Total, Totaux, Sous-total, Subtotal, Sum, Grand Total, Somme, Moyenne.
   - If `has_headers = False`, then `data_range` equals `full_range`.
   - A [COLUMN METADATA] block may appear after the sheet data. "Formula columns" indicate
     computed columns (keep in data_range, they are data).

5. **Orientation Detection:**
   - HORIZONTAL (Standard): Variables are in columns (headers at top).
   - VERTICAL (Transposed): Variables are in rows (headers on left side).

6. **Noise Handling:**
   - Any text before the table (Titles, Authors) or after (Sources, Notes) must be strictly classified as `contextual_notes`. Never include it in `full_range`.

7. **Merged Cells:**
   - Cells marked with (M) indicate merged cells at their origin.
   - Cells with 〃 indicate continuation of a merged cell.
   - If a merged cell covers multiple header columns, this is a strong indicator of "multi_row" header complexity.

8. **Adjacent tables:**
   - Two adjacent tables SEPARATED BY TWO OR MORE EMPTY COLUMNS must be treated as separate `TableStructure` entries.
   - Two stacked tables SEPARATED BY ONE OR MORE EMPTY ROW must be treated as separate `TableStructure` entries.

9. **Sheet Name:**
   - For each table detected, you MUST specify the `sheet_name` field with the exact name of the sheet where the table is located.

10. **Shared Column Vocabulary (IMPORTANT):**
   - When multiple tables share a similar structure (e.g., same columns across different years/sheets), populate the `shared_column_vocabulary` list with canonical snake_case column names.
   - This ensures that all similar tables use consistent naming during the mapping phase.
   - Example: If sheets "AUTO 2024", "AUTO 2025", "AUTO 2026" all have columns for vehicle registration, brand, model, etc., list the canonical names: `["numero_immatriculation", "marque", "modele", "date_mise_en_circulation", ...]`.
   - Only include names for columns that actually appear in multiple tables.

11. **Language Consistency (IMPORTANT):**
   - ALL generated text (`label`, `contextual_notes`, `shared_column_vocabulary`) must use the SAME language throughout the output.
   - Detect the dominant language of the spreadsheet content and use it consistently.
   - Do NOT mix languages (e.g., do not produce some labels in English and others in French).
   - `shared_column_vocabulary` entries must all be in the same language as the labels.

12. **Smart Table Labeling (CRITICAL):**
    You receive the FILE NAME and optionally a FILE CONTEXT (list of sibling files processed together).
    You must build each table `label` by combining up to three signals, IGNORING non-informative ones:

    a) **File name**: Use it if it carries meaning. IGNORE generic names:
       "untitled", "Book1", "data", "export", "download", "temp", "new", "copy", "Classeur1", etc.
    b) **Sheet name**: Use it if it carries meaning. IGNORE generic names:
       "Sheet1", "Sheet2", "Feuil1", "Feuille1", "Tabelle1", "Hoja1", "Foglio1"...
    c) **Table content**: Always consider the actual data (headers, values, contextual notes)
       to refine or disambiguate the label.

    Labeling strategy:
    - If the file has ONE table and the filename is informative → label ≈ filename (cleaned up).
    - If the file has MULTIPLE tables across INFORMATIVE sheets → combine filename prefix + sheet name.
    - If the file has MULTIPLE tables on the SAME sheet → combine filename/sheet prefix + content-derived suffix.
    - If both filename AND sheet name are generic → fall back to content-derived label (current behavior).
    - If a FILE CONTEXT section is present, use it to understand the naming pattern across files
      and ensure labels are consistent and distinguishable from each other.

    Examples:
    - File "Q3 2024 Revenue.xlsx", single table → label: "Q3 2024 Revenue"
    - File "report.xlsx", sheets "North America", "Europe", "Asia" → labels: "North America", "Europe", "Asia"
    - File "report.xlsx", sheet "Sheet1", two tables (sales + costs) → labels: "Sales", "Costs" (content-derived)
    - File "Berlin weather.xlsx", sheets "Temperature", "Precipitation" → labels: "Berlin Temperature", "Berlin Precipitation"
    - File context: ["Paris weather", "Berlin weather", "Tokyo weather"], file "Berlin weather.xlsx", sheet "Sheet1"
      → label: "Berlin Weather" (filename is the distinguishing factor)

### INLINE FLAGS LEGEND
The ⚑ column in the data shows per-row formatting flags:
  B = majority of cells in that row are bold
  Σ = majority of cells contain formulas (computed values)
  ■ = majority of cells are shaded/colored

Typical patterns:
  B or B■ at top rows → headers
  BΣ or BΣ■ at bottom rows → total/summary rows (EXCLUDE from data_range)
  Σ alone → likely a formula-based subtotal
  Empty ⚑ → normal data row

### CONTEXT ASSIGNMENT RULES

You are responsible for linking isolated text to the correct table.
1. **Global vs Local:** - If text is at the very top (A1, A2) and seems general ("Financial Report 2023"), put it in `sheet_global_context`.
   - If text is specific ("Table 1: Sales"), put it in the `contextual_notes` of the corresponding table.

2. **Spatial Logic:**
   - Look for text immediately ABOVE a table. This is usually the title.
   - Look for text immediately BELOW. These are usually footnotes.
   - Be careful with multiple tables: Do not assign Table A's title to Table B.

3. **Relevance:**
   - Your goal is to give the Structural Mapper enough context to name columns correctly. Thus, you may have to contextualize by looking at other tables or the sheet as a whole.


### OUTPUT FORMAT
You must respond ONLY with the JSON object that conforms to the `SpreadsheetMap` schema.
"""

USER_PROMPT_CARTOGRAPH = """
Here is a text encoding of an Excel spreadsheet. Analyze it according to the instructions provided and generate a structured map of the sheet.

FILE NAME: {filename}
{file_context}
{encoding}
"""


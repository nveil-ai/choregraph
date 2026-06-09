# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Clément Baraille
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

# Pydantic schemas - Structural Mapper (declarative approach)
#
# These schemas describe the Structural Mapper's output. Rather than specifying
# pandas operations (imperative), they describe the SEMANTICS of the table's
# regions (declarative). The ETL compiler then translates this mapping into
# concrete operations.
from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field

class CellReference(BaseModel):
    """Hybrid reference to an Excel column.

    Excel coordinates (A, B, C) are used as the ground truth because they never
    change. Semantic names are applied at the END of the pipeline, once the
    structure is normalized.

    Attributes:
        col_idx: Excel column letter (A, B, C...) — the ground truth.
        detected_name: Value read from the header cell (None if no header).
        semantic_name: Proposed snake_case name for the final DataFrame.
        data_type: Expected data type for this column.
    """
    col_idx: str = Field(..., description="Excel column letter (A, B, C...). This is the ground truth.")
    detected_name: Optional[str] = Field(None, description="Header value seen in the cell, if any")
    semantic_name: str = Field(..., description="Exact name detected in the header. Only propose a snake_case name if the header is missing or completely ambiguous.")
    data_type: Literal["string", "int", "float", "date", "boolean"] = Field(
        "string", 
        description="Expected data type for this column"
    )


class AnchorZone(BaseModel):
    """Anchor zone — columns that stay FIXED during an unpivot.

    These columns define the IDENTITY of each row (ProductID, CustomerName,
    Date, Category, etc.). In pandas terms, they are the ``id_vars`` of ``melt()``.

    Example:
        For the table: | Product | Jan | Feb | Mar |
                       | Widget  | 100 | 120 | 110 |

        AnchorZone.columns = [CellReference(col_idx="A", semantic_name="Product")]
        because "Product" is the identifier, not a value to pivot.
    """
    columns: List[CellReference] = Field(
        ..., 
        description="List of columns that define row identity (will become id_vars in melt)"
    )


class PivotZone(BaseModel):
    """Pivot zone — headers that are actually DATA to "unfold".

    These columns represent a hidden dimension currently spread horizontally
    that should become a single vertical column. In pandas terms, they are the
    ``value_vars`` of ``melt()``.

    Examples of hidden dimensions:
    - Years: 2019, 2020, 2021 -> becomes a "year" column.
    - Months: Jan, Feb, Mar -> becomes a "month" column.
    - Days: Day 1, Day 2, Day 3 -> becomes a "day" column.
    - Categories repeated horizontally.

    Attributes:
        col_range_start: First column of the range (e.g. "B").
        col_range_end: Last column of the range (e.g. "M").
        target_name: Name of the NEW column created after unpivot.
        extracted_type: Data type of the values extracted from the headers.
    """
    col_range_start: str = Field(..., description="First column letter of the pivot range (e.g., 'B')")
    col_range_end: str = Field(..., description="Last column letter of the pivot range (e.g., 'M')")
    target_name: str = Field(
        ..., 
        description="Name for the NEW column created after unpivot (e.g., 'month', 'year', 'day')"
    )
    extracted_type: Literal["string", "int", "float", "date"] = Field(
        "string",
        description="Data type of the values extracted from headers"
    )


class ValueZone(BaseModel):
    """Value zone — the numeric/data body of the table.

    These cells hold the measures/metrics/observations. After an unpivot they
    become a single value column.

    Example:
        In | Product | Jan | Feb |
           | Widget  | 100 | 120 |

        100 and 120 belong to the ValueZone.
        After unpivot: | Product | Month | Sales |
                       | Widget  | Jan   | 100   |
                       | Widget  | Feb   | 120   |

        "Sales" is the ValueZone.target_name.
    """
    target_name: str = Field(
        ..., 
        description="Name for the value column after unpivot (e.g., 'sales', 'revenue', 'score')"
    )
    data_type: Literal["int", "float", "string", "boolean"] = Field(
        "float",
        description="Data type of the values in the body"
    )


class FormKeyValue(BaseModel):
    """A key-value pair for the FORM pattern.

    The FORM pattern is used for "spec sheet" style data where information is
    organized as label/value pairs rather than as a table.

    Example FORM structure:
        | A          | B       |
        | Last name  | Dupont  |
        | First name | Jean    |
        | Age        | 42      |

    Here each row is a key-value pair, not an observation.
    """
    key_column: str = Field(..., description="Excel column letter containing the keys/labels")
    value_column: str = Field(..., description="Excel column letter containing the values")
    extracted_pairs: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Optional: specific key-value pairs to extract"
    )


class RegionMapping(BaseModel):
    """Main output of the Structural Mapper — a DECLARATIVE description of regions.

    This schema does NOT say "run a melt" (imperative); it says "here are the
    semantic regions" (declarative). The ETL compiler then translates the
    mapping into pandas operations.

    Supported patterns:

    1. FLAT_LIST (no structural transformation):
       - anchors holds every column.
       - pivot_axis = None.
       - The ETL compiler only renames and casts types.

    2. CROSS_TAB (unpivot required):
       - anchors holds the identity columns.
       - pivot_axis defines the range of columns to pivot.
       - value_body defines the resulting value column.
       - The ETL compiler generates a melt().

    3. FORM (key-value extraction):
       - form_mapping holds the key-value pairs.
       - The ETL compiler transposes and restructures.
    """
    table_id: str = Field(..., description="Table ID (must match TableStructure.table_id)")

    pattern: Literal["flat_list", "cross_tab", "form", "transposed_grid"] = Field(
        ...,
        description="""
        Pattern identified for this table:
        - flat_list: standard table, no structural transformation.
        - cross_tab: dimension spread horizontally -> unpivot required.
        - form: key-value pairs -> extraction and transposition.
        - transposed_grid: variables are in ROWS (column A), observations in COLUMNS. Requires a simple TRANSPOSE.
        """
    )
    
    header_row_index: Optional[int] = Field(
        None, 
        description="0-indexed row containing the headers (within the extracted sub-table). None if no headers."
    )
    
    # === REGIONS ===
    anchors: AnchorZone = Field(
        ...,
        description="Columns that define row identity (always required, even if just one column)"
    )
    
    pivot_axis: Optional[PivotZone] = Field(
        None,
        description="For cross_tab: the headers that are actually data values to unpivot. "
                    "For transposed_grid: specifies the observation column name (target_name) and the range of observation columns."
    )
    
    value_body: Optional[List[ValueZone]] = Field(
        None,
        description="ONLY for cross_tab pattern: describes the value cells. Use a single ValueZone for simple cross-tabs, or multiple for tables with several measure columns (e.g., Sales and Units)."
    )
    
    form_mapping: Optional[FormKeyValue] = Field(
        None,
        description="ONLY for form pattern: key-value pair configuration"
    )
    
    # === FILTERING ===
    rows_to_exclude_keywords: Optional[List[str]] = Field(
        None,
        description="Keywords to filter out from ALL string anchor columns using case-insensitive substring matching (e.g., ['Total', 'Subtotal', 'Sum'])"
    )



# Prompts - Structural Mapper

SYSTEM_PROMPT_MAPPER = """
You are a Structural Data Analyst specializing in understanding the semantic structure of Excel tables.

### YOUR MISSION
Given a table structure (from the Cartographer) and the raw data, produce a `RegionMapping` that identifies:
1. **ANCHOR ZONE**: Which columns define the IDENTITY of each row
2. **PIVOT ZONE**: Which columns are actually DATA that should become a single column (if any)
3. **VALUE ZONE**: What the numeric values represent (if cross_tab pattern)


### CRITICAL PRINCIPLES

1. **ALWAYS USE EXCEL COLUMN LETTERS**
   - Use `col_idx` = "A", "B", "C" as the primary reference
   - `detected_name` = what you SEE in the header (can be null)
   - `semantic_name` = The EXACT name detected in the header. Only propose a snake_case name if the header is missing or completely ambiguous.

2. **PATTERN DETECTION**

   **FLAT_LIST Pattern:**
   - Each column is a DIFFERENT variable
   - Example columns: "Name", "Age", "City", "Salary"
   - No columns follow a sequence pattern
   - SET: pattern="flat_list", pivot_axis=None
   - ALL columns go into anchors (they all define the row)
   - **INCLUDES tables with multi-row grouped headers** where a parent cell (e.g., "2024")
     spans multiple sub-columns with DIFFERENT names (e.g., "Coût", "Franchises", "Primes").
     After header flattening, these become prefixed names like "2024 Coût", "2024 Franchises".
     → Treat as flat_list. ALL columns go into anchors with their flattened names.

   **CROSS_TAB Pattern:**
   - The first column(s) contain ENTITY INSTANCES of the same type (products, countries, customers)
   - Other columns are VALUES of a hidden dimension (years, months, categories)
   - Each entity has ONE value per time period / category
   - SET: pattern="cross_tab", define pivot_axis and value_body
   - Anchor columns = those that DON'T follow the pattern
   - **CRITICAL: Only use cross_tab when each pivot column contains a SINGLE metric.**
     If a year/month group spans MULTIPLE sub-columns with different names
     (e.g., "2024: Coût, Franchises, Primes"), this is flat_list, NOT cross_tab.

   **TRANSPOSED_GRID Pattern (CRITICAL — financial statements, scorecards):**
   - The first data column lists DIVERSE VARIABLE NAMES (heterogeneous labels like
     Revenue, COGS, Gross Profit, Operating Income, etc.).
   - The column headers are OBSERVATIONS (fiscal years, quarters, entities).
   - Each row is a DIFFERENT variable, NOT an instance of the same type.
   - SET: pattern="transposed_grid".
   - Anchors should point to the column containing the variable names (typically column A).
   - pivot_axis: SET target_name to the name for the observation identifier column
     (e.g., "fiscal_year", "quarter", "entity"). Set col_range_start/col_range_end to
     the range of observation columns (e.g., B to K).
   - value_body: null (not needed — each variable name becomes its own column after transpose).
   - **⚠ cross_tab vs transposed_grid:** If the first column contains diverse metric/field names
     (like in a P&L statement), use transposed_grid. If it contains homogeneous entity names
     (like product names), use cross_tab.
   - Example:
     | Metric  | FY '09 | FY '10 |  → transpose → | fiscal_year | Revenue | COGS  |
     | Revenue | 30990  | 35119  |                 | FY '09      | 30990   | 11088 |
     | COGS    | 11088  | 12693  |                 | FY '10      | 35119   | 12693 |

   **FORM Pattern:**
   - Key-value pairs in 2 columns
   - First column = labels, Second column = values
   - Example: | "Name" | "John" | → key="Name", value="John"
   - SET: pattern="form", define form_mapping

3. **HANDLING MISSING HEADERS**
   - If `header_row_index` is null (no headers), you must INVENT semantic_name
   - Look at the DATA in the column to infer what it represents
   - Example: Column A has "Score dos", "Score migraine" → semantic_name = "metric_type"

4. **COLUMN RANGE CALCULATION**
   For pivot_axis, specify:
   - col_range_start: First column of the repeated pattern (e.g., "B")
   - col_range_end: Last column of the repeated pattern (e.g., "AK")
   
5. **ROWS TO EXCLUDE (CRITICAL — always check for totals)**
   Use the **⚑ (inline flags) column** as PRIMARY signal:
   - `BΣ` or `BΣ■` → bold + formula row = almost certainly a total/summary. Add keywords from that row.
   - `Σ` alone → formula-based subtotal. Check if it contains aggregation keywords.
   - Empty ⚑ → normal data row, do not exclude.
   - Keywords are searched across ALL string identity columns (not just the first one).

   Also look for keywords in the raw data: Total, Totaux, Sous-total, Subtotal, Sum, Grand Total,
   TOTAL, Somme, Moyenne, Average.
   - Add ALL matching keywords you find to `rows_to_exclude_keywords`
   - This applies to ALL patterns (flat_list, cross_tab, transposed_grid)
   - Even if only one total row exists, it MUST be excluded to avoid polluting the dataset
   - The ETL Compiler will filter them out using case-insensitive substring matching on ALL string anchor columns

6. **CONTEXT USAGE**
   - Use the provided context to inform your decisions, especially for:
   - Inferring semantic names when headers are missing
   - Determining the hidden dimension in cross_tab patterns

7. **SHARED VOCABULARY (CRITICAL FOR CONSISTENCY)**
   - When a shared column vocabulary is provided, you MUST use those exact names for matching columns in `semantic_name`, `pivot_axis.target_name`, and `value_body.target_name`.
   - This ensures that similar tables from the same file produce identical column names.
   - Example: If the vocabulary includes "numero_immatriculation" and the header says "N° Immat.", use "numero_immatriculation" as the `semantic_name`.
   - Only apply vocabulary names to columns that semantically match. Do not force a vocabulary name onto an unrelated column.

8. **LANGUAGE CONSISTENCY (IMPORTANT)**
   - ALL `semantic_name`, `target_name`, and generated text must use the SAME language throughout.
   - Detect the dominant language of the spreadsheet content and stick to it.
   - Do NOT mix languages (e.g., do not use "year" for one table and "annee" for another).
   - When the shared vocabulary provides names in a specific language, follow that language.

### INLINE FLAGS LEGEND
The ⚑ column in the data shows per-row formatting flags:
  B = majority of cells in that row are bold
  Σ = majority of cells contain formulas (computed values)
  ■ = majority of cells are shaded/colored

Typical patterns:
  B or B■ at top rows → headers
  BΣ or BΣ■ at bottom rows → total/summary rows (add keywords to rows_to_exclude_keywords)
  Σ alone → likely a formula-based subtotal
  Empty ⚑ → normal data row

### EXAMPLES

**Example 1: FLAT_LIST**
Input: | ID | Name | Age | City |
       | 1  | John | 25  | Paris |

Output:
{
  "pattern": "flat_list",
  "header_row_index": 0,
  "anchors": {
    "columns": [
      {"col_idx": "A", "detected_name": "ID", "semantic_name": "ID", "data_type": "int"},
      {"col_idx": "B", "detected_name": "Name", "semantic_name": "Name", "data_type": "string"},
      {"col_idx": "C", "detected_name": "Age", "semantic_name": "Age", "data_type": "int"},
      {"col_idx": "D", "detected_name": "City", "semantic_name": "City", "data_type": "string"}
    ]
  },
  "pivot_axis": null,
  "value_body": null
}

**Example 2: CROSS_TAB (Years)**
Input: | Product | 2019 | 2020 | 2021 |
       | Widget  | 100  | 150  | 200  |

Output:
{
  "pattern": "cross_tab",
  "header_row_index": 0,
  "anchors": {
    "columns": [
      {"col_idx": "A", "detected_name": "Product", "semantic_name": "Product", "data_type": "string"}
    ]
  },
  "pivot_axis": {
    "col_range_start": "B",
    "col_range_end": "D",
    "target_name": "year",
    "extracted_type": "int"
  },
  "value_body": [
    {"target_name": "sales", "data_type": "float"}
  ]
}


**Example 3: TRANSPOSED_GRID (Financial Statement)**
Input: |                          | FY '09 | FY '10 | FY '11 |
       | NET OPERATING REVENUES   | 30990  | 35119  | 46542  |
       | Cost of goods sold       | 11088  | 12693  | 18215  |

Output:
{
  "pattern": "transposed_grid",
  "header_row_index": 0,
  "anchors": {
    "columns": [
      {"col_idx": "A", "detected_name": null, "semantic_name": "metric", "data_type": "string"}
    ]
  },
  "pivot_axis": {
    "col_range_start": "B",
    "col_range_end": "D",
    "target_name": "fiscal_year",
    "extracted_type": "string"
  },
  "value_body": null
}


### OUTPUT FORMAT
Respond ONLY with a valid JSON conforming to the RegionMapping schema.
"""

USER_PROMPT_MAPPER = """
[TABLE STRUCTURE FROM CARTOGRAPHER]
{table_structure}

[RAW TABLE DATA - First rows]
{raw_table_md}

[GLOBAL SHEET CONTEXT]
{global_context}

[LOCAL TABLE CONTEXT]
{local_context}

[CURRENT COLUMNS - Excel letters with detected values]
{columns_info}

[SHARED COLUMN VOCABULARY]
{shared_vocabulary}

Analyze this table and produce a RegionMapping that describes the semantic regions.
"""


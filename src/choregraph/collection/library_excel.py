# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Clément Baraille
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LLM-assisted Excel data tidying.

Uses Google Gemini to analyze messy Excel spreadsheet layouts and restructure
them into tidy DataFrames. Handles multi-table detection via flood-fill,
column mapping with Excel coordinates, transposition, and melting.
"""
from pydantic import BaseModel
import re
import pandas as pd
from typing import List, Dict
from collections import deque
from choregraph.parser import _sanitize_name

DIM_SAMPLE = [25, 50]  # Dimensions of the sample given to the LLM for analysis.
WARNING = "[WARNING]"
ERROR = "[ERROR]"
DEBUG = "[DEBUG]"
INFO = "[INFO]"

class NewTableStructure(BaseModel):
    column_name: str
    data_range: str
    transpose: bool = False


class MeltParameters(BaseModel):
    id_vars: List[str]
    value_vars: List[str]
    var_name: str
    value_name: str


class LLMTableStructure(BaseModel):
    table_structures: List[NewTableStructure]
    column_names_index: str
    melt_parameters: MeltParameters = {}

def df_to_markdown(df):
    header_cells = [str(col).replace("|", "\\|") for col in df.columns]
    markdown = "| " + " | ".join(header_cells) + " |\n"

    for _, row in df.iterrows():
        values = [
            str(value).replace("|", "\\|") if pd.notna(value) else ""
            for value in row.values
        ]
        markdown += "| " + " | ".join(values) + " |\n"
    return markdown

def get_table_structure(df):
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from langchain_core.messages import HumanMessage, SystemMessage
    from .excel.main import _build_excel_llm
    from ..llm_config import select_provider

    # Auto-detect the configured provider (same boot order as the rest of the
    # stack); the model + kwargs come from the provider yaml's `defaults:`.
    sel = select_provider()
    if sel is None:
        raise RuntimeError(
            "No LLM provider configured for Excel table-structure analysis. "
            "Set a provider key/endpoint — run the setup wizard."
        )
    llm = _build_excel_llm(
        sel["provider"], sel["api_key"],
        base_url=sel["base_url"], model_override=sel["model_override"],
    )
    llm_structured = llm.with_structured_output(LLMTableStructure)
    callback = UsageMetadataCallbackHandler()

    system_prompt = r"""
Your role is to analyze a Markdown table (derived from Excel/Pandas) and determine the necessary structure to achieve a **wide-tidy data format**, i.e., where a column represents a variable, a row represents an observation, and a cell represents a value.

    ---
    ## TIDY DATA RULES
    ---

    1.  **Goal (Wide Format):** Unless the table is already in a long-tidy format, structure the output in a **wide format** where distinct metrics or categories become their own columns (do not melt them into a single 'Metric' column).
    2.  **No Change Rule:** If the provided table is already tidy, return an **empty list** for the `table_structures`.
    3.  **Naming:** Identify and name the columns of the final tidy table. **Do not invent column names if the original table does not contain explicit headers.**

    ---
    ## OUTPUT & ADDRESSING RULES
    ---

    * **Output:** Return **ONLY** a list of objects representing the final column structures. To do this, simply tell me, how the columns in my final table should be named and the data ranges associated (with excel coordinates). 
    Warning, **a tidy table may not contain headers, avoid inventing column names if not necessary**.
    * **Excel Range Convention:** Specify the data range for each column using Excel coordinates.
    * **Addressing End-of-Range:**
        * Use the **\$** symbol for the last row (e.g., `A2:A$`).
        * Use the **\*** symbol for the last column (e.g., `B1:*1`).

    Here are the detailed rules you must follow:
    ---
    ## 1. COLUMN METADATA & HEADER LOCATION
    ---

    * **Final Column Structure (List of Objects):** Based on the column definitions resulting from the previous step, for each column:
        * Specify if it resulted from a **direct transpose only** (boolean: `true` if it's a direct copy of a row/column from the original table without renaming, deletion, or hierarchy changes; `false` otherwise).
    * **Header Location:** Identify the Excel row or column letter/number that contains the **column names** in the *original* table. Return only the **single letter/number** (e.g., `1`, `A`). Return `"None"` if the original table has no explicit header.

    ---
    ## 2. CONDITIONAL MELT PROPOSAL (Pandas)
    ---

    Your task is to propose the `pandas.melt` function signature **ONLY IF** all the following conditions are met:
    1.  The resulting table is in a **wide format**.
    2.  The columns have **explicit names**.
    3.  You deem it relevant, meaning the wide columns clearly represent **repeated measures of the same variable** (e.g., years, months, product categories).
    
    **NEVER propose a melt if:**
    * The wide table already represents distinct variables (different units/categories).
    * The columns lack explicit labels.
    * It would melt two different variables (with distinct units or categories) into a single column.
    
    * **Output:** Populate the `MeltParameters` field with a dictionary containing the `id_vars`, `value_vars`, `var_name`, and `value_name` parameters for the `pandas.melt` function call. If the melt is not relevant, return an **empty dictionary**.
    Note: 
    As a reminder, the melt function has this signature 
    "pandas.melt(frame, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None, ignore_index=True)" with the following parameters:
    * id_vars
    scalar, tuple, list, or ndarray, optional
    Column(s) to use as identifier variables.
    * value_vars
    scalar, tuple, list, or ndarray, optional
    Column(s) to unpivot. If not specified, uses all columns that are not set as id_vars.
    * var_name
    scalar, default None
    Name to use for the 'variable' column. If None it uses frame.columns.name or 'variable'.
    * value_name
    scalar, default 'value'
    Name to use for the 'value' column, cannot be an existing column label.

    ---
    Your final output must integrate all required information from the previous analysis steps into the specified structure.
    """
    df_head_md=df_to_markdown(df.iloc[: DIM_SAMPLE[0], : DIM_SAMPLE[1]])
    user_prompt = f"""
    Here is my table:
    {df_head_md}

    For each column created, you will indicate if it results only from a direct transpose, without any other operation (including renaming column, deleting hierarchical headers, etc.) or not.
    Also, please identify where the column names are located in the original table, by providing **only** the row (or column if the table is transposed) excel representation (**single letter/number**, not couple or anything else). If the table does not contain any header, please return "None".
    """
    response = llm_structured.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)], config={"callbacks": [callback]})
    print(f"{INFO} Usage metadata: {callback.usage_metadata}")
    return response

def excel_to_df_indices(excel_coord, df):
    """Convert Excel column coordinates (like 'A1' or 'B5') to DataFrame indices."""
    match = re.match(r"([A-Z\*]+)([0-9\$]+)", excel_coord)
    if not match:
        raise ValueError(f"Invalid Excel coordinate: {excel_coord}")

    col_str, row_str = match.groups()
    if row_str == "$":
        row_idx = df.shape[0] - 1
    else:
        row_idx = int(row_str) - 1
    if col_str == "*":
        col_idx = df.shape[1] - 1
    else:
        col_idx = 0
        for c in col_str:
            col_idx = col_idx * 26 + (ord(c) - ord("A") + 1)
        col_idx -= 1  # 0-indexed

    return row_idx, col_idx


def excel_range_to_indices(range_str, df):
    """Convert a range of Excel coordinates (like 'A1:C5') to DataFrame indices."""
    if ":" not in range_str:
        row_idx, col_idx = excel_to_df_indices(range_str, df)
        return row_idx, col_idx, row_idx, col_idx
    start, end = range_str.split(":")
    start_row, start_col = excel_to_df_indices(start, df)
    end_row, end_col = excel_to_df_indices(end, df)

    return start_row, start_col, end_row, end_col


def df_index_to_excel_col(col_idx):
    """Convert a 0-indexed column index to Excel letters."""
    col_idx += 1  # Excel is 1-indexed for columns
    result = ""
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        result = chr(65 + remainder) + result
    return result


def indices_to_excel_range(start_row, start_col, end_row, end_col):
    """
    Convert DataFrame indices to an Excel range string (like 'A1:C5').
    """
    # Convert column indices to Excel letters
    start_col_str = df_index_to_excel_col(start_col)
    end_col_str = df_index_to_excel_col(end_col)

    start_row_str = str(start_row + 1)
    end_row_str = str(end_row + 1)

    # Combine to form the range string
    return f"{start_col_str}{start_row_str}:{end_col_str}{end_row_str}"


def get_data_by_excel_range(df, range_str):
    """Extract data from DataFrame using Excel-style range (e.g., 'A1:C5')"""
    start_row, start_col, end_row, end_col = excel_range_to_indices(range_str, df)
    
    # Adjust for dataframe offset if it's a slice from a larger sheet
    # (flood fill returns slices with their original index/columns preserved)
    try:
        # Check for RangeIndex start/stop which indicates an offset from the original sheet
        if hasattr(df.index, 'start'):
            start_row -= df.index.start
            end_row -= df.index.start
        if hasattr(df.columns, 'start'):
            start_col -= df.columns.start
            end_col -= df.columns.start
    except Exception:
        pass
        
    return df.iloc[max(0, start_row) : end_row + 1, max(0, start_col) : end_col + 1]



def expand_row_range(structure: LLMTableStructure, df):
    for item in structure.table_structures:
        start_row, start_col, end_row, end_col = excel_range_to_indices(
            item.data_range, df
        )
        # Fallback if the LLM did not use the syntax '*' '$'
        if (
            start_col == end_col and end_row == DIM_SAMPLE[0]
        ):  # If only one column is selected and it goes to the end of the sample, extend to the end of the df
            item.data_range = indices_to_excel_range(
                start_row, start_col, df.shape[0], end_col
            )
    return structure


def expand_column_range(structure: LLMTableStructure, df):
    curr_start_row, curr_end_row = None, None
    for item in structure.table_structures[
        -3:
    ]:  # Check the last 3 columns created by the LLM and collect the row range if they are consistent (arbitrary way to determine the row range used to expand columns)
        start_row, start_col, end_row, end_col = excel_range_to_indices(
            item.data_range, df
        )
        if start_col == end_col:
            if curr_start_row is None:
                curr_start_row, curr_end_row = start_row, end_row
            else:
                if curr_start_row == start_row and curr_end_row == end_row:
                    pass
                else:
                    curr_start_row, curr_end_row = min(curr_start_row, start_row), max(
                        curr_end_row, end_row
                    )
    try:
        header_row = (
            max(0, int(structure.column_names_index) - 2)
            if structure.column_names_index.lower() != "none"
            else 0
        )
    except ValueError:
        print(
            ERROR,
            f"The column_names_index detected by the LLM ({structure.column_names_index}) should be an integer or 'None' when the table is not transposed. We will consider that the header row is 0.",
        )
        header_row = 0
    for index in range(
        len(structure.table_structures), df.shape[1]
    ):  # For each missing (due to sampling) column in the structure, we add a new element describing the resulting column
        structure.table_structures.append(
            NewTableStructure(
                column_name=df.iloc[header_row, index],
                data_range=indices_to_excel_range(
                    curr_start_row, index, curr_end_row, index
                ),
                transpose=False,
            )
        )

    return structure


def expand_column_range_transposed(structure: LLMTableStructure, df):
    curr_start_col, curr_end_col = None, None
    if structure.column_names_index.lower().isalpha():
        if structure.column_names_index.lower() == "none":
            header_col = 0
    else:
        print(
            ERROR,
            f"The column_names_index detected by the LLM ({structure.column_names_index}) should be an integer or 'None' when the table is not transposed. We will consider that the header column is 'A'.",
        )
        header_col = "A"
    curr_start_col, curr_end_col = None, None
    for item in structure.table_structures[
        -3:
    ]:  # Check the last 3 columns created by the LLM and collect the row range if they are consistent (arbitrary way to determine the row range used to expand columns)
        start_row, start_col, end_row, end_col = excel_range_to_indices(
            item.data_range, df
        )
        if start_row == end_row:
            if curr_start_col is None:
                curr_start_col, curr_end_col = start_col, end_col
            else:
                if curr_start_col == start_col and curr_end_col == end_col:
                    pass
                else:
                    curr_start_col, curr_end_col = min(curr_start_col, start_col), max(
                        curr_end_col, end_col
                    )
    if (
        end_row == start_row == DIM_SAMPLE[0]
    ):  # If only one row is selected and it goes to the end of the sample, extend to the end of the df
        for line_index in range(len(structure.table_structures) + 1, df.shape[0] + 1):
            excel_range_new_line = indices_to_excel_range(
                line_index, curr_start_col, line_index, curr_end_col
            )
            structure.table_structures.append(
                NewTableStructure(
                    column_name=df.iloc[line_index - 1, 0],
                    data_range=excel_range_new_line,
                    transpose=True,
                )
            )
    return structure


def expand_row_range_transposed(structure: LLMTableStructure, df):
    for item in structure.table_structures:
        start_row, start_col, end_row, end_col = excel_range_to_indices(
            item.data_range, df
        )
        # Fallback if the LLM did not use the syntax '*' '$'
        if (
            start_row == end_row and end_col == DIM_SAMPLE[1] - 1
        ):  # If only one row is selected and it goes to the end of the sample, extend to the end of the df
            item.data_range = indices_to_excel_range(
                start_row, start_col, end_row, df.shape[1]
            )
    return structure


def complete_llm_response(structure: LLMTableStructure, df):
    """
    * If the table is transposed: we need to enrich the table_structures object with as many elements as there are missing rows in the sample.
    Also, if columns are missing in the sample, we need to modify the existing value ranges in table_structures.
    For instance, if we give a (10, 10) sample of a (50, 20) transposed table, we need to add 40 elements in table_structures and make sure the ranges cover the 20 columns of the original df.

    * If the table is not transposed: the tables structures object must be enriched with as many elements as there are missing columns in the sample.
    Also, if rows are missing from the sample, the existing value ranges in table_structures need to be modified.
    For instance, if we give a (10, 10) sample of a (50, 20) table, we need to add 10 elements in table_structures and make sure the ranges cover the 50 lines of the original df.
    """
    LAST_ITEMS_TO_CHECK_TRANSPOSITION = 3
    last_are_transposed = sum(
        [
            item.transpose
            for item in structure.table_structures[-LAST_ITEMS_TO_CHECK_TRANSPOSITION:]
        ]
    )
    if last_are_transposed == LAST_ITEMS_TO_CHECK_TRANSPOSITION:
        struct = expand_column_range_transposed(structure, df)
        struct = expand_row_range_transposed(struct, df)
    else:
        struct = expand_row_range(structure, df)
        struct = expand_column_range(struct, df)
    return struct


def create_tidy_df(df, structure: LLMTableStructure):
    """Create a tidy DataFrame based on the provided structure."""
    # print(DEBUG, f"Original DataFrame shape: {df}")
    # print(DEBUG, f"LLM structure received: {structure}")
    if (
        DIM_SAMPLE[0] < df.shape[0] or DIM_SAMPLE[1] < df.shape[1]
    ) and structure.table_structures != []:
        print(
            WARNING,
            "WARNING: The sample provided to the LLM was smaller than the original dataframe, so we need to automatically complete the structure inferred by the LLM to cover the entire dataframe",
        )
        structure = complete_llm_response(structure, df)
    tidy_data = {}
    if structure.table_structures == []:
        return df
    for item in structure.table_structures:
        column_name = item.column_name
        data_range = item.data_range
        # Extract data using the specified range
        data = get_data_by_excel_range(df, data_range)

        # Flatten the data if it's a DataFrame or Series
        if isinstance(data, pd.DataFrame):
            data = data.values.flatten()
        elif isinstance(data, pd.Series):
            data = data.values

        tidy_data[column_name] = data

    # Create a new DataFrame from the tidy data
    tidy_df = pd.DataFrame(tidy_data)
    try:
        if (
            structure.melt_parameters != {}
            and structure.melt_parameters.id_vars != []
            and structure.melt_parameters.value_vars != []
        ):
            print(
                INFO,
                f"The LLM suggested to melt the dataframe with parameters: {structure.melt_parameters}",
            )
            tidy_df = melt_df(tidy_df, structure.melt_parameters)
    except Exception as e:
        print(
            ERROR,
            f"Error while melting the dataframe with parameters {structure.melt_parameters}: {e}",
        )
    return tidy_df


def melt_df(df, melt_params: MeltParameters):
    if melt_params == {}:
        return df
    try:
        melted_df = pd.melt(
            df,
            id_vars=melt_params.id_vars,
            value_vars=melt_params.value_vars,
            var_name=melt_params.var_name,
            value_name=melt_params.value_name,
        )
        return melted_df
    except Exception as e:
        print(
            ERROR,
            f"Error while melting the dataframe with parameters {melt_params}: {e}",
        )
        return df
    
def _find_tables_flood_fill(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Extract all tables from the df of an excel file (using flood-fill / cross kernel) even if:
    - Columns are not entirely empty between tables
    - Multiple tables appear on the same line
    - Blocks must be grouped vertically only if the columns match exactly
    
    Uses 4-connectivity BFS to discover connected regions of non-empty cells.
    """
    import numpy as np
    
    visited = np.zeros(df.shape, dtype=bool)
    tables = []

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iat[i, j]
            if pd.notna(val) and val != "" and not visited[i, j]:
                # Starting point of a new table
                q = deque([(i, j)])
                cells = []
                visited[i, j] = True
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    # Explore neighbors (4-connectivity)
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < df.shape[0] and 0 <= cc < df.shape[1]:
                            val_neighbor = df.iat[rr, cc]
                            if pd.notna(val_neighbor) and val_neighbor != "" and not visited[rr, cc]:
                                visited[rr, cc] = True
                                q.append((rr, cc))

                # Create a DataFrame from the collected cells
                rows = [r for r, c in cells]
                cols = [c for r, c in cells]
                rmin, rmax = min(rows), max(rows)
                cmin, cmax = min(cols), max(cols)
                block = df.iloc[rmin : rmax + 1, cmin : cmax + 1]
                tables.append(block.reset_index(drop=True))
    
    clean_tables = []
    for block in tables:
        block = block.dropna(axis=1, how="all")  # Delete empty columns
        if block.empty or block.shape[0] < 2:
            continue
            
        # Use first row as header
        block.columns = block.iloc[0]
        block = block.drop(0).reset_index(drop=True)
        clean_tables.append(block)
    
    return clean_tables


def _clean_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame to ensure Parquet compatibility.
    
    - Ensures unique column names (converts to strings)
    - Converts NaN/None column names to 'Column_N'
    - Converts mixed-type object columns to strings
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame ready for Parquet serialization
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # 1. Convert column names to strings and handle NaN/duplicates
    new_columns = []
    seen = {}
    for i, col in enumerate(df.columns):
        # Convert to string, handling NaN/None
        if pd.isna(col):
            col_str = f"Column_{i}"
        else:
            col_str = str(col).strip()
            if not col_str:
                col_str = f"Column_{i}"
        
        # Ensure uniqueness
        if col_str in seen:
            seen[col_str] += 1
            col_str = f"{col_str}_{seen[col_str]}"
        else:
            seen[col_str] = 0
        
        new_columns.append(col_str)
    
    df.columns = new_columns
    
    # 2. Convert object columns with mixed types to string
    for col in df.columns:
        if df[col].dtype == object:
            # Check if there are mixed types by attempting conversion
            try:
                # Convert everything to string to avoid Parquet errors
                df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)
            except Exception:
                pass
    
    return df


def _clean_excel_df_with_tidying(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Clean the DataFrame extracted from an Excel file using AI-based tidying.
    Returns (cleaned_df, feedback_message).
    """
    try:
        table_structure = get_table_structure(df)
        if not table_structure:
            return df, ""
        
        df = create_tidy_df(df, table_structure)
        return df, ""
        
    except ImportError as e:
        # Tidying module not available - return original
        return df, f"Tidying not available: {e}"
    except Exception as e:
        feedback = ("It seems that your excel file contains a table which was not in a tidy format, "
                   "but the automatic tidying process failed. Please try to manually tidy your table "
                   "(one column corresponds to one variable and one row corresponds to one observation) "
                   "before uploading it again.")
        return df, feedback


def analyze_excel_structure(file_path: str) -> List[str]:
    """
    Analyze an Excel file to determine how many tables it contains.
    Returns a list of table keys like ['sheet1_table1', 'sheet1_table2', 'sheet2_table1'].
    """
    import openpyxl
    
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        sheet_names = workbook.sheetnames
        workbook.close()
        
        table_keys = []
        for sheet_idx, sheet_name in enumerate(sheet_names):
            # Read sheet without headers to detect tables
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            tables = _find_tables_flood_fill(df)
            
            for table_idx, table_df in enumerate(tables):
                # Skip empty or too small tables (must match tidy_excel_data logic)
                if table_df.empty or table_df.shape[0] < 1:
                    continue
                # Create a sanitized key
                safe_sheet = _sanitize_name(sheet_name)
                table_keys.append(f"{safe_sheet}_table{table_idx + 1}")
        
        # If no tables found, return at least one default output
        if not table_keys:
            table_keys = ["table1"]
            
        return table_keys
        
    except Exception as e:
        print(f"analyze_excel_structure failed: {e}")
        # Fallback: assume single table
        return ["table1"]


def tidy_excel_data(data: Dict[str, pd.DataFrame], apply_tidying: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Process Excel data loaded as a dict of {sheet_name: DataFrame}.
    
    For each sheet:
    1. Detect multiple tables using flood-fill algorithm
    2. Optionally apply AI-based tidying to each table (in parallel)
    3. Return a dict mapping table keys to cleaned DataFrames
    
    Args:
        data: Dict from ExcelDataset.load() with sheet_name=None
              Format: {"Sheet1": df1, "Sheet2": df2, ...}
        apply_tidying: Whether to apply AI-based tidying (default True)
    
    Returns:
        Dict mapping table keys to DataFrames, e.g.:
        {"sheet1_table1": df1, "sheet1_table2": df2, "sheet2_table1": df3}
    """
    import concurrent.futures
    from functools import partial
    
    # First pass: collect all tables to process
    all_tables = []  # List of (sheet_name, table_idx, table_df)
    
    print(f"tidy_excel_data starting. Apply tidying: {apply_tidying}")
    if isinstance(data, pd.DataFrame):
        print("tidy_excel_data received a single DataFrame. Converting to dict.")
        data = {"sheet1": data}

    print(f"tidy_excel_data received data with sheets: {list(data.keys())}")
    for sheet_name, sheet_df in data.items():
        print(f"Processing sheet '{sheet_name}' with shape {sheet_df.shape}")
        # Detect tables in this sheet using flood-fill
        tables = _find_tables_flood_fill(sheet_df)
        print(f"Found {len(tables)} tables in sheet '{sheet_name}'")
        
        for table_idx, table_df in enumerate(tables):
            # Skip empty or too small tables
            if table_df.empty or table_df.shape[0] < 1:
                print(f"Skipping empty/small table {table_idx+1} in sheet '{sheet_name}'")
                continue
            all_tables.append((sheet_name, table_idx, table_df))
    print(f"Total tables found: {len(all_tables)}")

    def process_single_table(table_info: tuple, apply_tidying: bool) -> tuple:
        """Process a single table, optionally applying tidying."""
        try:
            sheet_name, table_idx, table_df = table_info
            
            # Create key for this table
            safe_sheet = _sanitize_name(sheet_name)
            table_key = f"{safe_sheet}_table{table_idx + 1}"
            
            print(f"Processing table '{table_key}' (shape: {table_df.shape})")
            
            # Optionally apply tidying
            if apply_tidying:
                print(f"Applying tidying to '{table_key}'...")
                table_df, feedback = _clean_excel_df_with_tidying(table_df)
                if feedback:
                    print(f"Tidying feedback for '{table_key}': {feedback}")
            
            # Skip if table became empty after tidying
            if table_df.empty or table_df.shape[0] < 1:
                print(f"Table '{table_key}' is empty after tidying/processing")
                return (table_key, None)
            
            # Clean DataFrame for Parquet compatibility (handles mixed types)
            table_df = _clean_dataframe_for_parquet(table_df)
            
            return (table_key, table_df)
        except Exception as e:
            print(f"Critical error processing table {table_info[1]+1} in sheet '{table_info[0]}': {e}")
            return (table_key, None)
    
    result = {}
    
    if len(all_tables) > 1:
        print(f"Using ThreadPoolExecutor for {len(all_tables)} tables")
        # Parallel processing for multiple tables
        process_func = partial(process_single_table, apply_tidying=apply_tidying)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_func, all_tables))
        
        for table_key, table_df in results:
            if table_df is not None:
                result[table_key] = table_df
    else:
        print("Using sequential processing for single table")
        # Sequential processing for single table (no overhead)
        for table_info in all_tables:
            table_key, table_df = process_single_table(table_info, apply_tidying)
            if table_df is not None:
                result[table_key] = table_df

    # Ensure we return at least the keys we found during initial scan, 
    # even if tidying failed or processing returned None.
    # This prevents Kedro key mismatch errors.
    for sheet_name, table_idx, _ in all_tables:
        safe_sheet = _sanitize_name(sheet_name)
        table_key = f"{safe_sheet}_table{table_idx + 1}"
        if table_key not in result:
            print(f"Key '{table_key}' missing from result, providing empty fallback")
            result[table_key] = pd.DataFrame()
    
    # Final fallback: ensure at least one output if absolutely nothing found
    if not result:
        print("No tables found at all, returning fallback 'table1'")
        result["table1"] = pd.DataFrame()
    
    print(f"tidy_excel_data finished. Returning keys: {list(result.keys())}")
    # PartitionedDataset in wrapper.py handles persistence - no need for _last_results workaround
    
    return result

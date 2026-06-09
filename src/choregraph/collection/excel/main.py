# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Clément Baraille
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from .utils import (
    get_compatible_excel_path,
    cleanup_temp_files,
    extract_sub_table,
    get_llm_friendly_columns,
    print_table_structure,
    print_spreadsheet_map,
    format_results_for_choregraph,
    _align_column_names,
)
from .cartograph import (
    TableStructure,
    SpreadsheetMap,
    SYSTEM_PROMPT_CARTOGRAPH,
    USER_PROMPT_CARTOGRAPH,
)
from .mapper import (
    RegionMapping,
    SYSTEM_PROMPT_MAPPER as system_prompt_mapper,
    USER_PROMPT_MAPPER as user_prompt_mapper,
)
from .etl_engine import (
    compile_region_mapping_to_pipeline,
    execute_pipeline,
)
from .encoder import SpreadsheetEncoder

logger = logging.getLogger(__name__)

# Timeouts (seconds)
LLM_CALL_TIMEOUT = 120        # Max time for a single LLM call (Cartographer or Mapper)
TABLE_PROCESSING_TIMEOUT = 300  # Max time to process one table (including all retries)


# Provider plumbing — model names, provider-native kwargs, boot-order
# auto-detection and credentials — lives in `choregraph.llm_config`, which
# reads the ai_service provider yaml files (the single source of truth,
# shared with CSV characterization and the ai_service). Excel tidying uses
# the main `defaults:` profile (a richer model than CSV's cheap `minimal:`).
from ...llm_config import (
    PROVIDER_ENV_KEY as _PROVIDER_ENV_KEY,
    build_chat_model,
    resolve_api_key as _resolve_api_key,
    select_provider,
)


def _build_excel_llm(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
    model_override: Optional[str] = None,
) -> BaseChatModel:
    """Instantiate the chat model used by Cartograph + Mapper.

    Model name and provider-native kwargs (`thinking_level` for Gemini,
    `thinking={...}` for Anthropic, etc.) come from the provider yaml's
    `defaults:` profile — nothing is hardcoded here.

    `base_url` overrides the default provider endpoint (OpenAI-compatible
    proxies); `model_override` overrides the yaml model (local/custom
    endpoints carry the real model name in env).
    """
    return build_chat_model(
        provider, api_key, "defaults",
        base_url=base_url, model_override=model_override,
    )


def _invoke_llm_with_timeout(llm_structured, messages, timeout: int = LLM_CALL_TIMEOUT):
    """Invoke a structured LLM call with a timeout guard.

    Runs the synchronous ``llm_structured.invoke()`` inside a single-thread
    pool so that we can enforce a hard timeout.  If the call exceeds
    *timeout* seconds, a ``TimeoutError`` is raised.

    Uses ``shutdown(wait=False, cancel_futures=True)`` to avoid blocking
    the caller when the LLM call exceeds the timeout — the orphan thread
    will finish in the background but won't hold up the retry loop or
    the outer per-table executor.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(llm_structured.invoke, messages)
    try:
        result = future.result(timeout=timeout)
        pool.shutdown(wait=False)
        return result
    except (TimeoutError, Exception):
        future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise


def process_single_table_declarative(
    table: TableStructure,
    sheet_global_context: List[str],
    sheets_dict: Dict[str, "pd.DataFrame"],
    encoder: SpreadsheetEncoder,
    api_key: str,
    max_retries: int = 3,
    shared_column_vocabulary: List[str] = None,
    *,
    provider: str = "google_genai",
    base_url: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single table with the declarative approach.

    Pipeline:
    1. Extract sub-table from the sheet DataFrame
    2. Call the Structural Mapper (LLM) -> RegionMapping
    3. Compile ETL (deterministic) -> TransformationPipeline
    4. Execute the pipeline
    5. Retry with error feedback on failure
    """
    table_id = table.table_id
    label = table.label
    sheet_name = table.sheet_name

    if sheet_name not in sheets_dict:
        logger.warning("Sheet '%s' not found. Available: %s", sheet_name, list(sheets_dict.keys()))
        return {"table_id": table_id, "label": label, "status": "error", "message": f"Sheet '{sheet_name}' not found"}

    df_initial = sheets_dict[sheet_name]

    try:
        # STEP 1: Extract sub-table
        full_range = table.full_range.range
        df_sub_full = extract_sub_table(df_initial, full_range)

        logger.debug("Extracted sub-table %s: %s", table_id, df_sub_full.shape)

        # Prepare mapper context
        table_structure_text = print_table_structure(table)
        raw_table_md = encoder.render_range_as_markdown(sheet_name, full_range)
        columns_info = get_llm_friendly_columns(df_sub_full)
        global_ctx = "\n".join(sheet_global_context)
        local_ctx = "\n".join(table.contextual_notes) if table.contextual_notes else "None"

        # Build shared vocabulary text for the mapper prompt
        if shared_column_vocabulary:
            shared_vocab_text = "Use these canonical column names when applicable:\n" + \
                ", ".join(shared_column_vocabulary)
        else:
            shared_vocab_text = "No shared vocabulary provided."

        # STEP 2: Structural Mapper (LLM)
        from langchain_core.messages import HumanMessage, SystemMessage
        llm_mapper = _build_excel_llm(provider, api_key, base_url=base_url, model_override=model_override)
        llm_mapper_structured = llm_mapper.with_structured_output(
            schema=RegionMapping, method="json_schema"
        )

        attempts = 0
        last_errors = []

        while attempts < max_retries:
            error_feedback = ""
            if last_errors:
                error_feedback = (
                    f"\n\nPREVIOUS ATTEMPT FAILED WITH ERROR:\n{last_errors[-1]}\n\n"
                    "IMPORTANT: Analyze this error carefully and adjust your RegionMapping:\n"
                    "- If it's a KeyError, the column name doesn't match. Check the actual columns.\n"
                    "- After promote_header, column names change from Excel letters to header values.\n"
                    "- Make sure pivot_axis col_range matches actual column letters in the data.\n"
                )

            try:
                response_mapper = _invoke_llm_with_timeout(
                    llm_mapper_structured,
                    [
                        SystemMessage(content=system_prompt_mapper),
                        HumanMessage(
                            content=user_prompt_mapper.format(
                                table_structure=table_structure_text,
                                raw_table_md=raw_table_md,
                                columns_info=columns_info,
                                global_context=global_ctx,
                                local_context=local_ctx,
                                shared_vocabulary=shared_vocab_text,
                            )
                            + error_feedback
                        ),
                    ],
                )
            except TimeoutError:
                logger.warning("Mapper LLM call timed out for %s (attempt %d)", table_id, attempts + 1)
                attempts += 1
                last_errors.append(f"LLM call timed out after {LLM_CALL_TIMEOUT}s")
                continue
            except Exception as e:
                logger.warning("Mapper call failed for %s: %s", table_id, e)
                attempts += 1
                last_errors.append(str(e))
                continue

            # STEP 3: ETL Compilation (deterministic)
            try:
                pipeline = compile_region_mapping_to_pipeline(response_mapper, table, df_sub_full)
                logger.info(
                    "Compiled %d steps for %s: %s",
                    len(pipeline.steps),
                    table_id,
                    [s.function_name for s in pipeline.steps],
                )
            except Exception as e:
                logger.warning("Compilation failed for %s: %s", table_id, e)
                attempts += 1
                last_errors.append(f"Compilation error: {str(e)}")
                continue

            # STEP 4: Execute pipeline
            exec_result = execute_pipeline(pipeline, df_sub_full)

            if exec_result["status"] == "success":
                result_df = exec_result["final_df"]
                logger.info(
                    "Success for %s after %d attempt(s): %d rows x %d cols",
                    table_id, attempts + 1, result_df.shape[0], result_df.shape[1],
                )
                return {"table_id": table_id, "label": label, "status": "success", "df": result_df}
            else:
                last_errors.append(exec_result["errors"][0] if exec_result.get("errors") else "Unknown error")
                attempts += 1
                logger.info(
                    "Attempt %d failed for %s. %s",
                    attempts, table_id,
                    "Retrying..." if attempts < max_retries else "Max retries reached.",
                )

        return {"table_id": table_id, "label": label, "status": "failed", "errors": last_errors}

    except Exception as e:
        logger.error("Critical error on table %s: %s", table_id, e, exc_info=True)
        return {"table_id": table_id, "label": label, "status": "error", "message": str(e)}


def tidy_excel_data(
    path_excel: str,
    api_key: str = None,
    max_retries: int = 3,
    file_context: str = None,
    previous_table_names: List[str] = None,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entry point — process an Excel file into tidy DataFrames.

    Can be used standalone or invoked via a Kedro pipeline node.

    Args:
        path_excel: Path to the Excel file (.xlsx, .xls, .ods).
        api_key: Provider API key.  If ``None``, resolved from the provider
                 env var or ``.env`` file (see ``_resolve_api_key``).
        max_retries: Max LLM retry attempts per table on pipeline failure.
        file_context: Optional string listing sibling filenames being
                      processed alongside this file.  Helps the Cartographer
                      LLM produce distinctive, context-aware table labels.
        previous_table_names: Optional list of table names from a previous
                      processing run (e.g. reupload).  The Cartographer LLM
                      will try to reuse these labels when the detected tables
                      match structurally, preserving compatibility with
                      existing choregraph/specifications.xml references.
        provider: LLM provider name. When ``None`` (e.g. the kedro file
                  pipeline), the active provider is auto-detected from the
                  configured env vars — the same boot order as the rest of
                  the stack. ai_service passes it explicitly.
        base_url / model_override: forwarded to the chat model; auto-filled
                  from the detected provider when ``provider`` is ``None``.

    Returns:
        Dict mapping ``table_id`` -> ``pd.DataFrame`` for each
        successfully processed table.
    """
    # Explicit provider (ai_service) wins; otherwise auto-detect the
    # configured provider so the file-pipeline node uses the user's choice.
    if provider is None:
        sel = select_provider()
        if sel is None:
            raise RuntimeError(
                "No LLM provider configured for Excel tidying. Set a provider "
                "key/endpoint (GOOGLE_API_KEY, OPENAI_API_KEY, OLLAMA_BASE_URL+"
                "MODEL, …) — run the setup wizard."
            )
        provider = sel["provider"]
        api_key = api_key or sel["api_key"]
        base_url = base_url or sel["base_url"]
        model_override = model_override or sel["model_override"]

    api_key = _resolve_api_key(api_key, provider=provider)
    if not api_key:
        env_var = _PROVIDER_ENV_KEY.get(provider, "GOOGLE_API_KEY")
        raise RuntimeError(
            f"API key for provider {provider!r} not found. "
            f"Pass api_key parameter or set {env_var} env var."
        )

    xlsx_path = get_compatible_excel_path(path_excel)

    try:
        # STEP 1+2: Encode and build DataFrames from the same openpyxl workbook
        encoder = SpreadsheetEncoder(xlsx_path)
        encoding = encoder.encode_all_sheets()
        sheets_dict = encoder.get_dataframes()

        logger.info("Loaded %d sheet(s): %s", len(sheets_dict), list(sheets_dict.keys()))

        # STEP 3: Cartography (with timeout)
        logger.info("Cartographing the file...")

        filename = Path(xlsx_path).stem

        file_context_section = ""
        if file_context:
            file_context_section = (
                "FILE CONTEXT (other files being processed alongside this one):\n"
                f"{file_context}\n"
            )

        if previous_table_names:
            names_list = ", ".join(f'"{n}"' for n in previous_table_names)
            file_context_section += (
                "\nPREVIOUS TABLE NAMES (from a prior processing of this same file):\n"
                f"{names_list}\n"
                "IMPORTANT: You MUST reuse these exact labels for tables that match "
                "structurally (same sheet, same data area). Only use new labels for "
                "genuinely new tables. This preserves downstream compatibility.\n"
            )

        from langchain_core.messages import HumanMessage, SystemMessage
        llm_cartograph = _build_excel_llm(provider, api_key, base_url=base_url, model_override=model_override)
        llm_cartograph_structured = llm_cartograph.with_structured_output(
            schema=SpreadsheetMap, method="json_schema"
        )

        try:
            response_cartograph = _invoke_llm_with_timeout(
                llm_cartograph_structured,
                [
                    SystemMessage(content=SYSTEM_PROMPT_CARTOGRAPH),
                    HumanMessage(content=USER_PROMPT_CARTOGRAPH.format(
                        encoding=encoding,
                        filename=filename,
                        file_context=file_context_section,
                    )),
                ],
            )
        except TimeoutError:
            raise RuntimeError(
                f"Cartographer LLM call timed out after {LLM_CALL_TIMEOUT}s. "
                "The file may be too large or the API is unresponsive."
            )

        logger.info(print_spreadsheet_map(response_cartograph))

        # STEP 4: Process each table with per-table timeout
        num_tables = len(response_cartograph.detected_tables)
        shared_vocab = response_cartograph.shared_column_vocabulary
        logger.info("Processing %d table(s)...", num_tables)
        if shared_vocab:
            logger.info("Shared column vocabulary: %s", shared_vocab)

        final_results = []

        executor = ThreadPoolExecutor(max_workers=min(max(num_tables, 1), 4))
        try:
            futures = {
                executor.submit(
                    process_single_table_declarative,
                    table=table,
                    sheet_global_context=response_cartograph.sheet_global_context,
                    sheets_dict=sheets_dict,
                    encoder=encoder,
                    api_key=api_key,
                    max_retries=max_retries,
                    shared_column_vocabulary=shared_vocab,
                    provider=provider,
                    base_url=base_url,
                    model_override=model_override,
                ): table
                for table in response_cartograph.detected_tables
            }

            try:
                for future in as_completed(futures, timeout=TABLE_PROCESSING_TIMEOUT * max(num_tables, 1)):
                    table = futures[future]
                    try:
                        result = future.result(timeout=TABLE_PROCESSING_TIMEOUT)
                    except TimeoutError:
                        logger.error("Table %s timed out after %ds", table.table_id, TABLE_PROCESSING_TIMEOUT)
                        result = {
                            "table_id": table.table_id,
                            "label": table.label,
                            "status": "error",
                            "message": f"Processing timed out after {TABLE_PROCESSING_TIMEOUT}s",
                        }
                    except Exception as e:
                        logger.error("Unhandled error processing table %s: %s", table.table_id, e)
                        result = {"table_id": table.table_id, "label": table.label, "status": "error", "message": str(e)}
                    final_results.append(result)
            except TimeoutError:
                logger.error("Global timeout reached — cancelling remaining futures")
                for future in futures:
                    future.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        # Check for futures that never completed (global timeout)
        completed_ids = {r["table_id"] for r in final_results}
        for table in response_cartograph.detected_tables:
            if table.table_id not in completed_ids:
                logger.error("Table %s was never completed (global timeout)", table.table_id)
                final_results.append({
                    "table_id": table.table_id,
                    "label": table.label,
                    "status": "error",
                    "message": "Global processing timeout — table was never completed",
                })

        formatted = format_results_for_choregraph(final_results)
        return _align_column_names(formatted)

    finally:
        cleanup_temp_files()

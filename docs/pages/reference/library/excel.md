# Excel Transforms

LLM-assisted Excel data tidying. Encodes each spreadsheet sheet into a compact
textual form, then uses an LLM cartographer to locate the tables and a
structural mapper to restructure each one into a tidy DataFrame. The provider
and model are auto-detected via `choregraph.llm_config` (shared with the
ai_service); ai_service passes them explicitly.

::: choregraph.collection.excel.main
    options:
      members_order: source
      show_source: true
      allow_inspection: false

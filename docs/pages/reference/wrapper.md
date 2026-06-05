# Wrapper

Generates a complete Kedro project directory structure on disk from a `ChoregraphSpec`.
Produces `pyproject.toml`, `settings.py`, `catalog.yml`, and `pipeline_registry.py` so that
Kedro sessions can execute the pipeline. Acts as the Single Source of Truth bridge between
Choregraph's XML model and Kedro's file-based configuration.

::: choregraph.wrapper
    options:
      members_order: source
      show_source: true

# Core Transforms

The core transform function library — 50 DataFrame operations registered in `TRANSFORM_REGISTRY`.
These functions are used by the [Builder](../builder.md) when constructing Kedro pipeline nodes
from the XML specification.

All functions follow a consistent pattern: accept a DataFrame (and parameters), return a
DataFrame (or scalar). Functions with `return_mask=True` support return both a filtered result
and a boolean mask.

::: choregraph.library
    options:
      members_order: source
      show_source: true
      filters:
        - "!^_"
        - "!^TRANSFORM_REGISTRY"

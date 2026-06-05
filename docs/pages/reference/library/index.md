# Transform Library

Choregraph provides 48 built-in transform functions registered in `TRANSFORM_REGISTRY`.
The [Builder](../builder.md) looks up functions by name when constructing Kedro pipeline nodes
from the XML specification.

## Categories

### Filtering

| Function | Description |
|----------|-------------|
| `filter_less_than` | Keep rows where column < value |
| `filter_greater_than` | Keep rows where column > value |
| `filter_equal` | Keep rows where column == value |
| `filter_not_equal` | Keep rows where column != value |
| `filter_in_range` | Keep rows where min_value ≤ column ≤ max_value |

### Top / Bottom

| Function | Description |
|----------|-------------|
| `get_top_n` | Top N rows by column value |
| `get_top_percentage` | Top percentage of rows by column value |
| `get_bottom_n` | Bottom N rows by column value |
| `get_bottom_percentage` | Bottom percentage of rows by column value |

### Aggregation

| Function | Description |
|----------|-------------|
| `aggregate_mean` | Mean of numeric columns, optionally grouped |
| `aggregate_count` | Row count, optionally grouped |
| `aggregate_sum` | Sum of numeric columns, optionally grouped |
| `aggregate_median` | Median of numeric columns, optionally grouped |

### Min / Max

| Function | Description |
|----------|-------------|
| `calculate_min` | Minimum value from a column or list |
| `calculate_max` | Maximum value from a column or list |

### Column Operations

| Function | Description |
|----------|-------------|
| `select_columns` | Keep only specified columns |
| `drop_columns` | Remove specified columns |
| `rename_column` | Rename a single column |
| `add_label` | Add a constant-value label column |

### Row Operations

| Function | Description |
|----------|-------------|
| `slice_rows` | Keep a range of rows by index |
| `sort_values` | Sort by one or more columns |
| `sample_rows` | Random sample of rows |
| `count_rows` | Count rows, optionally grouped |

### Calculations

| Function | Description |
|----------|-------------|
| `calc_distance` | Euclidean distance from a reference point |
| `calc_ratio` | Ratio between two columns |
| `arithmetic_op` | Arithmetic between columns or constants (+, -, *, /) |

### Reshaping

| Function | Description |
|----------|-------------|
| `melt` | Unpivot columns into rows |
| `hierarchical_rollup` | Aggregate hierarchical data at multiple levels |
| `concat_partitions` | Concatenate partitioned datasets |

### Advanced

| Function | Description |
|----------|-------------|
| `normalize_column` | Min-max or z-score normalization |
| `discretize` | Bin continuous values (uniform or quantile) |
| `execute_code` | Execute user-provided Python code on a DataFrame |

### Time Series

| Function | Description |
|----------|-------------|
| `extract_date_part` | Extract year, month, day, etc. from datetime column |
| `rolling_statistics` | Rolling window aggregation (mean, sum, etc.) |
| `lag_lead` | Shift column values forward or backward |
| `offset_datetime` | Offset datetime column by a time delta |
| `forecast_time_series` | Simple time series forecasting |

### Multi-Input

| Function | Description |
|----------|-------------|
| `join` | Join multiple DataFrames on a key |
| `union` | Vertically stack (concatenate) DataFrames |

### JSON

| Function | Description |
|----------|-------------|
| `flatten_json` | Flatten nested JSON structures |

### Image

| Function | Description |
|----------|-------------|
| `image_to_dataframe` | Convert image pixels to a DataFrame |
| `extract_channel` | Extract a single color channel from image data |
| `image_metadata` | Extract image metadata (dimensions, format, etc.) |

### Geolocation

| Function | Description |
|----------|-------------|
| `geocode_location` | Enrich with lat/lon from location names |
| `get_country_contours` | Join country boundary geometries |

### NLP

| Function | Description |
|----------|-------------|
| `nlp_binarize_labels_auto` | Unsupervised multi-label binarization |
| `nlp_binarize_labels_hinted` | Supervised binarization with fuzzy hint matching |

### Excel

| Function | Description |
|----------|-------------|
| `tidy_excel_data` | LLM-assisted multi-table Excel tidying |

---

## Detailed Reference

- [Core Transforms](core.md) — Filtering, aggregation, column/row ops, calculations, advanced, multi-input, JSON
- [Excel Transforms](excel.md) — LLM-assisted Excel processing
- [Geo Collection](geo.md) — Geocoding and country boundaries
- [NLP Collection](nlp.md) — Text label binarization
